"""FEVM integration tests for the dynamic VS tool schema on the
``retail_consumer_goods.hardware_store.products_index`` index.

Purpose: prove the hallucination fix isn't specific to the commerce_swarm
config. The archived regression trace ``fc785d795b77675ac0e42fe5296b523a``
was on ``agent-commerce-super-dao``; here we drive the same class of
regression against a completely different config, catalog, schema, and
column set — same fix, same outcome.

Prompt shapes are grounded in the live source-table data
(``retail_consumer_goods.hardware_store.products``) so the filters match
what an actual user would trigger: real brands (DEWALT, CRAFTSMAN,
HILLMAN), real ``merchandise_class`` values (WRENCHES, ELECTRICAL FITTINGS).

To run:
    uv run pytest -m integration tests/dao_ai/test_vector_search_hardware_store_fevm.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import pytest
from pydantic import ValidationError

from dao_ai.config import (
    AiSearchRetrieverModel,
    IndexModel,
    SchemaModel,
    SearchParametersModel,
    VectorSearchEndpoint,
    VectorStoreModel,
)
from dao_ai.tools.vector_search import create_vector_search_tool

TEST_PROFILE = os.getenv("DAO_AI_TEST_PROFILE", "fevm")


def _has_fevm_profile() -> bool:
    if os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"):
        return True
    cfg = os.path.expanduser("~/.databrickscfg")
    if not os.path.exists(cfg):
        return False
    try:
        with open(cfg) as f:
            return f"[{TEST_PROFILE}]" in f.read()
    except Exception:
        return False


SKIP_MSG = (
    f"Requires DATABRICKS_CONFIG_PROFILE={TEST_PROFILE} in ~/.databrickscfg "
    "or DATABRICKS_HOST + DATABRICKS_TOKEN env vars."
)

# Columns declared in examples/99_complete_applications/hardware_store/hardware_store.yaml.
# The live products_index has these exact columns.
HARDWARE_STORE_COLUMNS = [
    "product_id",
    "sku",
    "upc",
    "brand_name",
    "product_name",
    "merchandise_class",
    "class_cd",
    "description",
]


def _hardware_store_retriever(
    *, columns: list[str] | None = HARDWARE_STORE_COLUMNS
) -> AiSearchRetrieverModel:
    """Build a retriever mirroring ``hardware_store.yaml`` — same index,
    endpoint, primary_key, and (by default) column set."""
    schema = SchemaModel(
        catalog_name="retail_consumer_goods",
        schema_name="hardware_store",
    )
    return AiSearchRetrieverModel(
        vector_store=VectorStoreModel(
            index=IndexModel(name="products_index", schema=schema),
            endpoint=VectorSearchEndpoint(name="dbdemos_vs_endpoint"),
            primary_key="product_id",
            embedding_source_column="description",
            columns=columns,
        ),
        search_parameters=SearchParametersModel(num_results=3),
    )


@pytest.fixture(autouse=True)
def _use_fevm_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route all workspace calls through the fevm profile."""
    for var in ("DATABRICKS_TOKEN", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", TEST_PROFILE)


@pytest.fixture
def local_tracing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Enable local MLflow tracing so we can inspect the tool-call span
    without depending on a workspace experiment. Autolog LangChain so the
    dao-ai VS tool emits a proper span with inputs/outputs.
    """
    monkeypatch.setenv("MLFLOW_TRACE_SAMPLING_RATIO", "1")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mlflow.set_experiment("test-hw-vs-tool-schema")
    mlflow.tracing.enable()
    mlflow.langchain.autolog(run_tracer_inline=True)
    try:
        yield
    finally:
        mlflow.langchain.autolog(disable=True)
        mlflow.tracing.disable()


def _tool_call_span_inputs(trace) -> dict:
    """Return the VS tool's span inputs from an MLflow trace.

    Assumes exactly one tool-call span; raises AssertionError otherwise.
    """
    assert trace is not None, "MLflow trace not found"
    tool_spans = [
        s
        for s in trace.data.spans
        if s.attributes.get("mlflow.spanType") == "TOOL"
        or (s.name and "hw_search" in s.name)
    ]
    assert tool_spans, (
        f"no tool span in trace {trace.info.request_id}; "
        f"spans: {[s.name for s in trace.data.spans]}"
    )
    span = tool_spans[0]
    return span.inputs or {}


@pytest.mark.integration
@pytest.mark.skipif(not _has_fevm_profile(), reason=SKIP_MSG)
class TestHardwareStoreDynamicSchema:
    """Hallucination-proof filter enum, verified on a live FEVM index that
    is NOT the one the regression trace was captured against."""

    def test_dynamic_schema_hallucination_blocked(self) -> None:
        """The archetypal hallucination — LLM emits a column key that
        isn't on the index — must be rejected by pydantic before the VS
        API is ever called. Same failure mode as the commerce_swarm
        regression, on a different config."""
        retriever = _hardware_store_retriever()
        tool = create_vector_search_tool(retriever=retriever, name="hw_search")

        # ``product_name`` is the real column — LLM would hallucinate
        # ``name``. That must be pre-empted at schema-validation time.
        with pytest.raises(ValidationError):
            tool.invoke(
                {
                    "query": "screwdriver set",
                    "filters": [{"key": "name NOT LIKE", "value": "%hammer%"}],
                }
            )

    def test_valid_filter_reaches_index_live_with_trace_assertions(
        self, local_tracing
    ) -> None:
        """End-to-end: valid filter shape reaches the VS API and the
        MLflow trace records the exact ``filters[].key`` values we sent.
        Asserts (a) no ``ValidationError``, (b) no VS API "not present in
        index" event, (c) the trace span's inputs contain the filter we
        sent (proves the tool call is what actually reached the API)."""
        retriever = _hardware_store_retriever()
        tool = create_vector_search_tool(retriever=retriever, name="hw_search")

        expected_filter_key = "brand_name LIKE"
        with mlflow.start_span(name="hw_test_outer") as outer_span:
            trace_id = outer_span.request_id
            result = tool.invoke(
                {
                    "query": "wrench",
                    "filters": [
                        {"key": expected_filter_key, "value": "%"},
                    ],
                }
            )

        # (a) No exception + non-error content.
        content = result.content if hasattr(result, "content") else result
        if isinstance(content, str):
            assert "not present in index" not in content, content
            if content.startswith("["):
                json.loads(content)

        # (b) The trace records the tool call with the exact filter key.
        trace = mlflow.get_trace(trace_id)
        span_inputs = _tool_call_span_inputs(trace)
        # Tool inputs can be nested under a ToolCall wrapper — search the
        # inputs JSON for the expected filter key.
        inputs_blob = json.dumps(span_inputs)
        assert expected_filter_key in inputs_blob, (
            f"expected filter key {expected_filter_key!r} in span inputs; "
            f"got {inputs_blob[:400]}"
        )

        # (c) No span carries a VS API "not present in index" event.
        for span in trace.data.spans:
            for event in getattr(span, "events", []) or []:
                event_name = getattr(event, "name", "") or ""
                event_attrs = getattr(event, "attributes", {}) or {}
                joined = event_name + " " + json.dumps(event_attrs)
                assert "not present in index" not in joined, (
                    f"span {span.name} carries a not-present-in-index event: {joined}"
                )

    def test_yaml_column_intersection_dropped_with_warning(self) -> None:
        """YAML declares a bogus column not on the live index → dropped
        from the Literal enum before the tool is built. The LLM cannot
        emit a filter using it. Regression guard for the WARN+drop path
        against a real describe() payload."""
        retriever = _hardware_store_retriever(
            columns=[
                "product_id",
                "product_name",
                "brand_name",
                "nonexistent_column_xyz",
            ]
        )
        tool = create_vector_search_tool(retriever=retriever, name="hw_search")

        enum = tool.args_schema.model_json_schema()["$defs"]["DynamicFilterItem"][
            "properties"
        ]["key"]["enum"]
        assert "product_name" in enum
        assert "brand_name" in enum
        assert not any(k.startswith("nonexistent_column_xyz") for k in enum), (
            f"bogus YAML column leaked into enum: {enum!r}"
        )
