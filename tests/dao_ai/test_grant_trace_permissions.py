"""Tests for ``_grant_trace_permissions_to_principal``.

The helper must issue the exact set of UC grants the Databricks docs
prescribe for MLflow trace persistence:

    * USE_CATALOG on the catalog
    * USE_SCHEMA on the schema
    * SELECT + MODIFY on each of the four OTEL tables

Docs: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog

Implementation uses the raw REST endpoint (via ``WorkspaceClient.api_client.do``)
rather than the typed ``grants.update()`` SDK method because the latter
serializes ``SecurableType.TABLE`` incorrectly on some SDK versions
(``Invalid input: SECURABLETYPE.TABLE is not a valid securable type``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_helper_grants_all_documented_privileges() -> None:
    from dao_ai.providers.databricks import _grant_trace_permissions_to_principal

    with patch("databricks.sdk.WorkspaceClient") as mock_wc:
        mock_client = MagicMock()
        mock_wc.return_value = mock_client
        _grant_trace_permissions_to_principal(
            principal="sp-uuid",
            catalog_name="cat",
            schema_name="sch",
            table_prefix="exp123",
        )

    # 1 catalog + 1 schema + 4 tables = 6 REST calls.
    assert mock_client.api_client.do.call_count == 6

    calls = mock_client.api_client.do.call_args_list
    # Catalog
    method, path = calls[0].args[0], calls[0].args[1]
    body = calls[0].kwargs["body"]
    assert method == "PATCH"
    assert path == "/api/2.1/unity-catalog/permissions/catalog/cat"
    assert body["changes"] == [{"principal": "sp-uuid", "add": ["USE_CATALOG"]}]
    # Schema
    body = calls[1].kwargs["body"]
    assert (
        calls[1].args[1] == "/api/2.1/unity-catalog/permissions/schema/cat.sch"
    )
    assert body["changes"] == [{"principal": "sp-uuid", "add": ["USE_SCHEMA"]}]
    # Tables
    table_calls = calls[2:]
    paths = {c.args[1] for c in table_calls}
    assert paths == {
        f"/api/2.1/unity-catalog/permissions/table/cat.sch.exp123_otel_{s}"
        for s in ("spans", "logs", "metrics", "annotations")
    }
    for c in table_calls:
        assert c.kwargs["body"]["changes"] == [
            {"principal": "sp-uuid", "add": ["SELECT", "MODIFY"]}
        ]


@pytest.mark.unit
def test_helper_survives_individual_grant_failures() -> None:
    from dao_ai.providers.databricks import _grant_trace_permissions_to_principal

    call_count = 0

    def flaky_do(*args: Any, **kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated per-table failure")

    with patch("databricks.sdk.WorkspaceClient") as mock_wc:
        mock_client = MagicMock()
        mock_client.api_client.do.side_effect = flaky_do
        mock_wc.return_value = mock_client
        _grant_trace_permissions_to_principal(
            principal="sp-uuid",
            catalog_name="cat",
            schema_name="sch",
            table_prefix="exp",
        )

    # All 6 calls were attempted despite one failing.
    assert mock_client.api_client.do.call_count == 6


@pytest.mark.unit
def test_resolve_prefix_prefers_explicit_over_experiment_id() -> None:
    from dao_ai.config import (
        AgentModel,
        AppConfig,
        AppModel,
        LLMModel,
        OrchestrationModel,
        RegisteredModelModel,
        SchemaModel,
        SupervisorModel,
        TraceLocationModel,
    )
    from dao_ai.providers.databricks import _resolve_trace_table_prefix

    schema = SchemaModel(catalog_name="cat", schema_name="sch")
    cfg = AppConfig(
        app=AppModel(
            name="app",
            registered_model=RegisteredModelModel(schema=schema, name="m"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=LLMModel(name="t"))
            ),
            agents=[AgentModel(name="ag", model=LLMModel(name="t"))],
            trace_location=TraceLocationModel(
                schema=schema, warehouse="wh", table_prefix="my_agent"
            ),
        ),
    )
    assert _resolve_trace_table_prefix(cfg, "9999") == "my_agent"


@pytest.mark.unit
def test_resolve_prefix_falls_back_to_experiment_id() -> None:
    from dao_ai.config import (
        AgentModel,
        AppConfig,
        AppModel,
        LLMModel,
        OrchestrationModel,
        RegisteredModelModel,
        SchemaModel,
        SupervisorModel,
        TraceLocationModel,
    )
    from dao_ai.providers.databricks import _resolve_trace_table_prefix

    schema = SchemaModel(catalog_name="cat", schema_name="sch")
    cfg = AppConfig(
        app=AppModel(
            name="app",
            registered_model=RegisteredModelModel(schema=schema, name="m"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=LLMModel(name="t"))
            ),
            agents=[AgentModel(name="ag", model=LLMModel(name="t"))],
            trace_location=TraceLocationModel(schema=schema, warehouse="wh"),
        ),
    )
    assert _resolve_trace_table_prefix(cfg, "9999") == "9999"


@pytest.mark.unit
def test_resolve_prefix_raises_when_neither_source_available() -> None:
    from dao_ai.config import (
        AgentModel,
        AppConfig,
        AppModel,
        LLMModel,
        OrchestrationModel,
        RegisteredModelModel,
        SchemaModel,
        SupervisorModel,
        TraceLocationModel,
    )
    from dao_ai.providers.databricks import _resolve_trace_table_prefix

    schema = SchemaModel(catalog_name="cat", schema_name="sch")
    cfg = AppConfig(
        app=AppModel(
            name="app",
            registered_model=RegisteredModelModel(schema=schema, name="m"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=LLMModel(name="t"))
            ),
            agents=[AgentModel(name="ag", model=LLMModel(name="t"))],
            trace_location=TraceLocationModel(schema=schema, warehouse="wh"),
        ),
    )
    with pytest.raises(ValueError, match="cannot resolve OTEL trace-table prefix"):
        _resolve_trace_table_prefix(cfg, None)
