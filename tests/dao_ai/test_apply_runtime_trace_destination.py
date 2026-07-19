"""Unit tests for apply_runtime_trace_destination's UC table-name qualification.

Regression guard for the Apps UC-trace-span drop: dao-ai must set the
``UnityCatalog`` destination's ``_otel_*_table_name`` fields to FULLY-QUALIFIED
three-level names. ``UnityCatalog.full_otel_spans_table_name`` returns those
private fields verbatim (it does not auto-qualify like ``UCSchemaLocation``), so
``mlflow.tracing.utils.get_active_spans_table_name()`` — which the OTEL span
exporter consults — returns ``None`` (silent skip) when they are unset, and the
trace server rejects a bare table name with ``Invalid full table name``.
"""

from __future__ import annotations

import pytest

from mlflow.tracing.provider import _MLFLOW_TRACE_USER_DESTINATION
from mlflow.tracing.utils import get_active_spans_table_name

from dao_ai.config import AppConfig, AppModel, SchemaModel, TraceLocationModel
from dao_ai.providers.databricks import apply_runtime_trace_destination


def _config(trace_location: TraceLocationModel | None) -> AppConfig:
    # ``model_construct`` bypasses AppModel's "at least one agent" validator —
    # these tests exercise only the trace-destination logic, not agent wiring.
    app = AppModel.model_construct(name="test_app", trace_location=trace_location)
    return AppConfig.model_construct(app=app)


def _config_with_prefix(prefix: str | None) -> AppConfig:
    schema = SchemaModel(catalog_name="cat", schema_name="sch")
    tl = TraceLocationModel(schema=schema, warehouse="wh-123", table_prefix=prefix)
    return _config(tl)


@pytest.mark.unit
class TestApplyRuntimeTraceDestination:
    def teardown_method(self) -> None:
        _MLFLOW_TRACE_USER_DESTINATION.set(None)

    def test_prefixed_sets_fully_qualified_spans_table(self) -> None:
        apply_runtime_trace_destination(_config_with_prefix("myprefix"))
        dest = _MLFLOW_TRACE_USER_DESTINATION.get()
        assert dest is not None
        # The exporter reads this; it must be the 3-level name, not bare.
        assert dest.full_otel_spans_table_name == "cat.sch.myprefix_otel_spans"
        assert get_active_spans_table_name() == "cat.sch.myprefix_otel_spans"

    def test_no_prefix_clears_contextvar_for_experiment_resolver(self) -> None:
        # Without a table_prefix, dao-ai clears the ContextVar so MLflow's
        # _resolve_experiment_uc_location fallback computes the experiment-id
        # prefix from the tracking store.
        apply_runtime_trace_destination(_config_with_prefix(None))
        assert _MLFLOW_TRACE_USER_DESTINATION.get() is None

    def test_no_trace_location_is_noop(self) -> None:
        _MLFLOW_TRACE_USER_DESTINATION.set(None)
        apply_runtime_trace_destination(_config(None))
        assert _MLFLOW_TRACE_USER_DESTINATION.get() is None
