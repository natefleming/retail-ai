"""Tests confirming ``build_auth_policy`` never adds OTEL tables.

MLflow tracing writers on Model Serving endpoints use a separate
authentication path that ``agents.deploy(resources=…)`` auto-auth does
not cover. Declaring OTEL tables in the model's ``SystemAuthPolicy``
therefore has no effect on trace persistence, so we intentionally omit
them. See ``TraceLocationModel.as_resources`` docstring.
"""

from __future__ import annotations

import pytest
from mlflow.models.resources import DatabricksTable

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
from dao_ai.providers.databricks import build_auth_policy


def _build_config(trace_location) -> AppConfig:
    schema = SchemaModel(catalog_name="cat", schema_name="sch")
    return AppConfig(
        app=AppModel(
            name="test-app",
            registered_model=RegisteredModelModel(schema=schema, name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=LLMModel(name="test"))
            ),
            agents=[AgentModel(name="test", model=LLMModel(name="test"))],
            trace_location=trace_location,
        ),
    )


def _otel_table_names(policy_resources) -> set[str]:
    return {
        r.name
        for r in policy_resources
        if isinstance(r, DatabricksTable) and "_otel_" in r.name
    }


@pytest.mark.unit
def test_no_otel_tables_declared_with_trace_location() -> None:
    cfg = _build_config(
        TraceLocationModel(
            schema=SchemaModel(catalog_name="cat", schema_name="sch"),
            warehouse="wh",
        )
    )
    policy = build_auth_policy(cfg)
    assert _otel_table_names(policy.system_auth_policy.resources) == set()


@pytest.mark.unit
def test_no_otel_tables_declared_with_explicit_prefix() -> None:
    cfg = _build_config(
        TraceLocationModel(
            schema=SchemaModel(catalog_name="cat", schema_name="sch"),
            warehouse="wh",
            table_prefix="my_agent",
        )
    )
    policy = build_auth_policy(cfg)
    assert _otel_table_names(policy.system_auth_policy.resources) == set()


@pytest.mark.unit
def test_no_trace_location_still_empty_otel_set() -> None:
    cfg = _build_config(trace_location=None)
    policy = build_auth_policy(cfg)
    assert _otel_table_names(policy.system_auth_policy.resources) == set()


@pytest.mark.unit
def test_build_auth_policy_single_arg_signature() -> None:
    """Guard against re-threading ``experiment_id`` back in. The trace-table
    declarations are empirically ineffective; the deploy site should not
    have to know the experiment_id just to build the auth policy."""
    import inspect

    sig = inspect.signature(build_auth_policy)
    assert list(sig.parameters) == ["config"]
