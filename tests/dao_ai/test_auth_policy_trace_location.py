"""Tests for the OTEL trace-table contribution to the auth policy.

When ``config.app.trace_location`` is set and the deploy site supplies an
``experiment_id``, :func:`build_auth_policy` should include one
``DatabricksTable`` entry per OTEL table that MLflow materializes
(``_otel_{spans,logs,metrics,annotations}``) — so ``agents.deploy``
auto-grants the Model Serving SP without a manual ``GRANT`` step.
"""

from __future__ import annotations

from typing import Optional

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

_OTEL_SUFFIXES: tuple[str, ...] = ("annotations", "logs", "metrics", "spans")


def _build_config(trace_location: Optional[TraceLocationModel]) -> AppConfig:
    """Minimal AppConfig with optional trace_location attached to AppModel."""
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


@pytest.fixture
def cfg_no_prefix() -> AppConfig:
    return _build_config(
        TraceLocationModel(
            schema=SchemaModel(catalog_name="cat", schema_name="sch"),
            warehouse="wh",
        )
    )


@pytest.fixture
def cfg_with_prefix() -> AppConfig:
    return _build_config(
        TraceLocationModel(
            schema=SchemaModel(catalog_name="cat", schema_name="sch"),
            warehouse="wh",
            table_prefix="my_agent",
        )
    )


@pytest.mark.unit
def test_includes_otel_tables_when_experiment_id_passed(
    cfg_no_prefix: AppConfig,
) -> None:
    policy = build_auth_policy(cfg_no_prefix, experiment_id="999")
    assert _otel_table_names(policy.system_auth_policy.resources) == {
        f"cat.sch.999_otel_{s}" for s in _OTEL_SUFFIXES
    }


@pytest.mark.unit
def test_omits_otel_tables_when_no_experiment_id_and_no_prefix(
    cfg_no_prefix: AppConfig,
) -> None:
    policy = build_auth_policy(cfg_no_prefix, experiment_id=None)
    assert _otel_table_names(policy.system_auth_policy.resources) == set()


@pytest.mark.unit
def test_prefers_explicit_table_prefix_over_experiment_id(
    cfg_with_prefix: AppConfig,
) -> None:
    policy = build_auth_policy(cfg_with_prefix, experiment_id="999")
    assert _otel_table_names(policy.system_auth_policy.resources) == {
        f"cat.sch.my_agent_otel_{s}" for s in _OTEL_SUFFIXES
    }


@pytest.mark.unit
def test_no_trace_location_emits_no_otel_tables() -> None:
    cfg = _build_config(trace_location=None)
    policy = build_auth_policy(cfg, experiment_id="999")
    assert _otel_table_names(policy.system_auth_policy.resources) == set()


@pytest.mark.unit
def test_default_signature_remains_callable_without_experiment_id() -> None:
    # Existing callers passing only `config` must not break.
    cfg = AppConfig()
    policy = build_auth_policy(cfg)
    assert list(policy.system_auth_policy.resources) == []
    assert list(policy.user_auth_policy.api_scopes) == []
