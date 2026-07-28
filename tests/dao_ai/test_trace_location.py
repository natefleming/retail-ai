"""Tests for :class:`dao_ai.config.TraceLocationModel`.

``as_resources`` returns ``[]`` — MLflow trace persistence from Model
Serving endpoints created via ``agents.deploy`` is a documented
Databricks platform limitation. See ``TraceLocationModel.as_resources``
docstring for the empirical finding. On Databricks Apps, trace
persistence works via an explicit post-deploy grant against the App's
own runtime SP.
"""

from __future__ import annotations

import pytest

from dao_ai.config import SchemaModel, TraceLocationModel


@pytest.fixture
def loc_no_prefix() -> TraceLocationModel:
    return TraceLocationModel(
        schema=SchemaModel(catalog_name="cat", schema_name="sch"),
        warehouse="abc123",
    )


@pytest.fixture
def loc_with_prefix() -> TraceLocationModel:
    return TraceLocationModel(
        schema=SchemaModel(catalog_name="cat", schema_name="sch"),
        warehouse="abc123",
        table_prefix="my_agent",
    )


@pytest.mark.unit
def test_as_resources_always_empty(
    loc_no_prefix: TraceLocationModel, loc_with_prefix: TraceLocationModel
) -> None:
    """Neither prefix nor experiment_id changes the outcome — MS trace
    tables are unreachable via auth_policy declarations by design."""
    assert list(loc_no_prefix.as_resources()) == []
    assert list(loc_with_prefix.as_resources()) == []


@pytest.mark.unit
def test_resolved_table_prefix_when_set(loc_with_prefix: TraceLocationModel) -> None:
    assert loc_with_prefix.resolved_table_prefix == "my_agent"


@pytest.mark.unit
def test_resolved_table_prefix_when_unset(loc_no_prefix: TraceLocationModel) -> None:
    assert loc_no_prefix.resolved_table_prefix is None


@pytest.mark.unit
def test_catalog_and_schema_names(loc_no_prefix: TraceLocationModel) -> None:
    assert loc_no_prefix.catalog_name == "cat"
    assert loc_no_prefix.schema_name == "sch"


# =============================================================================
# F1: MLFLOW_TRACING_SQL_WAREHOUSE_ID env-var injection guard
# =============================================================================
# set_databricks_env_vars must never inject a None warehouse id. A warehouse
# given by NAME resolves lazily (ensure_resolved → live API), so warehouse_id
# is None at config-load; injecting it poisons environment_vars and the Model
# Serving pyfunc's schema re-validation rejects the None. An id-based warehouse
# resolves immediately and must still be injected.

from dao_ai.config import AppConfig  # noqa: E402

_BASE = """\
resources:
  warehouses:
    w: &w
{warehouse}
  models:
    m: &m
      name: databricks-claude-sonnet-4-5
schemas:
  s: &s
    catalog_name: cat
    schema_name: sch
agents:
  a: &a
    name: agent_one
    model: *m
    prompt: hi
app:
  name: trace_wh_test
  registered_model:
    schema: *s
    name: trace_wh_test
  trace_location:
    schema: *s
    warehouse: *w
  agents: [*a]
"""


def _env(tmp_path, warehouse: str) -> dict:
    p = tmp_path / "c.yaml"
    p.write_text(_BASE.format(warehouse=warehouse))
    cfg = AppConfig.from_file(str(p), initialize=False)
    return cfg.app.environment_vars


@pytest.mark.unit
def test_name_based_warehouse_does_not_inject_none(tmp_path) -> None:
    """A NAME-based warehouse must NOT put MLFLOW_TRACING_SQL_WAREHOUSE_ID in
    the env at load time (it would be None). The key is simply absent."""
    env = _env(tmp_path, '      name: "Serverless Starter Warehouse"')
    assert "MLFLOW_TRACING_SQL_WAREHOUSE_ID" not in env


@pytest.mark.unit
def test_id_based_warehouse_injects_id(tmp_path) -> None:
    """An ID-based warehouse resolves immediately and IS injected."""
    env = _env(tmp_path, "      warehouse_id: d58e5fb998498840")
    assert env.get("MLFLOW_TRACING_SQL_WAREHOUSE_ID") == "d58e5fb998498840"
