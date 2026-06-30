"""Tests for :class:`dao_ai.config.TraceLocationModel`.

Focused on :meth:`TraceLocationModel.as_resources`, which now emits one
``DatabricksTable`` per OTEL Delta table that MLflow's ``UnityCatalog``
trace location materializes. The auth-policy layer relies on these
entries so ``agents.deploy`` auto-grants the Model Serving SP on the
trace tables — no manual ``GRANT`` step required.
"""

from __future__ import annotations

import pytest
from mlflow.models.resources import DatabricksTable

from dao_ai.config import SchemaModel, TraceLocationModel

_OTEL_SUFFIXES: tuple[str, ...] = ("annotations", "logs", "metrics", "spans")


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


def _table_names(loc: TraceLocationModel, **kwargs: object) -> list[str]:
    resources = list(loc.as_resources(**kwargs))  # type: ignore[arg-type]
    assert all(isinstance(r, DatabricksTable) for r in resources)
    return sorted(r.name for r in resources)


@pytest.mark.unit
def test_as_resources_with_explicit_prefix_emits_four_tables(
    loc_with_prefix: TraceLocationModel,
) -> None:
    names = _table_names(loc_with_prefix, experiment_id="999")
    assert names == [f"cat.sch.my_agent_otel_{s}" for s in _OTEL_SUFFIXES]


@pytest.mark.unit
def test_as_resources_without_prefix_falls_back_to_experiment_id(
    loc_no_prefix: TraceLocationModel,
) -> None:
    names = _table_names(loc_no_prefix, experiment_id="2931483616868130")
    assert names == [
        f"cat.sch.2931483616868130_otel_{s}" for s in _OTEL_SUFFIXES
    ]


@pytest.mark.unit
def test_as_resources_explicit_prefix_wins_over_experiment_id(
    loc_with_prefix: TraceLocationModel,
) -> None:
    names = _table_names(loc_with_prefix, experiment_id="999")
    assert all("my_agent_otel_" in n for n in names)
    assert not any(n.endswith("999_otel_spans") for n in names)


@pytest.mark.unit
def test_as_resources_returns_empty_when_neither_known(
    loc_no_prefix: TraceLocationModel,
) -> None:
    assert list(loc_no_prefix.as_resources(experiment_id=None)) == []


@pytest.mark.unit
def test_as_resources_default_arg_returns_empty_without_prefix(
    loc_no_prefix: TraceLocationModel,
) -> None:
    # Called with no kwargs — default experiment_id=None, no prefix set.
    assert list(loc_no_prefix.as_resources()) == []


@pytest.mark.unit
def test_as_resources_default_arg_uses_prefix_when_only_prefix_set(
    loc_with_prefix: TraceLocationModel,
) -> None:
    names = _table_names(loc_with_prefix)
    assert names == [f"cat.sch.my_agent_otel_{s}" for s in _OTEL_SUFFIXES]
