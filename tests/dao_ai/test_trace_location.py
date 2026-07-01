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
