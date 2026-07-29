"""Regression tests (F5) for Databricks Apps resource-name sanitization.

Dict keys for genie-derived tables/functions can exceed the Apps 2–30 char
resource-name rule (e.g. ``tbl_3978880469080159_trace_unified`` = 34 chars, or
``<room>_<catalog>_<schema>_<table>`` from the primary genie validator). Those
keys are emitted verbatim as the resource ``name`` and made ``databricks bundle
deploy`` fail with 400 INVALID_PARAMETER_VALUE. Both emit points must sanitize +
uniquify the name.
"""

from __future__ import annotations

import pytest

from dao_ai.apps.resources import (
    _extract_function_resources,
    _extract_table_resources,
    _unique_resource_name,
)
from dao_ai.config import FunctionModel, TableModel


@pytest.mark.unit
class TestUniqueResourceName:
    def test_long_name_truncated_to_30(self) -> None:
        # The exact matrix failure key (34 chars).
        name = _unique_resource_name("tbl_3978880469080159_trace_unified", set())
        assert 2 <= len(name) <= 30

    def test_short_name_unchanged(self) -> None:
        used: set[str] = set()
        assert _unique_resource_name("tbl_products", used) == "tbl_products"

    def test_collision_gets_distinct_suffix_within_budget(self) -> None:
        used: set[str] = set()
        base = "tbl_" + ("segment_" * 6)  # > 30 chars, shared prefix
        a = _unique_resource_name(base, used)
        b = _unique_resource_name(base + "_other", used)
        assert a != b
        assert len(a) <= 30 and len(b) <= 30

    def test_dots_and_hyphens_normalized(self) -> None:
        name = _unique_resource_name("cat.sch-tbl", set())
        assert "." not in name and "-" not in name


@pytest.mark.unit
class TestExtractResourcesSanitizeNames:
    def test_table_resource_name_within_limit(self) -> None:
        tables = {
            # 34-char key — the matrix repro.
            "tbl_3978880469080159_trace_unified": TableModel(
                name="cat.sch.3978880469080159_trace_unified"
            ),
            "tbl_products": TableModel(name="cat.sch.products"),
        }
        resources = _extract_table_resources(tables)
        names = [r["name"] for r in resources]
        assert all(2 <= len(n) <= 30 for n in names)
        # table_name (the actual UC target) is preserved untouched.
        assert any(
            r["table_name"] == "cat.sch.3978880469080159_trace_unified"
            for r in resources
        )

    def test_function_resource_name_within_limit(self) -> None:
        functions = {
            "fn_a_very_long_function_name_exceeding_limit": FunctionModel(
                name="cat.sch.a_very_long_function_name_exceeding_limit"
            ),
        }
        resources = _extract_function_resources(functions)
        assert all(2 <= len(r["name"]) <= 30 for r in resources)

    def test_two_long_table_keys_stay_distinct(self) -> None:
        tables = {
            "tbl_" + ("x_prefix_" * 5) + "alpha": TableModel(name="cat.sch.alpha"),
            "tbl_" + ("x_prefix_" * 5) + "beta": TableModel(name="cat.sch.beta"),
        }
        names = [r["name"] for r in _extract_table_resources(tables)]
        assert len(names) == len(set(names))  # no collision
        assert all(len(n) <= 30 for n in names)
