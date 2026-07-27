"""Unit tests for `coerce_filter_values` — the LLM-filter-robustness helper.

Covers the common failure mode where LLM decomposition emits a string value
on a non-string column (e.g. `{"priority": "high"}` on an int column) —
Postgres 500s in the raw path; this helper coerces where possible and drops
the entry with a warning where not, so the query still runs.
"""

from datetime import datetime

import pytest

from dao_ai.config import ColumnInfo
from dao_ai.tools.instructed_pipeline import (
    _base_column_name,
    coerce_filter_values,
)


@pytest.mark.unit
class TestBaseColumnName:
    def test_bare_column(self) -> None:
        assert _base_column_name("priority") == "priority"

    def test_suffix_stripped(self) -> None:
        assert _base_column_name("priority >=") == "priority"
        assert _base_column_name("priority NOT") == "priority"
        assert _base_column_name("brand LIKE") == "brand"
        assert _base_column_name("brand NOT LIKE") == "brand"

    def test_column_containing_suffix_word_preserved(self) -> None:
        # "note" doesn't end in " NOT" (space matters).
        assert _base_column_name("note") == "note"


@pytest.mark.unit
class TestCoerceNumber:
    cols = [ColumnInfo(name="priority", type="number")]

    def test_int_string_to_int(self) -> None:
        assert coerce_filter_values({"priority": "3"}, self.cols) == {"priority": 3}

    def test_float_string_to_float(self) -> None:
        assert coerce_filter_values({"priority": "3.14"}, self.cols) == {
            "priority": 3.14
        }

    def test_int_passes_through(self) -> None:
        assert coerce_filter_values({"priority": 5}, self.cols) == {"priority": 5}

    def test_float_passes_through(self) -> None:
        assert coerce_filter_values({"priority": 1.5}, self.cols) == {"priority": 1.5}

    def test_bad_string_dropped(self, caplog) -> None:
        out = coerce_filter_values({"priority": "high"}, self.cols)
        assert out == {}

    def test_suffixed_key_coerced(self) -> None:
        assert coerce_filter_values({"priority >=": "3"}, self.cols) == {
            "priority >=": 3
        }

    def test_list_of_int_strings(self) -> None:
        assert coerce_filter_values({"priority": ["1", "2", "3"]}, self.cols) == {
            "priority": [1, 2, 3]
        }

    def test_list_with_bad_element_drops_entire_entry(self) -> None:
        out = coerce_filter_values({"priority": ["1", "high", "3"]}, self.cols)
        assert out == {}

    def test_bool_rejected_as_number(self) -> None:
        # bool is technically int subclass in Python — reject to avoid
        # sending True/False into a numeric column.
        assert coerce_filter_values({"priority": True}, self.cols) == {}


@pytest.mark.unit
class TestCoerceBoolean:
    cols = [ColumnInfo(name="active", type="boolean")]

    def test_true_string(self) -> None:
        assert coerce_filter_values({"active": "true"}, self.cols) == {"active": True}
        assert coerce_filter_values({"active": "TRUE"}, self.cols) == {"active": True}

    def test_false_string(self) -> None:
        assert coerce_filter_values({"active": "false"}, self.cols) == {"active": False}

    def test_numeric_string(self) -> None:
        assert coerce_filter_values({"active": "1"}, self.cols) == {"active": True}
        assert coerce_filter_values({"active": "0"}, self.cols) == {"active": False}

    def test_yes_no(self) -> None:
        assert coerce_filter_values({"active": "yes"}, self.cols) == {"active": True}
        assert coerce_filter_values({"active": "no"}, self.cols) == {"active": False}

    def test_bool_passes_through(self) -> None:
        assert coerce_filter_values({"active": True}, self.cols) == {"active": True}

    def test_unknown_dropped(self) -> None:
        assert coerce_filter_values({"active": "maybe"}, self.cols) == {}


@pytest.mark.unit
class TestCoerceDatetime:
    cols = [ColumnInfo(name="ts", type="datetime")]

    def test_iso_date_kept(self) -> None:
        assert coerce_filter_values({"ts": "2026-07-09"}, self.cols) == {
            "ts": "2026-07-09"
        }

    def test_iso_datetime_kept(self) -> None:
        assert coerce_filter_values({"ts": "2026-07-09T10:30:00"}, self.cols) == {
            "ts": "2026-07-09T10:30:00"
        }

    def test_datetime_instance_becomes_iso(self) -> None:
        dt = datetime(2026, 7, 9, 10, 30)
        assert coerce_filter_values({"ts": dt}, self.cols) == {
            "ts": "2026-07-09T10:30:00"
        }

    def test_unparseable_dropped(self) -> None:
        assert coerce_filter_values({"ts": "yesterday"}, self.cols) == {}


@pytest.mark.unit
class TestCoercePassthroughs:
    def test_string_untouched(self) -> None:
        cols = [ColumnInfo(name="brand", type="string")]
        assert coerce_filter_values({"brand": "DEWALT"}, cols) == {"brand": "DEWALT"}
        # Integer on a string column stays an integer (no coercion for
        # string type — the backend will str-coerce or reject).
        assert coerce_filter_values({"brand": 5}, cols) == {"brand": 5}

    def test_array_untouched(self) -> None:
        cols = [ColumnInfo(name="tags", type="array")]
        assert coerce_filter_values({"tags": ["cordless", "brushless"]}, cols) == {
            "tags": ["cordless", "brushless"]
        }

    def test_unknown_column_passes_through(self) -> None:
        cols = [ColumnInfo(name="brand", type="string")]
        # The LLM might invent a column; let the backend reject.
        assert coerce_filter_values({"invented_col": "x"}, cols) == {
            "invented_col": "x"
        }

    def test_empty_filters(self) -> None:
        assert coerce_filter_values({}, [ColumnInfo(name="x", type="number")]) == {}

    def test_empty_columns(self) -> None:
        # No columns declared → passthrough (nothing to coerce against).
        assert coerce_filter_values({"anything": "value"}, []) == {"anything": "value"}


@pytest.mark.unit
class TestCoerceMixed:
    def test_multi_column_filter(self) -> None:
        cols = [
            ColumnInfo(name="brand", type="string"),
            ColumnInfo(name="price", type="number"),
            ColumnInfo(name="in_stock", type="boolean"),
        ]
        out = coerce_filter_values(
            {"brand": "DEWALT", "price": "99.99", "in_stock": "true"},
            cols,
        )
        assert out == {"brand": "DEWALT", "price": 99.99, "in_stock": True}

    def test_one_bad_entry_dropped_others_kept(self) -> None:
        cols = [
            ColumnInfo(name="brand", type="string"),
            ColumnInfo(name="price", type="number"),
        ]
        out = coerce_filter_values({"brand": "DEWALT", "price": "cheap"}, cols)
        assert out == {"brand": "DEWALT"}
