"""Pure-unit tests for LongRunningStore helpers that don't require a live DB."""

from __future__ import annotations

import pytest

from dao_ai.long_running.store import ResponseStatus, _coerce_json, _valid_identifier


def test_valid_identifier_allows_simple_names():
    assert _valid_identifier("dao_ai_responses") == "dao_ai_responses"
    assert _valid_identifier("table123") == "table123"


@pytest.mark.parametrize(
    "bad",
    ["", "drop table users;", "foo-bar", "foo bar", "foo'--", "foo.bar"],
)
def test_valid_identifier_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        _valid_identifier(bad)


def test_response_status_is_terminal():
    assert ResponseStatus.COMPLETED.is_terminal
    assert ResponseStatus.FAILED.is_terminal
    assert ResponseStatus.CANCELLED.is_terminal
    assert not ResponseStatus.QUEUED.is_terminal
    assert not ResponseStatus.IN_PROGRESS.is_terminal


def test_coerce_json_passthrough_for_dict_and_list():
    assert _coerce_json({"a": 1}) == {"a": 1}
    assert _coerce_json([1, 2]) == [1, 2]
    assert _coerce_json(None) is None


def test_coerce_json_parses_json_strings():
    assert _coerce_json('{"a": 1}') == {"a": 1}


def test_coerce_json_returns_str_on_parse_failure():
    assert _coerce_json("not json") == "not json"


def test_coerce_json_handles_bytes():
    assert _coerce_json(b'{"a": 1}') == {"a": 1}
