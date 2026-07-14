"""Tests for AuditToolkit — the BaseToolkit-shaped bundle of audit query tools."""

from __future__ import annotations

import pytest
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_core.tools import BaseTool, tool

from dao_ai.config import AuditModel, DatabaseModel
from dao_ai.tools.audit_query import (
    AuditToolkit,
    as_tool_list,
    as_toolkit,
    create_audit_query_tools,
    create_audit_toolkit,
)


@tool
def _sample_tool(name: str) -> str:
    """Sample tool used to exercise shape adapters in tests."""
    return f"hello {name}"


@tool
def _second_sample_tool(x: int) -> int:
    """Second sample tool."""
    return x + 1


def test_toolkit_is_langchain_base_toolkit() -> None:
    """AuditToolkit must be a BaseToolkit so dao-ai's factory tool resolver
    expands it via get_tools() in ``dao_ai.tools.python.create_factory_tool``.
    """
    audit = AuditModel(database=DatabaseModel(project="test-lake"))
    toolkit = create_audit_toolkit(audit)
    assert isinstance(toolkit, AuditToolkit)
    assert isinstance(toolkit, BaseToolkit)


def test_toolkit_get_tools_returns_all_three() -> None:
    audit = AuditModel(database=DatabaseModel(project="test-lake"))
    toolkit = create_audit_toolkit(audit)
    tools = toolkit.get_tools()
    names = {t.name for t in tools}
    assert names == {
        "query_audit_receipts",
        "get_audit_receipt_by_id",
        "verify_audit_hash_chain",
    }


def test_toolkit_accepts_dict_config() -> None:
    """Same YAML-anchor coercion story as the individual factories."""
    toolkit = create_audit_toolkit(
        {"database": {"project": "test-lake"}, "table": "custom_receipts"}
    )
    assert len(toolkit.get_tools()) == 3


def test_get_tools_returns_defensive_copy() -> None:
    """get_tools() must not expose the internal list — mutation isolation."""
    audit = AuditModel(database=DatabaseModel(project="test-lake"))
    toolkit = create_audit_toolkit(audit)
    tools_first = toolkit.get_tools()
    tools_first.pop()
    tools_second = toolkit.get_tools()
    assert len(tools_second) == 3


class TestAsToolList:
    """Coverage for the ``BaseTool | Sequence[BaseTool] | BaseToolkit`` shape adapter."""

    def test_none_returns_empty_list(self) -> None:
        assert as_tool_list(None) == []

    def test_single_base_tool(self) -> None:
        result = as_tool_list(_sample_tool)
        assert result == [_sample_tool]

    def test_sequence_of_tools_list(self) -> None:
        result = as_tool_list([_sample_tool, _second_sample_tool])
        assert result == [_sample_tool, _second_sample_tool]

    def test_sequence_of_tools_tuple(self) -> None:
        result = as_tool_list((_sample_tool, _second_sample_tool))
        assert result == [_sample_tool, _second_sample_tool]

    def test_toolkit_expanded_via_get_tools(self) -> None:
        tk = AuditToolkit(tools=[_sample_tool, _second_sample_tool])
        result = as_tool_list(tk)
        assert result == [_sample_tool, _second_sample_tool]

    def test_string_rejected(self) -> None:
        """A str is a Sequence[str] — must NOT sneak through as a tool sequence."""
        with pytest.raises(TypeError):
            as_tool_list("not a tool")  # type: ignore[arg-type]

    def test_non_tool_in_sequence_rejected(self) -> None:
        with pytest.raises(TypeError):
            as_tool_list([_sample_tool, "oops"])  # type: ignore[list-item]

    def test_random_object_rejected(self) -> None:
        with pytest.raises(TypeError):
            as_tool_list(object())  # type: ignore[arg-type]


class TestAsToolkit:
    def test_returns_existing_toolkit_unchanged(self) -> None:
        """Passing a BaseToolkit through must preserve identity."""
        tk = AuditToolkit(tools=[_sample_tool])
        result = as_toolkit(tk)
        assert result is tk

    def test_wraps_single_tool(self) -> None:
        result = as_toolkit(_sample_tool)
        assert isinstance(result, BaseToolkit)
        assert result.get_tools() == [_sample_tool]

    def test_wraps_sequence(self) -> None:
        result = as_toolkit([_sample_tool, _second_sample_tool])
        assert isinstance(result, BaseToolkit)
        assert result.get_tools() == [_sample_tool, _second_sample_tool]


class TestFactoryExtraTools:
    """The factories accept an ``extra_tools`` shape and bundle it into the output."""

    def test_create_audit_query_tools_bundles_extras(self) -> None:
        audit = AuditModel(database=DatabaseModel(project="test-lake"))
        combined: list[BaseTool] = create_audit_query_tools(
            audit, extra_tools=_sample_tool
        )
        names = {t.name for t in combined}
        assert names == {
            "query_audit_receipts",
            "get_audit_receipt_by_id",
            "verify_audit_hash_chain",
            "_sample_tool",
        }

    def test_create_audit_toolkit_bundles_sequence(self) -> None:
        audit = AuditModel(database=DatabaseModel(project="test-lake"))
        toolkit = create_audit_toolkit(
            audit, extra_tools=[_sample_tool, _second_sample_tool]
        )
        names = {t.name for t in toolkit.get_tools()}
        assert names == {
            "query_audit_receipts",
            "get_audit_receipt_by_id",
            "verify_audit_hash_chain",
            "_sample_tool",
            "_second_sample_tool",
        }

    def test_create_audit_toolkit_bundles_toolkit(self) -> None:
        audit = AuditModel(database=DatabaseModel(project="test-lake"))
        extras = AuditToolkit(tools=[_sample_tool])
        toolkit = create_audit_toolkit(audit, extra_tools=extras)
        names = {t.name for t in toolkit.get_tools()}
        assert "query_audit_receipts" in names and "_sample_tool" in names

    def test_create_audit_toolkit_no_extras_still_3(self) -> None:
        audit = AuditModel(database=DatabaseModel(project="test-lake"))
        toolkit = create_audit_toolkit(audit)
        assert len(toolkit.get_tools()) == 3
