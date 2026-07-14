"""Tests for AuditToolkit — the BaseToolkit-shaped bundle of audit query tools."""

from __future__ import annotations

from langchain_community.agent_toolkits.base import BaseToolkit

from dao_ai.config import AuditModel, DatabaseModel
from dao_ai.tools.audit_query import AuditToolkit, create_audit_toolkit


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
