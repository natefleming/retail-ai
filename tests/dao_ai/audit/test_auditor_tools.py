"""Tests for the extended AuditToolkit (auditor-oriented tools).

These tools all execute live SQL against Lakebase, so pure-Python tests
cover the factory shape, tool descriptions, and window-clause builder.
End-to-end execution against a real database is exercised via the FEVM
verification path in ``tests/dao_ai/audit/test_lakebase_sink.py``-style
integration tests (marker-gated).
"""

from __future__ import annotations

from dao_ai.config import AuditModel, DatabaseModel
from dao_ai.tools.audit_query import (
    AuditToolkit,
    _build_window_clauses,
    create_audit_toolkit,
    create_find_security_incidents_tool,
    create_get_approver_activity_tool,
    create_get_thread_timeline_tool,
    create_summarize_audit_activity_tool,
)


class TestFactoriesReturnBaseTools:
    """Each factory must return a callable BaseTool with the documented name."""

    def _audit(self) -> AuditModel:
        return AuditModel(database=DatabaseModel(project="test-lake"))

    def test_summarize_audit_activity_factory(self) -> None:
        tool_obj = create_summarize_audit_activity_tool(self._audit())
        assert tool_obj.name == "summarize_audit_activity"
        assert callable(getattr(tool_obj, "coroutine", None))

    def test_find_security_incidents_factory(self) -> None:
        tool_obj = create_find_security_incidents_tool(self._audit())
        assert tool_obj.name == "find_security_incidents"
        assert callable(getattr(tool_obj, "coroutine", None))

    def test_get_thread_timeline_factory(self) -> None:
        tool_obj = create_get_thread_timeline_tool(self._audit())
        assert tool_obj.name == "get_thread_timeline"
        assert callable(getattr(tool_obj, "coroutine", None))

    def test_get_approver_activity_factory(self) -> None:
        tool_obj = create_get_approver_activity_tool(self._audit())
        assert tool_obj.name == "get_approver_activity"
        assert callable(getattr(tool_obj, "coroutine", None))


class TestToolkitBundlesAuditorTools:
    def test_toolkit_ships_all_seven_tools(self) -> None:
        toolkit: AuditToolkit = create_audit_toolkit(
            AuditModel(database=DatabaseModel(project="test-lake"))
        )
        names = {t.name for t in toolkit.get_tools()}
        assert names == {
            "query_audit_receipts",
            "get_audit_receipt_by_id",
            "verify_audit_hash_chain",
            "summarize_audit_activity",
            "find_security_incidents",
            "get_thread_timeline",
            "get_approver_activity",
        }


class TestBuildWindowClauses:
    """Shared helper covers since/until/thread_id/tool_name shape."""

    def test_no_filters_returns_empty_clauses(self) -> None:
        clauses, params = _build_window_clauses()
        assert clauses == []
        assert params == []

    def test_since_filter_added(self) -> None:
        clauses, params = _build_window_clauses(since="2026-07-14T00:00:00Z")
        assert clauses == ["recorded_at >= %s"]
        assert len(params) == 1

    def test_until_filter_added(self) -> None:
        clauses, params = _build_window_clauses(until="2026-07-15T00:00:00Z")
        assert clauses == ["recorded_at < %s"]
        assert len(params) == 1

    def test_thread_id_and_tool_name(self) -> None:
        clauses, params = _build_window_clauses(
            thread_id="t-1", tool_name="refund"
        )
        assert "thread_id = %s" in clauses
        assert "tool_name = %s" in clauses
        assert params == ["t-1", "refund"]

    def test_all_filters_stacked(self) -> None:
        clauses, params = _build_window_clauses(
            since="2026-07-14T00:00:00Z",
            until="2026-07-15T00:00:00Z",
            thread_id="t-1",
            tool_name="refund",
        )
        assert len(clauses) == 4
        assert len(params) == 4
