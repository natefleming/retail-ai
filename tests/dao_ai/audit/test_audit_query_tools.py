"""Tests for the audit-query agent tools."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from dao_ai.config import AuditModel, DatabaseModel
from dao_ai.tools.audit_query import (
    _coerce_audit_model,
    _parse_iso,
    _row_to_receipt,
    create_audit_query_tools,
    create_get_audit_receipt_by_id_tool,
    create_query_audit_receipts_tool,
    create_verify_audit_hash_chain_tool,
)


class TestAuditModelCoercion:
    def test_coerce_from_model_returns_same_object(self) -> None:
        model = AuditModel(database=DatabaseModel(project="test-lake"))
        assert _coerce_audit_model(model) is model

    def test_coerce_from_dict(self) -> None:
        raw = {
            "database": {"project": "test-lake"},
            "table": "custom_receipts",
        }
        model = _coerce_audit_model(raw)
        assert model.table == "custom_receipts"
        assert model.database.project == "test-lake"

    def test_coerce_from_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _coerce_audit_model(42)  # type: ignore[arg-type]

    def test_coerce_from_dict_validates_constraints(self) -> None:
        """dict → model must honour AuditModel validators (extra=forbid, TTL bounds)."""
        with pytest.raises(ValueError):
            _coerce_audit_model(
                {"database": {"project": "test-lake"}, "nonce_ttl_seconds": 10}
            )


class TestFactoryReturnsBaseTool:
    def test_query_tool_factory(self) -> None:
        audit = AuditModel(database=DatabaseModel(project="test-lake"))
        tool_obj = create_query_audit_receipts_tool(audit)
        assert tool_obj.name == "query_audit_receipts"
        assert callable(getattr(tool_obj, "coroutine", None))

    def test_get_by_id_tool_factory(self) -> None:
        audit = {"database": {"project": "test-lake"}}
        tool_obj = create_get_audit_receipt_by_id_tool(audit)
        assert tool_obj.name == "get_audit_receipt_by_id"
        assert callable(getattr(tool_obj, "coroutine", None))

    def test_verify_hash_chain_tool_factory(self) -> None:
        audit = AuditModel(database=DatabaseModel(project="test-lake"))
        tool_obj = create_verify_audit_hash_chain_tool(audit)
        assert tool_obj.name == "verify_audit_hash_chain"

    def test_bundle_factory_returns_full_toolkit(self) -> None:
        """create_audit_query_tools returns every tool in the toolkit
        (three basic query tools + four auditor-oriented tools)."""
        audit = AuditModel(database=DatabaseModel(project="test-lake"))
        tools = create_audit_query_tools(audit)
        names = {t.name for t in tools}
        assert names == {
            "query_audit_receipts",
            "get_audit_receipt_by_id",
            "verify_audit_hash_chain",
            "summarize_audit_activity",
            "find_security_incidents",
            "get_thread_timeline",
            "get_approver_activity",
        }


class TestRowNormalization:
    def test_row_drops_sensitive_columns(self) -> None:
        row: dict[str, Any] = {
            "receipt_id": "abc",
            "args_jcs": '{"a":1}',
            "args_hash": "hash",
            "obo_access_token": "eyJraWQ.....",
            "obo_token_sub": "user-1",
            "decision": "approve",
        }
        cleaned = _row_to_receipt(row)
        assert "obo_access_token" not in cleaned
        assert "args_jcs" not in cleaned
        assert cleaned["args_hash"] == "hash"
        assert cleaned["obo_token_sub"] == "user-1"
        assert cleaned["decision"] == "approve"

    def test_datetime_serialised_as_iso(self) -> None:
        now = datetime(2026, 7, 14, 12, 30, 45, tzinfo=timezone.utc)
        row = {"receipt_id": "abc", "recorded_at": now}
        cleaned = _row_to_receipt(row)
        assert cleaned["recorded_at"] == "2026-07-14T12:30:45+00:00"

    def test_decision_detail_string_json_is_parsed(self) -> None:
        row = {
            "receipt_id": "abc",
            "decision_detail": '{"message": "not authorised"}',
        }
        cleaned = _row_to_receipt(row)
        assert cleaned["decision_detail"] == {"message": "not authorised"}


class TestIsoParsing:
    def test_naive_iso(self) -> None:
        result = _parse_iso("2026-07-14T00:00:00")
        assert result.year == 2026

    def test_z_suffix_treated_as_utc(self) -> None:
        result = _parse_iso("2026-07-14T00:00:00Z")
        assert result.tzinfo is not None

    def test_offset_preserved(self) -> None:
        result = _parse_iso("2026-07-14T00:00:00+05:00")
        assert result.utcoffset() is not None
