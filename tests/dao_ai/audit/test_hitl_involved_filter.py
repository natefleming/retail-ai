"""Tests for the ``hitl_involved`` filter + computed field on query_audit_receipts."""

from __future__ import annotations

from typing import Any

from dao_ai.tools.audit_query import _row_had_hitl, _row_to_receipt


class TestRowHadHitl:
    """Coverage for the boolean synthesis rule."""

    def test_audit_only_execution_returns_false(self) -> None:
        row: dict[str, Any] = {
            "receipt_kind": "execution",
            "args_hash_at_interrupt": None,
            "decision": None,
        }
        assert _row_had_hitl(row) is False

    def test_hitl_approve_execution_returns_true(self) -> None:
        row: dict[str, Any] = {
            "receipt_kind": "execution",
            "args_hash_at_interrupt": "a" * 64,
            "decision": "approve",
        }
        assert _row_had_hitl(row) is True

    def test_hitl_edit_execution_returns_true(self) -> None:
        row: dict[str, Any] = {
            "receipt_kind": "execution",
            "args_hash_at_interrupt": "a" * 64,
            "decision": "edit",
        }
        assert _row_had_hitl(row) is True

    def test_rejection_returns_true(self) -> None:
        row: dict[str, Any] = {
            "receipt_kind": "rejection",
            "args_hash_at_interrupt": None,
            "decision": "reject",
        }
        assert _row_had_hitl(row) is True

    def test_respond_returns_true(self) -> None:
        row: dict[str, Any] = {
            "receipt_kind": "rejection",
            "args_hash_at_interrupt": None,
            "decision": "respond",
        }
        assert _row_had_hitl(row) is True

    def test_args_mismatch_rejection_returns_true(self) -> None:
        """Even a fail-closed args-mismatch rejection is HITL-flavoured."""
        row: dict[str, Any] = {
            "receipt_kind": "rejection",
            "args_hash_at_interrupt": "a" * 64,
            "decision": None,
            "execution_status": "args_mismatch",
        }
        assert _row_had_hitl(row) is True

    def test_partial_hitl_evidence_still_marks_hitl(self) -> None:
        """Any single HITL signal is sufficient — no AND requirement."""
        # Only args_hash_at_interrupt is set (edge case where decision decoration
        # didn't complete but the interrupt-time stash was captured).
        row: dict[str, Any] = {
            "receipt_kind": "execution",
            "args_hash_at_interrupt": "a" * 64,
            "decision": None,
        }
        assert _row_had_hitl(row) is True


class TestRowToReceipt:
    """Verify the normaliser sets ``hitl_involved`` on every row."""

    def test_hitl_involved_flag_present_on_audit_only(self) -> None:
        row: dict[str, Any] = {
            "receipt_id": "abc",
            "receipt_kind": "execution",
            "args_hash": "h",
            "args_hash_at_interrupt": None,
            "decision": None,
        }
        cleaned = _row_to_receipt(row)
        assert cleaned["hitl_involved"] is False

    def test_hitl_involved_flag_true_on_hitl_row(self) -> None:
        row: dict[str, Any] = {
            "receipt_id": "abc",
            "receipt_kind": "execution",
            "args_hash": "h",
            "args_hash_at_interrupt": "a" * 64,
            "decision": "approve",
        }
        cleaned = _row_to_receipt(row)
        assert cleaned["hitl_involved"] is True

    def test_hitl_involved_flag_true_on_rejection(self) -> None:
        row: dict[str, Any] = {
            "receipt_id": "abc",
            "receipt_kind": "rejection",
            "args_hash": "h",
            "args_hash_at_interrupt": None,
            "decision": "reject",
        }
        cleaned = _row_to_receipt(row)
        assert cleaned["hitl_involved"] is True
