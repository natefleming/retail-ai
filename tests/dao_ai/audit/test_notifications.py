"""Tests for the client-facing audit-receipt notification envelope."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from dao_ai.audit import (
    AUDIT_RECEIPT_CHANNEL,
    AuditReceipt,
    ExecutionStatus,
    ReceiptKind,
    args_hash_of,
    build_receipt_notification,
    dispatch_audit_receipt_notification,
)
from dao_ai.audit.base import canonical_jcs


def _receipt(
    *,
    kind: ReceiptKind = ReceiptKind.EXECUTION,
    decision: str | None = None,
    hitl: bool = False,
    args: dict[str, object] | None = None,
) -> AuditReceipt:
    args_dict: dict[str, object] = args or {"amount": 42}
    return AuditReceipt(
        receipt_id=uuid.uuid4().hex,
        receipt_kind=kind,
        thread_id="thread-notif",
        agent_id="agent-A",
        mlflow_trace_id="trace-1",
        tool_call_id="call-1",
        tool_name="refund_customer",
        args_jcs=canonical_jcs(args_dict),
        args_hash=args_hash_of(args_dict),
        args_hash_at_interrupt=(args_hash_of(args_dict) if hitl else None),
        decision=decision,
        approver_sub="alice@example.com" if hitl else None,
        confirmed_via="chat_ui" if hitl else None,
        execution_status=ExecutionStatus.OK,
        recorded_at=datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc),
    )


class TestBuildReceiptNotification:
    def test_channel_and_server_name_stable(self) -> None:
        envelope = build_receipt_notification(_receipt())
        assert envelope["channel"] == AUDIT_RECEIPT_CHANNEL
        assert envelope["server_name"] == "dao_ai.audit"

    def test_execution_receipt_no_hitl(self) -> None:
        envelope = build_receipt_notification(_receipt())
        assert envelope["receipt_kind"] == "execution"
        assert envelope["hitl_involved"] is False
        assert envelope["decision"] is None
        assert envelope["approver_sub"] is None
        assert envelope["tool_name"] == "refund_customer"
        assert envelope["thread_id"] == "thread-notif"
        assert envelope["execution_status"] == "ok"

    def test_hitl_approve_execution(self) -> None:
        envelope = build_receipt_notification(
            _receipt(decision="approve", hitl=True)
        )
        assert envelope["hitl_involved"] is True
        assert envelope["decision"] == "approve"
        assert envelope["approver_sub"] == "alice@example.com"
        assert envelope["confirmed_via"] == "chat_ui"

    def test_rejection_receipt(self) -> None:
        envelope = build_receipt_notification(
            _receipt(kind=ReceiptKind.REJECTION, decision="reject", hitl=True)
        )
        assert envelope["receipt_kind"] == "rejection"
        assert envelope["hitl_involved"] is True
        assert envelope["decision"] == "reject"

    def test_recorded_at_serialized_as_iso(self) -> None:
        envelope = build_receipt_notification(_receipt())
        assert envelope["recorded_at"] == "2026-07-14T12:00:00+00:00"

    def test_sensitive_fields_absent(self) -> None:
        """Raw JWT + args + nonce + chain fields must NOT be on the wire event."""
        envelope = build_receipt_notification(_receipt(hitl=True))
        # These are always dropped from the client-facing envelope.
        for forbidden in (
            "obo_access_token",
            "args_jcs",
            "args_hash_at_interrupt",
            "args_hash_at_resume",
            "nonce",
            "nonce_exp",
            "prev_hash",
            "this_hash",
            "displayed_summary",
        ):
            assert forbidden not in envelope, (
                f"envelope must not carry {forbidden}"
            )


class TestDispatchAuditReceiptNotification:
    """The dispatcher is best-effort — no-op when there's no callback ctx."""

    def test_no_config_is_noop(self) -> None:
        # Should complete without raising.
        asyncio.run(
            dispatch_audit_receipt_notification(_receipt(), config=None)
        )

    def test_config_without_callbacks_is_noop(self) -> None:
        asyncio.run(
            dispatch_audit_receipt_notification(
                _receipt(), config={"configurable": {}}
            )
        )
