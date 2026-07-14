"""
Client-facing notifications for audit-receipt writes.

Every successful receipt write dispatches a ``dao_ai.audit.receipt``
custom event through LangChain's callback manager. The streaming
response agent (``LanggraphResponsesAgent.apredict_stream``) collects
those events, forwards them as ``response.output_item.added(status=
"in_progress")`` stream frames, and also accumulates them into
``custom_outputs["mcp_events"]`` for non-streaming replay.

The event carries only the non-sensitive receipt metadata a UI needs
to display "audit trail captured for tool X":

- ``receipt_id`` — matches the Lakebase row
- ``receipt_kind`` — execution / rejection / approval
- ``tool_name`` / ``tool_call_id``
- ``decision`` (HITL only)
- ``hitl_involved``
- ``execution_status``
- ``recorded_at`` (ISO)
- ``thread_id``
- ``mlflow_trace_id``

Deliberately excluded (sensitive or agent-noise):

- ``obo_access_token`` — raw JWT, never streamed to client
- ``args_jcs`` / ``args_hash_at_interrupt`` / ``args_hash_at_resume`` — internal
- ``nonce`` / ``nonce_exp`` — server-only
- ``prev_hash`` / ``this_hash`` — chain-verification queries fetch from Lakebase
- ``displayed_summary`` — already shown at interrupt time; not needed again
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.runnables.config import RunnableConfig
from loguru import logger

if TYPE_CHECKING:
    from dao_ai.audit.base import AuditReceipt


AUDIT_RECEIPT_CHANNEL: str = "dao_ai.audit.receipt"


def build_receipt_notification(receipt: "AuditReceipt") -> dict[str, Any]:
    """Return the JSON-serialisable, client-safe envelope for a receipt."""
    recorded_at: Any = receipt.recorded_at
    return {
        "channel": AUDIT_RECEIPT_CHANNEL,
        "server_name": "dao_ai.audit",
        "receipt_id": receipt.receipt_id,
        "receipt_kind": (
            receipt.receipt_kind.value
            if hasattr(receipt.receipt_kind, "value")
            else str(receipt.receipt_kind)
        ),
        "tool_name": receipt.tool_name,
        "tool_call_id": receipt.tool_call_id,
        "thread_id": receipt.thread_id,
        "agent_id": receipt.agent_id,
        "mlflow_trace_id": receipt.mlflow_trace_id,
        "decision": receipt.decision,
        "hitl_involved": _hitl_involved(receipt),
        "execution_status": (
            receipt.execution_status.value
            if receipt.execution_status is not None
            and hasattr(receipt.execution_status, "value")
            else receipt.execution_status
        ),
        "confirmed_via": receipt.confirmed_via,
        "approver_sub": receipt.approver_sub,
        "recorded_at": (
            recorded_at.isoformat()
            if isinstance(recorded_at, datetime)
            else recorded_at
        ),
    }


def _hitl_involved(receipt: "AuditReceipt") -> bool:
    """Mirrors the Postgres GENERATED column semantics for the streaming envelope."""
    if receipt.args_hash_at_interrupt is not None:
        return True
    kind: Any = receipt.receipt_kind
    kind_value: str = (
        kind.value if hasattr(kind, "value") else str(kind)
    )
    if kind_value == "rejection":
        return True
    if isinstance(receipt.decision, str) and receipt.decision in {
        "approve",
        "edit",
        "reject",
        "respond",
    }:
        return True
    return False


async def dispatch_audit_receipt_notification(
    receipt: "AuditReceipt",
    config: Optional[RunnableConfig] = None,
) -> None:
    """
    Emit an audit-receipt notification through LangChain's callback manager.

    When ``config`` is None or contains no callback handlers (e.g. batch
    predict without a graph.astream context), this is a silent no-op.

    Never blocks the audit write path — any dispatch exception is
    logged and swallowed so the receipt has already landed by the time
    we get here anyway.
    """
    envelope: dict[str, Any] = build_receipt_notification(receipt)
    try:
        await adispatch_custom_event(AUDIT_RECEIPT_CHANNEL, envelope, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Audit receipt notification dispatch failed (non-fatal)",
            receipt_id=receipt.receipt_id,
            error=repr(exc)[:120],
        )
