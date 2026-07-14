"""Tests for AuditedHumanInTheLoopMiddleware._process_decision decoration.

Covers the path where LangChain's HITL calls back into
_process_decision AFTER a human resume and BEFORE the tool executes.
The subclass override there is what populates the receipt's `decision`,
`decision_detail`, `confirmed_via`, and (for edit) `edited_args_hash`
fields.
"""

from __future__ import annotations

from typing import Any, Optional

from dao_ai.audit import AuditSinkManager, args_hash_of
from dao_ai.config import (
    AuditModel,
    DatabaseModel,
    HumanInTheLoopModel,
)
from dao_ai.middleware.audit_hitl import AuditedHumanInTheLoopMiddleware
from dao_ai.middleware.audit_receipt import AuditStash, AuditStashEntry


def _seed_stash(
    thread_id: str,
    tool_call_id: str,
    args: dict[str, Any],
    tool_name: str = "refund",
) -> None:
    """Simulate the interrupt-time stash population."""
    from datetime import datetime, timedelta, timezone

    entry = AuditStashEntry(
        tool_name=tool_name,
        args_hash_at_interrupt=args_hash_of(args),
        nonce="nonce-abc",
        nonce_exp=datetime.now(timezone.utc) + timedelta(seconds=300),
        displayed_summary="Approve me",
    )
    AuditStash.put(thread_id, tool_call_id, entry)


def _build_middleware(monkeypatch: Any) -> AuditedHumanInTheLoopMiddleware:
    class _FakeSink:
        def __init__(self, cfg: AuditModel) -> None:
            self.cfg = cfg

    monkeypatch.setattr(
        AuditSinkManager,
        "for_config",
        classmethod(lambda cls, cfg: _FakeSink(cfg)),  # type: ignore[arg-type]
    )
    audit_cfg = AuditModel(database=DatabaseModel(project="test-lake"))
    hitl_cfg = HumanInTheLoopModel(
        review_prompt="Approve?", allowed_decisions=["approve", "edit", "reject", "respond"]
    )
    return AuditedHumanInTheLoopMiddleware(
        interrupt_on={
            "refund": {
                "allowed_decisions": ["approve", "edit", "reject", "respond"]
            }
        },
        audited_tools={"refund": audit_cfg},
        hitl_configs={"refund": hitl_cfg},
    )


def _tool_call(tool_call_id: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": "refund", "args": args, "id": tool_call_id, "type": "tool_call"}


def _interrupt_config() -> dict[str, Any]:
    return {"allowed_decisions": ["approve", "edit", "reject", "respond"]}


def test_process_decision_approve_populates_stash(monkeypatch: Any) -> None:
    AuditStash.reset()
    middleware = _build_middleware(monkeypatch)
    tool_call_id = "call-approve-1"
    original_args = {"amount": 42, "customer_id": "C-1"}
    _seed_stash("t-1", tool_call_id, original_args)

    middleware._process_decision(
        {"type": "approve"},
        _tool_call(tool_call_id, original_args),  # type: ignore[arg-type]
        _interrupt_config(),
    )

    entry = AuditStash.take("t-1", tool_call_id)
    assert entry is not None
    assert entry.decision == "approve"
    assert entry.confirmed_via == "chat_ui"
    assert entry.edited_args_hash is None


def test_process_decision_edit_captures_edited_args_hash(monkeypatch: Any) -> None:
    AuditStash.reset()
    middleware = _build_middleware(monkeypatch)
    tool_call_id = "call-edit-1"
    original_args = {"amount": 42, "customer_id": "C-1"}
    edited_args = {"amount": 20, "customer_id": "C-1"}
    _seed_stash("t-1", tool_call_id, original_args)

    middleware._process_decision(
        {
            "type": "edit",
            "edited_action": {"name": "refund", "args": edited_args},
        },
        _tool_call(tool_call_id, original_args),  # type: ignore[arg-type]
        _interrupt_config(),
    )

    entry = AuditStash.take("t-1", tool_call_id)
    assert entry is not None
    assert entry.decision == "edit"
    assert entry.confirmed_via == "chat_ui"
    assert entry.edited_args_hash == args_hash_of(edited_args)
    assert entry.edited_args_jcs is not None
    assert entry.decision_detail is not None
    assert entry.decision_detail["edited_action"]["args"] == edited_args


def test_process_decision_reject_populates_message(monkeypatch: Any) -> None:
    AuditStash.reset()
    middleware = _build_middleware(monkeypatch)
    tool_call_id = "call-reject-1"
    original_args = {"amount": 42, "customer_id": "C-1"}
    _seed_stash("t-1", tool_call_id, original_args)

    middleware._process_decision(
        {"type": "reject", "message": "Not authorised."},
        _tool_call(tool_call_id, original_args),  # type: ignore[arg-type]
        _interrupt_config(),
    )

    entry = AuditStash.take("t-1", tool_call_id)
    assert entry is not None
    assert entry.decision == "reject"
    assert entry.decision_detail == {"message": "Not authorised."}


def test_process_decision_respond_captures_synthetic_reply(monkeypatch: Any) -> None:
    AuditStash.reset()
    middleware = _build_middleware(monkeypatch)
    tool_call_id = "call-respond-1"
    original_args = {"amount": 42, "customer_id": "C-1"}
    _seed_stash("t-1", tool_call_id, original_args)

    middleware._process_decision(
        {"type": "respond", "message": "Reviewer answered on behalf of tool."},
        _tool_call(tool_call_id, original_args),  # type: ignore[arg-type]
        _interrupt_config(),
    )

    entry = AuditStash.take("t-1", tool_call_id)
    assert entry is not None
    assert entry.decision == "respond"
    assert entry.decision_detail == {
        "message": "Reviewer answered on behalf of tool."
    }


def test_process_decision_unaudited_tool_is_noop(monkeypatch: Any) -> None:
    """A tool NOT in ``audited_tools`` must not touch the stash."""
    AuditStash.reset()
    middleware = _build_middleware(monkeypatch)
    tool_call_id = "call-other-1"

    middleware._process_decision(
        {"type": "approve"},
        {"name": "not_audited_tool", "args": {}, "id": tool_call_id, "type": "tool_call"},  # type: ignore[arg-type]
        _interrupt_config(),
    )
    # No stash entry was ever placed for a non-audited tool.
    assert AuditStash.take("t-1", tool_call_id) is None


def test_thread_id_from_stash_recovers_thread(monkeypatch: Any) -> None:
    AuditStash.reset()
    _seed_stash("thread-alpha", "call-A", {"x": 1})
    _seed_stash("thread-beta", "call-B", {"x": 2})

    assert (
        AuditedHumanInTheLoopMiddleware._thread_id_from_stash("call-A")
        == "thread-alpha"
    )
    assert (
        AuditedHumanInTheLoopMiddleware._thread_id_from_stash("call-B")
        == "thread-beta"
    )
    assert (
        AuditedHumanInTheLoopMiddleware._thread_id_from_stash("call-missing")
        == "unknown-thread"
    )
    # Clean up.
    AuditStash.reset()
