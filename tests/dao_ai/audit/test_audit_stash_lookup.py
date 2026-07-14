"""Tests for AuditStash tool-name-keyed lookup.

Covers the fallback path used by ``dao_ai.hitl._record_hitl_non_executions``
when recovering the interrupt-time stash for ``reject`` / ``respond``
decisions: the snapshot walk to reconstruct tool_call_id is fragile
across checkpointer serialization paths, so the tap looks up by
``(thread_id, tool_name)`` instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dao_ai.middleware.audit_receipt import AuditStash, AuditStashEntry


def _entry(tool_name: str) -> AuditStashEntry:
    return AuditStashEntry(
        tool_name=tool_name,
        args_hash_at_interrupt="a" * 64,
        nonce="nonce-abc",
        nonce_exp=datetime.now(timezone.utc) + timedelta(seconds=300),
        displayed_summary="review me",
    )


def test_take_by_tool_name_returns_call_id_and_entry() -> None:
    AuditStash.reset()
    AuditStash.put("thread-1", "call-A", _entry("refund"))
    result = AuditStash.take_by_tool_name("thread-1", "refund")
    assert result is not None
    tool_call_id, entry = result
    assert tool_call_id == "call-A"
    assert entry.tool_name == "refund"
    # Entry is removed on take — subsequent takes return None.
    assert AuditStash.take_by_tool_name("thread-1", "refund") is None


def test_take_by_tool_name_no_match_returns_none() -> None:
    AuditStash.reset()
    AuditStash.put("thread-1", "call-A", _entry("refund"))
    assert AuditStash.take_by_tool_name("thread-1", "cancel") is None
    # Original entry is untouched.
    assert AuditStash.take("thread-1", "call-A") is not None


def test_take_by_tool_name_scoped_by_thread() -> None:
    """A tool_name match in a different thread must NOT be returned."""
    AuditStash.reset()
    AuditStash.put("thread-A", "call-A", _entry("refund"))
    AuditStash.put("thread-B", "call-B", _entry("refund"))
    result_a = AuditStash.take_by_tool_name("thread-A", "refund")
    assert result_a is not None
    assert result_a[0] == "call-A"
    # thread-B entry is still there.
    result_b = AuditStash.take_by_tool_name("thread-B", "refund")
    assert result_b is not None
    assert result_b[0] == "call-B"


def test_take_by_tool_name_first_match_wins_when_duplicates() -> None:
    """
    In LangChain HITL, tool_name is unique per interrupt per thread, so
    duplicates should never occur — but if they do, take_by_tool_name
    returns some deterministic entry and removes only that one.
    """
    AuditStash.reset()
    AuditStash.put("thread-1", "call-A", _entry("refund"))
    AuditStash.put("thread-1", "call-B", _entry("refund"))
    result1 = AuditStash.take_by_tool_name("thread-1", "refund")
    assert result1 is not None
    result2 = AuditStash.take_by_tool_name("thread-1", "refund")
    assert result2 is not None
    # Both eventually consumed, in some order.
    assert {result1[0], result2[0]} == {"call-A", "call-B"}
    assert AuditStash.take_by_tool_name("thread-1", "refund") is None
