"""Tests for AuditReceipt schema and HashChain linkage."""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

from dao_ai.audit import (
    AuditReceipt,
    ExecutionStatus,
    ReceiptKind,
    args_hash_of,
)
from dao_ai.audit.chain import HashChain


class _StubSink:
    """Minimal in-memory sink used to exercise HashChain without Lakebase."""

    def __init__(self) -> None:
        self.records: list[AuditReceipt] = []
        self.head_by_thread: dict[str, Optional[str]] = {}

    async def head_hash(self, thread_id: str) -> Optional[str]:
        return self.head_by_thread.get(thread_id)


def _receipt(thread_id: str = "t-1", args: dict | None = None) -> AuditReceipt:
    args = args or {"amount": 42, "customer_id": "C-1"}
    from dao_ai.audit.base import canonical_jcs

    return AuditReceipt(
        receipt_id=uuid.uuid4().hex,
        receipt_kind=ReceiptKind.EXECUTION,
        thread_id=thread_id,
        tool_name="refund_customer",
        args_jcs=canonical_jcs(args),
        args_hash=args_hash_of(args),
        execution_status=ExecutionStatus.OK,
    )


class TestReceiptHashStability:
    def test_this_hash_is_deterministic(self) -> None:
        r = _receipt()
        h1 = r.compute_this_hash()
        h2 = r.compute_this_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_field_change_changes_hash(self) -> None:
        r1 = _receipt()
        r2 = _receipt()
        # Force identical mutable fields so the tool_name change is the only
        # meaningful delta between the two receipts.
        r2.recorded_at = r1.recorded_at
        r2.receipt_id = r1.receipt_id
        assert r1.compute_this_hash() == r2.compute_this_hash()
        r2.tool_name = "different_tool"
        assert r1.compute_this_hash() != r2.compute_this_hash()

    def test_this_hash_excludes_itself(self) -> None:
        """Setting this_hash must not affect the computed value."""
        r = _receipt()
        first = r.compute_this_hash()
        r.this_hash = "poisoned"
        second = r.compute_this_hash()
        assert first == second


def test_hash_chain_links_within_thread() -> None:
    async def scenario() -> tuple[AuditReceipt, AuditReceipt]:
        sink = _StubSink()
        chain = HashChain(sink)
        r1 = await chain.link_and_seal(_receipt())
        r2 = await chain.link_and_seal(_receipt())
        return r1, r2

    sealed_1, sealed_2 = asyncio.run(scenario())
    assert sealed_1.prev_hash is None
    assert sealed_1.this_hash != ""
    assert sealed_2.prev_hash == sealed_1.this_hash
    assert sealed_2.this_hash != sealed_1.this_hash


def test_hash_chain_isolated_by_thread() -> None:
    async def scenario() -> tuple[AuditReceipt, AuditReceipt]:
        sink = _StubSink()
        chain = HashChain(sink)
        sa = await chain.link_and_seal(_receipt(thread_id="thread-A"))
        sb = await chain.link_and_seal(_receipt(thread_id="thread-B"))
        return sa, sb

    sa, sb = asyncio.run(scenario())
    # Each thread starts with its own NULL prev_hash — no cross-linkage.
    assert sa.prev_hash is None
    assert sb.prev_hash is None
    assert sa.this_hash != sb.this_hash


def test_hash_chain_cold_start_uses_sink_head() -> None:
    async def scenario() -> AuditReceipt:
        sink = _StubSink()
        sink.head_by_thread["t-cold"] = "seed-hash"
        chain = HashChain(sink)
        return await chain.link_and_seal(_receipt(thread_id="t-cold"))

    sealed = asyncio.run(scenario())
    assert sealed.prev_hash == "seed-hash"


def test_hash_chain_concurrent_receipts_serialise() -> None:
    """Concurrent link_and_seal calls on the same thread must form a valid linear chain."""

    async def scenario() -> list[AuditReceipt]:
        sink = _StubSink()
        chain = HashChain(sink)

        async def one() -> AuditReceipt:
            return await chain.link_and_seal(_receipt(thread_id="race"))

        return await asyncio.gather(*(one() for _ in range(5)))

    sealed = asyncio.run(scenario())
    hashes = {r.this_hash for r in sealed}
    assert len(hashes) == 5
    # The chain forms a valid linear sequence: every prev_hash must either be
    # None (the first receipt) or match some earlier receipt's this_hash.
    seen: set[Optional[str]] = {None}
    for r in sealed:
        assert r.prev_hash in seen
        seen.add(r.this_hash)
