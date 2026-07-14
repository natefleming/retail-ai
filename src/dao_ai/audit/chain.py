"""
Hash-chain helpers for the audit ledger.

Each receipt carries ``prev_hash`` pointing at the previous receipt's
``this_hash`` for the same ``thread_id``. Auditors can walk the chain to
detect any post-hoc mutation of a receipt: changing a row invalidates the
chain from that row forward.

This module tracks the most-recent ``this_hash`` per ``thread_id`` in an
in-process cache. Cold starts fall back to a single query against the sink
for the head-of-chain.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Protocol, runtime_checkable

from dao_ai.audit.base import AuditReceipt


@runtime_checkable
class HeadHashProvider(Protocol):
    """Minimum surface HashChain needs from a sink — enables typed testing stubs."""

    async def head_hash(self, thread_id: str) -> Optional[str]: ...


class HashChain:
    """
    In-process cache of the head-of-chain hash per ``thread_id``.

    Threading model: single asyncio loop; per-thread state is guarded by
    ``asyncio.Lock`` so concurrent receipts within one thread serialise
    their ``prev_hash`` / ``this_hash`` computation.
    """

    def __init__(self, sink: HeadHashProvider) -> None:
        self._sink: HeadHashProvider = sink
        self._heads: dict[str, Optional[str]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, thread_id: str) -> asyncio.Lock:
        lock: Optional[asyncio.Lock] = self._locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[thread_id] = lock
        return lock

    async def link_and_seal(self, receipt: AuditReceipt) -> AuditReceipt:
        """
        Set ``receipt.prev_hash`` from the current head and ``receipt.this_hash``
        from a fresh canonical hash of the sealed body.
        """
        thread_id: str = receipt.thread_id
        async with self._lock_for(thread_id):
            if thread_id not in self._heads:
                # Cold start — query sink for the last hash on this thread.
                self._heads[thread_id] = await self._sink.head_hash(thread_id)
            prev: Optional[str] = self._heads[thread_id]
            receipt.prev_hash = prev
            receipt.this_hash = receipt.compute_this_hash()
            self._heads[thread_id] = receipt.this_hash
            return receipt

    def reset(self) -> None:
        """Clear the in-process cache. Test-only helper."""
        self._heads.clear()
        self._locks.clear()
