"""
Server-issued single-use approval nonce lifecycle.

Nonces bind an approval decision to a specific ``(thread_id, tool_call_id)``
so a captured approval envelope cannot be replayed on a different call or
reused after the fact. Consumption is atomic via ``UPDATE ... WHERE
used_at IS NULL RETURNING`` so the sink guarantees at-most-once use.

The nonce table is created idempotently alongside the receipts table by
``LakebaseAuditSink.ensure_schema``.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable


class AuditNonceError(RuntimeError):
    """Raised when a nonce is missing, expired, reused, or malformed."""


@runtime_checkable
class NonceStore(Protocol):
    """Minimum sink surface NonceIssuer relies on — enables typed testing stubs."""

    async def record_nonce(
        self,
        *,
        nonce: str,
        thread_id: str,
        tool_call_id: str,
        expires_at: datetime,
    ) -> None: ...

    async def consume_nonce(
        self,
        *,
        nonce: str,
        thread_id: str,
        tool_call_id: str,
    ) -> bool: ...


class NonceIssuer:
    """
    Issues and consumes single-use nonces bound to
    ``(thread_id, tool_call_id)``. Fail-closed on any anomaly.
    """

    def __init__(self, sink: NonceStore, ttl_seconds: int) -> None:
        self._sink: NonceStore = sink
        self._ttl_seconds: int = ttl_seconds

    def _new_nonce(self) -> str:
        return secrets.token_urlsafe(32)

    async def issue(
        self,
        thread_id: str,
        tool_call_id: str,
    ) -> tuple[str, datetime]:
        """Issue a fresh nonce; return ``(nonce, exp)`` where ``exp`` is UTC."""
        nonce: str = self._new_nonce()
        exp: datetime = datetime.now(timezone.utc) + timedelta(
            seconds=self._ttl_seconds
        )
        await self._sink.record_nonce(
            nonce=nonce,
            thread_id=thread_id,
            tool_call_id=tool_call_id,
            expires_at=exp,
        )
        return nonce, exp

    async def consume(
        self,
        *,
        nonce: str,
        thread_id: str,
        tool_call_id: str,
    ) -> None:
        """
        Atomically consume ``nonce``, validating it matches
        ``(thread_id, tool_call_id)`` and is unexpired and unused.

        Raises ``AuditNonceError`` on any failure — never fail-open.
        """
        ok: bool = await self._sink.consume_nonce(
            nonce=nonce,
            thread_id=thread_id,
            tool_call_id=tool_call_id,
        )
        if not ok:
            raise AuditNonceError(
                f"Nonce {nonce[:8]}... is missing, expired, reused, or bound to a "
                f"different tool call."
            )
