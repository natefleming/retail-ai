"""Tests for NonceIssuer lifecycle using a Protocol-satisfying stub sink."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from dao_ai.audit import AuditNonceError
from dao_ai.audit.nonces import NonceIssuer


class _StubNonceSink:
    """In-memory implementation of NonceStore for tests."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[str, str, datetime, datetime | None]] = {}

    async def record_nonce(
        self,
        *,
        nonce: str,
        thread_id: str,
        tool_call_id: str,
        expires_at: datetime,
    ) -> None:
        self._records[nonce] = (thread_id, tool_call_id, expires_at, None)

    async def consume_nonce(
        self,
        *,
        nonce: str,
        thread_id: str,
        tool_call_id: str,
    ) -> bool:
        entry = self._records.get(nonce)
        if entry is None:
            return False
        rec_thread, rec_tool_call, exp, used_at = entry
        if rec_thread != thread_id or rec_tool_call != tool_call_id:
            return False
        if used_at is not None:
            return False
        if exp <= datetime.now(timezone.utc):
            return False
        self._records[nonce] = (
            rec_thread,
            rec_tool_call,
            exp,
            datetime.now(timezone.utc),
        )
        return True


def test_issue_and_consume_happy_path() -> None:
    async def scenario() -> None:
        sink = _StubNonceSink()
        issuer = NonceIssuer(sink, ttl_seconds=300)
        nonce, exp = await issuer.issue("thread-1", "call-1")
        assert nonce
        assert exp > datetime.now(timezone.utc)
        await issuer.consume(nonce=nonce, thread_id="thread-1", tool_call_id="call-1")

    asyncio.run(scenario())


def test_reuse_rejected() -> None:
    async def scenario() -> None:
        sink = _StubNonceSink()
        issuer = NonceIssuer(sink, ttl_seconds=300)
        nonce, _ = await issuer.issue("thread-1", "call-1")
        await issuer.consume(nonce=nonce, thread_id="thread-1", tool_call_id="call-1")
        with pytest.raises(AuditNonceError):
            await issuer.consume(
                nonce=nonce, thread_id="thread-1", tool_call_id="call-1"
            )

    asyncio.run(scenario())


def test_missing_nonce_rejected() -> None:
    async def scenario() -> None:
        sink = _StubNonceSink()
        issuer = NonceIssuer(sink, ttl_seconds=300)
        with pytest.raises(AuditNonceError):
            await issuer.consume(
                nonce="never-issued", thread_id="thread-1", tool_call_id="call-1"
            )

    asyncio.run(scenario())


def test_wrong_tool_call_id_rejected() -> None:
    async def scenario() -> None:
        sink = _StubNonceSink()
        issuer = NonceIssuer(sink, ttl_seconds=300)
        nonce, _ = await issuer.issue("thread-1", "call-1")
        with pytest.raises(AuditNonceError):
            await issuer.consume(
                nonce=nonce, thread_id="thread-1", tool_call_id="different-call"
            )

    asyncio.run(scenario())


def test_wrong_thread_id_rejected() -> None:
    async def scenario() -> None:
        sink = _StubNonceSink()
        issuer = NonceIssuer(sink, ttl_seconds=300)
        nonce, _ = await issuer.issue("thread-1", "call-1")
        with pytest.raises(AuditNonceError):
            await issuer.consume(
                nonce=nonce, thread_id="different-thread", tool_call_id="call-1"
            )

    asyncio.run(scenario())


def test_expired_nonce_rejected() -> None:
    async def scenario() -> None:
        sink = _StubNonceSink()
        issuer = NonceIssuer(sink, ttl_seconds=300)
        # Manually inject an expired nonce.
        expired_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        await sink.record_nonce(
            nonce="expired-nonce",
            thread_id="thread-1",
            tool_call_id="call-1",
            expires_at=expired_at,
        )
        with pytest.raises(AuditNonceError):
            await issuer.consume(
                nonce="expired-nonce", thread_id="thread-1", tool_call_id="call-1"
            )

    asyncio.run(scenario())
