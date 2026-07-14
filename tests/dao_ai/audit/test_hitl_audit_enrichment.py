"""Tests for AuditedHumanInTheLoopMiddleware enrichment behaviour."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from langchain_core.messages import ToolCall

from dao_ai.audit import AuditReceipt, AuditSinkManager
from dao_ai.config import (
    AuditModel,
    DatabaseModel,
    HumanInTheLoopModel,
)
from dao_ai.middleware.audit_hitl import AuditedHumanInTheLoopMiddleware
from dao_ai.middleware.audit_receipt import AuditStash


class _FakeSinkNonces:
    """Nonce-issuing surface substituted onto FakeSink.nonces."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl_seconds = ttl_seconds
        self.issued: list[tuple[str, str, str, datetime]] = []

    async def issue(self, thread_id: str, tool_call_id: str) -> tuple[str, datetime]:
        nonce = f"nonce-{tool_call_id}"
        exp = datetime.now(timezone.utc) + timedelta(seconds=self._ttl_seconds)
        self.issued.append((nonce, thread_id, tool_call_id, exp))
        return nonce, exp


class _FakeSink:
    """Stand-in for LakebaseAuditSink used by unit tests."""

    def __init__(self) -> None:
        self.nonces = _FakeSinkNonces()
        self.records: list[AuditReceipt] = []

    async def record(self, receipt: AuditReceipt) -> None:
        self.records.append(receipt)

    async def head_hash(self, thread_id: str) -> Optional[str]:
        return None


class _FakeContext:
    """Minimal Context stand-in for the Runtime returned to the middleware."""

    def __init__(self, thread_id: str) -> None:
        self.thread_id: str = thread_id
        self.user_id: Optional[str] = "test-user"
        self.headers: Optional[dict[str, Any]] = {}


class _FakeRuntime:
    """Minimal Runtime[Context] stand-in for the middleware."""

    def __init__(self, thread_id: str) -> None:
        self.context = _FakeContext(thread_id)
        self.config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id}
        }


def _build_middleware(monkeypatch: Any) -> tuple[AuditedHumanInTheLoopMiddleware, _FakeSink]:
    """Wire an AuditedHumanInTheLoopMiddleware whose sink is a fake."""
    fake_sink = _FakeSink()
    monkeypatch.setattr(
        AuditSinkManager,
        "for_config",
        classmethod(lambda cls, cfg: fake_sink),  # type: ignore[arg-type]
    )
    audit_cfg = AuditModel(database=DatabaseModel(project="test-lake"))
    hitl_cfg = HumanInTheLoopModel(
        review_prompt="Approve refund?",
        allowed_decisions=["approve", "reject"],
    )
    middleware = AuditedHumanInTheLoopMiddleware(
        interrupt_on={"refund": {"allowed_decisions": ["approve", "reject"]}},
        audited_tools={"refund": audit_cfg},
        hitl_configs={"refund": hitl_cfg},
    )
    return middleware, fake_sink


def test_create_action_and_config_populates_stash(monkeypatch: Any) -> None:
    AuditStash.reset()
    middleware, fake_sink = _build_middleware(monkeypatch)

    tool_call: ToolCall = {
        "name": "refund",
        "args": {"amount": 42, "customer_id": "C-1"},
        "id": "call-abc",
    }
    runtime = _FakeRuntime(thread_id="t-1")

    async def scenario() -> None:
        # Must run inside a loop so _issue_nonce_sync uses the running-loop path.
        middleware._create_action_and_config(  # type: ignore[arg-type]
            tool_call,
            {"allowed_decisions": ["approve", "reject"]},
            state={"messages": []},  # not used at this depth
            runtime=runtime,  # type: ignore[arg-type]
        )

    asyncio.run(scenario())

    stash = AuditStash.take("t-1", "call-abc")
    assert stash is not None
    assert len(stash.args_hash_at_interrupt) == 64
    # Nonces are process-local (v1) — generated via secrets.token_urlsafe(32),
    # not persisted to Lakebase. Sink stub is no longer invoked.
    assert stash.nonce and len(stash.nonce) >= 32
    assert stash.nonce_exp > datetime.now(timezone.utc)
    assert "Approve refund?" in stash.displayed_summary
    assert "refund" in stash.displayed_summary
    assert fake_sink.nonces.issued == []


def test_non_audited_tool_skips_enrichment(monkeypatch: Any) -> None:
    AuditStash.reset()
    fake_sink = _FakeSink()
    monkeypatch.setattr(
        AuditSinkManager,
        "for_config",
        classmethod(lambda cls, cfg: fake_sink),  # type: ignore[arg-type]
    )
    audit_cfg = AuditModel(database=DatabaseModel(project="test-lake"))
    middleware = AuditedHumanInTheLoopMiddleware(
        interrupt_on={
            "refund": {"allowed_decisions": ["approve", "reject"]},
            "lookup": {"allowed_decisions": ["approve", "reject"]},
        },
        audited_tools={"refund": audit_cfg},  # lookup is NOT audited
        hitl_configs={
            "refund": HumanInTheLoopModel(review_prompt="Approve refund?"),
            "lookup": HumanInTheLoopModel(review_prompt="Approve lookup?"),
        },
    )
    runtime = _FakeRuntime(thread_id="t-1")

    tool_call: ToolCall = {"name": "lookup", "args": {"q": "x"}, "id": "call-lookup"}

    async def scenario() -> None:
        middleware._create_action_and_config(  # type: ignore[arg-type]
            tool_call,
            {"allowed_decisions": ["approve", "reject"]},
            state={"messages": []},
            runtime=runtime,  # type: ignore[arg-type]
        )

    asyncio.run(scenario())

    # No stash entry, no nonce issued — the base HITL behaviour is untouched.
    assert AuditStash.take("t-1", "call-lookup") is None
    assert fake_sink.nonces.issued == []


def test_stash_entry_carries_displayed_summary_with_review_prompt(
    monkeypatch: Any,
) -> None:
    AuditStash.reset()
    middleware, _ = _build_middleware(monkeypatch)

    async def scenario() -> None:
        middleware._create_action_and_config(  # type: ignore[arg-type]
            {"name": "refund", "args": {"a": 1}, "id": "call-x"},
            {"allowed_decisions": ["approve", "reject"]},
            state={"messages": []},
            runtime=_FakeRuntime(thread_id="t-x"),  # type: ignore[arg-type]
        )

    asyncio.run(scenario())
    stash = AuditStash.take("t-x", "call-x")
    assert stash is not None
    # displayed_summary is harness-generated, not model-generated.
    assert stash.displayed_summary.startswith("Approve refund?")
