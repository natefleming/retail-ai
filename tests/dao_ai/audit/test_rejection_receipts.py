"""Tests for the HITL audit-rejection tap in dao_ai.hitl.decide_graph_turn."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import AIMessage

from dao_ai.audit import AuditReceipt, AuditSinkManager
from dao_ai.config import (
    AuditModel,
    DatabaseModel,
    HumanInTheLoopModel,
    PythonFunctionModel,
    ToolModel,
)
from dao_ai.hitl import decide_graph_turn


class _FakeSink:
    """Records receipts written by the rejection tap."""

    def __init__(self) -> None:
        self.records: list[AuditReceipt] = []

    async def record(self, receipt: AuditReceipt) -> None:
        self.records.append(receipt)

    async def head_hash(self, thread_id: str) -> Optional[str]:
        return None


class _FakeSnapshot:
    """Stand-in for LangGraph StateSnapshot."""

    def __init__(
        self,
        *,
        interrupts: tuple[Any, ...],
        messages: list[Any],
    ) -> None:
        self.interrupts = interrupts
        self.values: dict[str, Any] = {"messages": messages}


class _FakeInterrupt:
    """Stand-in for langgraph.types.Interrupt."""

    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value


class _FakeGraph:
    """CompiledStateGraph stub — provides aget_state + checkpointer."""

    def __init__(self, snapshot: Optional[_FakeSnapshot] = None) -> None:
        self._snapshot: Optional[_FakeSnapshot] = snapshot
        self.checkpointer = object()  # non-None triggers snapshot path

    async def aget_state(self, config: dict[str, Any]) -> Optional[_FakeSnapshot]:
        return self._snapshot


def _make_refund_tool(audit: Optional[AuditModel]) -> ToolModel:
    return ToolModel(
        name="refund",
        function=PythonFunctionModel(
            name="tests.fixtures.refund",
            human_in_the_loop=HumanInTheLoopModel(
                review_prompt="Approve refund?",
                allowed_decisions=["approve", "reject"],
            ),
            audit=audit,
        ),
    )


def _hitl_snapshot_with_pending_refund() -> _FakeSnapshot:
    """Build a snapshot where a refund tool call is awaiting a decision."""
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "refund",
                "args": {"amount": 42, "customer_id": "C-1"},
                "id": "tool-call-refund-1",
                "type": "tool_call",
            }
        ],
    )
    interrupt_value: dict[str, Any] = {
        "action_requests": [
            {
                "name": "refund",
                "args": {"amount": 42, "customer_id": "C-1"},
                "description": "Approve refund?",
            }
        ],
        "review_configs": [
            {
                "action_name": "refund",
                "allowed_decisions": ["approve", "reject"],
            }
        ],
    }
    return _FakeSnapshot(
        interrupts=(_FakeInterrupt(interrupt_value),),
        messages=[ai_message],
    )


def test_rejection_tap_writes_receipt_for_audited_hitl_tool(monkeypatch: Any) -> None:
    """A `reject` decision on an audited HITL tool must produce a rejection receipt."""
    fake_sink = _FakeSink()
    monkeypatch.setattr(
        AuditSinkManager,
        "for_config",
        classmethod(lambda cls, cfg: fake_sink),  # type: ignore[arg-type]
    )
    # Fake as_tools so PythonFunctionModel doesn't try to import.
    import dao_ai.config as _cfg

    class _FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(
        _cfg.PythonFunctionModel,
        "as_tools",
        lambda self, **_: [_FakeTool("refund")],
    )

    audit_cfg = AuditModel(database=DatabaseModel(project="test-lake"))
    tool_models = [_make_refund_tool(audit=audit_cfg)]

    graph = _FakeGraph(snapshot=_hitl_snapshot_with_pending_refund())

    runtime_config: dict[str, Any] = {"configurable": {"thread_id": "thread-42"}}

    async def scenario() -> None:
        turn = await decide_graph_turn(
            graph=graph,  # type: ignore[arg-type]
            messages=[],
            custom_inputs={
                "decisions": [
                    {"type": "reject", "message": "Not authorised."}
                ]
            },
            runtime_config=runtime_config,
            tool_models=tool_models,
        )
        assert turn.resume_command is not None

    asyncio.run(scenario())

    assert len(fake_sink.records) == 1
    receipt = fake_sink.records[0]
    assert receipt.decision == "reject"
    assert receipt.tool_name == "refund"
    assert receipt.tool_call_id == "tool-call-refund-1"
    assert receipt.receipt_kind == "rejection"
    assert receipt.execution_status == "not_executed_rejected"
    assert (
        receipt.decision_detail == {"message": "Not authorised."}
        if receipt.decision_detail is not None
        else False
    )


def test_rejection_tap_ignores_non_audited_tool(monkeypatch: Any) -> None:
    """A `reject` decision on a HITL-only tool (no audit) must NOT produce a receipt."""
    fake_sink = _FakeSink()
    monkeypatch.setattr(
        AuditSinkManager,
        "for_config",
        classmethod(lambda cls, cfg: fake_sink),  # type: ignore[arg-type]
    )
    import dao_ai.config as _cfg

    class _FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(
        _cfg.PythonFunctionModel,
        "as_tools",
        lambda self, **_: [_FakeTool("refund")],
    )

    tool_models = [_make_refund_tool(audit=None)]  # no audit
    graph = _FakeGraph(snapshot=_hitl_snapshot_with_pending_refund())
    runtime_config: dict[str, Any] = {"configurable": {"thread_id": "thread-42"}}

    async def scenario() -> None:
        await decide_graph_turn(
            graph=graph,  # type: ignore[arg-type]
            messages=[],
            custom_inputs={
                "decisions": [{"type": "reject", "message": "Nope."}]
            },
            runtime_config=runtime_config,
            tool_models=tool_models,
        )

    asyncio.run(scenario())
    assert fake_sink.records == []


def test_rejection_tap_ignores_non_reject_decisions(monkeypatch: Any) -> None:
    """Approve/edit/respond decisions must not trigger the rejection tap."""
    fake_sink = _FakeSink()
    monkeypatch.setattr(
        AuditSinkManager,
        "for_config",
        classmethod(lambda cls, cfg: fake_sink),  # type: ignore[arg-type]
    )
    import dao_ai.config as _cfg

    class _FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(
        _cfg.PythonFunctionModel,
        "as_tools",
        lambda self, **_: [_FakeTool("refund")],
    )

    audit_cfg = AuditModel(database=DatabaseModel(project="test-lake"))
    tool_models = [_make_refund_tool(audit=audit_cfg)]
    graph = _FakeGraph(snapshot=_hitl_snapshot_with_pending_refund())
    runtime_config: dict[str, Any] = {"configurable": {"thread_id": "thread-42"}}

    async def scenario() -> None:
        await decide_graph_turn(
            graph=graph,  # type: ignore[arg-type]
            messages=[],
            custom_inputs={
                "decisions": [{"type": "approve"}]
            },
            runtime_config=runtime_config,
            tool_models=tool_models,
        )

    asyncio.run(scenario())
    # Approvals write receipts at execution time via AuditReceiptMiddleware,
    # not from the rejection tap.
    assert fake_sink.records == []
