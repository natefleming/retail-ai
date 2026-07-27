"""Tests for MCP callback → outer LangGraph stream forwarding via LangChain's
callback manager.

Covers the wire contract:
- MCP progress notifications and audit receipts from a tool call reach an
  ``AsyncCallbackHandler`` registered on the outer ``RunnableConfig``.
- ``apredict_stream`` attaches its own collector handler and translates
  captured envelopes into ``response.output_item.added(status="in_progress")``
  events between astream chunks.
- Envelopes accumulate into ``custom_outputs["mcp_events"]`` on the terminal
  ``response.output_item.done`` event for non-streaming replay.
- When no runnable context is present (e.g. batch predict without a graph
  stream), the callbacks still emit MLflow span events but skip dispatch.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from mlflow.types.responses import ResponsesAgentRequest
from mlflow.types.responses_helpers import Message

from dao_ai.models import LanggraphResponsesAgent
from dao_ai.tools.mcp_callbacks import (
    DaoAiProgressCallback,
)


def _run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_graph() -> MagicMock:
    graph = MagicMock()
    graph.ainvoke = AsyncMock()
    graph.astream = AsyncMock()
    graph.aget_state = AsyncMock()
    graph.checkpointer = None
    snapshot = MagicMock()
    snapshot.interrupts = ()
    snapshot.values = {}
    graph.aget_state.return_value = snapshot
    return graph


def _make_request() -> ResponsesAgentRequest:
    return ResponsesAgentRequest(
        input=[Message(role="user", content="Run the MCP tool")],
        custom_inputs={
            "configurable": {
                "thread_id": "mcp-stream-test",
                "user_id": "test_user",
            }
        },
    )


def test_apredict_stream_forwards_mcp_envelopes_via_collector():
    """Envelopes pushed onto the collector's queue during astream are
    drained between chunks and yielded as
    ``response.output_item.added(status="in_progress")`` events.

    Exercises both surviving producers on the shared forwarding path: MCP
    progress (``mcp.*``) and audit receipts (``dao_ai.audit.*``)."""

    progress_envelope: dict[str, Any] = {
        "channel": "mcp.progress",
        "server_name": "genie",
        "tool_name": "run_genie_query",
        "progress": 0.3,
        "total": 1.0,
        "message": "Fetched 3/10 docs",
    }
    audit_envelope: dict[str, Any] = {
        "channel": "dao_ai.audit.receipt",
        "action": "genie.query",
        "status": "completed",
    }

    captured_collector: dict[str, Any] = {}

    async def mock_astream(*args, **kwargs):
        # Capture the collector attached to the config so we can
        # simulate a tool dispatching custom events during the run.
        cfg = kwargs.get("config") or (args[2] if len(args) > 2 else {})
        for cb in cfg.get("callbacks", []) or []:
            if type(cb).__name__ == "_McpEventCollector":
                captured_collector["cb"] = cb
                break

        collector = captured_collector.get("cb")
        assert collector is not None, "apredict_stream must attach the collector"
        # Simulate two envelopes arriving mid-stream.
        from uuid import uuid4

        await collector.on_custom_event(
            "mcp.progress", progress_envelope, run_id=uuid4()
        )
        await collector.on_custom_event(
            "dao_ai.audit.receipt", audit_envelope, run_id=uuid4()
        )
        yield (
            ("agent",),
            "messages",
            [MagicMock(content="done", type="ai")],
        )

    graph = _mock_graph()
    graph.astream = MagicMock(side_effect=lambda *a, **kw: mock_astream(*a, **kw))
    agent = LanggraphResponsesAgent(graph)

    async def _test():
        events = []
        async for event in agent.apredict_stream(_make_request()):
            events.append(event)
        return events

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None
        events = _run_async(_test())

    added = [e for e in events if e.type == "response.output_item.added"]
    assert len(added) == 2, f"expected 2 added events, got {len(added)}: {events}"
    assert all(e.item["status"] == "in_progress" for e in added)
    assert added[0].item["name"] == "mcp.progress"
    assert added[0].item["input"] == progress_envelope
    assert added[1].item["name"] == "dao_ai.audit.receipt"
    assert added[1].item["input"] == audit_envelope
    assert added[0].item["id"] != added[1].item["id"]

    done = [e for e in events if e.type == "response.output_item.done"]
    assert len(done) == 1
    assert done[0].custom_outputs["mcp_events"] == [progress_envelope, audit_envelope]


def test_apredict_stream_collector_ignores_non_mcp_events():
    """Custom events not matching the ``mcp.*`` channel prefix must be
    dropped — other tools may also dispatch via the same callback manager."""

    async def mock_astream(*args, **kwargs):
        cfg = kwargs.get("config") or {}
        collector = None
        for cb in cfg.get("callbacks", []) or []:
            if type(cb).__name__ == "_McpEventCollector":
                collector = cb
                break
        assert collector is not None
        from uuid import uuid4

        # Non-dict, dict without channel, and non-mcp channel — all ignored.
        await collector.on_custom_event("other", "string payload", run_id=uuid4())
        await collector.on_custom_event(
            "other", {"unrelated": "payload"}, run_id=uuid4()
        )
        await collector.on_custom_event(
            "other", {"channel": "not.mcp", "x": 1}, run_id=uuid4()
        )
        yield (
            ("agent",),
            "messages",
            [MagicMock(content="hi", type="ai")],
        )

    graph = _mock_graph()
    graph.astream = MagicMock(side_effect=lambda *a, **kw: mock_astream(*a, **kw))
    agent = LanggraphResponsesAgent(graph)

    async def _test():
        events = []
        async for event in agent.apredict_stream(_make_request()):
            events.append(event)
        return events

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None
        events = _run_async(_test())

    added = [e for e in events if e.type == "response.output_item.added"]
    assert added == [], f"non-mcp events should be dropped, got {added}"
    done = [e for e in events if e.type == "response.output_item.done"][0]
    assert "mcp_events" not in done.custom_outputs


def test_progress_callback_dispatches_via_callback_manager():
    """``DaoAiProgressCallback`` captures the RunnableConfig at construction
    and dispatches envelopes through ``adispatch_custom_event`` on notify."""

    from langchain_mcp_adapters.callbacks import CallbackContext

    fake_config = {"callbacks": [MagicMock()]}
    with (
        patch("dao_ai.tools.mcp_callbacks.ensure_config", return_value=fake_config),
        patch(
            "dao_ai.tools.mcp_callbacks.adispatch_custom_event",
            new_callable=AsyncMock,
        ) as dispatch,
        patch("dao_ai.tools.mcp_callbacks._add_span_event") as span,
    ):
        cb = DaoAiProgressCallback()
        ctx = CallbackContext(server_name="genie", tool_name="q")
        _run_async(cb(0.5, 1.0, "half done", ctx))

    dispatch.assert_called_once()
    args, kwargs = dispatch.call_args
    assert args[0] == "mcp.progress"
    envelope = args[1]
    assert envelope["channel"] == "mcp.progress"
    assert envelope["server_name"] == "genie"
    assert envelope["progress"] == 0.5
    assert envelope["message"] == "half done"
    assert kwargs["config"] is fake_config
    span.assert_called_once_with("mcp.progress", envelope)


def test_callback_dispatch_safe_outside_runnable_context():
    """When ``ensure_config`` returns no callbacks (no runnable context /
    batch predict) the callback still emits its span event but does not
    dispatch (dispatch would raise inside adispatch_custom_event)."""

    from langchain_mcp_adapters.callbacks import CallbackContext

    def _boom(_config):
        raise RuntimeError("not inside a runnable context")

    with (
        patch("dao_ai.tools.mcp_callbacks.ensure_config", side_effect=_boom),
        patch(
            "dao_ai.tools.mcp_callbacks.adispatch_custom_event",
            new_callable=AsyncMock,
        ) as dispatch,
        patch("dao_ai.tools.mcp_callbacks._add_span_event") as span,
    ):
        cb = DaoAiProgressCallback()
        ctx = CallbackContext(server_name="s", tool_name="t")
        _run_async(cb(0.1, 1.0, "tick", ctx))

    # Span still fires (observability preserved).
    span.assert_called_once()
    # Dispatch skipped because config capture failed at __init__.
    dispatch.assert_not_called()
