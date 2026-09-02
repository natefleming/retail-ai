"""Integration tests for the anatomy the streaming agent exposes:

- Tool-call lifecycle surfaces as ``function_call`` / ``function_call_output``
  ``response.output_item.added`` events (with live ``duration_ms``) and is
  mirrored into ``custom_outputs["tool_calls"]`` as one merged record per call.
- Reasoning is streamed on a SEPARATE channel
  (``response.reasoning_summary_text.delta`` + a final ``reasoning`` item) and
  mirrored into ``custom_outputs["reasoning"]`` — it is NOT folded into the
  answer text (no legacy markdown blockquote).

Driven with the same fake-graph harness as ``test_mcp_stream_forwarding``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from langchain_core.messages import AIMessage
from mlflow.types.responses import ResponsesAgentRequest
from mlflow.types.responses_helpers import Message

from dao_ai.models import LanggraphResponsesAgent


def _run_async(coro):
    """Drive a coroutine on a fresh loop, then restore a fresh current loop so
    the suite stays order-independent regardless of ``asyncio.run`` usage in
    sibling tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


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
        input=[Message(role="user", content="do the thing")],
        custom_inputs={"configurable": {"thread_id": "anatomy-test", "user_id": "u"}},
    )


def _collect_events(agent: LanggraphResponsesAgent):
    async def _test():
        events = []
        async for event in agent.apredict_stream(_make_request()):
            events.append(event)
        return events

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None
        return _run_async(_test())


def test_tool_lifecycle_surfaces_as_function_call_items_with_duration():
    tool_run = uuid4()

    async def mock_astream(*args, **kwargs):
        cfg = kwargs.get("config") or {}
        collector = next(
            cb
            for cb in (cfg.get("callbacks") or [])
            if type(cb).__name__ == "_DaoAiStreamCollector"
        )
        await collector.on_tool_start(
            {"name": "search_docs"}, "q", run_id=tool_run, inputs={"query": "hi"}
        )
        await collector.on_tool_end("found 2 docs", run_id=tool_run)
        yield (("agent",), "messages", [AIMessage(content="Here is the answer.")])

    graph = _mock_graph()
    graph.astream = MagicMock(side_effect=lambda *a, **kw: mock_astream(*a, **kw))
    agent = LanggraphResponsesAgent(graph)

    events = _collect_events(agent)

    added = [e for e in events if e.type == "response.output_item.added"]
    calls = [e for e in added if e.item.get("type") == "function_call"]
    outputs = [e for e in added if e.item.get("type") == "function_call_output"]
    assert len(calls) == 1
    assert calls[0].item["name"] == "search_docs"
    assert calls[0].item["call_id"] == str(tool_run)
    assert len(outputs) == 1
    assert outputs[0].item["call_id"] == str(tool_run)
    assert outputs[0].item["status"] == "completed"
    assert isinstance(outputs[0].item["duration_ms"], float)

    done = [e for e in events if e.type == "response.output_item.done"][0]
    tool_calls = done.custom_outputs["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["call_id"] == str(tool_run)
    assert tool_calls[0]["name"] == "search_docs"
    assert tool_calls[0]["status"] == "completed"
    assert "duration_ms" in tool_calls[0]


def test_error_mid_stream_preserves_partial_answer_and_tool_calls():
    """If the graph raises after streaming some content/tools, the terminal
    done event should still carry the partial answer and the captured
    tool-call anatomy in custom_outputs (so the UI keeps the trace/Timeline)."""
    tool_run = uuid4()

    async def mock_astream(*args, **kwargs):
        cfg = kwargs.get("config") or {}
        collector = next(
            cb
            for cb in (cfg.get("callbacks") or [])
            if type(cb).__name__ == "_DaoAiStreamCollector"
        )
        await collector.on_tool_start(
            {"name": "search_docs"}, "q", run_id=tool_run, inputs={}
        )
        await collector.on_tool_end("found it", run_id=tool_run)
        yield (("agent",), "messages", [AIMessage(content="Partial answer before ")])
        raise RuntimeError("upstream 400")

    graph = _mock_graph()
    graph.astream = MagicMock(side_effect=lambda *a, **kw: mock_astream(*a, **kw))
    agent = LanggraphResponsesAgent(graph)

    events = _collect_events(agent)
    done = [e for e in events if e.type == "response.output_item.done"][0]
    answer = "".join(part.get("text", "") for part in done.item["content"])
    assert "Partial answer before" in answer  # partial content preserved
    assert done.custom_outputs is not None
    tool_calls = done.custom_outputs.get("tool_calls", [])
    assert any(t["call_id"] == str(tool_run) for t in tool_calls)


def test_reasoning_streamed_on_separate_channel_not_in_answer_text():
    async def mock_astream(*args, **kwargs):
        yield (
            ("agent",),
            "messages",
            [
                AIMessage(
                    content=[
                        {"type": "reasoning", "reasoning": "let me think"},
                        {"type": "text", "text": "The answer is 42."},
                    ]
                )
            ],
        )

    graph = _mock_graph()
    graph.astream = MagicMock(side_effect=lambda *a, **kw: mock_astream(*a, **kw))
    agent = LanggraphResponsesAgent(graph)

    events = _collect_events(agent)

    reasoning_deltas = [
        e for e in events if e.type == "response.reasoning_summary_text.delta"
    ]
    assert reasoning_deltas, "expected a separate reasoning delta channel"
    assert reasoning_deltas[0].delta == "let me think"

    text_deltas = [e for e in events if e.type == "response.output_text.delta"]
    joined_text = "".join(e.delta for e in text_deltas)
    assert joined_text == "The answer is 42."
    # Reasoning must NOT leak into the answer text as a legacy blockquote.
    assert ">" not in joined_text
    assert "let me think" not in joined_text

    added = [e for e in events if e.type == "response.output_item.added"]
    reasoning_items = [e for e in added if e.item.get("type") == "reasoning"]
    assert len(reasoning_items) == 1

    done = [e for e in events if e.type == "response.output_item.done"][0]
    assert done.custom_outputs["reasoning"] == "let me think"
    # The final answer item carries only the answer text (no reasoning blockquote).
    answer_text = "".join(part.get("text", "") for part in done.item["content"])
    assert answer_text == "The answer is 42."
    assert ">" not in answer_text
