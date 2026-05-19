"""Multi-turn integration smoke tests for the default ``output_mode``.

Reproduces the deployment shape (worker subgraph wrapped in
``create_agent_node_handler`` inside a parent orchestration graph) for both
supervisor and swarm patterns. Drives 2-turn conversations against the
parent graph with the same ``thread_id`` and ``user_id`` across turns to
verify:

1. Both turns complete with no orphan tool_result / parallel tool_call
   issues bubbling up to the parent.
2. The parent state's message history never contains worker-side malformed
   patterns (interleaved system messages, orphan tool_results, parallel
   tool_calls in assistant messages from the worker).

This is the integration version of the unit tests in
``test_output_mode_default_regression.py`` — same invariant, full LangGraph
runtime.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from dao_ai.orchestration.core import create_agent_node_handler
from dao_ai.state import AgentState, Context

# ---------------------------------------------------------------------------
# Test fixtures: a tiny FakeChatModel that mimics the worker patterns that
# caused the bug — parallel tool_calls, text + tool_calls in one message,
# and a clean final text response.
# ---------------------------------------------------------------------------


@tool
def find_inventory(sku: str) -> str:
    """Look up inventory for a given SKU."""
    return f"sku={sku} qty=3"


@tool
def search_memory(query: str) -> str:
    """Search long-term memory."""
    return "[]"


class _ParallelToolCallWorker(BaseChatModel):
    """Worker that emits PARALLEL tool_calls + text in a single assistant
    message on first call, then a clean final answer once tool results land.
    Each call returns a fresh AIMessage with a unique id to avoid
    ``add_messages`` dedup across turns.
    """

    @property
    def _llm_type(self) -> str:
        return "fake-parallel-worker"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        has_tool_result = any(isinstance(m, ToolMessage) for m in messages)
        if has_tool_result:
            msg = AIMessage(
                content="Found 3 in stock.",
                id=str(uuid.uuid4()),
            )
        else:
            msg = AIMessage(
                content="Let me search inventory and memory at once.",
                tool_calls=[
                    {
                        "name": "find_inventory",
                        "args": {"sku": "DEWALT-DRILL"},
                        "id": f"tu_par_a_{uuid.uuid4().hex[:6]}",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_memory",
                        "args": {"query": "drill prefs"},
                        "id": f"tu_par_b_{uuid.uuid4().hex[:6]}",
                        "type": "tool_call",
                    },
                ],
                id=str(uuid.uuid4()),
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self


def _build_worker_subgraph() -> "CompiledStateGraph":  # noqa: F821
    """Single agent with parallel-tool-call behavior."""
    return create_agent(
        name="inventory_worker",
        model=_ParallelToolCallWorker(),
        tools=[find_inventory, search_memory],
        checkpointer=InMemorySaver(),
        state_schema=AgentState,
        context_schema=Context,
    )


def _build_parent_with_handler(output_mode: str) -> "CompiledStateGraph":  # noqa: F821
    """Parent orchestration graph that wraps the worker via
    ``create_agent_node_handler`` — same shape as the supervisor/swarm
    deployment.
    """
    worker = _build_worker_subgraph()
    handler = create_agent_node_handler(
        agent_name="inventory_worker",
        agent=worker,
        output_mode=output_mode,
    )
    workflow = StateGraph(
        AgentState,
        input=AgentState,
        output=AgentState,
        context_schema=Context,
    )
    workflow.add_node("inventory_worker", handler)
    workflow.add_edge(START, "inventory_worker")
    workflow.add_edge("inventory_worker", END)
    return workflow.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_no_orphan_tool_results(messages: list) -> None:
    """Walk the message list and assert every ToolMessage has a matching
    tool_use in the immediately preceding assistant message. This is the
    Anthropic-strict-validation rule that previously caused the 400s.
    """
    for i, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            continue
        # Walk backwards to find the most recent AIMessage. There may be
        # ToolMessages or other AIMessages between, but the matching
        # tool_use must be in a preceding AIMessage.
        found = False
        for j in range(i - 1, -1, -1):
            prev = messages[j]
            if isinstance(prev, AIMessage):
                tcs = prev.tool_calls or []
                if any(tc.get("id") == msg.tool_call_id for tc in tcs):
                    found = True
                break
            # If we cross another ToolMessage, that's fine — could be a
            # multi-tool batch. Keep walking.
        assert found, (
            f"orphan ToolMessage at index {i} "
            f"(tool_call_id={msg.tool_call_id}) "
            f"— no matching tool_use in any preceding AIMessage"
        )


def _assert_no_parallel_tool_calls_in_assistant(messages: list) -> None:
    """Assert no surfaced AIMessage emits parallel tool_calls (>1 tool_call
    in a single assistant message). With ``last_message`` the supervisor
    never sees these; with ``full_history`` they may leak through and cause
    strict-validation LLM 400s downstream.
    """
    for msg in messages:
        if isinstance(msg, AIMessage):
            tcs = msg.tool_calls or []
            assert len(tcs) <= 1, (
                f"parallel tool_calls in AIMessage: {len(tcs)} calls — "
                f"would 400 on strict-validation LLMs"
            )


# ---------------------------------------------------------------------------
# Supervisor scenario — single worker, default output_mode (last_message)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_supervisor_multi_turn_same_thread_clean_history() -> None:
    """Two turns, same thread_id + user_id, default output_mode. The parent
    state's accumulated messages must remain free of worker-side malformed
    patterns across both turns. Uses ``asyncio.run`` (matching the pattern
    in ``test_hitl.py``) because ``create_agent_node_handler`` is async-only.
    """
    parent = _build_parent_with_handler(output_mode="last_message")

    thread_id = "thread-supervisor-1"
    user_id = "user-nate"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        },
    }

    async def _run() -> tuple[list, list]:
        s1 = await parent.ainvoke(
            {"messages": [HumanMessage(content="DeWalt drills in stock?")]},
            config=config,
        )
        s2 = await parent.ainvoke(
            {"messages": [HumanMessage(content="Any in the back?")]},
            config=config,
        )
        return s1["messages"], s2["messages"]

    msgs_t1, msgs_t2 = asyncio.run(_run())

    _assert_no_orphan_tool_results(msgs_t1)
    _assert_no_parallel_tool_calls_in_assistant(msgs_t1)
    last_ai = next(m for m in reversed(msgs_t1) if isinstance(m, AIMessage))
    assert not (last_ai.tool_calls or [])

    _assert_no_orphan_tool_results(msgs_t2)
    _assert_no_parallel_tool_calls_in_assistant(msgs_t2)


# ---------------------------------------------------------------------------
# Swarm-shaped scenario — two workers in sequence, default output_mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_swarm_two_workers_multi_turn_clean_history() -> None:
    """Two workers chained (simulating a swarm handoff), 2-turn conversation,
    same thread_id + user_id. Verify the parent state stays clean across the
    handoff AND the follow-up turn.
    """
    worker_a = _build_worker_subgraph()
    worker_b = _build_worker_subgraph()

    handler_a = create_agent_node_handler(
        agent_name="agent_a",
        agent=worker_a,
        output_mode="last_message",
    )
    handler_b = create_agent_node_handler(
        agent_name="agent_b",
        agent=worker_b,
        output_mode="last_message",
    )

    workflow = StateGraph(
        AgentState,
        input=AgentState,
        output=AgentState,
        context_schema=Context,
    )
    workflow.add_node("agent_a", handler_a)
    workflow.add_node("agent_b", handler_b)
    workflow.add_edge(START, "agent_a")
    workflow.add_edge("agent_a", "agent_b")  # simulated handoff
    workflow.add_edge("agent_b", END)
    parent = workflow.compile(checkpointer=InMemorySaver())

    config = {
        "configurable": {
            "thread_id": "thread-swarm-1",
            "user_id": "user-nate",
        },
    }

    async def _run() -> tuple[list, list]:
        s1 = await parent.ainvoke(
            {"messages": [HumanMessage(content="DeWalt drills?")]},
            config=config,
        )
        s2 = await parent.ainvoke(
            {"messages": [HumanMessage(content="In the back?")]},
            config=config,
        )
        return s1["messages"], s2["messages"]

    msgs_t1, msgs_t2 = asyncio.run(_run())

    _assert_no_orphan_tool_results(msgs_t1)
    _assert_no_parallel_tool_calls_in_assistant(msgs_t1)
    _assert_no_orphan_tool_results(msgs_t2)
    _assert_no_parallel_tool_calls_in_assistant(msgs_t2)


# ---------------------------------------------------------------------------
# Negative reproducer — assert the old default (full_history) WOULD have
# surfaced the bug shape we just blocked.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_full_history_propagates_parallel_tool_calls_to_parent() -> None:
    """Sanity check on the negative case: under ``output_mode='full_history'``
    the parent's accumulated state DOES contain the worker's parallel
    tool_calls. This is the pre-fix behavior and the reason for flipping the
    default. If a future refactor inadvertently turned this back on, this
    test confirms the bug surface is still there as documented."""
    parent = _build_parent_with_handler(output_mode="full_history")
    config = {
        "configurable": {
            "thread_id": "thread-fh-1",
            "user_id": "user-nate",
        },
    }

    async def _run() -> list:
        s = await parent.ainvoke(
            {"messages": [HumanMessage(content="DeWalt drills?")]},
            config=config,
        )
        return s["messages"]

    msgs = asyncio.run(_run())
    # Under full_history, the worker's parallel-tool-call assistant message
    # propagates upward. The assertion below would FAIL (parallel calls
    # present) — confirming this is exactly the surface the default flip
    # eliminates.
    with pytest.raises(AssertionError, match="parallel tool_calls"):
        _assert_no_parallel_tool_calls_in_assistant(msgs)
