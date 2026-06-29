"""Tests for AgentModel.surface_to_user and v3 streaming gate behavior.

These tests pin the contract relied on by ``dao_ai.models.LanggraphResponsesAgent``
and ``dao_ai.models.ChatModel`` after the LangGraph 1.2.6 v3 streaming swap:

1. ``AsyncChatModelStream.node`` equals the LangGraph node name (== the dao-ai
   agent name when nodes are added via ``workflow.add_node(name, handler)``).
2. The visibility gate skips text deltas for agents marked
   ``surface_to_user=False`` while still letting their state updates propagate
   downstream (so a composer can read their outputs).
3. The ``AgentModel.surface_to_user`` field defaults to True and is honored by
   the per-agent visibility map published from ``create_swarm_graph``.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict

import pytest
from langchain_core.language_models.chat_model_stream import AsyncChatModelStream
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.stream.run_stream import AsyncGraphRunStream

from dao_ai.config import AgentModel, InferenceEndpointModel
from dao_ai.state import Context


class _GraphState(TypedDict, total=False):
    messages: list[BaseMessage]


def _build_two_agent_graph() -> CompiledStateGraph:
    """Build a deterministic worker→composer graph with FakeListChatModel.

    Each node is registered with the agent name we want to assert appears on
    ``AsyncChatModelStream.node`` — that's the entire load-bearing contract.
    """

    worker_model: FakeListChatModel = FakeListChatModel(
        responses=["secret credit headroom $42,000"]
    )
    composer_model: FakeListChatModel = FakeListChatModel(
        responses=["Your account is in good standing."]
    )

    async def worker(
        state: _GraphState, config: RunnableConfig
    ) -> dict[str, list[BaseMessage]]:
        reply: BaseMessage = await worker_model.ainvoke(
            state.get("messages", []), config=config
        )
        # Tag with the agent name so downstream visibility checks have a key
        # to gate on even if .node attribution falters.
        if isinstance(reply, AIMessage):
            reply = reply.model_copy(update={"name": "worker"})
        return {"messages": state.get("messages", []) + [reply]}

    async def composer(
        state: _GraphState, config: RunnableConfig
    ) -> dict[str, list[BaseMessage]]:
        reply: BaseMessage = await composer_model.ainvoke(
            state.get("messages", []), config=config
        )
        if isinstance(reply, AIMessage):
            reply = reply.model_copy(update={"name": "composer"})
        return {"messages": state.get("messages", []) + [reply]}

    sg: StateGraph = StateGraph(_GraphState)
    sg.add_node("worker", worker)
    sg.add_node("composer", composer)
    sg.set_entry_point("worker")
    sg.add_edge("worker", "composer")
    sg.set_finish_point("composer")
    return sg.compile()


@pytest.mark.unit
def test_agent_model_surface_to_user_defaults_true() -> None:
    """AgentModel.surface_to_user must default to True so existing configs are unaffected."""
    agent: AgentModel = AgentModel(
        name="alpha",
        model=InferenceEndpointModel(name="fake-endpoint"),
    )
    assert agent.surface_to_user is True


@pytest.mark.unit
def test_agent_model_surface_to_user_false_round_trip() -> None:
    """surface_to_user=False round-trips through pydantic for YAML configs."""
    agent: AgentModel = AgentModel(
        name="alpha",
        model=InferenceEndpointModel(name="fake-endpoint"),
        surface_to_user=False,
    )
    assert agent.surface_to_user is False
    assert agent.model_dump()["surface_to_user"] is False


@pytest.mark.unit
def test_context_agent_visibility_round_trips() -> None:
    """Context.agent_visibility must round-trip via from_runnable_config so the
    map published by create_swarm_graph reaches the streaming layer."""
    ctx: Context = Context.from_runnable_config(
        {
            "configurable": {
                "user_id": "u1",
                "thread_id": "t1",
                "agent_visibility": {"worker": False, "composer": True},
            }
        }
    )
    assert ctx.agent_visibility == {"worker": False, "composer": True}


@pytest.mark.unit
def test_v3_chat_model_stream_node_equals_agent_name() -> None:
    """AsyncChatModelStream.node must equal the LangGraph node name.

    This is the load-bearing assumption behind the visibility gate in
    ``models.py``: we look up ``visibility[chat.node]`` to decide whether to
    yield ``chat.text`` deltas to the user stream.
    """

    async def _run() -> list[str]:
        graph: CompiledStateGraph = _build_two_agent_graph()
        stream: AsyncGraphRunStream = await graph.astream_events(
            {"messages": [HumanMessage(content="hi")]},
            version="v3",
        )
        seen_nodes: list[str] = []
        chat: AsyncChatModelStream
        async for chat in stream.messages:
            async for _ in chat.text:
                pass
            if chat.node is not None:
                seen_nodes.append(chat.node)
        return seen_nodes

    nodes: list[str] = asyncio.run(_run())
    # Each agent emits at least one stream chunk attributed to its node; some
    # provider stacks (e.g. FakeListChatModel via Pregel) may emit multiple
    # chunks per node. Assert both names appear and worker precedes composer.
    assert "worker" in nodes and "composer" in nodes, nodes
    assert nodes.index("worker") < nodes.index("composer"), nodes


@pytest.mark.unit
def test_surface_to_user_gate_silences_worker_text() -> None:
    """Emulate the gate in ``LanggraphResponsesAgent.apredict_stream``: when
    ``visibility[worker] is False`` the worker's text deltas must not surface
    to the user stream, but its state-propagated message must still be visible
    to downstream consumers via ``stream.values``."""
    visibility: dict[str, bool] = {"worker": False, "composer": True}

    async def _run() -> tuple[dict[str, str], list[BaseMessage]]:
        graph: CompiledStateGraph = _build_two_agent_graph()
        stream: AsyncGraphRunStream = await graph.astream_events(
            {"messages": [HumanMessage(content="hi")]},
            version="v3",
        )
        surfaced_tokens: dict[str, str] = {"worker": "", "composer": ""}
        final_messages: list[BaseMessage] = []

        async def _gate_text() -> None:
            chat: AsyncChatModelStream
            async for chat in stream.messages:
                node: str | None = chat.node
                allow: bool = (
                    node is None or visibility.get(node, True) is not False
                )
                async for token in chat.text:
                    if allow and node is not None:
                        surfaced_tokens[node] += token

        async def _capture_values() -> None:
            state: Any
            async for state in stream.values:
                if isinstance(state, dict):
                    final_messages[:] = list(state.get("messages", []) or [])

        await asyncio.gather(_gate_text(), _capture_values())
        return surfaced_tokens, final_messages

    surfaced_tokens, final_messages = asyncio.run(_run())

    # Worker emitted text but the gate silenced it
    assert surfaced_tokens["worker"] == ""

    # Composer's text reached the user stream
    assert "good standing" in surfaced_tokens["composer"]

    # Worker's AIMessage IS present in the final state — downstream agents and
    # the composer can read it. Visibility gating is a *display* concern, not
    # a state-propagation one.
    worker_msgs: list[AIMessage] = [
        m
        for m in final_messages
        if isinstance(m, AIMessage) and m.name == "worker"
    ]
    assert len(worker_msgs) == 1
    # FakeListChatModel emits content as either a str or a list of block dicts;
    # normalize before assertion so the test is shape-agnostic.
    raw: Any = worker_msgs[0].content
    flattened: str
    if isinstance(raw, str):
        flattened = raw
    elif isinstance(raw, list):
        flattened = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw
        )
    else:
        flattened = str(raw)
    assert "credit headroom" in flattened
