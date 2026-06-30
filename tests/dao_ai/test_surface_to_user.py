"""Tests for AgentModel.surface_to_user and the stream-mode gate.

Pins the contracts ``dao_ai.models.LanggraphResponsesAgent`` and
``dao_ai.models.LanggraphChatModel`` rely on:

1. ``AgentModel.surface_to_user`` defaults to ``True`` and round-trips
   through pydantic for YAML configs.
2. ``Context.agent_visibility`` round-trips via ``from_runnable_config``
   so the map published by ``create_swarm_graph`` reaches the streaming
   layer.
3. With ``stream_mode=["messages","updates"]`` the messages channel
   emits ``(AIMessageChunk, metadata)`` tuples whose
   ``metadata["langgraph_node"]`` equals the LangGraph node name.
4. The dao-ai visibility gate (``AgentFilter`` + ``AgentResolver``) drops
   text deltas for agents with ``surface_to_user=False`` while still
   letting their state diffs propagate.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from dao_ai.config import AgentModel, InferenceEndpointModel
from dao_ai.state import Context
from dao_ai.streaming import AgentFilter, AgentResolver


class _GraphState(TypedDict, total=False):
    messages: list[BaseMessage]


def _build_two_agent_graph() -> CompiledStateGraph:
    """Build a deterministic worker→composer graph with FakeListChatModel."""

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
    agent: AgentModel = AgentModel(
        name="alpha",
        model=InferenceEndpointModel(name="fake-endpoint"),
    )
    assert agent.surface_to_user is True


@pytest.mark.unit
def test_agent_model_surface_to_user_false_round_trip() -> None:
    agent: AgentModel = AgentModel(
        name="alpha",
        model=InferenceEndpointModel(name="fake-endpoint"),
        surface_to_user=False,
    )
    assert agent.surface_to_user is False
    assert agent.model_dump()["surface_to_user"] is False


@pytest.mark.unit
def test_context_agent_visibility_round_trips() -> None:
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
def test_stream_mode_messages_metadata_carries_node_name() -> None:
    """Pin the load-bearing fact: ``stream_mode=['messages','updates']``
    surfaces the agent's LangGraph node name in
    ``metadata['langgraph_node']`` — that's the attribution channel the
    visibility filter uses."""
    async def _run() -> list[str]:
        graph: CompiledStateGraph = _build_two_agent_graph()
        seen_nodes: list[str] = []
        async for chunk_kind, payload in graph.astream(
            {"messages": [HumanMessage(content="hi")]},
            stream_mode=["messages", "updates"],
        ):
            if chunk_kind != "messages":
                continue
            _msg, metadata = payload
            node: Any = metadata.get("langgraph_node")
            if isinstance(node, str):
                seen_nodes.append(node)
        return seen_nodes

    nodes: list[str] = asyncio.run(_run())
    assert "worker" in nodes and "composer" in nodes, nodes
    assert nodes.index("worker") < nodes.index("composer"), nodes


@pytest.mark.unit
def test_surface_to_user_gate_silences_worker_text() -> None:
    """Emulate the runtime gate: ``AgentFilter`` constructed from a
    visibility map drops worker chunks while letting composer chunks
    through, and the worker's state update still propagates so
    downstream agents see its output."""
    visibility: dict[str, bool] = {"worker": False, "composer": True}

    async def _run() -> tuple[dict[str, str], list[BaseMessage]]:
        graph: CompiledStateGraph = _build_two_agent_graph()
        agent_filter: AgentFilter = AgentFilter.from_visibility(visibility)
        resolver: AgentResolver = AgentResolver()
        surfaced: dict[str, str] = {"worker": "", "composer": ""}
        final_messages: list[BaseMessage] = []

        async for chunk_kind, payload in graph.astream(
            {"messages": [HumanMessage(content="hi")]},
            stream_mode=["messages", "updates"],
        ):
            if chunk_kind == "updates":
                if isinstance(payload, dict):
                    resolver.record_update(payload)
                    for _node, state in payload.items():
                        if isinstance(state, dict):
                            msgs = state.get("messages") or []
                            for m in msgs:
                                if isinstance(m, BaseMessage):
                                    final_messages.append(m)
                continue
            msg, metadata = payload
            node: str | None = metadata.get("langgraph_node") if metadata else None
            agent: str | None = resolver.attribute(msg, metadata)
            # Fallback to node name when chunk.name not set yet
            attributed: str | None = agent or node
            if not agent_filter.allows(attributed):
                continue
            if attributed in surfaced:
                content = msg.content if isinstance(msg.content, str) else ""
                surfaced[attributed] += content

        return surfaced, final_messages

    surfaced_tokens, final_messages = asyncio.run(_run())
    assert surfaced_tokens["worker"] == ""
    assert "good standing" in surfaced_tokens["composer"]
    worker_msgs: list[AIMessage] = [
        m
        for m in final_messages
        if isinstance(m, AIMessage) and m.name == "worker"
    ]
    assert len(worker_msgs) >= 1
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
