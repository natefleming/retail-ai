"""Multi-turn sticky-``active_agent`` regression tests for the swarm pattern.

These tests protect the canonical langgraph-swarm semantic that was restored
when the ``is_terminal`` reset behavior was reverted. Specifically:

1. After a deterministic-handoff pipeline runs (``entry -> middle -> end``),
   ``active_agent`` lands on the last node.
2. On a subsequent user turn with the same ``thread_id``, the swarm router
   sticky-resumes at that last node — it does *not* fall through to
   ``default_agent``.
3. Declaring a node with an empty outbound handoff list
   (``handoffs: {end: []}``) — the shape ``commerce_swarm.yaml`` uses for
   composer — must NOT be inferred as a terminal reset trigger. This is the
   specific reversion of the removed structural ``is_terminal`` detection.

The scaffold builds a real ``create_swarm_graph`` (so router + edge wiring is
under test) but stubs the four "expensive" build dependencies to keep the
test hermetic:

    create_checkpointer            -> InMemorySaver
    create_store                   -> None
    create_extraction_manager_...  -> (None, None)
    create_agent_node              -> tiny stub CompiledStateGraph per agent

No LLM, no Databricks resources, no external I/O.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from dao_ai.config import AgentModel, AppConfig, LLMModel
from dao_ai.state import AgentState, Context


def _make_stub_subgraph(agent_name: str) -> CompiledStateGraph:
    """Return a minimal CompiledStateGraph that appends a tagged AIMessage.

    Simulates the shape of ``create_agent_node``'s output without needing a
    real chat model. The handler that wraps this subgraph in
    ``create_agent_node_handler`` extracts the last message and tags it with
    the agent's name (belt-and-suspenders: we already tag here too).
    """

    async def _node(state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(content=f"[{agent_name} ran]", name=agent_name),
            ]
        }

    workflow: StateGraph = StateGraph(
        AgentState,
        input=AgentState,
        output=AgentState,
        context_schema=Context,
    )
    workflow.add_node("run", _node)
    workflow.add_edge(START, "run")
    workflow.add_edge("run", END)
    return workflow.compile()


def _make_swarm_config(handoffs: dict | None = None) -> AppConfig:
    """Three-agent deterministic pipeline: entry -> middle -> end, end: []."""
    agents = [
        AgentModel(name="entry", model=LLMModel(name="test-model")),
        AgentModel(name="middle", model=LLMModel(name="test-model")),
        AgentModel(name="end", model=LLMModel(name="test-model")),
    ]
    default_handoffs: dict = {
        "entry": [{"agent": "middle", "is_deterministic": True}],
        "middle": [{"agent": "end", "is_deterministic": True}],
        "end": [],
    }
    config_dict: dict = {
        "app": {
            "name": "test_app",
            "registered_model": {"name": "test_model"},
            "agents": agents,
            "orchestration": {
                "swarm": {
                    "default_agent": "entry",
                    "handoffs": handoffs if handoffs is not None else default_handoffs,
                }
            },
        }
    }
    return AppConfig(**config_dict)


@pytest.fixture
def swarm_graph() -> CompiledStateGraph:
    """Build a swarm graph with in-memory checkpointer + stub subgraphs.

    Uses the real ``create_swarm_graph`` so router, deterministic-handoff
    handler, and edge wiring are exercised. External build dependencies are
    replaced with lightweight fakes so no LLM or workspace is needed.
    """
    checkpointer: InMemorySaver = InMemorySaver()

    def _fake_create_agent_node(agent: AgentModel, **_kwargs) -> CompiledStateGraph:
        return _make_stub_subgraph(agent.name)

    with (
        patch(
            "dao_ai.orchestration.swarm.create_checkpointer",
            return_value=checkpointer,
        ),
        patch("dao_ai.orchestration.swarm.create_store", return_value=None),
        patch(
            "dao_ai.orchestration.swarm.create_extraction_manager_and_executor",
            return_value=(None, None),
        ),
        patch(
            "dao_ai.orchestration.swarm.create_agent_node",
            side_effect=_fake_create_agent_node,
        ),
    ):
        from dao_ai.orchestration.swarm import create_swarm_graph

        config: AppConfig = _make_swarm_config()
        yield create_swarm_graph(config)


def _run(graph: CompiledStateGraph, message: str, thread_id: str) -> dict:
    """Invoke the graph with a single user message on the given thread."""
    context: Context = Context(thread_id=thread_id, user_id="u")
    invoke_config: dict = {
        "configurable": {"thread_id": thread_id, "user_id": "u"},
    }
    return asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=invoke_config,
            context=context,
        )
    )


def _ai_message_names(state: dict) -> list[str | None]:
    """Extract the ``name`` field from every AIMessage in state.messages."""
    return [
        getattr(m, "name", None)
        for m in state.get("messages", [])
        if isinstance(m, AIMessage)
    ]


@pytest.mark.unit
def test_pipeline_lands_active_agent_at_last_deterministic_target(
    swarm_graph: CompiledStateGraph,
) -> None:
    """After entry -> middle -> end runs, ``active_agent`` sticks at ``end``."""
    result: dict = _run(swarm_graph, "hello", "sticky-t1")

    assert _ai_message_names(result) == ["entry", "middle", "end"], (
        f"Expected pipeline to run all three agents once; "
        f"got AIMessage.name sequence {_ai_message_names(result)}"
    )
    assert result.get("active_agent") == "end", (
        f"Expected active_agent='end' after deterministic pipeline; "
        f"got {result.get('active_agent')!r}"
    )


@pytest.mark.unit
def test_second_turn_stickily_resumes_at_last_active_agent(
    swarm_graph: CompiledStateGraph,
) -> None:
    """Same ``thread_id`` on turn 2 must resume at ``end``, not restart at ``entry``.

    Under canonical swarm semantics, ``active_agent`` persists via the
    checkpointer, and the router uses it as the entry point on the next turn.
    If the removed ``is_terminal`` reset ever comes back, this test fails
    because turn 2 will run the full pipeline again.
    """
    thread_id: str = "sticky-t2"

    r1: dict = _run(swarm_graph, "turn one", thread_id)
    turn1_names: list[str | None] = _ai_message_names(r1)
    assert turn1_names == ["entry", "middle", "end"]
    assert r1.get("active_agent") == "end"

    r2: dict = _run(swarm_graph, "turn two", thread_id)
    turn2_names: list[str | None] = _ai_message_names(r2)

    # Only 'end' should have run on turn 2; entry+middle stay from turn 1.
    new_ai_msgs: list[str | None] = turn2_names[len(turn1_names) :]
    assert new_ai_msgs == ["end"], (
        f"Turn 2 must sticky-resume at 'end' only. Got new AIMessages from "
        f"{new_ai_msgs!r}. If this includes 'entry' or 'middle', the router "
        f"reset active_agent — is_terminal-style reset behavior has been "
        f"reintroduced."
    )
    assert r2.get("active_agent") == "end"


@pytest.mark.unit
def test_empty_outbound_handoffs_does_not_trigger_reset(
    swarm_graph: CompiledStateGraph,
) -> None:
    """``handoffs: {end: []}`` must NOT be inferred as a terminal reset.

    This is the exact shape ``commerce_swarm.yaml`` uses for its composer.
    Under the pre-revert structural inference, an empty handoffs list caused
    the router to clear ``active_agent`` after the node ran. The
    ``swarm_graph`` fixture uses ``end: []`` so if the inference ever comes
    back, this test's turn-2 assertion will fail with the pipeline running
    top-to-bottom again.
    """
    thread_id: str = "sticky-t3"

    r1: dict = _run(swarm_graph, "hello", thread_id)
    assert r1.get("active_agent") == "end", (
        "Under canonical swarm, an empty outbound handoffs list must not "
        "clear active_agent after the node completes its turn."
    )

    r2: dict = _run(swarm_graph, "follow up", thread_id)
    assert r2.get("active_agent") == "end", (
        f"Empty-handoffs config must NOT trigger a between-turn reset. "
        f"Got active_agent={r2.get('active_agent')!r} on turn 2 — expected "
        f"'end' (sticky). The reverted is_terminal structural detection "
        f"appears to have been reintroduced."
    )
