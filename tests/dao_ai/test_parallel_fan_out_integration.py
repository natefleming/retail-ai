"""Integration test for parallel fan-out swarm topology.

Builds a swarm ``source -> {worker_a, worker_b, worker_c} -> join`` using the
real ``create_swarm_graph`` (so router + edge wiring + sibling handler
wrapping is under test) with stub agent subgraphs. External build
dependencies are replaced with lightweight fakes so no LLM or workspace is
needed.

The source stub returns a ``Command`` that mimics the state produced when
an LLM invokes three parallel handoff tools in a single turn (parent-level
``goto`` list). LangGraph runs the three siblings in the same superstep,
each sibling has a static edge to the shared join, and the join runs
exactly once.

Assertions:

1. The source produces exactly one AIMessage and hands off to all three
   siblings in the same superstep.
2. Each sibling produces exactly one AIMessage tagged with its own name.
3. The join agent runs exactly once and appears last in the AIMessage sequence.
4. ``active_agent`` lands on the join.
5. The join's inbound message tail is a HumanMessage (the bridge appended by
   ``_create_deterministic_handler`` when a sibling routes to the join).
6. On a follow-up turn with the same ``thread_id``, the router resumes at
   the join (not the source) — proves the sibling wrapper correctly
   persists ``active_agent`` for checkpoint resume.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from dao_ai.config import AgentModel, AppConfig, LLMModel
from dao_ai.state import AgentState, Context


def _make_source_subgraph(sibling_names: tuple[str, ...]) -> CompiledStateGraph:
    """Source stub that mimics real parallel-handoff-tool output.

    Emits an AIMessage with N tool_calls plus N paired ToolMessages, AND
    populates the ``parallel_dispatches`` state field — exactly what the
    real ``create_handoff_tool(parallel=True)`` produces when the LLM
    invokes N parallel handoff tools in one turn (each tool returns a
    Command with ``update={parallel_dispatches: [target]}``, and the
    reducer merges them). The source-agent wrapper reads this field to
    dispatch a single fan-out Command with ``goto=[Send(sibling), ...]``.
    """
    from dao_ai.orchestration import PARALLEL_DISPATCH_STATE_KEY

    async def _node(state: AgentState) -> dict:
        tool_calls: list[dict] = [
            {"name": f"handoff_to_{sibling}", "args": {}, "id": f"call_{sibling}"}
            for sibling in sibling_names
        ]
        ai_msg: AIMessage = AIMessage(
            content="Fanning out to workers.",
            name="source",
            tool_calls=tool_calls,
        )
        tool_msgs: list[ToolMessage] = [
            ToolMessage(
                content=f"Transferred to {sibling}",
                tool_call_id=f"call_{sibling}",
            )
            for sibling in sibling_names
        ]
        return {
            "messages": [ai_msg, *tool_msgs],
            PARALLEL_DISPATCH_STATE_KEY: list(sibling_names),
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


def _make_stub_subgraph(agent_name: str) -> CompiledStateGraph:
    """Return a minimal CompiledStateGraph that appends a tagged AIMessage."""

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


SIBLINGS: tuple[str, ...] = ("worker_a", "worker_b", "worker_c")


def _make_fanout_config() -> AppConfig:
    """source fans out to worker_a/b/c which all converge on join."""
    agents = [
        AgentModel(name="source", model=LLMModel(name="test-model")),
        AgentModel(name="worker_a", model=LLMModel(name="test-model")),
        AgentModel(name="worker_b", model=LLMModel(name="test-model")),
        AgentModel(name="worker_c", model=LLMModel(name="test-model")),
        AgentModel(name="join", model=LLMModel(name="test-model")),
    ]
    handoffs: dict = {
        "source": [
            {
                "agents": ["worker_a", "worker_b", "worker_c"],
                "join": "join",
            }
        ],
        "worker_a": [],
        "worker_b": [],
        "worker_c": [],
        "join": [],
    }
    return AppConfig(
        **{
            "app": {
                "name": "test_app",
                "registered_model": {"name": "test_model"},
                "agents": agents,
                "orchestration": {
                    "swarm": {
                        "default_agent": "source",
                        "handoffs": handoffs,
                    }
                },
            }
        }
    )


@pytest.fixture
def swarm_graph() -> CompiledStateGraph:
    """Build a fan-out swarm graph with stub subgraphs."""
    checkpointer: InMemorySaver = InMemorySaver()

    def _fake_create_agent_node(agent: AgentModel, **_kwargs) -> CompiledStateGraph:
        if agent.name == "source":
            return _make_source_subgraph(SIBLINGS)
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

        config: AppConfig = _make_fanout_config()
        yield create_swarm_graph(config)


def _run(graph: CompiledStateGraph, message: str, thread_id: str) -> dict:
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
    return [
        m.name if isinstance(m, AIMessage) else None
        for m in state.get("messages", [])
        if isinstance(m, AIMessage)
    ]


@pytest.mark.unit
def test_fan_out_runs_all_siblings_once_and_join_once(
    swarm_graph: CompiledStateGraph,
) -> None:
    result: dict = _run(swarm_graph, "hello", "fanout-t1")
    names: list[str | None] = _ai_message_names(result)

    # source AIMessage first, then the three siblings (order not
    # guaranteed across concurrent branches), then join last.
    assert names[0] == "source", f"first AIMessage must be source, got {names!r}"
    assert names[-1] == "join", f"last AIMessage must be join, got {names!r}"

    middle: list[str | None] = names[1:-1]
    assert sorted(middle) == sorted(SIBLINGS), (
        f"exactly the three siblings must appear between source and join once "
        f"each; got {middle!r}"
    )


@pytest.mark.unit
def test_fan_out_active_agent_lands_at_join(
    swarm_graph: CompiledStateGraph,
) -> None:
    result: dict = _run(swarm_graph, "hello", "fanout-t2")
    assert result.get("active_agent") == "join", (
        f"active_agent must land at 'join' after fan-out; "
        f"got {result.get('active_agent')!r}"
    )


@pytest.mark.unit
def test_fan_out_join_receives_human_message_bridge(
    swarm_graph: CompiledStateGraph,
) -> None:
    """The join agent's inbound tail must be a HumanMessage bridge.

    Each sibling's handler is wrapped by ``_create_deterministic_handler``
    with target=join. The wrapper appends a HumanMessage bridge whenever
    the sibling's turn ends with an AIMessage — required so downstream
    LLMs don't see an assistant tail. Because all three siblings wrap
    concurrently, multiple bridges end up in the message list; the join
    still sees a HumanMessage-tail just BEFORE its own AIMessage.
    """
    result: dict = _run(swarm_graph, "please synthesize", "fanout-t3")
    messages: list[BaseMessage] = result.get("messages", [])
    assert messages, "graph must produce messages"

    # Locate the join's AIMessage and confirm the preceding message is a
    # HumanMessage bridge (any of the sibling-emitted bridges).
    join_idx: int = next(
        i
        for i, m in enumerate(messages)
        if isinstance(m, AIMessage) and m.name == "join"
    )
    prior: BaseMessage = messages[join_idx - 1]
    assert isinstance(prior, HumanMessage), (
        f"join's inbound tail must be a HumanMessage bridge; got {type(prior).__name__}"
    )
    assert prior.name == "__deterministic_handoff__", (
        f"bridge should be tagged as a deterministic-handoff bridge; got name={prior.name!r}"
    )


@pytest.mark.unit
def test_follow_up_turn_resumes_at_join(
    swarm_graph: CompiledStateGraph,
) -> None:
    """Turn 2 on the same thread must resume at join, not restart at source.

    Proves the sibling handler wrapper persists ``active_agent = join`` via
    the checkpointer (the sticky-swarm resume path). If the wrapper is
    ever removed, this test fails because turn 2 will fan out again.
    """
    thread_id: str = "fanout-t4"

    r1: dict = _run(swarm_graph, "turn one", thread_id)
    turn1_names: list[str | None] = _ai_message_names(r1)
    assert r1.get("active_agent") == "join"
    assert turn1_names[-1] == "join"

    r2: dict = _run(swarm_graph, "turn two", thread_id)
    turn2_names: list[str | None] = _ai_message_names(r2)
    new_names: list[str | None] = turn2_names[len(turn1_names) :]

    # Only join should have run on turn 2 — no re-fan-out.
    assert new_names == ["join"], (
        f"Turn 2 must sticky-resume at 'join' only. Got new AIMessages from "
        f"{new_names!r}. If this includes 'source' or a sibling name, "
        f"active_agent was not persisted correctly by the sibling handler wrapper."
    )
    assert r2.get("active_agent") == "join"


@pytest.mark.unit
def test_join_agent_receives_all_sibling_outputs() -> None:
    """The join agent's input must contain each sibling's tagged AIMessage.

    Fan-out is only useful if the join actually *synthesizes* the workers'
    outputs — not if it just runs after them. This test proves the join's
    inbound message list contains one AIMessage per sibling, each carrying
    that sibling's distinctive marker content and tagged with the
    sibling's name. Without this, a real LLM synthesizer would have no
    material to reduce.

    We capture the join's input via a mutable list closed over by the
    join stub, so the assertion inspects exactly what the join agent's
    handler saw in ``state["messages"]``.
    """
    captured_join_inputs: list[list[BaseMessage]] = []
    siblings: tuple[str, ...] = ("worker_a", "worker_b", "worker_c")

    def _make_marker_sibling(agent_name: str) -> CompiledStateGraph:
        marker: str = f"MARKER_FROM_{agent_name.upper()}"

        async def _node(state: AgentState) -> dict:
            return {
                "messages": [
                    AIMessage(content=marker, name=agent_name),
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

    def _make_capturing_join() -> CompiledStateGraph:
        async def _node(state: AgentState) -> dict:
            captured_join_inputs.append(list(state.get("messages", [])))
            return {
                "messages": [
                    AIMessage(content="synthesized", name="join"),
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

    def _fake_create_agent_node(agent: AgentModel, **_kwargs) -> CompiledStateGraph:
        if agent.name == "source":
            return _make_source_subgraph(siblings)
        if agent.name == "join":
            return _make_capturing_join()
        return _make_marker_sibling(agent.name)

    with (
        patch(
            "dao_ai.orchestration.swarm.create_checkpointer",
            return_value=InMemorySaver(),
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

        graph = create_swarm_graph(_make_fanout_config())
        _run(graph, "please synthesize", "verify-synth-t1")

    assert captured_join_inputs, "join must have run at least once"
    join_msgs: list[BaseMessage] = captured_join_inputs[-1]

    # For each sibling, an AIMessage tagged with that sibling's name,
    # carrying its distinctive marker, must be present in the join's input.
    for sibling in siblings:
        expected_marker: str = f"MARKER_FROM_{sibling.upper()}"
        matches = [
            m
            for m in join_msgs
            if isinstance(m, AIMessage)
            and m.name == sibling
            and expected_marker in (m.content if isinstance(m.content, str) else "")
        ]
        assert len(matches) == 1, (
            f"join must see exactly one AIMessage from '{sibling}' containing "
            f"'{expected_marker}'. Got {len(matches)} matches. Full name/type "
            f"sequence at join: {[(type(m).__name__, getattr(m, 'name', None)) for m in join_msgs]}"
        )

    # The synthesizer must NOT have run before any sibling — proves ordering.
    ai_sequence: list[str | None] = [
        m.name for m in join_msgs if isinstance(m, AIMessage)
    ]
    assert "join" not in ai_sequence, (
        f"join's own AIMessage must not appear in its OWN input — that would "
        f"indicate the join ran before it collected sibling outputs. "
        f"AIMessage.name sequence at join: {ai_sequence!r}"
    )


@pytest.mark.unit
def test_source_gets_parallel_handoff_tools_wired_to_agent_node() -> None:
    """The source agent must receive per-sibling handoff tools (integration).

    This complements the mock-based test in ``test_parallel_handoffs.py`` by
    exercising the real ``create_swarm_graph`` code path with a spy on
    ``create_agent_node`` inputs.
    """
    captured_kwargs: dict[str, dict] = {}

    def _spy_create_agent_node(agent: AgentModel, **kwargs) -> CompiledStateGraph:
        captured_kwargs[agent.name] = kwargs
        if agent.name == "source":
            return _make_source_subgraph(SIBLINGS)
        return _make_stub_subgraph(agent.name)

    with (
        patch(
            "dao_ai.orchestration.swarm.create_checkpointer",
            return_value=InMemorySaver(),
        ),
        patch("dao_ai.orchestration.swarm.create_store", return_value=None),
        patch(
            "dao_ai.orchestration.swarm.create_extraction_manager_and_executor",
            return_value=(None, None),
        ),
        patch(
            "dao_ai.orchestration.swarm.create_agent_node",
            side_effect=_spy_create_agent_node,
        ),
    ):
        from dao_ai.orchestration.swarm import create_swarm_graph

        create_swarm_graph(_make_fanout_config())

    source_tools = captured_kwargs["source"]["additional_tools"]
    tool_names: list[str] = sorted(t.name for t in source_tools)
    assert tool_names == [
        "handoff_to_worker_a",
        "handoff_to_worker_b",
        "handoff_to_worker_c",
    ], f"source should get one handoff tool per sibling, got {tool_names!r}"
