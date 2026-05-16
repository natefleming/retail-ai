"""Unit tests for :func:`dao_ai.hitl.decide_graph_turn`.

The helper consolidates the three HITL branches that the Responses agent
and the A2A executor both need: explicit decisions, snapshot-based resume
of an interrupted graph, and fresh invocation. These tests pin the
branch semantics so the two call sites cannot drift.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.types import Command

from dao_ai.hitl import GraphTurn, decide_graph_turn


def _make_graph(*, checkpointer: bool = False, aget_state_return=None) -> MagicMock:
    g = MagicMock()
    g.checkpointer = MagicMock() if checkpointer else None
    g.aget_state = AsyncMock(return_value=aget_state_return)
    return g


@pytest.mark.unit
def test_graphturn_should_skip_graph_property():
    assert GraphTurn(validation_error_message="x").should_skip_graph is True
    assert GraphTurn(graph_input={"messages": []}).should_skip_graph is False
    assert GraphTurn(resume_command=Command(resume={})).should_skip_graph is False


@pytest.mark.unit
def test_graphturn_stream_input_raises_when_no_payload():
    with pytest.raises(RuntimeError):
        _ = GraphTurn(validation_error_message="oops").stream_input


@pytest.mark.unit
def test_decide_graph_turn_explicit_decisions_takes_resume_branch():
    graph = _make_graph(checkpointer=False)

    result = asyncio.run(
        decide_graph_turn(
            graph=graph,
            messages=[{"role": "user", "content": "hi"}],
            custom_inputs={"decisions": [{"type": "approve"}]},
            runtime_config={"configurable": {"thread_id": "t1"}},
        )
    )

    assert isinstance(result.resume_command, Command)
    assert result.resume_command.resume == {"decisions": [{"type": "approve"}]}
    assert result.graph_input is None


@pytest.mark.unit
def test_decide_graph_turn_no_checkpointer_takes_fresh_invocation():
    graph = _make_graph(checkpointer=False)

    result = asyncio.run(
        decide_graph_turn(
            graph=graph,
            messages=[{"role": "user", "content": "hi"}],
            custom_inputs=None,
            runtime_config={"configurable": {"thread_id": "t1"}},
        )
    )

    assert result.graph_input == {"messages": [{"role": "user", "content": "hi"}]}
    assert result.resume_command is None
    graph.aget_state.assert_not_called()


@pytest.mark.unit
def test_decide_graph_turn_carries_genie_session_ids():
    graph = _make_graph(checkpointer=False)
    session_input = {"genie_conversation_ids": {"space_1": "conv_1"}}

    result = asyncio.run(
        decide_graph_turn(
            graph=graph,
            messages=[{"role": "user", "content": "hi"}],
            custom_inputs=None,
            runtime_config={"configurable": {"thread_id": "t1"}},
            session_input=session_input,
        )
    )

    assert result.graph_input == {
        "messages": [{"role": "user", "content": "hi"}],
        "genie_conversation_ids": {"space_1": "conv_1"},
    }


@pytest.mark.unit
def test_decide_graph_turn_snapshot_non_interrupted_takes_fresh_path():
    class Snap:
        interrupts = ()

    graph = _make_graph(checkpointer=True, aget_state_return=Snap())

    result = asyncio.run(
        decide_graph_turn(
            graph=graph,
            messages=[{"role": "user", "content": "hi"}],
            custom_inputs=None,
            runtime_config={"configurable": {"thread_id": "t1"}},
        )
    )

    assert result.graph_input is not None
    assert result.resume_command is None
    graph.aget_state.assert_called_once()
