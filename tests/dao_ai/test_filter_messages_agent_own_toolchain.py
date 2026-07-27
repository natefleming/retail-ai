"""Ensure the filter bridge does NOT get injected between an agent's
own ``AIMessage(tool_calls=X)`` and its paired ``ToolMessage(X)`` — that
would create a Claude ``tool_use ids were found without tool_result
blocks immediately after`` 400 (a different constraint from the prefill
one).

If ``test_own_ai_tool_tail_is_ai_message_no_bridge_appended`` ever fails,
tighten ``filter_messages_for_agent``: only append the bridge when the
trailing ``AIMessage`` does NOT carry ``tool_calls`` — a tool-call tail
means the next graph step is the tools node, and a synthetic
``HumanMessage`` inserted between them would split the tool_use from
its tool_result and provoke the ``tool_use ids without tool_result``
400.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from dao_ai.orchestration.core import filter_messages_for_agent


@pytest.mark.unit
def test_own_tool_chain_stays_adjacent_no_bridge_between() -> None:
    """Filter must keep own ``AIMessage(tool_calls)`` immediately
    followed by its paired ``ToolMessage`` — no bridge sneaking in
    between."""
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            name="general",
            tool_calls=[
                {"id": "own1", "name": "search_memory", "args": {"query": "hi"}}
            ],
        ),
        ToolMessage(content="[]", tool_call_id="own1"),
    ]
    filtered = filter_messages_for_agent(messages, current_agent_name="general")

    # Every own AIMessage with tool_calls must be immediately followed
    # by its paired ToolMessage — no synthetic bridge in between.
    for i, msg in enumerate(filtered):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            assert i + 1 < len(filtered), (
                "own AIMessage(tool_calls) is the tail with no ToolMessage — "
                "the paired result should have been kept by the filter"
            )
            next_msg = filtered[i + 1]
            assert isinstance(next_msg, ToolMessage), (
                f"expected ToolMessage after own AIMessage(tool_calls); "
                f"got {type(next_msg).__name__}"
            )
            expected_ids = {
                tc.get("id") for tc in msg.tool_calls if isinstance(tc, dict)
            }
            assert next_msg.tool_call_id in expected_ids


@pytest.mark.unit
def test_own_ai_tool_tail_no_bridge_synthetic_result_inserted_instead() -> None:
    """Own agent's ``AIMessage(tool_calls)`` at the message-tail with no
    matching ``ToolMessage`` in state (the parallel-tool-call orphan
    scenario, or a partial checkpoint restore) must be paired with a
    synthetic ``ToolMessage`` immediately after — NOT a
    ``HumanMessage`` bridge.

    The filter appends an ``__orphan_placeholder__`` ToolMessage so
    Claude sees ``tool_use → tool_result`` adjacency AND the messages
    array ends on a ``tool`` role (which the API treats as a
    ``user``-role message, satisfying the "conversation must end with a
    user message" prefill constraint too).
    """
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            name="general",
            tool_calls=[
                {"id": "own2", "name": "search_memory", "args": {"query": "hi"}}
            ],
        ),
    ]
    filtered = filter_messages_for_agent(messages, current_agent_name="general")

    assert isinstance(filtered[-1], ToolMessage), (
        f"tail should be a synthetic ToolMessage; got {type(filtered[-1]).__name__}"
    )
    assert filtered[-1].tool_call_id == "own2"
    assert filtered[-1].name == "__orphan_placeholder__"
    # AND no HumanMessage bridge should have been inserted between the
    # AIMessage and its synthetic ToolMessage.
    ai_idx = next(
        i for i, m in enumerate(filtered) if isinstance(m, AIMessage) and m.tool_calls
    )
    assert isinstance(filtered[ai_idx + 1], ToolMessage), (
        "synthetic ToolMessage must sit immediately after its parent AIMessage"
    )
    assert not any(
        isinstance(m, HumanMessage) and m.name == "__filter_bridge__" for m in filtered
    ), "no bridge should be inserted for an own tool_call orphan"


@pytest.mark.unit
def test_peer_ai_tail_without_tool_calls_still_gets_bridge() -> None:
    """Sanity: a peer ``AIMessage`` (content-only, no tool_calls) at the
    tail still needs the bridge — this is the prefill-error case, not
    the tool_use case."""
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="peer summary", name="planner"),
    ]
    filtered = filter_messages_for_agent(messages, current_agent_name="general")
    assert isinstance(filtered[-1], HumanMessage)
    assert filtered[-1].name == "__filter_bridge__"


@pytest.mark.unit
def test_orphan_tool_call_gets_synthetic_placeholder() -> None:
    """Repro for the supervisor credit_limit 400: LLM emits parallel
    tool_calls (``search_memory`` + ``handoff_to_credit_limit``) in one
    AIMessage; only the handoff's ToolMessage lands in state because the
    handoff Command routes control out of the tool node. Filter must
    inject a synthetic placeholder for the orphan tool_call_id so the
    tool_use/tool_result adjacency Claude enforces stays intact."""
    messages = [
        HumanMessage(content="What is my credit limit?"),
        AIMessage(
            content="",
            name="supervisor",
            tool_calls=[
                {
                    "id": "orphan-call",
                    "name": "search_memory",
                    "args": {"query": "credit"},
                },
                {"id": "handoff-call", "name": "handoff_to_credit_limit", "args": {}},
            ],
        ),
        ToolMessage(content="Transferred to credit_limit", tool_call_id="handoff-call"),
    ]
    filtered = filter_messages_for_agent(messages, current_agent_name="supervisor")

    # Find the AIMessage — its tool_calls must all have adjacent ToolMessages.
    ai_index = next(
        i for i, m in enumerate(filtered) if isinstance(m, AIMessage) and m.tool_calls
    )
    following = filtered[ai_index + 1 :]
    tool_ids_seen = {m.tool_call_id for m in following if isinstance(m, ToolMessage)}
    expected_ids = {tc["id"] for tc in filtered[ai_index].tool_calls}
    assert expected_ids <= tool_ids_seen, (
        f"every tool_call must have a paired ToolMessage; "
        f"missing: {expected_ids - tool_ids_seen}"
    )

    # And the synthetic placeholder must be tagged so ops can spot it.
    placeholder = next(
        m
        for m in filtered
        if isinstance(m, ToolMessage) and m.tool_call_id == "orphan-call"
    )
    assert placeholder.name == "__orphan_placeholder__"


@pytest.mark.unit
def test_orphan_placeholders_stay_adjacent_to_parent_ai() -> None:
    """The placeholder must sit IMMEDIATELY after its parent AIMessage
    (not at the end of the list). Claude's ``tool_use → tool_result``
    adjacency is turn-scoped: intervening HumanMessage / AIMessage
    between tool_use and its result would also 400."""
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            name="supervisor",
            tool_calls=[
                {"id": "a", "name": "search_memory", "args": {"query": "x"}},
                {"id": "b", "name": "handoff_to_general", "args": {}},
            ],
        ),
        ToolMessage(content="ok", tool_call_id="b"),
    ]
    filtered = filter_messages_for_agent(messages, current_agent_name="supervisor")

    # Sequence must be [AIMessage, ToolMessage(a), ToolMessage(b)] with no gaps.
    ai_idx = next(
        i for i, m in enumerate(filtered) if isinstance(m, AIMessage) and m.tool_calls
    )
    assert isinstance(filtered[ai_idx + 1], ToolMessage)
    assert isinstance(filtered[ai_idx + 2], ToolMessage)
    tool_ids = {filtered[ai_idx + 1].tool_call_id, filtered[ai_idx + 2].tool_call_id}
    assert tool_ids == {"a", "b"}


@pytest.mark.unit
def test_no_orphan_no_placeholder() -> None:
    """When every own tool_call has a real ToolMessage, no placeholder is
    added — filter is only defensive, not intrusive."""
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            name="general",
            tool_calls=[
                {"id": "own1", "name": "search_memory", "args": {"query": "x"}}
            ],
        ),
        ToolMessage(content="[]", tool_call_id="own1"),
    ]
    filtered = filter_messages_for_agent(messages, current_agent_name="general")
    placeholders = [
        m
        for m in filtered
        if isinstance(m, ToolMessage) and m.name == "__orphan_placeholder__"
    ]
    assert placeholders == [], f"unexpected placeholders: {placeholders}"
