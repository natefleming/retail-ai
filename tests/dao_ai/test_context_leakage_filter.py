"""Regression tests for context-leakage prevention in ``filter_messages_for_agent``.

Two orthogonal policies added to the filter:

1. ``SystemMessage`` in shared history is dropped by default. Every agent's
   system prompt / memory-context injection is applied fresh per invocation
   by middleware / the prompt template, so a ``SystemMessage`` from a prior
   agent's turn is context noise.

2. Peer ``AIMessage`` from an agent named in ``internal_agents`` is dropped
   when the current agent is NOT itself internal. Internal agents can still
   see each other's outputs (a planner needs to read a supervisor's intent
   classification to route).

Together these mirror Anthropic's "distilled handoffs" pattern and
LangGraph's documented swarm private-history recipe: internal reasoning is
architecturally invisible to downstream customer-facing agents.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from dao_ai.orchestration.core import filter_messages_for_agent


def _names(messages: list[BaseMessage]) -> list[tuple[str, str | None]]:
    """Compact repr: (type, name) tuples for readable assertions."""
    return [(type(m).__name__, getattr(m, "name", None)) for m in messages]


@pytest.mark.unit
def test_system_messages_are_dropped_by_default() -> None:
    """``SystemMessage`` in shared history is always dropped from the view."""
    msgs: list[BaseMessage] = [
        SystemMessage(content="You are a helpful supervisor."),
        HumanMessage(content="hello"),
        AIMessage(content="hi there", name="general"),
        SystemMessage(content="## Memories\n- user is Nate"),
    ]
    out = filter_messages_for_agent(msgs, current_agent_name="general")
    assert not any(isinstance(m, SystemMessage) for m in out), (
        f"SystemMessages should be filtered out; got {_names(out)}"
    )
    # Non-system messages are preserved in order.
    non_bridge = [m for m in out if getattr(m, "name", None) != "__filter_bridge__"]
    assert _names(non_bridge) == [("HumanMessage", None), ("AIMessage", "general")]


@pytest.mark.unit
def test_peer_internal_ai_message_dropped_for_non_internal_current() -> None:
    """Non-internal agent (composer) must not see internal agent (supervisor) output."""
    msgs: list[BaseMessage] = [
        HumanMessage(content="show me my orders"),
        AIMessage(
            content="INTENT: order_history | CONFIDENCE: 0.95 | NOTES: …",
            name="supervisor",
        ),
        AIMessage(content="No orders found.", name="order_history"),
    ]
    out = filter_messages_for_agent(
        msgs,
        current_agent_name="composer",
        internal_agents=frozenset({"supervisor", "planner"}),
    )
    # Supervisor's INTENT line is filtered — composer literally cannot
    # pattern-match on a string it never receives.
    assert not any(getattr(m, "name", None) == "supervisor" for m in out), (
        f"supervisor's AIMessage should have been filtered; got {_names(out)}"
    )
    # Order history (non-internal) still visible.
    assert any(getattr(m, "name", None) == "order_history" for m in out)


@pytest.mark.unit
def test_peer_internal_ai_message_kept_for_internal_current() -> None:
    """Internal agent (planner) MUST see other internal agent (supervisor) output.

    Without this rule, planner couldn't read supervisor's intent classification
    to make its routing decision.
    """
    msgs: list[BaseMessage] = [
        HumanMessage(content="show me my orders"),
        AIMessage(
            content="INTENT: order_history | CONFIDENCE: 0.95 | NOTES: …",
            name="supervisor",
        ),
    ]
    out = filter_messages_for_agent(
        msgs,
        current_agent_name="planner",
        internal_agents=frozenset({"supervisor", "planner"}),
    )
    assert any(getattr(m, "name", None) == "supervisor" for m in out), (
        f"planner should see supervisor's output; got {_names(out)}"
    )


@pytest.mark.unit
def test_own_ai_message_preserved_for_internal_agent() -> None:
    """Any agent (internal or not) always sees its own prior AIMessages."""
    msgs: list[BaseMessage] = [
        HumanMessage(content="q"),
        AIMessage(content="prior supervisor output", name="supervisor"),
    ]
    out = filter_messages_for_agent(
        msgs,
        current_agent_name="supervisor",
        internal_agents=frozenset({"supervisor"}),
    )
    # Own output survives even though supervisor is in internal_agents.
    assert any(getattr(m, "name", None) == "supervisor" for m in out)


@pytest.mark.unit
def test_peer_non_internal_ai_message_preserved_with_tool_calls_stripped() -> None:
    """Existing behavior for non-internal peers is unchanged."""
    peer_msg = AIMessage(
        content="Here is the answer.",
        name="order_history",
        tool_calls=[{"id": "tc1", "name": "search_orders", "args": {}}],
    )
    msgs: list[BaseMessage] = [HumanMessage(content="q"), peer_msg]
    out = filter_messages_for_agent(
        msgs,
        current_agent_name="composer",
        internal_agents=frozenset({"supervisor"}),
    )
    ai_out = [m for m in out if isinstance(m, AIMessage)]
    assert len(ai_out) == 1
    assert ai_out[0].name == "order_history"
    # tool_calls stripped for peer content
    assert not ai_out[0].tool_calls


@pytest.mark.unit
def test_backward_compat_no_internal_agents_no_system_messages() -> None:
    """With ``internal_agents=None`` and no SystemMessages, output is unchanged.

    Pins the existing behavior for configs that don't set ``internal: true``
    on any agent. Note: the existing filter appends a ``__filter_bridge__``
    HumanMessage when the tail is a content-only AIMessage (strict-mode
    provider workaround) — that behavior is preserved.
    """
    msgs: list[BaseMessage] = [
        HumanMessage(content="q"),
        AIMessage(content="a1", name="planner"),
        AIMessage(content="a2", name="composer"),
    ]
    out_new = filter_messages_for_agent(msgs, current_agent_name="composer")
    # Non-bridge portion matches the prior filter behavior.
    non_bridge = [m for m in out_new if getattr(m, "name", None) != "__filter_bridge__"]
    assert _names(non_bridge) == [
        ("HumanMessage", None),
        ("AIMessage", "planner"),
        ("AIMessage", "composer"),
    ]


@pytest.mark.unit
def test_own_tool_exchange_pairs_correctly_for_internal_agent() -> None:
    """Internal agents' own tool exchanges still pair via tool_call_id."""
    own_ai = AIMessage(
        content="",
        name="supervisor",
        tool_calls=[{"id": "tc_1", "name": "some_tool", "args": {}}],
    )
    tool_result = ToolMessage(content="tool ok", tool_call_id="tc_1")
    msgs: list[BaseMessage] = [HumanMessage(content="q"), own_ai, tool_result]
    out = filter_messages_for_agent(
        msgs,
        current_agent_name="supervisor",
        internal_agents=frozenset({"supervisor"}),
    )
    # Own AIMessage(tool_calls) + paired ToolMessage both preserved.
    tool_ids_seen: list[str] = [
        m.tool_call_id for m in out if isinstance(m, ToolMessage)
    ]
    assert "tc_1" in tool_ids_seen


@pytest.mark.unit
def test_dropped_internal_peer_does_not_leave_orphan_tool_pairings() -> None:
    """Filtering out an internal peer's AIMessage doesn't create orphan ToolMessages.

    A peer agent's ToolMessages are never associated with the current agent's
    ``own_tool_call_ids`` set, so they were being dropped by the tool-pairing
    rule already. Verify the new internal-agent filter doesn't accidentally
    keep them.
    """
    peer_ai = AIMessage(
        content="",
        name="supervisor",
        tool_calls=[{"id": "tc_peer", "name": "peer_tool", "args": {}}],
    )
    peer_tool_result = ToolMessage(content="peer tool ok", tool_call_id="tc_peer")
    msgs: list[BaseMessage] = [
        HumanMessage(content="q"),
        peer_ai,
        peer_tool_result,
    ]
    out = filter_messages_for_agent(
        msgs,
        current_agent_name="composer",
        internal_agents=frozenset({"supervisor"}),
    )
    # No orphan ToolMessage from the filtered-out peer.
    assert not any(isinstance(m, ToolMessage) for m in out), _names(out)
    # No supervisor AIMessage either.
    assert not any(getattr(m, "name", None) == "supervisor" for m in out)


@pytest.mark.unit
def test_empty_internal_agents_still_drops_system_messages() -> None:
    """The SystemMessage-drop rule is unconditional; not gated on internal_agents."""
    msgs: list[BaseMessage] = [
        SystemMessage(content="stale prompt from prior turn"),
        HumanMessage(content="q"),
    ]
    out = filter_messages_for_agent(msgs, current_agent_name="composer")
    assert not any(isinstance(m, SystemMessage) for m in out)


@pytest.mark.unit
def test_untagged_peer_ai_message_not_filtered() -> None:
    """AIMessages without a ``name`` field are treated as peers with content-visible.

    Guards against future writers that forget to tag their AIMessage — they
    should still appear (not silently vanish) so any leak would still be
    visible for observation rather than silently dropped.
    """
    msgs: list[BaseMessage] = [
        HumanMessage(content="q"),
        AIMessage(content="untagged content"),  # no name
    ]
    out = filter_messages_for_agent(
        msgs,
        current_agent_name="composer",
        internal_agents=frozenset({"supervisor"}),
    )
    ai_seen = [m for m in out if isinstance(m, AIMessage)]
    assert len(ai_seen) == 1
    assert ai_seen[0].content == "untagged content"
