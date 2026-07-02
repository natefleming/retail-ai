"""Reproducers for the AIMessage-tail bug in ``filter_messages_for_agent``.

Newer Claude models (Opus 4.6+, Sonnet 4.5+) removed assistant-message
prefill support. When ``filter_messages_for_agent`` runs for a worker
agent that inherits a peer's tool exchange (e.g. planner emits
``handoff_to_general`` → planner's ``AIMessage(tool_calls=…)`` is kept
as content-only, but its paired ``ToolMessage`` is dropped because
``general`` doesn't own the tool_call_id), the filtered tail becomes an
``AIMessage``. The downstream LLM call then 400s with
``This model does not support assistant message prefill. The
conversation must end with a user message.``

The fix: after filtering, if the tail is an ``AIMessage``, append a
synthetic ``HumanMessage`` bridge that normalizes the tail without
affecting downstream reasoning.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from dao_ai.orchestration.core import filter_messages_for_agent


@pytest.mark.unit
def test_planner_handoff_shape_ends_with_non_ai_tail() -> None:
    """Repro of trace ``8997d313f46fe28b5e418dba7adce538`` shape.

    Message sequence coming into ``general`` after a deterministic
    supervisor→planner handoff and an agentic planner→general handoff:

      1. HumanMessage (user query)
      2. AIMessage(supervisor's classification, content-only)
      3. HumanMessage(deterministic bridge to planner)
      4. AIMessage(planner) with tool_calls=[handoff_to_general]
      5. ToolMessage(handoff_to_general result)

    After filtering for ``current_agent_name="general"``:
      - Supervisor's AIMessage is kept content-only.
      - Planner's tool-call AIMessage is silently dropped (no content).
      - Planner's ToolMessage is dropped (general doesn't own the id).

    Invariant: the filtered tail MUST NOT be an ``AIMessage`` — that
    triggers Claude's "assistant message prefill" 400 on modern
    endpoints.
    """
    messages: list[BaseMessage] = [
        HumanMessage(content="remember my dog's name is ripley"),
        AIMessage(
            content="INTENT: general | CONFIDENCE: 0.91",
            name="supervisor",
        ),
        HumanMessage(
            content="[automated deterministic handoff to planner]",
            name="__deterministic_handoff__",
        ),
        AIMessage(
            content="",
            name="planner",
            tool_calls=[
                {
                    "id": "call_handoff_general",
                    "name": "handoff_to_general",
                    "args": {},
                }
            ],
        ),
        ToolMessage(
            content="Handed off to general.",
            tool_call_id="call_handoff_general",
        ),
    ]

    filtered = filter_messages_for_agent(messages, current_agent_name="general")

    assert not isinstance(
        filtered[-1], AIMessage
    ), f"tail must not be AIMessage; got {type(filtered[-1]).__name__}"


@pytest.mark.unit
def test_planner_with_content_and_dropped_toolmsg_gets_bridge() -> None:
    """Alternate shape where planner emits **content plus tool_calls**
    (a common Claude pattern — model narrates its choice then invokes
    a tool). The filter keeps the content-only AIMessage; the
    ToolMessage is dropped because general doesn't own the id. Tail
    becomes an AIMessage → bridge is appended.
    """
    messages: list[BaseMessage] = [
        HumanMessage(content="hi"),
        AIMessage(
            content="Routing to general to remember dog's name.",
            name="planner",
            tool_calls=[
                {
                    "id": "call_h1",
                    "name": "handoff_to_general",
                    "args": {},
                }
            ],
        ),
        ToolMessage(content="handoff done", tool_call_id="call_h1"),
    ]
    filtered = filter_messages_for_agent(messages, current_agent_name="general")
    assert isinstance(filtered[-1], HumanMessage)
    assert filtered[-1].name == "__filter_bridge__"


@pytest.mark.unit
def test_agent_receiving_pure_ai_tail_gets_bridge() -> None:
    """A minimal repro: prior turn produced an AIMessage; new agent
    inherits the state; without the bridge, the LLM call 400s."""
    messages: list[BaseMessage] = [
        HumanMessage(content="hi"),
        AIMessage(content="Hello!", name="supervisor"),
    ]
    filtered = filter_messages_for_agent(messages, current_agent_name="general")
    assert isinstance(filtered[-1], HumanMessage)


@pytest.mark.unit
def test_no_bridge_appended_when_tail_is_already_human() -> None:
    """Idempotent: don't stack bridges when the tail is already a
    HumanMessage."""
    messages: list[BaseMessage] = [
        HumanMessage(content="user message 1"),
        AIMessage(content="agent reply", name="supervisor"),
        HumanMessage(content="user message 2"),
    ]
    filtered = filter_messages_for_agent(messages, current_agent_name="general")
    # Only original HumanMessages preserved — no extra bridge.
    assert sum(
        1 for m in filtered if isinstance(m, HumanMessage) and m.name != "__filter_bridge__"
    ) == 2
    assert not any(
        isinstance(m, HumanMessage) and m.name == "__filter_bridge__"
        for m in filtered
    )


@pytest.mark.unit
def test_own_ai_tool_exchange_preserved_with_tool_tail() -> None:
    """When the agent's OWN tool exchange is present, both AIMessage
    (tool_calls) and ToolMessage are kept — tail is ToolMessage, no
    bridge needed."""
    messages: list[BaseMessage] = [
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
    assert isinstance(filtered[-1], ToolMessage)


@pytest.mark.unit
def test_empty_history_is_untouched() -> None:
    """Empty message list stays empty — no bridge synthesized from thin
    air."""
    assert filter_messages_for_agent([], current_agent_name="general") == []


@pytest.mark.unit
def test_bridge_content_names_target_agent_and_echoes_user_query() -> None:
    """The bridge content includes the target agent name AND echoes the
    last real user query verbatim — this prevents downstream LLMs from
    interpreting the bridge as an empty user message and answering
    "your message came through empty!". Log/trace inspection stays
    legible because the ``name="__filter_bridge__"`` marker is intact.
    """
    messages: list[BaseMessage] = [
        HumanMessage(content="What is my credit limit?"),
        AIMessage(content="prior reply", name="supervisor"),
    ]
    filtered = filter_messages_for_agent(messages, current_agent_name="credit_limit")
    assert filtered[-1].name == "__filter_bridge__"
    # Target agent named for clarity.
    assert "credit_limit" in filtered[-1].content
    # Real user query echoed verbatim so downstream LLM has clean grounding.
    assert "What is my credit limit?" in filtered[-1].content


@pytest.mark.unit
def test_bridge_falls_back_when_no_agent_name() -> None:
    """Legacy callers pass ``None`` — bridge still appended with a
    generic target label."""
    messages: list[BaseMessage] = [
        HumanMessage(content="hi"),
        AIMessage(content="prior reply", name="supervisor"),
    ]
    filtered = filter_messages_for_agent(messages, current_agent_name=None)
    assert isinstance(filtered[-1], HumanMessage)
    assert "next agent" in filtered[-1].content


@pytest.mark.unit
def test_article_invariant_last_message_is_not_assistant() -> None:
    """Article invariant: ``messages[-1]`` MUST NOT be an assistant
    message on Opus 4.6 / Sonnet 4.5+ endpoints. Exercise every
    peer-AIMessage-tail shape we know about and assert the tail never
    comes back as ``AIMessage`` regardless of whether tool_calls
    or content-only.
    """
    peer_shapes: list[list[BaseMessage]] = [
        # (a) content-only peer AIMessage — filter keeps as-is
        [
            HumanMessage(content="hi"),
            AIMessage(content="peer reply", name="planner"),
        ],
        # (b) peer AIMessage with content + tool_calls — filter keeps content only
        [
            HumanMessage(content="hi"),
            AIMessage(
                content="routing note",
                name="planner",
                tool_calls=[
                    {"id": "call1", "name": "handoff_to_x", "args": {}}
                ],
            ),
        ],
        # (c) peer AIMessage with tool_calls only (no content) — filter drops entirely
        [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                name="planner",
                tool_calls=[
                    {"id": "call1", "name": "handoff_to_x", "args": {}}
                ],
            ),
        ],
    ]
    for msgs in peer_shapes:
        filtered = filter_messages_for_agent(msgs, current_agent_name="general")
        assert not isinstance(filtered[-1], AIMessage), (
            f"tail must not be AIMessage; got {type(filtered[-1]).__name__} "
            f"for input {[type(m).__name__ for m in msgs]}"
        )
