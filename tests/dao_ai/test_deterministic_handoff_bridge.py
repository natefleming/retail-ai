"""Regression test for the deterministic-handoff bridge content.

``_create_deterministic_handler`` in ``src/dao_ai/orchestration/swarm.py``
appends a ``HumanMessage`` bridge whenever the wrapped agent's turn
ends with an ``AIMessage`` (so the downstream LLM doesn't see an
assistant-tail and 400 on the newer Claude prefill constraint). The
bridge content matters — a cryptic ``[automated deterministic handoff
to X]`` made downstream agents reply ``"looks like that message came
through as a system handoff rather than a question from you"`` and the
composer then produced non-sensical customer-facing text. The bridge
now echoes the last real user query verbatim so downstream LLMs have
grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from dao_ai.orchestration.swarm import _create_deterministic_handler


@dataclass
class _StubRuntime:
    context: Any = None


async def _identity_handler(state, runtime):
    return dict(state)


@pytest.mark.unit
def test_deterministic_bridge_echoes_last_user_query() -> None:
    import asyncio

    handler = _create_deterministic_handler(_identity_handler, "planner")
    state = {
        "messages": [
            HumanMessage(content="can you remember that my dogs name is ripley"),
            AIMessage(
                content="INTENT: general | CONFIDENCE: 0.97",
                name="supervisor",
            ),
        ]
    }
    result = asyncio.run(handler(state, _StubRuntime()))

    tail = result["messages"][-1]
    assert isinstance(tail, HumanMessage)
    assert tail.name == "__deterministic_handoff__"
    # Bridge should mention the target agent AND echo the user query.
    assert "planner" in tail.content
    assert "can you remember that my dogs name is ripley" in tail.content


@pytest.mark.unit
def test_deterministic_bridge_falls_back_when_no_user_query() -> None:
    import asyncio

    handler = _create_deterministic_handler(_identity_handler, "composer")
    state = {
        "messages": [
            AIMessage(content="some peer output", name="planner"),
        ]
    }
    result = asyncio.run(handler(state, _StubRuntime()))

    tail = result["messages"][-1]
    assert isinstance(tail, HumanMessage)
    assert tail.name == "__deterministic_handoff__"
    assert "composer" in tail.content


@pytest.mark.unit
def test_deterministic_bridge_skips_earlier_bridges_when_finding_user_query() -> None:
    """The bridge should not echo a PRIOR bridge (of any type) — it
    should walk past ``__deterministic_handoff__`` and ``__filter_bridge__``
    named HumanMessages to find the actual user query."""
    import asyncio

    handler = _create_deterministic_handler(_identity_handler, "credit_limit")
    state = {
        "messages": [
            HumanMessage(content="What is my credit limit?"),
            AIMessage(content="INTENT: credit_limit", name="supervisor"),
            HumanMessage(
                content="[automated deterministic handoff to planner]",
                name="__deterministic_handoff__",
            ),
            AIMessage(content="Routing to credit_limit", name="planner"),
        ]
    }
    result = asyncio.run(handler(state, _StubRuntime()))

    tail = result["messages"][-1]
    assert isinstance(tail, HumanMessage)
    assert "What is my credit limit?" in tail.content
    # Confirm it did NOT echo the bridge content.
    assert "[automated deterministic handoff to planner]" not in tail.content


@pytest.mark.unit
def test_deterministic_bridge_only_fires_on_ai_tail() -> None:
    """No bridge if the tail is already a HumanMessage or ToolMessage."""
    import asyncio

    handler = _create_deterministic_handler(_identity_handler, "planner")
    state = {
        "messages": [
            HumanMessage(content="hi"),
        ]
    }
    result = asyncio.run(handler(state, _StubRuntime()))
    assert not any(
        isinstance(m, HumanMessage) and m.name == "__deterministic_handoff__"
        for m in result["messages"]
    )
