"""Tests for the swarm router's terminal-agent reset behavior.

When the prior turn ended at a terminal agent (either ``handoffs: []`` or
``AgentModel.is_terminal: true``), the next user turn must restart at
``default_agent`` instead of stickily resuming at the terminal agent.
"""

from __future__ import annotations

import pytest

from dao_ai.orchestration.swarm import _create_swarm_router


@pytest.mark.unit
def test_router_routes_to_active_agent_when_not_terminal() -> None:
    router = _create_swarm_router(
        default_agent="supervisor",
        agent_names=["supervisor", "planner", "composer"],
        terminal_agents=frozenset({"composer"}),
    )
    # Mid-pipeline active agent (planner) — should resume sticky.
    assert router({"active_agent": "planner"}) == "planner"


@pytest.mark.unit
def test_router_falls_through_to_default_when_active_is_terminal() -> None:
    router = _create_swarm_router(
        default_agent="supervisor",
        agent_names=["supervisor", "planner", "composer"],
        terminal_agents=frozenset({"composer"}),
    )
    # Prior turn ended at composer (terminal) — next turn restarts.
    assert router({"active_agent": "composer"}) == "supervisor"


@pytest.mark.unit
def test_router_routes_to_default_when_no_active_agent() -> None:
    router = _create_swarm_router(
        default_agent="supervisor",
        agent_names=["supervisor", "planner", "composer"],
        terminal_agents=frozenset({"composer"}),
    )
    assert router({}) == "supervisor"


@pytest.mark.unit
def test_router_routes_to_default_when_active_agent_unknown() -> None:
    router = _create_swarm_router(
        default_agent="supervisor",
        agent_names=["supervisor", "planner", "composer"],
        terminal_agents=frozenset({"composer"}),
    )
    # Unknown name — safety fallback.
    assert router({"active_agent": "ghost"}) == "supervisor"


@pytest.mark.unit
def test_empty_terminal_set_preserves_legacy_sticky_behavior() -> None:
    router = _create_swarm_router(
        default_agent="supervisor",
        agent_names=["supervisor", "planner", "composer"],
        terminal_agents=frozenset(),
    )
    # No terminal agents → composer is just another sticky node.
    assert router({"active_agent": "composer"}) == "composer"
