"""
Tests for swarm handoff_constraints (the ``requires`` field on AgentModel).

Covers:
- Config-time validators: self-reference, unknown agent, cycle in requires DAG,
  deterministic handoff to constrained target.
- Runtime behavior of ``create_handoff_tool``: refusal when prereqs are unmet,
  pass-through when satisfied, no constraint when ``requires`` is empty/None.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from dao_ai.config import AgentModel, AppConfig, LLMModel
from dao_ai.orchestration.core import create_handoff_tool


# =============================================================================
# AgentModel.requires — local validators (self-reference)
# =============================================================================


@pytest.mark.unit
class TestRequiresFieldOnAgentModel:
    """Tests for the ``requires`` field on AgentModel."""

    def test_requires_defaults_to_empty_list(self) -> None:
        agent = AgentModel(name="a", model=LLMModel(name="m"))
        assert agent.requires == []

    def test_requires_accepts_list_of_names(self) -> None:
        agent = AgentModel(
            name="checkout", model=LLMModel(name="m"), requires=["browse", "cart"]
        )
        assert agent.requires == ["browse", "cart"]

    def test_self_reference_rejected(self) -> None:
        with pytest.raises(ValueError, match="self-reference"):
            AgentModel(name="loop", model=LLMModel(name="m"), requires=["loop"])


# =============================================================================
# AppModel.validate_agent_requires — cross-agent validators
# =============================================================================


def _minimal_app(agents: list[dict], handoffs: dict | None = None) -> dict:
    """Build a minimal AppConfig dict with the given agents and handoffs."""
    swarm: dict = {"default_agent": agents[0]["name"]}
    if handoffs is not None:
        swarm["handoffs"] = handoffs
    return {
        "app": {
            "name": "test_app",
            "registered_model": {"name": "test_model"},
            "agents": agents,
            "orchestration": {"swarm": swarm},
        }
    }


@pytest.mark.unit
class TestRequiresCrossAgentValidators:
    """Tests for AppModel-level validators on agent ``requires``."""

    def test_unknown_agent_in_requires_is_rejected(self) -> None:
        cfg = _minimal_app(
            agents=[
                {"name": "a", "model": {"name": "m"}},
                {
                    "name": "b",
                    "model": {"name": "m"},
                    "requires": ["does_not_exist"],
                },
            ],
        )
        with pytest.raises(ValueError, match="does_not_exist"):
            AppConfig(**cfg)

    def test_known_agent_in_requires_is_accepted(self) -> None:
        cfg = _minimal_app(
            agents=[
                {"name": "a", "model": {"name": "m"}},
                {"name": "b", "model": {"name": "m"}, "requires": ["a"]},
            ],
        )
        # No exception
        config = AppConfig(**cfg)
        b = next(x for x in config.app.agents if x.name == "b")
        assert b.requires == ["a"]

    def test_two_node_cycle_is_rejected(self) -> None:
        # A requires B, B requires A
        cfg = _minimal_app(
            agents=[
                {"name": "a", "model": {"name": "m"}, "requires": ["b"]},
                {"name": "b", "model": {"name": "m"}, "requires": ["a"]},
            ],
        )
        with pytest.raises(ValueError, match="Cycle detected"):
            AppConfig(**cfg)

    def test_three_node_cycle_is_rejected(self) -> None:
        # A -> B -> C -> A
        cfg = _minimal_app(
            agents=[
                {"name": "a", "model": {"name": "m"}, "requires": ["b"]},
                {"name": "b", "model": {"name": "m"}, "requires": ["c"]},
                {"name": "c", "model": {"name": "m"}, "requires": ["a"]},
            ],
        )
        with pytest.raises(ValueError, match="Cycle detected"):
            AppConfig(**cfg)

    def test_dag_without_cycle_is_accepted(self) -> None:
        # checkout requires cart; cart requires browse — chain, no cycle
        cfg = _minimal_app(
            agents=[
                {"name": "browse", "model": {"name": "m"}},
                {"name": "cart", "model": {"name": "m"}, "requires": ["browse"]},
                {"name": "checkout", "model": {"name": "m"}, "requires": ["cart"]},
            ],
        )
        # No exception
        AppConfig(**cfg)


# =============================================================================
# Deterministic handoff to constrained target
# =============================================================================


@pytest.mark.unit
class TestDeterministicHandoffValidator:
    """Tests for ``validate_no_deterministic_handoff_to_constrained``."""

    def test_deterministic_handoff_to_constrained_target_rejected(self) -> None:
        cfg = _minimal_app(
            agents=[
                {"name": "a", "model": {"name": "m"}},
                {"name": "cart", "model": {"name": "m"}},
                {
                    "name": "checkout",
                    "model": {"name": "m"},
                    "requires": ["cart"],
                },
            ],
            handoffs={
                "a": [
                    {"agent": "checkout", "is_deterministic": True},
                ],
            },
        )
        with pytest.raises(
            ValueError, match="deterministic handoff to 'checkout'"
        ):
            AppConfig(**cfg)

    def test_agentic_handoff_to_constrained_target_allowed(self) -> None:
        # Same shape as above but is_deterministic: false (default).
        cfg = _minimal_app(
            agents=[
                {"name": "a", "model": {"name": "m"}},
                {"name": "cart", "model": {"name": "m"}},
                {
                    "name": "checkout",
                    "model": {"name": "m"},
                    "requires": ["cart"],
                },
            ],
            handoffs={
                "a": ["checkout"],  # agentic shorthand
            },
        )
        # No exception
        AppConfig(**cfg)

    def test_deterministic_handoff_to_unconstrained_target_allowed(self) -> None:
        # Constrained target exists but the deterministic edge points elsewhere.
        cfg = _minimal_app(
            agents=[
                {"name": "a", "model": {"name": "m"}},
                {"name": "cart", "model": {"name": "m"}},
                {
                    "name": "checkout",
                    "model": {"name": "m"},
                    "requires": ["cart"],
                },
            ],
            handoffs={
                "a": [
                    {"agent": "cart", "is_deterministic": True},
                ],
            },
        )
        # No exception
        AppConfig(**cfg)


# =============================================================================
# Runtime: create_handoff_tool with requires
# =============================================================================


def _runtime(messages: list, tool_call_id: str = "call_test") -> SimpleNamespace:
    """Build a minimal stand-in for ToolRuntime.

    The handoff tool body uses only ``runtime.tool_call_id`` and
    ``runtime.state.get("messages", [])``. SimpleNamespace + dict is enough.
    """
    return SimpleNamespace(tool_call_id=tool_call_id, state={"messages": messages})


@pytest.mark.unit
class TestHandoffToolRuntimeCheck:
    """Tests for the ``requires`` runtime check inside ``create_handoff_tool``."""

    def test_no_requires_proceeds_normally(self) -> None:
        """Backward compatibility: tools without requires behave as before."""
        tool = create_handoff_tool("checkout", "checkout agent")
        runtime = _runtime([HumanMessage(content="hi")])

        result = tool.func(runtime)

        assert result.goto == "checkout"
        assert result.update["active_agent"] == "checkout"

    def test_requires_satisfied_proceeds_normally(self) -> None:
        tool = create_handoff_tool("checkout", "checkout agent", requires=["cart"])
        runtime = _runtime(
            [
                HumanMessage(content="hi"),
                AIMessage(content="picking items", name="cart"),
            ]
        )

        result = tool.func(runtime)

        assert result.goto == "checkout"
        assert result.update["active_agent"] == "checkout"

    def test_requires_unmet_returns_refusal(self) -> None:
        tool = create_handoff_tool("checkout", "checkout agent", requires=["cart"])
        runtime = _runtime(
            [
                HumanMessage(content="ready to pay"),
                AIMessage(content="routing", name="triage"),
            ]
        )

        result = tool.func(runtime)

        # No routing / no active_agent change.
        assert getattr(result, "goto", ()) in ((), None)
        assert "active_agent" not in result.update

        # Refusal ToolMessage in the update.
        msgs = result.update["messages"]
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        content = tool_msgs[0].content
        assert "Cannot hand off to 'checkout'" in content
        assert "cart" in content
        assert "Missing" in content

    def test_partial_coverage_lists_only_missing(self) -> None:
        """When some prereqs are met and some aren't, refusal names the missing."""
        tool = create_handoff_tool(
            "checkout", "checkout agent", requires=["cart", "verify_age"]
        )
        runtime = _runtime(
            [
                HumanMessage(content="ready"),
                AIMessage(content="picked items", name="cart"),
                # verify_age has NOT run
            ]
        )

        result = tool.func(runtime)

        msgs = result.update["messages"]
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        content = tool_msgs[0].content
        # The refusal must explicitly identify the missing prereq.
        assert "verify_age" in content
        # And ``cart`` must show up in the "called so far" report.
        assert "cart" in content

    def test_empty_requires_list_is_unconstrained(self) -> None:
        """Explicit empty list behaves like no constraint."""
        tool = create_handoff_tool("checkout", "checkout agent", requires=[])
        runtime = _runtime([HumanMessage(content="hi")])

        result = tool.func(runtime)

        assert result.goto == "checkout"
        assert result.update["active_agent"] == "checkout"

    def test_unnamed_aimessages_do_not_satisfy_requires(self) -> None:
        """An AIMessage without a name doesn't count toward prereqs.

        This protects against false positives if some upstream agent emits
        an AIMessage without the agent-name tag.
        """
        tool = create_handoff_tool("checkout", "checkout agent", requires=["cart"])
        runtime = _runtime(
            [
                HumanMessage(content="hi"),
                AIMessage(content="orphan"),  # no name
            ]
        )

        result = tool.func(runtime)

        # Should refuse — cart is not in called_agents because the AIMessage
        # has no name to attribute to.
        assert getattr(result, "goto", ()) in ((), None)
        assert "active_agent" not in result.update
