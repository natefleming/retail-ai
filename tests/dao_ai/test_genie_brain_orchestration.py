"""A Genie-brain agent (``model: {genie_room: ...}``) under an orchestrator.

The brain's model runs its own tool loop server-side and never calls a client
tool, so orchestration must not hand it one it cannot use — and must not try
to route *with* it:

1. ``create_supervisor_graph`` gives every worker ``handoff_to_supervisor``
   except a Genie-brain worker, which gets no additional tools.
2. ``AppModel.set_default_orchestration`` never picks a ``GenieAgentModel`` as
   the supervisor's model: it borrows the first LLM-backed agent's, and with
   none it raises a message that says what to declare.

Hermetic: the supervisor graph is built with the same stubs as the swarm
multi-turn tests (in-memory checkpointer, no store, no extraction, a recorded
``create_agent_node`` fake). No LLM, no workspace, no I/O.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from dao_ai.config import (
    AppConfig,
    GenieAgentModel,
    InferenceEndpointModel,
)


def _capture_warnings(fn: Callable[[], Any]) -> list[str]:
    """Run ``fn`` with a temporary WARNING sink; return captured messages.

    The sink renders ``{extra}`` explicitly: the agent name is a structured
    field, and loguru's default format drops it. This mirrors the real stderr
    sink (``dao_ai.logging.configure_logging``), which also ends in ``{extra}``.
    """
    msgs: list[str] = []
    sink_id = logger.add(
        lambda m: msgs.append(m), level="WARNING", format="{message}{extra}"
    )
    try:
        fn()
    finally:
        logger.remove(sink_id)
    return msgs


SPACE_A: str = "01f05dd06c421ad6b522bf7a517cf6d2"
SPACE_B: str = "01f05dd06c421ad6b522bf7a517cf6d3"


def _brain(name: str, space_id: str) -> dict[str, Any]:
    return {"name": name, "model": {"genie_room": {"agent_id": space_id}}, "tools": []}


def _llm(name: str) -> dict[str, Any]:
    return {"name": name, "model": {"name": "test-model"}, "tools": []}


def _config(agents: list[dict[str, Any]], **app_extra: Any) -> AppConfig:
    return AppConfig(
        **{
            "resources": {
                "genie_rooms": {
                    "a": {"agent_id": SPACE_A},
                    "b": {"agent_id": SPACE_B},
                }
            },
            "app": {"name": "genie_brain_test", "agents": agents, **app_extra},
        }
    )


# =============================================================================
# 1. Supervisor: no handoff tool for a brain worker
# =============================================================================


@pytest.mark.unit
class TestSupervisorGivesBrainNoHandoffTool:
    def test_llm_worker_gets_handoff_brain_gets_none(self) -> None:
        from dao_ai.orchestration.supervisor import create_supervisor_graph

        config = _config(
            [_llm("billing"), _brain("sellout", SPACE_A)],
            orchestration={"supervisor": {"model": {"name": "test-model"}}},
        )
        additional_tools_by_agent: dict[str, list[str]] = {}

        def _fake_create_agent_node(agent: Any, **kwargs: Any) -> Any:
            additional_tools_by_agent[agent.name] = [
                t.name for t in kwargs.get("additional_tools") or []
            ]
            return MagicMock(name=f"subgraph:{agent.name}")

        with (
            patch(
                "dao_ai.orchestration.supervisor.create_checkpointer", return_value=None
            ),
            patch("dao_ai.orchestration.supervisor.create_store", return_value=None),
            patch(
                "dao_ai.orchestration.supervisor.create_extraction_manager_and_executor",
                return_value=(None, None),
            ),
            patch(
                "dao_ai.orchestration.supervisor.create_agent_node",
                side_effect=_fake_create_agent_node,
            ),
        ):
            create_supervisor_graph(config)

        assert additional_tools_by_agent == {
            "billing": ["handoff_to_supervisor"],
            "sellout": [],
        }


# =============================================================================
# 2. Default orchestration never routes with a brain
# =============================================================================


@pytest.mark.unit
class TestDefaultOrchestrationSkipsGenieBrain:
    def test_llm_first_is_unchanged(self) -> None:
        config = _config([_llm("billing"), _brain("sellout", SPACE_A)])
        supervisor = config.app.orchestration.supervisor
        assert supervisor is not None
        assert isinstance(supervisor.model, InferenceEndpointModel)
        assert supervisor.model.name == "test-model"

    def test_brain_first_borrows_the_llm_agents_model(self) -> None:
        config = _config([_brain("sellout", SPACE_A), _llm("billing")])
        supervisor = config.app.orchestration.supervisor
        assert supervisor is not None
        assert isinstance(supervisor.model, InferenceEndpointModel)
        assert supervisor.model.name == "test-model"
        # The brains stay brains.
        assert isinstance(config.app.agents[0].model, GenieAgentModel)

    def test_only_brains_raises_with_what_to_declare(self) -> None:
        with pytest.raises(ValueError, match="no LLM for a supervisor to route with"):
            _config([_brain("sellout", SPACE_A), _brain("inventory", SPACE_B)])

    def test_single_brain_still_defaults_to_swarm(self) -> None:
        config = _config([_brain("sellout", SPACE_A)])
        assert config.app.orchestration.supervisor is None
        assert config.app.orchestration.swarm is not None


# =============================================================================
# 3. A brain cannot use what it is handed — say so at config load
# =============================================================================


@pytest.mark.unit
class TestBrainRejectsUnusableConfig:
    """``bind_tools`` silently discards everything it is handed, so a config
    that declares client tools or structured output on a Genie brain builds
    cleanly and then can never honor either. Both are rejected at config load
    instead, where the message can name the agent."""

    def test_declared_tools_are_rejected(self) -> None:
        brain = _brain("sellout", SPACE_A)
        brain["tools"] = [{"name": "order_lookup", "function": {"name": "a.b.c"}}]
        with pytest.raises(ValueError, match="sellout"):
            _config([brain])

    def test_declared_tools_message_names_tools_and_the_alternative(self) -> None:
        brain = _brain("sellout", SPACE_A)
        brain["tools"] = [{"name": "order_lookup", "function": {"name": "a.b.c"}}]
        with pytest.raises(ValueError) as excinfo:
            _config([brain])
        message = str(excinfo.value)
        assert "order_lookup" in message
        # Points at the shape that does support tools: a `type: genie` tool
        # hanging off an LLM-backed agent.
        assert "type: genie" in message

    def test_response_format_is_rejected(self) -> None:
        brain = _brain("sellout", SPACE_A)
        brain["response_format"] = {"response_schema": '{"type": "object"}'}
        with pytest.raises(ValueError, match="response_format"):
            _config([brain])

    def test_an_llm_agent_keeps_tools_and_response_format(self) -> None:
        agent = _llm("billing")
        agent["tools"] = [{"name": "order_lookup", "function": {"name": "a.b.c"}}]
        agent["response_format"] = {"response_schema": '{"type": "object"}'}
        config = _config([agent])
        assert len(config.app.agents[0].tools) == 1
        assert config.app.agents[0].response_format is not None


# =============================================================================
# 4. The all-brains message must not steer at a swarm that cannot work
# =============================================================================


@pytest.mark.unit
class TestAllBrainsMessageIsActionable:
    def test_points_at_supervisor_model(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _config([_brain("sellout", SPACE_A), _brain("inventory", SPACE_B)])
        assert "orchestration.supervisor.model" in str(excinfo.value)

    def test_any_swarm_suggestion_is_qualified_as_deterministic(self) -> None:
        """A bare swarm of brains is a silently dead app: the handoff tools are
        discarded, ``active_agent`` is never written, and every turn routes to
        the default agent forever. Only deterministic handoffs work."""
        with pytest.raises(ValueError) as excinfo:
            _config([_brain("sellout", SPACE_A), _brain("inventory", SPACE_B)])
        message = str(excinfo.value)
        if "swarm" in message:
            assert "is_deterministic" in message


# =============================================================================
# 5. Swarm: an agentic handoff out of a brain is a dead end
# =============================================================================


@pytest.mark.unit
class TestSwarmBrainHandoffs:
    """In a swarm, ``_handoffs_for_agent`` passes agentic handoff tools as the
    worker's ``additional_tools``. A Genie model discards them, so it emits no
    tool call, ``active_agent`` is never written, and the swarm router sends
    every subsequent turn back to the same agent — a dead app with no error.
    A deterministic handoff is a real graph edge, so it still works.
    """

    def _swarm(self, handoffs: dict[str, Any], default_agent: str) -> AppConfig:
        return _config(
            [_brain("sellout", SPACE_A), _llm("billing")],
            orchestration={
                "swarm": {"default_agent": default_agent, "handoffs": handoffs}
            },
        )

    def test_agentic_handoff_out_of_a_brain_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            self._swarm({"sellout": ["billing"]}, default_agent="sellout")
        message = str(excinfo.value)
        assert "sellout" in message
        assert "is_deterministic" in message

    def test_deterministic_handoff_out_of_a_brain_is_allowed(self) -> None:
        config = self._swarm(
            {"sellout": [{"agent": "billing", "is_deterministic": True}]},
            default_agent="sellout",
        )
        assert config.app.orchestration.swarm is not None

    def test_a_brain_with_no_outbound_handoffs_is_allowed(self) -> None:
        """A brain as a swarm leaf is fine — it answers and the turn ends."""
        config = self._swarm({"billing": ["sellout"]}, default_agent="billing")
        assert config.app.orchestration.swarm is not None

    def test_agentic_handoff_out_of_an_llm_agent_is_untouched(self) -> None:
        config = self._swarm({"billing": ["sellout"]}, default_agent="billing")
        handoffs = config.app.orchestration.swarm.handoffs or {}
        assert "billing" in handoffs


# =============================================================================
# 6. Supervisor: say out loud that a brain worker is a graph sink
# =============================================================================


@pytest.mark.unit
class TestSupervisorWarnsBrainIsASink:
    """A brain worker has no ``handoff_to_supervisor`` and no outgoing edge, so
    control cannot return to the supervisor mid-turn: the brain answers and the
    turn ends. That is intended, but it means a supervisor cannot compose a
    brain with another agent inside one turn — and nothing in the build says so.
    Warn once per brain worker, naming it.
    """

    def _build(self, agents: list[dict[str, Any]]) -> Callable[[], Any]:
        from dao_ai.orchestration.supervisor import create_supervisor_graph

        config = _config(
            agents,
            orchestration={"supervisor": {"model": {"name": "test-model"}}},
        )

        def _run() -> None:
            with (
                patch(
                    "dao_ai.orchestration.supervisor.create_checkpointer",
                    return_value=None,
                ),
                patch(
                    "dao_ai.orchestration.supervisor.create_store", return_value=None
                ),
                patch(
                    "dao_ai.orchestration.supervisor.create_extraction_manager_and_executor",
                    return_value=(None, None),
                ),
                patch(
                    "dao_ai.orchestration.supervisor.create_agent_node",
                    side_effect=lambda agent, **kwargs: MagicMock(
                        name=f"subgraph:{agent.name}"
                    ),
                ),
            ):
                create_supervisor_graph(config)

        return _run

    def test_brain_worker_is_named_in_a_warning(self) -> None:
        msgs = _capture_warnings(
            self._build([_llm("billing"), _brain("sellout", SPACE_A)])
        )
        assert any("sellout" in m for m in msgs), msgs

    def test_the_warning_says_control_cannot_return_mid_turn(self) -> None:
        msgs = _capture_warnings(
            self._build([_llm("billing"), _brain("sellout", SPACE_A)])
        )
        brain_msgs = [m for m in msgs if "sellout" in m]
        assert brain_msgs, msgs
        # The point is the consequence, not the mechanism: a reader has to learn
        # that the turn ends here rather than routing on.
        assert any("turn" in m for m in brain_msgs), brain_msgs

    def test_an_all_llm_supervisor_names_no_agent(self) -> None:
        msgs = _capture_warnings(self._build([_llm("billing"), _llm("returns")]))
        assert not [m for m in msgs if "billing" in m or "returns" in m], msgs
