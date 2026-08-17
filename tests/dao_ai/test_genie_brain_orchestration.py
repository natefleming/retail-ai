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

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dao_ai.config import (
    AppConfig,
    GenieAgentModel,
    InferenceEndpointModel,
)

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
