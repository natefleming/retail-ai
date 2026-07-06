"""Tests for parallel fan-out handoff functionality (cohort syntax).

Config shape under test::

    handoffs:
      triage_agent:
      - agents: [pricing_agent, inventory_agent, policy_agent]
        join: synthesizer_agent
      - escalation_agent               # single-target agentic peer

This module covers:

- Per-entry shape validation on ``HandoffRouteModel``:
  ``agents``/``agent`` mutual exclusion, ``agents`` requires ``join``,
  ``is_deterministic`` invalid on cohort entries, minimum 2 siblings,
  distinct siblings, join ≠ sibling.
- Cross-entry cohort validators on ``SwarmModel``: self-source, cross-cohort
  collision (same sibling in two cohorts with different joins), nested
  fan-out (sibling that is also a cohort source).
- Cycle detection through cohort edges (parallel + join treated as
  unconditional).
- ``_handoffs_for_agent`` resolution: cohort entries produce N parallel
  handoff tools + ``parallel_targets`` + ``parallel_join``; single-target
  entries unchanged.
- Swarm graph wiring: source agent gets the per-sibling parallel handoff
  tools.
- YAML round-trip through ``AppConfig``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from dao_ai.config import (
    AgentModel,
    AppConfig,
    HandoffRouteModel,
    LLMModel,
    SwarmModel,
)


# =============================================================================
# HandoffRouteModel per-entry shape
# =============================================================================


@pytest.mark.unit
class TestHandoffRouteModelShape:
    """Per-entry ``HandoffRouteModel.validate_shape`` invariants."""

    def test_single_target_entry_accepted(self) -> None:
        route = HandoffRouteModel(agent="worker_a")
        assert route.agent == "worker_a"
        assert route.agents is None
        assert route.join is None
        assert route.is_deterministic is False

    def test_deterministic_single_target_accepted(self) -> None:
        route = HandoffRouteModel(agent="synth", is_deterministic=True)
        assert route.is_deterministic is True

    def test_cohort_entry_accepted(self) -> None:
        route = HandoffRouteModel(
            agents=["worker_a", "worker_b", "worker_c"],
            join="synth",
        )
        assert route.agent is None
        assert route.agents == ["worker_a", "worker_b", "worker_c"]
        assert route.join == "synth"

    def test_agent_and_agents_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="cannot set both"):
            HandoffRouteModel(
                agent="x",
                agents=["y", "z"],
                join="j",
            )

    def test_neither_agent_nor_agents_rejected(self) -> None:
        with pytest.raises(ValueError, match="must set either"):
            HandoffRouteModel(join="j")

    def test_cohort_requires_join(self) -> None:
        with pytest.raises(ValueError, match=r"cohort.*requires ``join``"):
            HandoffRouteModel(agents=["a", "b"])

    def test_join_without_agents_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"``join`` is only valid on a cohort"):
            HandoffRouteModel(agent="x", join="j")

    def test_is_deterministic_invalid_on_cohort(self) -> None:
        with pytest.raises(ValueError, match="not meaningful on a cohort"):
            HandoffRouteModel(
                agents=["a", "b"],
                join="j",
                is_deterministic=True,
            )

    def test_cohort_of_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two siblings"):
            HandoffRouteModel(agents=["only_one"], join="j")

    def test_duplicate_siblings_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be distinct"):
            HandoffRouteModel(
                agents=["worker_a", "worker_b", "worker_a"],
                join="synth",
            )

    def test_join_cannot_also_be_sibling(self) -> None:
        with pytest.raises(ValueError, match="cannot also be a sibling"):
            HandoffRouteModel(
                agents=["worker_a", "worker_b"],
                join="worker_a",
            )

    def test_cohort_accepts_agent_model_refs(self) -> None:
        a = AgentModel(name="worker_a", model=LLMModel(name="m"))
        b = AgentModel(name="worker_b", model=LLMModel(name="m"))
        j = AgentModel(name="join", model=LLMModel(name="m"))
        route = HandoffRouteModel(agents=[a, b], join=j)
        assert route.agents is not None
        assert route.agents[0].name == "worker_a"
        assert route.join.name == "join"


# =============================================================================
# SwarmModel cross-entry cohort validators
# =============================================================================


@pytest.mark.unit
class TestSwarmParallelCohortValidators:
    """Cross-entry invariants enforced on ``SwarmModel``."""

    def test_valid_cohort_accepted(self) -> None:
        swarm = SwarmModel(
            handoffs={
                "source": [
                    HandoffRouteModel(
                        agents=["worker_a", "worker_b", "worker_c"],
                        join="synth",
                    ),
                ]
            }
        )
        assert swarm.handoffs is not None

    def test_cohort_with_agentic_peer_accepted(self) -> None:
        swarm = SwarmModel(
            handoffs={
                "source": [
                    HandoffRouteModel(
                        agents=["worker_a", "worker_b"],
                        join="synth",
                    ),
                    "escalation",  # agentic peer
                ]
            }
        )
        assert swarm.handoffs is not None

    def test_source_cannot_be_own_cohort_sibling(self) -> None:
        with pytest.raises(ValueError, match="cannot appear in its own parallel"):
            SwarmModel(
                handoffs={
                    "source": [
                        HandoffRouteModel(
                            agents=["source", "worker_b"], join="synth"
                        ),
                    ]
                }
            )

    def test_source_cannot_be_own_cohort_join(self) -> None:
        with pytest.raises(ValueError, match="cannot be its own cohort"):
            SwarmModel(
                handoffs={
                    "source": [
                        HandoffRouteModel(
                            agents=["worker_a", "worker_b"], join="source"
                        ),
                    ]
                }
            )

    def test_sibling_in_two_cohorts_with_different_joins_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="different join targets"
        ):
            SwarmModel(
                handoffs={
                    "source_1": [
                        HandoffRouteModel(
                            agents=["shared", "worker_a"], join="join_1"
                        ),
                    ],
                    "source_2": [
                        HandoffRouteModel(
                            agents=["shared", "worker_b"], join="join_2"
                        ),
                    ],
                }
            )

    def test_sibling_shared_by_cohorts_with_same_join_allowed(self) -> None:
        # Not the recommended shape but not itself incoherent.
        swarm = SwarmModel(
            handoffs={
                "source_1": [
                    HandoffRouteModel(agents=["shared", "worker_a"], join="join"),
                ],
                "source_2": [
                    HandoffRouteModel(agents=["shared", "worker_b"], join="join"),
                ],
            }
        )
        assert swarm.handoffs is not None

    def test_nested_fan_out_rejected(self) -> None:
        # 'inner_src' is a sibling of the outer cohort AND the source of its own cohort.
        with pytest.raises(ValueError, match="Nested parallel fan-out"):
            SwarmModel(
                handoffs={
                    "outer_src": [
                        HandoffRouteModel(
                            agents=["inner_src", "worker_x"], join="outer_join"
                        ),
                    ],
                    "inner_src": [
                        HandoffRouteModel(
                            agents=["worker_1", "worker_2"], join="inner_join"
                        ),
                    ],
                }
            )


# =============================================================================
# Cycle detection through cohort edges
# =============================================================================


@pytest.mark.unit
class TestSwarmCohortCycleDetection:
    """Parallel + join edges are unconditional; cycles through them are rejected."""

    def test_parallel_edge_in_cycle_rejected(self) -> None:
        # source -[parallel]-> worker -[agentic]-> source
        with pytest.raises(ValueError, match="parallel handoff inside a cycle"):
            SwarmModel(
                handoffs={
                    "source": [
                        HandoffRouteModel(agents=["worker", "peer"], join="synth"),
                    ],
                    "worker": ["source"],
                }
            )

    def test_join_edge_in_cycle_rejected(self) -> None:
        # source -[cohort]-> synth -> source (via agentic)
        with pytest.raises(
            ValueError, match=r"(deterministic|parallel) handoff inside a cycle"
        ):
            SwarmModel(
                handoffs={
                    "source": [
                        HandoffRouteModel(agents=["worker_a", "worker_b"], join="synth"),
                    ],
                    "synth": ["source"],
                }
            )

    def test_acyclic_cohort_allowed(self) -> None:
        swarm = SwarmModel(
            handoffs={
                "source": [
                    HandoffRouteModel(agents=["worker_a", "worker_b"], join="synth"),
                ],
                "synth": [],
                "worker_a": [],
                "worker_b": [],
            }
        )
        assert swarm.handoffs is not None


# =============================================================================
# _handoffs_for_agent resolution
# =============================================================================


@pytest.mark.unit
class TestHandoffsForAgentCohort:
    """Cohort entries expand into N parallel tools + parallel_targets/join."""

    def _make_config(
        self,
        agents: list[AgentModel],
        handoffs: dict,
    ) -> AppConfig:
        return AppConfig(
            **{
                "app": {
                    "name": "test_app",
                    "registered_model": {"name": "test_model"},
                    "agents": agents,
                    "orchestration": {
                        "swarm": {
                            "default_agent": agents[0].name,
                            "handoffs": handoffs,
                        }
                    },
                }
            }
        )

    def test_cohort_entry_yields_per_sibling_tools_and_join(self) -> None:
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="source", model=LLMModel(name="m")),
            AgentModel(name="worker_a", model=LLMModel(name="m")),
            AgentModel(name="worker_b", model=LLMModel(name="m")),
            AgentModel(name="synth", model=LLMModel(name="m")),
        ]
        config = self._make_config(
            agents,
            {
                "source": [
                    {"agents": ["worker_a", "worker_b"], "join": "synth"},
                ]
            },
        )

        result = _handoffs_for_agent(agents[0], config)
        tool_names = sorted(t.name for t in result.tools)
        assert tool_names == ["handoff_to_worker_a", "handoff_to_worker_b"]
        assert result.parallel_targets == ("worker_a", "worker_b")
        assert result.parallel_join == "synth"

    def test_cohort_plus_agentic_peer(self) -> None:
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="source", model=LLMModel(name="m")),
            AgentModel(name="worker_a", model=LLMModel(name="m")),
            AgentModel(name="synth", model=LLMModel(name="m")),
            AgentModel(name="escalation", model=LLMModel(name="m")),
        ]
        config = self._make_config(
            agents,
            {
                "source": [
                    {"agents": ["worker_a", "synth"], "join": "escalation"},
                    "escalation",  # Peer entry (agentic)
                ]
            },
        )

        # We used escalation as the join above only to keep the fixture small; more
        # importantly, a plain string entry ALSO exists on the same source.
        result = _handoffs_for_agent(agents[0], config)
        tool_names = sorted(t.name for t in result.tools)
        # Two cohort tools (worker_a, synth) + one peer tool (escalation).
        assert tool_names == ["handoff_to_escalation", "handoff_to_synth", "handoff_to_worker_a"]
        assert set(result.parallel_targets) == {"worker_a", "synth"}
        assert result.parallel_join == "escalation"

    def test_single_target_deterministic_unchanged(self) -> None:
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="source", model=LLMModel(name="m")),
            AgentModel(name="target", model=LLMModel(name="m")),
        ]
        config = self._make_config(
            agents,
            {"source": [{"agent": "target", "is_deterministic": True}]},
        )

        result = _handoffs_for_agent(agents[0], config)
        assert result.deterministic_target == "target"
        assert result.parallel_targets == ()
        assert result.parallel_join is None

    def test_cohort_with_agent_model_refs(self) -> None:
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        source = AgentModel(name="source", model=LLMModel(name="m"))
        worker_a = AgentModel(name="worker_a", model=LLMModel(name="m"))
        worker_b = AgentModel(name="worker_b", model=LLMModel(name="m"))
        synth = AgentModel(name="synth", model=LLMModel(name="m"))

        config = self._make_config(
            [source, worker_a, worker_b, synth],
            {
                "source": [
                    HandoffRouteModel(agents=[worker_a, worker_b], join=synth),
                ]
            },
        )

        result = _handoffs_for_agent(source, config)
        assert result.parallel_targets == ("worker_a", "worker_b")
        assert result.parallel_join == "synth"


# =============================================================================
# Swarm Graph Construction
# =============================================================================


@pytest.mark.unit
@patch("dao_ai.orchestration.swarm.create_agent_node")
@patch("dao_ai.orchestration.swarm.create_store")
@patch("dao_ai.orchestration.swarm.create_checkpointer")
class TestSwarmGraphCohortWiring:
    """The parent graph gets per-sibling tools on the source + sibling→join edges."""

    def _build(self, mock_create_agent_node: Mock):
        mock_create_agent_node.return_value = MagicMock()
        agents = [
            AgentModel(name="source", model=LLMModel(name="m")),
            AgentModel(name="worker_a", model=LLMModel(name="m")),
            AgentModel(name="worker_b", model=LLMModel(name="m")),
            AgentModel(name="synth", model=LLMModel(name="m")),
        ]
        config = AppConfig(
            **{
                "app": {
                    "name": "test_app",
                    "registered_model": {"name": "test_model"},
                    "agents": agents,
                    "orchestration": {
                        "swarm": {
                            "default_agent": "source",
                            "handoffs": {
                                "source": [
                                    {
                                        "agents": ["worker_a", "worker_b"],
                                        "join": "synth",
                                    }
                                ]
                            },
                        }
                    },
                }
            }
        )
        from dao_ai.orchestration.swarm import create_swarm_graph

        return create_swarm_graph(config), mock_create_agent_node

    def test_source_gets_parallel_handoff_tools(
        self,
        mock_checkpointer: Mock,
        mock_store: Mock,
        mock_create_agent_node: Mock,
    ) -> None:
        mock_checkpointer.return_value = None
        mock_store.return_value = None
        _, mock_create_agent_node = self._build(mock_create_agent_node)

        source_call = next(
            c
            for c in mock_create_agent_node.call_args_list
            if c.kwargs["agent"].name == "source"
        )
        tool_names = sorted(t.name for t in source_call.kwargs["additional_tools"])
        assert tool_names == ["handoff_to_worker_a", "handoff_to_worker_b"]


# =============================================================================
# YAML round-trip
# =============================================================================


@pytest.mark.unit
class TestParallelHandoffYAML:
    """Loading a cohort config through ``AppConfig`` should preserve the shape."""

    def test_load_cohort_from_dict(self) -> None:
        config_dict = {
            "app": {
                "name": "fan_out_app",
                "registered_model": {"name": "test_model"},
                "agents": [
                    {"name": "source", "model": {"name": "m"}},
                    {"name": "worker_a", "model": {"name": "m"}},
                    {"name": "worker_b", "model": {"name": "m"}},
                    {"name": "synth", "model": {"name": "m"}},
                ],
                "orchestration": {
                    "swarm": {
                        "default_agent": "source",
                        "handoffs": {
                            "source": [
                                {
                                    "agents": ["worker_a", "worker_b"],
                                    "join": "synth",
                                }
                            ]
                        },
                    }
                },
            }
        }
        config = AppConfig(**config_dict)
        entries = config.app.orchestration.swarm.handoffs["source"]
        assert len(entries) == 1
        entry = entries[0]
        assert isinstance(entry, HandoffRouteModel)
        assert entry.agents == ["worker_a", "worker_b"]
        assert entry.join == "synth"
        assert entry.agent is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
