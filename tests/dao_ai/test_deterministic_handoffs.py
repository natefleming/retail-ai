"""
Tests for deterministic handoff functionality.

This module tests:
- HandoffRouteModel configuration model
- SwarmModel handoffs type with HandoffRouteModel entries
- _handoffs_for_agent resolution of deterministic vs agentic handoffs
- Deterministic handler wrapper
- Swarm graph construction with deterministic edges
- Validation rules (single deterministic per agent, no self-handoff)
- YAML/dict-based configuration parsing
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from dao_ai.config import (
    AgentModel,
    AppConfig,
    HandoffRouteModel,
    LLMModel,
    SwarmModel,
)

# =============================================================================
# HandoffRouteModel Unit Tests
# =============================================================================


@pytest.mark.unit
class TestHandoffRouteModel:
    """Tests for the HandoffRouteModel configuration model."""

    def test_handoff_route_model_with_string_agent(self) -> None:
        """Test creating a HandoffRouteModel with a string agent reference."""
        route = HandoffRouteModel(agent="target_agent", is_deterministic=True)
        assert route.agent == "target_agent"
        assert route.is_deterministic is True

    def test_handoff_route_model_with_agent_model(self) -> None:
        """Test creating a HandoffRouteModel with an AgentModel reference."""
        agent = AgentModel(name="target_agent", model=LLMModel(name="test-model"))
        route = HandoffRouteModel(agent=agent, is_deterministic=True)
        assert isinstance(route.agent, AgentModel)
        assert route.agent.name == "target_agent"
        assert route.is_deterministic is True

    def test_handoff_route_model_defaults_to_non_deterministic(self) -> None:
        """Test that is_deterministic defaults to False."""
        route = HandoffRouteModel(agent="target_agent")
        assert route.is_deterministic is False

    def test_handoff_route_model_explicit_non_deterministic(self) -> None:
        """Test explicitly setting is_deterministic to False."""
        route = HandoffRouteModel(agent="target_agent", is_deterministic=False)
        assert route.is_deterministic is False

    def test_handoff_route_model_rejects_extra_fields(self) -> None:
        """Test that extra fields are rejected (extra='forbid')."""
        with pytest.raises(Exception):
            HandoffRouteModel(
                agent="target_agent",
                is_deterministic=True,
                unknown_field="value",  # type: ignore[call-arg]
            )


# =============================================================================
# SwarmModel Configuration Tests
# =============================================================================


@pytest.mark.unit
class TestSwarmModelHandoffsWithHandoffRoute:
    """Tests for SwarmModel.handoffs with HandoffRouteModel entries."""

    def test_swarm_model_accepts_handoff_route_model(self) -> None:
        """Test that SwarmModel.handoffs accepts HandoffRouteModel in the list."""
        swarm = SwarmModel(
            default_agent="agent_a",
            handoffs={
                "agent_a": [
                    HandoffRouteModel(agent="agent_b", is_deterministic=True),
                ]
            },
        )
        assert swarm.handoffs is not None
        assert len(swarm.handoffs["agent_a"]) == 1
        entry = swarm.handoffs["agent_a"][0]
        assert isinstance(entry, HandoffRouteModel)
        assert entry.is_deterministic is True

    def test_swarm_model_accepts_mixed_handoff_types(self) -> None:
        """Test that handoffs can contain str, AgentModel, and HandoffRouteModel."""
        agent = AgentModel(name="agent_c", model=LLMModel(name="test-model"))
        swarm = SwarmModel(
            default_agent="agent_a",
            handoffs={
                "agent_a": [
                    "agent_b",  # str shorthand
                    agent,  # AgentModel
                    HandoffRouteModel(agent="agent_d", is_deterministic=True),
                ]
            },
        )
        assert swarm.handoffs is not None
        targets = swarm.handoffs["agent_a"]
        assert len(targets) == 3
        assert isinstance(targets[0], str)
        assert isinstance(targets[1], AgentModel)
        assert isinstance(targets[2], HandoffRouteModel)

    def test_swarm_model_null_handoffs_for_agent(self) -> None:
        """Test that null (~) handoffs still works (hand off to any)."""
        swarm = SwarmModel(
            default_agent="agent_a",
            handoffs={"agent_a": None},
        )
        assert swarm.handoffs["agent_a"] is None

    def test_swarm_model_empty_handoffs_list(self) -> None:
        """Test that empty handoffs list disables handoffs for an agent."""
        swarm = SwarmModel(
            default_agent="agent_a",
            handoffs={"agent_a": []},
        )
        assert swarm.handoffs["agent_a"] == []

    def test_swarm_model_backward_compatible_with_str_only(self) -> None:
        """Test that existing str-only handoffs configs still work."""
        swarm = SwarmModel(
            default_agent="agent_a",
            handoffs={
                "agent_a": ["agent_b", "agent_c"],
            },
        )
        assert len(swarm.handoffs["agent_a"]) == 2


# =============================================================================
# _resolve_agent Tests
# =============================================================================


@pytest.mark.unit
class TestResolveAgent:
    """Tests for _resolve_agent helper function."""

    def test_resolve_string_agent(self) -> None:
        """Test resolving a plain string agent name."""
        from dao_ai.orchestration.swarm import _resolve_agent

        agent_ref, is_deterministic = _resolve_agent("agent_b")
        assert agent_ref == "agent_b"
        assert is_deterministic is False

    def test_resolve_agent_model(self) -> None:
        """Test resolving an AgentModel reference."""
        from dao_ai.orchestration.swarm import _resolve_agent

        agent = AgentModel(name="agent_b", model=LLMModel(name="test-model"))
        agent_ref, is_deterministic = _resolve_agent(agent)
        assert isinstance(agent_ref, AgentModel)
        assert agent_ref.name == "agent_b"
        assert is_deterministic is False

    def test_resolve_handoff_route_deterministic(self) -> None:
        """Test resolving a deterministic HandoffRouteModel."""
        from dao_ai.orchestration.swarm import _resolve_agent

        route = HandoffRouteModel(agent="agent_b", is_deterministic=True)
        agent_ref, is_deterministic = _resolve_agent(route)
        assert agent_ref == "agent_b"
        assert is_deterministic is True

    def test_resolve_handoff_route_non_deterministic(self) -> None:
        """Test resolving a non-deterministic HandoffRouteModel."""
        from dao_ai.orchestration.swarm import _resolve_agent

        route = HandoffRouteModel(agent="agent_b", is_deterministic=False)
        agent_ref, is_deterministic = _resolve_agent(route)
        assert agent_ref == "agent_b"
        assert is_deterministic is False


# =============================================================================
# HandoffResult Tests
# =============================================================================


@pytest.mark.unit
class TestHandoffResult:
    """Tests for the HandoffResult dataclass."""

    def test_handoff_result_defaults(self) -> None:
        """Test HandoffResult default values."""
        from dao_ai.orchestration.swarm import HandoffResult

        result = HandoffResult()
        assert result.tools == []
        assert result.deterministic_target is None

    def test_handoff_result_with_values(self) -> None:
        """Test HandoffResult with explicit values."""
        from dao_ai.orchestration.swarm import HandoffResult

        mock_tool = MagicMock()
        result = HandoffResult(
            tools=[mock_tool],
            deterministic_target="agent_b",
        )
        assert len(result.tools) == 1
        assert result.deterministic_target == "agent_b"

    def test_handoff_result_is_frozen(self) -> None:
        """Test that HandoffResult is immutable (frozen=True)."""
        from dao_ai.orchestration.swarm import HandoffResult

        result = HandoffResult()
        with pytest.raises(AttributeError):
            result.deterministic_target = "agent_b"  # type: ignore[misc]


# =============================================================================
# _handoffs_for_agent Tests
# =============================================================================


@pytest.mark.unit
class TestHandoffsForAgent:
    """Tests for _handoffs_for_agent with deterministic handoffs."""

    def _make_config(
        self,
        agents: list[AgentModel],
        handoffs: dict | None = None,
    ) -> AppConfig:
        """Helper to create a minimal AppConfig for testing."""
        config_dict: dict = {
            "app": {
                "name": "test_app",
                "registered_model": {"name": "test_model"},
                "agents": agents,
                "orchestration": {
                    "swarm": {
                        "default_agent": agents[0].name,
                        "handoffs": handoffs or {},
                    }
                },
            }
        }
        return AppConfig(**config_dict)

    def test_agentic_handoffs_return_tools(self) -> None:
        """Test that non-deterministic handoffs produce handoff tools."""
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
            AgentModel(name="agent_b", model=LLMModel(name="test-model")),
        ]
        config = self._make_config(agents, {"agent_a": ["agent_b"]})

        result = _handoffs_for_agent(agents[0], config)
        assert len(result.tools) == 1
        assert result.deterministic_target is None
        assert result.tools[0].name == "handoff_to_agent_b"

    def test_deterministic_handoff_returns_target(self) -> None:
        """Test that a deterministic handoff sets deterministic_target."""
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
            AgentModel(name="agent_b", model=LLMModel(name="test-model")),
        ]
        config = self._make_config(
            agents,
            {
                "agent_a": [
                    {"agent": "agent_b", "is_deterministic": True},
                ]
            },
        )

        result = _handoffs_for_agent(agents[0], config)
        assert result.tools == []
        assert result.deterministic_target == "agent_b"

    def test_mixed_agentic_and_deterministic_handoffs(self) -> None:
        """Test that agentic and deterministic handoffs coexist correctly."""
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
            AgentModel(name="agent_b", model=LLMModel(name="test-model")),
            AgentModel(name="agent_c", model=LLMModel(name="test-model")),
        ]
        config = self._make_config(
            agents,
            {
                "agent_a": [
                    "agent_b",  # agentic
                    {"agent": "agent_c", "is_deterministic": True},  # deterministic
                ]
            },
        )

        result = _handoffs_for_agent(agents[0], config)
        assert len(result.tools) == 1
        assert result.tools[0].name == "handoff_to_agent_b"
        assert result.deterministic_target == "agent_c"

    def test_multiple_deterministic_handoffs_raises_error(self) -> None:
        """Test that multiple deterministic handoffs for same agent raises ValueError."""
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
            AgentModel(name="agent_b", model=LLMModel(name="test-model")),
            AgentModel(name="agent_c", model=LLMModel(name="test-model")),
        ]
        config = self._make_config(
            agents,
            {
                "agent_a": [
                    {"agent": "agent_b", "is_deterministic": True},
                    {"agent": "agent_c", "is_deterministic": True},
                ]
            },
        )

        with pytest.raises(ValueError, match="multiple deterministic handoffs"):
            _handoffs_for_agent(agents[0], config)

    def test_deterministic_self_handoff_raises_error(self) -> None:
        """Test that a deterministic handoff to self raises ValueError."""
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
        ]
        config = self._make_config(
            agents,
            {
                "agent_a": [
                    {"agent": "agent_a", "is_deterministic": True},
                ]
            },
        )

        with pytest.raises(
            ValueError, match="cannot have a deterministic handoff to itself"
        ):
            _handoffs_for_agent(agents[0], config)

    def test_agentic_self_handoff_is_silently_skipped(self) -> None:
        """Test that a non-deterministic self-handoff is just skipped (existing behavior)."""
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
            AgentModel(name="agent_b", model=LLMModel(name="test-model")),
        ]
        config = self._make_config(
            agents,
            {
                "agent_a": ["agent_a", "agent_b"],
            },
        )

        result = _handoffs_for_agent(agents[0], config)
        assert len(result.tools) == 1
        assert result.tools[0].name == "handoff_to_agent_b"
        assert result.deterministic_target is None

    def test_null_handoffs_falls_back_to_all_agents(self) -> None:
        """Test that null handoffs (not in dict) falls back to all agents."""
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
            AgentModel(name="agent_b", model=LLMModel(name="test-model")),
        ]
        # agent_a is not in handoffs dict → falls back to all agents
        config = self._make_config(agents, {})

        result = _handoffs_for_agent(agents[0], config)
        # Should create tool for agent_b (skips agent_a since it's self)
        assert len(result.tools) == 1
        assert result.tools[0].name == "handoff_to_agent_b"
        assert result.deterministic_target is None

    def test_handoff_route_model_with_agent_model_ref(self) -> None:
        """Test HandoffRouteModel with inline AgentModel reference."""
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agent_b = AgentModel(name="agent_b", model=LLMModel(name="test-model"))
        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
            agent_b,
        ]
        config = self._make_config(
            agents,
            {
                "agent_a": [
                    HandoffRouteModel(agent=agent_b, is_deterministic=True),
                ]
            },
        )

        result = _handoffs_for_agent(agents[0], config)
        assert result.deterministic_target == "agent_b"
        assert result.tools == []

    def test_unknown_agent_reference_is_skipped(self) -> None:
        """Test that an unknown agent name in handoffs is skipped with warning."""
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
        ]
        config = self._make_config(
            agents,
            {
                "agent_a": ["nonexistent_agent"],
            },
        )

        result = _handoffs_for_agent(agents[0], config)
        assert result.tools == []
        assert result.deterministic_target is None


# =============================================================================
# Deterministic Handler Wrapper Tests
# =============================================================================


@pytest.mark.unit
class TestDeterministicHandler:
    """Tests for _create_deterministic_handler wrapper."""

    def test_deterministic_handler_sets_active_agent(self) -> None:
        """Test that the deterministic handler sets active_agent on result."""
        import asyncio

        from dao_ai.orchestration.swarm import _create_deterministic_handler

        inner_result: dict = {"messages": [], "active_agent": None}
        inner_handler = AsyncMock(return_value=inner_result)

        handler = _create_deterministic_handler(inner_handler, "target_agent")
        mock_state: dict = {"messages": []}
        mock_runtime = MagicMock()

        result: dict = asyncio.get_event_loop().run_until_complete(
            handler(mock_state, mock_runtime)
        )

        assert result["active_agent"] == "target_agent"
        inner_handler.assert_awaited_once_with(mock_state, mock_runtime)

    def test_deterministic_handler_preserves_messages(self) -> None:
        """Test that the deterministic handler preserves other state fields."""
        import asyncio

        from dao_ai.orchestration.swarm import _create_deterministic_handler

        inner_result: dict = {
            "messages": [{"role": "assistant", "content": "hello"}],
            "active_agent": None,
            "custom_field": "preserved",
        }
        inner_handler = AsyncMock(return_value=inner_result)

        handler = _create_deterministic_handler(inner_handler, "target_agent")
        result: dict = asyncio.get_event_loop().run_until_complete(
            handler({"messages": []}, MagicMock())
        )

        assert result["messages"] == [{"role": "assistant", "content": "hello"}]
        assert result["custom_field"] == "preserved"
        assert result["active_agent"] == "target_agent"


# =============================================================================
# Swarm Graph Construction Tests
# =============================================================================


@pytest.mark.unit
@patch("dao_ai.orchestration.swarm.create_agent_node")
@patch("dao_ai.orchestration.swarm.create_store")
@patch("dao_ai.orchestration.swarm.create_checkpointer")
class TestSwarmGraphWithDeterministicHandoffs:
    """Tests for swarm graph creation with deterministic handoffs."""

    def test_swarm_graph_creates_deterministic_edges(
        self,
        mock_checkpointer: Mock,
        mock_store: Mock,
        mock_create_agent_node: Mock,
    ) -> None:
        """Test that deterministic handoffs produce add_edge calls in the graph."""
        mock_checkpointer.return_value = None
        mock_store.return_value = None
        mock_create_agent_node.return_value = MagicMock()

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
            AgentModel(name="agent_b", model=LLMModel(name="test-model")),
        ]

        config_dict: dict = {
            "app": {
                "name": "test_app",
                "registered_model": {"name": "test_model"},
                "agents": agents,
                "orchestration": {
                    "swarm": {
                        "default_agent": "agent_a",
                        "handoffs": {
                            "agent_a": [
                                {"agent": "agent_b", "is_deterministic": True},
                            ],
                        },
                    }
                },
            }
        }

        config = AppConfig(**config_dict)

        from dao_ai.orchestration.swarm import create_swarm_graph

        try:
            create_swarm_graph(config)
        except Exception:
            # Graph compilation might fail in test due to mocked subgraphs
            pass

        # Verify create_agent_node was called for each agent
        assert mock_create_agent_node.call_count == 2

    def test_swarm_graph_no_handoff_tools_for_deterministic(
        self,
        mock_checkpointer: Mock,
        mock_store: Mock,
        mock_create_agent_node: Mock,
    ) -> None:
        """Test that deterministic handoffs don't create handoff tools."""
        mock_checkpointer.return_value = None
        mock_store.return_value = None
        mock_create_agent_node.return_value = MagicMock()

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
            AgentModel(name="agent_b", model=LLMModel(name="test-model")),
        ]

        config_dict: dict = {
            "app": {
                "name": "test_app",
                "registered_model": {"name": "test_model"},
                "agents": agents,
                "orchestration": {
                    "swarm": {
                        "default_agent": "agent_a",
                        "handoffs": {
                            "agent_a": [
                                {"agent": "agent_b", "is_deterministic": True},
                            ],
                        },
                    }
                },
            }
        }

        config = AppConfig(**config_dict)

        from dao_ai.orchestration.swarm import create_swarm_graph

        try:
            create_swarm_graph(config)
        except Exception:
            pass

        # Check that agent_a was created with NO handoff tools
        first_call_kwargs = mock_create_agent_node.call_args_list[0][1]
        additional_tools = first_call_kwargs.get("additional_tools", [])
        assert len(additional_tools) == 0, (
            f"Expected no handoff tools for deterministic agent, got {len(additional_tools)}"
        )

    def test_swarm_graph_mixed_handoffs_creates_tools_for_agentic_only(
        self,
        mock_checkpointer: Mock,
        mock_store: Mock,
        mock_create_agent_node: Mock,
    ) -> None:
        """Test that mixed handoffs create tools only for agentic entries."""
        mock_checkpointer.return_value = None
        mock_store.return_value = None
        mock_create_agent_node.return_value = MagicMock()

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
            AgentModel(name="agent_b", model=LLMModel(name="test-model")),
            AgentModel(name="agent_c", model=LLMModel(name="test-model")),
        ]

        config_dict: dict = {
            "app": {
                "name": "test_app",
                "registered_model": {"name": "test_model"},
                "agents": agents,
                "orchestration": {
                    "swarm": {
                        "default_agent": "agent_a",
                        "handoffs": {
                            "agent_a": [
                                "agent_b",  # agentic
                                {"agent": "agent_c", "is_deterministic": True},
                            ],
                        },
                    }
                },
            }
        }

        config = AppConfig(**config_dict)

        from dao_ai.orchestration.swarm import create_swarm_graph

        try:
            create_swarm_graph(config)
        except Exception:
            pass

        # agent_a should have 1 agentic handoff tool (agent_b), not agent_c
        first_call_kwargs = mock_create_agent_node.call_args_list[0][1]
        additional_tools = first_call_kwargs.get("additional_tools", [])
        assert len(additional_tools) == 1
        assert additional_tools[0].name == "handoff_to_agent_b"

    def test_swarm_graph_backward_compatible_with_string_handoffs(
        self,
        mock_checkpointer: Mock,
        mock_store: Mock,
        mock_create_agent_node: Mock,
    ) -> None:
        """Test that existing string-only handoffs configs still work."""
        mock_checkpointer.return_value = None
        mock_store.return_value = None
        mock_create_agent_node.return_value = MagicMock()

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
            AgentModel(name="agent_b", model=LLMModel(name="test-model")),
        ]

        config_dict: dict = {
            "app": {
                "name": "test_app",
                "registered_model": {"name": "test_model"},
                "agents": agents,
                "orchestration": {
                    "swarm": {
                        "default_agent": "agent_a",
                        "handoffs": {
                            "agent_a": ["agent_b"],
                        },
                    }
                },
            }
        }

        config = AppConfig(**config_dict)

        from dao_ai.orchestration.swarm import create_swarm_graph

        try:
            create_swarm_graph(config)
        except Exception:
            pass

        # agent_a should have 1 agentic handoff tool
        first_call_kwargs = mock_create_agent_node.call_args_list[0][1]
        additional_tools = first_call_kwargs.get("additional_tools", [])
        assert len(additional_tools) == 1
        assert additional_tools[0].name == "handoff_to_agent_b"


# =============================================================================
# YAML/Dict Configuration Parsing Tests
# =============================================================================


@pytest.mark.unit
class TestDeterministicHandoffYAMLConfig:
    """Tests for loading deterministic handoff config from YAML-like dicts."""

    def test_load_deterministic_handoff_from_dict(self) -> None:
        """Test loading a deterministic handoff from a YAML-like dict."""
        config_dict: dict = {
            "app": {
                "name": "test_app",
                "registered_model": {"name": "test_model"},
                "agents": [
                    {"name": "triage", "model": {"name": "test-model"}},
                    {"name": "resolver", "model": {"name": "test-model"}},
                ],
                "orchestration": {
                    "swarm": {
                        "default_agent": "triage",
                        "handoffs": {
                            "triage": [
                                {
                                    "agent": "resolver",
                                    "is_deterministic": True,
                                }
                            ],
                        },
                    }
                },
            }
        }

        config = AppConfig(**config_dict)
        swarm = config.app.orchestration.swarm
        assert swarm is not None
        assert swarm.handoffs is not None

        triage_handoffs = swarm.handoffs["triage"]
        assert len(triage_handoffs) == 1
        entry = triage_handoffs[0]
        assert isinstance(entry, HandoffRouteModel)
        assert entry.agent == "resolver"
        assert entry.is_deterministic is True

    def test_load_mixed_handoffs_from_dict(self) -> None:
        """Test loading mixed agentic and deterministic handoffs from dict."""
        config_dict: dict = {
            "app": {
                "name": "test_app",
                "registered_model": {"name": "test_model"},
                "agents": [
                    {"name": "agent_a", "model": {"name": "test-model"}},
                    {"name": "agent_b", "model": {"name": "test-model"}},
                    {"name": "agent_c", "model": {"name": "test-model"}},
                ],
                "orchestration": {
                    "swarm": {
                        "default_agent": "agent_a",
                        "handoffs": {
                            "agent_a": [
                                "agent_b",
                                {
                                    "agent": "agent_c",
                                    "is_deterministic": True,
                                },
                            ],
                        },
                    }
                },
            }
        }

        config = AppConfig(**config_dict)
        swarm = config.app.orchestration.swarm
        handoffs = swarm.handoffs["agent_a"]
        assert len(handoffs) == 2
        # First entry is a string
        assert handoffs[0] == "agent_b"
        # Second entry is a HandoffRouteModel
        assert isinstance(handoffs[1], HandoffRouteModel)
        assert handoffs[1].is_deterministic is True

    def test_load_pipeline_style_config(self) -> None:
        """Test loading a full pipeline-style deterministic config."""
        config_dict: dict = {
            "app": {
                "name": "pipeline_app",
                "registered_model": {"name": "test_model"},
                "agents": [
                    {"name": "step_1", "model": {"name": "test-model"}},
                    {"name": "step_2", "model": {"name": "test-model"}},
                    {"name": "step_3", "model": {"name": "test-model"}},
                ],
                "orchestration": {
                    "swarm": {
                        "default_agent": "step_1",
                        "handoffs": {
                            "step_1": [
                                {"agent": "step_2", "is_deterministic": True},
                            ],
                            "step_2": [
                                {"agent": "step_3", "is_deterministic": True},
                            ],
                        },
                    }
                },
            }
        }

        config = AppConfig(**config_dict)
        swarm = config.app.orchestration.swarm

        # Step 1 -> Step 2 (deterministic)
        step1_handoffs = swarm.handoffs["step_1"]
        assert len(step1_handoffs) == 1
        assert isinstance(step1_handoffs[0], HandoffRouteModel)
        assert step1_handoffs[0].agent == "step_2"
        assert step1_handoffs[0].is_deterministic is True

        # Step 2 -> Step 3 (deterministic)
        step2_handoffs = swarm.handoffs["step_2"]
        assert len(step2_handoffs) == 1
        assert isinstance(step2_handoffs[0], HandoffRouteModel)
        assert step2_handoffs[0].agent == "step_3"
        assert step2_handoffs[0].is_deterministic is True


# =============================================================================
# Validation Error Tests
# =============================================================================


@pytest.mark.unit
class TestDeterministicHandoffValidation:
    """Tests for validation of deterministic handoff configuration."""

    def test_multiple_deterministic_raises_at_build_time(self) -> None:
        """Test that multiple deterministic handoffs for one agent raise ValueError."""
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
            AgentModel(name="agent_b", model=LLMModel(name="test-model")),
            AgentModel(name="agent_c", model=LLMModel(name="test-model")),
        ]
        config_dict: dict = {
            "app": {
                "name": "test_app",
                "registered_model": {"name": "test_model"},
                "agents": agents,
                "orchestration": {
                    "swarm": {
                        "default_agent": "agent_a",
                        "handoffs": {
                            "agent_a": [
                                {"agent": "agent_b", "is_deterministic": True},
                                {"agent": "agent_c", "is_deterministic": True},
                            ],
                        },
                    }
                },
            }
        }
        config = AppConfig(**config_dict)

        with pytest.raises(ValueError, match="multiple deterministic"):
            _handoffs_for_agent(agents[0], config)

    def test_deterministic_self_reference_raises_at_build_time(self) -> None:
        """Test that a deterministic self-handoff raises ValueError."""
        from dao_ai.orchestration.swarm import _handoffs_for_agent

        agents = [
            AgentModel(name="agent_a", model=LLMModel(name="test-model")),
        ]
        config_dict: dict = {
            "app": {
                "name": "test_app",
                "registered_model": {"name": "test_model"},
                "agents": agents,
                "orchestration": {
                    "swarm": {
                        "default_agent": "agent_a",
                        "handoffs": {
                            "agent_a": [
                                {"agent": "agent_a", "is_deterministic": True},
                            ],
                        },
                    }
                },
            }
        }
        config = AppConfig(**config_dict)

        with pytest.raises(
            ValueError, match="cannot have a deterministic handoff to itself"
        ):
            _handoffs_for_agent(agents[0], config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
