"""
Tests for swarm-level middleware functionality.

This module tests that middleware configured at the swarm level
is properly applied to all agents in the swarm.
"""

from copy import deepcopy
from unittest.mock import MagicMock, Mock, patch

import pytest

from dao_ai.config import (
    AgentModel,
    AppConfig,
    LLMModel,
    MiddlewareModel,
    SwarmModel,
)

# =============================================================================
# SwarmModel Configuration Tests
# =============================================================================


@pytest.mark.unit
class TestSwarmModelMiddlewareField:
    """Tests for the middleware field on SwarmModel."""

    def test_swarm_model_accepts_empty_middleware(self) -> None:
        """Test that SwarmModel can be created with empty middleware list."""
        swarm = SwarmModel(
            default_agent="test_agent",
            middleware=[],
        )

        assert swarm.middleware == []

    def test_swarm_model_accepts_middleware_list(self) -> None:
        """Test that SwarmModel can be created with middleware list."""
        middleware = [
            MiddlewareModel(
                name="dao_ai.middleware.test_middleware",
                args={"param": "value"},
            )
        ]

        swarm = SwarmModel(
            default_agent="test_agent",
            middleware=middleware,
        )

        assert len(swarm.middleware) == 1
        assert swarm.middleware[0].name == "dao_ai.middleware.test_middleware"
        assert swarm.middleware[0].args == {"param": "value"}

    def test_swarm_model_middleware_defaults_to_empty_list(self) -> None:
        """Test that middleware field defaults to empty list."""
        swarm = SwarmModel(
            default_agent="test_agent",
        )

        assert swarm.middleware == []

    def test_swarm_model_with_multiple_middleware(self) -> None:
        """Test that SwarmModel can accept multiple middleware items."""
        middleware = [
            MiddlewareModel(name="middleware1"),
            MiddlewareModel(name="middleware2"),
            MiddlewareModel(name="middleware3"),
        ]

        swarm = SwarmModel(
            default_agent="test_agent",
            middleware=middleware,
        )

        assert len(swarm.middleware) == 3
        assert swarm.middleware[0].name == "middleware1"
        assert swarm.middleware[1].name == "middleware2"
        assert swarm.middleware[2].name == "middleware3"


# =============================================================================
# Middleware Merging Tests
# =============================================================================


@pytest.mark.unit
class TestSwarmMiddlewareMerging:
    """Tests for merging swarm-level middleware with agent middleware."""

    def test_swarm_middleware_merged_with_agent_middleware(self) -> None:
        """Test that swarm middleware is prepended to agent middleware."""
        # Create swarm middleware
        swarm_middleware = [
            MiddlewareModel(name="swarm_middleware_1"),
            MiddlewareModel(name="swarm_middleware_2"),
        ]

        # Create agent with its own middleware
        agent = AgentModel(
            name="test_agent",
            model=LLMModel(name="test-model"),
            middleware=[
                MiddlewareModel(name="agent_middleware_1"),
                MiddlewareModel(name="agent_middleware_2"),
            ],
        )

        # Simulate merging (as done in create_swarm_graph)
        agent_with_middleware = deepcopy(agent)
        agent_with_middleware.middleware = swarm_middleware + agent.middleware

        # Verify order: swarm middleware first, then agent middleware
        assert len(agent_with_middleware.middleware) == 4
        assert agent_with_middleware.middleware[0].name == "swarm_middleware_1"
        assert agent_with_middleware.middleware[1].name == "swarm_middleware_2"
        assert agent_with_middleware.middleware[2].name == "agent_middleware_1"
        assert agent_with_middleware.middleware[3].name == "agent_middleware_2"

    def test_swarm_middleware_with_empty_agent_middleware(self) -> None:
        """Test that swarm middleware works when agent has no middleware."""
        swarm_middleware = [
            MiddlewareModel(name="swarm_middleware_1"),
        ]

        agent = AgentModel(
            name="test_agent",
            model=LLMModel(name="test-model"),
            middleware=[],
        )

        # Simulate merging
        agent_with_middleware = deepcopy(agent)
        agent_with_middleware.middleware = swarm_middleware + agent.middleware

        assert len(agent_with_middleware.middleware) == 1
        assert agent_with_middleware.middleware[0].name == "swarm_middleware_1"

    def test_empty_swarm_middleware_with_agent_middleware(self) -> None:
        """Test that agent middleware is preserved when swarm has no middleware."""
        swarm_middleware: list[MiddlewareModel] = []

        agent = AgentModel(
            name="test_agent",
            model=LLMModel(name="test-model"),
            middleware=[
                MiddlewareModel(name="agent_middleware_1"),
            ],
        )

        # When swarm middleware is empty, agent should be used as-is
        if swarm_middleware:
            agent_with_middleware = deepcopy(agent)
            agent_with_middleware.middleware = swarm_middleware + agent.middleware
        else:
            agent_with_middleware = agent

        assert len(agent_with_middleware.middleware) == 1
        assert agent_with_middleware.middleware[0].name == "agent_middleware_1"

    def test_deepcopy_prevents_original_agent_modification(self) -> None:
        """Test that merging doesn't modify the original agent."""
        swarm_middleware = [MiddlewareModel(name="swarm_middleware")]

        original_agent = AgentModel(
            name="test_agent",
            model=LLMModel(name="test-model"),
            middleware=[MiddlewareModel(name="agent_middleware")],
        )

        # Store original middleware count
        original_count = len(original_agent.middleware)

        # Simulate merging
        agent_with_middleware = deepcopy(original_agent)
        agent_with_middleware.middleware = (
            swarm_middleware + agent_with_middleware.middleware
        )

        # Original agent should be unchanged
        assert len(original_agent.middleware) == original_count
        assert len(agent_with_middleware.middleware) == 2


# =============================================================================
# Integration Tests with AppConfig
# =============================================================================


@pytest.mark.unit
class TestSwarmMiddlewareInAppConfig:
    """Tests for swarm middleware in complete AppConfig."""

    def test_app_config_with_swarm_middleware(self) -> None:
        """Test that AppConfig correctly handles swarm with middleware."""
        config_dict = {
            "app": {
                "name": "test_app",
                "registered_model": {
                    "name": "test_model",
                },
                "agents": [
                    {
                        "name": "agent1",
                        "model": {"name": "test-model"},
                    }
                ],
                "orchestration": {
                    "swarm": {
                        "default_agent": "agent1",
                        "middleware": [
                            {
                                "name": "dao_ai.middleware.test_middleware",
                                "args": {"param": "value"},
                            }
                        ],
                    }
                },
            }
        }

        config = AppConfig(**config_dict)

        assert config.app.orchestration.swarm is not None
        assert len(config.app.orchestration.swarm.middleware) == 1
        assert (
            config.app.orchestration.swarm.middleware[0].name
            == "dao_ai.middleware.test_middleware"
        )

    def test_app_config_swarm_without_middleware(self) -> None:
        """Test that swarm without middleware works (backward compatibility)."""
        config_dict = {
            "app": {
                "name": "test_app",
                "registered_model": {
                    "name": "test_model",
                },
                "agents": [
                    {
                        "name": "agent1",
                        "model": {"name": "test-model"},
                    }
                ],
                "orchestration": {
                    "swarm": {
                        "default_agent": "agent1",
                    }
                },
            }
        }

        config = AppConfig(**config_dict)

        assert config.app.orchestration.swarm is not None
        assert config.app.orchestration.swarm.middleware == []


# =============================================================================
# Swarm Graph Creation Tests
# =============================================================================


@pytest.mark.unit
@patch("dao_ai.orchestration.swarm.create_agent_node")
@patch("dao_ai.orchestration.swarm.create_store")
@patch("dao_ai.orchestration.swarm.create_checkpointer")
class TestSwarmGraphWithMiddleware:
    """Tests for swarm graph creation with middleware."""

    def test_swarm_graph_applies_middleware_to_all_agents(
        self,
        mock_checkpointer: Mock,
        mock_store: Mock,
        mock_create_agent_node: Mock,
    ) -> None:
        """Test that swarm middleware is applied to all agents."""
        mock_checkpointer.return_value = None
        mock_store.return_value = None
        mock_create_agent_node.return_value = MagicMock()

        # Create config with swarm middleware
        swarm_middleware = [
            MiddlewareModel(name="swarm_middleware"),
        ]

        agent1 = AgentModel(
            name="agent1",
            model=LLMModel(name="test-model"),
            middleware=[],
        )
        agent2 = AgentModel(
            name="agent2",
            model=LLMModel(name="test-model"),
            middleware=[MiddlewareModel(name="agent2_middleware")],
        )

        config_dict = {
            "app": {
                "name": "test_app",
                "registered_model": {
                    "name": "test_model",
                },
                "agents": [agent1, agent2],
                "orchestration": {
                    "swarm": {
                        "default_agent": "agent1",
                        "middleware": swarm_middleware,
                    }
                },
            }
        }

        config = AppConfig(**config_dict)

        # Import and call create_swarm_graph
        from dao_ai.orchestration.swarm import create_swarm_graph

        try:
            create_swarm_graph(config)
        except Exception:
            # Graph compilation might fail in test, but we can still check the calls
            pass

        # Verify create_agent_node was called for each agent
        assert mock_create_agent_node.call_count == 2

        # Check that agents were passed with merged middleware
        # First call should be agent1 with swarm middleware
        first_call_agent = mock_create_agent_node.call_args_list[0][1]["agent"]
        assert len(first_call_agent.middleware) == 1
        assert first_call_agent.middleware[0].name == "swarm_middleware"

        # Second call should be agent2 with swarm + agent middleware
        second_call_agent = mock_create_agent_node.call_args_list[1][1]["agent"]
        assert len(second_call_agent.middleware) == 2
        assert second_call_agent.middleware[0].name == "swarm_middleware"
        assert second_call_agent.middleware[1].name == "agent2_middleware"


# =============================================================================
# YAML Configuration Tests
# =============================================================================


@pytest.mark.unit
class TestSwarmMiddlewareYAMLConfig:
    """Tests for loading swarm middleware from YAML."""

    def test_load_swarm_middleware_from_yaml_dict(self) -> None:
        """Test loading swarm with middleware from YAML-like dict."""
        yaml_config = {
            "middleware": {
                "store_validation": {
                    "name": "dao_ai.middleware.create_custom_field_validation_middleware",
                    "args": {
                        "fields": [{"name": "store_num", "description": "Store number"}]
                    },
                }
            },
            "app": {
                "name": "test_app",
                "registered_model": {
                    "name": "test_model",
                },
                "agents": [
                    {
                        "name": "agent1",
                        "model": {"name": "test-model"},
                    }
                ],
                "orchestration": {
                    "swarm": {
                        "default_agent": "agent1",
                        "middleware": [
                            {
                                "name": "dao_ai.middleware.create_custom_field_validation_middleware",
                                "args": {
                                    "fields": [
                                        {
                                            "name": "store_num",
                                            "description": "Store number",
                                        }
                                    ]
                                },
                            }
                        ],
                    }
                },
            },
        }

        config = AppConfig(**yaml_config)

        assert config.app.orchestration.swarm is not None
        assert len(config.app.orchestration.swarm.middleware) == 1
        middleware = config.app.orchestration.swarm.middleware[0]
        assert (
            middleware.name
            == "dao_ai.middleware.create_custom_field_validation_middleware"
        )
        assert "fields" in middleware.args
        assert len(middleware.args["fields"]) == 1
        assert middleware.args["fields"][0]["name"] == "store_num"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
