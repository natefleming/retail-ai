"""
Graph creation utilities for DAO AI multi-agent orchestration.

This module provides backwards-compatible imports from the refactored
orchestration package. New code should import directly from:
    - dao_ai.orchestration
    - dao_ai.orchestration.supervisor
    - dao_ai.orchestration.swarm

See: https://docs.langchain.com/oss/python/langchain/multi-agent
"""

from langgraph.graph.state import CompiledStateGraph

from dao_ai.config import AppConfig
from dao_ai.orchestration import create_orchestration_graph


def create_dao_ai_graph(config: AppConfig) -> CompiledStateGraph:
    """
    Create the main DAO AI graph based on the orchestration configuration.

    This factory function creates either a supervisor or swarm graph
    depending on the configuration.

    Args:
        config: The application configuration

    Returns:
        A compiled LangGraph state machine

    Note:
        This function is provided for backwards compatibility.
        New code should use `create_orchestration_graph` from
        `dao_ai.orchestration` instead.
    """
    return create_orchestration_graph(config)
