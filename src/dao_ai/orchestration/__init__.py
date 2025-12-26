"""
Orchestration patterns for DAO AI multi-agent systems.

This package provides factory functions for creating LangGraph workflows
that orchestrate multiple agents using the supervisor and swarm patterns.

Supervisor Pattern:
    A central supervisor coordinates specialized worker agents. The supervisor
    hands off control to agents who then control the conversation. Agents can
    hand back to the supervisor when done or hand off to other agents.

Swarm Pattern:
    Agents can directly transfer control to each other using handoff tools.
    The active agent changes, and the user may continue interacting with
    the new agent. This provides decentralized, peer-to-peer collaboration.

Both patterns use Command(goto=...) for routing between agent nodes in
the workflow graph.

See: https://docs.langchain.com/oss/python/langchain/multi-agent
See: https://github.com/langchain-ai/langgraph-supervisor-py
See: https://github.com/langchain-ai/langgraph-swarm-py
"""

from dao_ai.orchestration.core import (
    SUPERVISOR_NODE,
    OutputMode,
    create_agent_node_handler,
    create_checkpointer,
    create_handoff_tool,
    create_orchestration_graph,
    create_store,
    extract_agent_response,
    filter_messages_for_agent,
    get_handoff_description,
)

__all__ = [
    # Constants
    "SUPERVISOR_NODE",
    "OutputMode",
    # Core utilities
    "create_store",
    "create_checkpointer",
    "filter_messages_for_agent",
    "extract_agent_response",
    "create_agent_node_handler",
    "create_handoff_tool",
    "get_handoff_description",
    # Main factory
    "create_orchestration_graph",
]
