from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from loguru import logger


def create_app_info_tool(
    app_name: str,
    description: str,
    agents: list[dict[str, Any]],
    sample_prompts: list[str] | None = None,
) -> BaseTool:
    """Create a tool that returns the application architecture and sample prompts.

    Designed for use as a supervisor factory tool. Agent dicts are typically
    resolved from YAML anchors and may contain full AgentModel fields; only
    ``name``, ``description``, and ``handoff_prompt`` are used.

    Args:
        app_name: Display name of the application.
        description: High-level description of the application.
        agents: List of agent config dicts (from YAML anchor resolution).
        sample_prompts: Optional list of example prompts users can try.

    Returns:
        A LangChain tool that returns a markdown-formatted architecture summary.
    """

    agent_entries: list[dict[str, str]] = []
    for agent in agents:
        entry: dict[str, str] = {
            "name": agent.get("name", "unknown"),
            "description": agent.get("description", ""),
        }
        handoff: str | None = agent.get("handoff_prompt")
        if handoff:
            entry["handoff_prompt"] = handoff.strip()
        agent_entries.append(entry)

    logger.trace(
        "Creating app info tool",
        app_name=app_name,
        agent_count=len(agent_entries),
    )

    @create_tool
    def app_info() -> str:
        """Get information about this application's architecture, available agents, and sample prompts.

        Use this tool when the user asks what this application can do, what agents
        are available, or needs example prompts to get started.

        Returns:
            A markdown-formatted summary of the application architecture.
        """
        lines: list[str] = [
            f"# {app_name}",
            "",
            description.strip(),
            "",
            "## Available Agents",
            "",
        ]

        for entry in agent_entries:
            lines.append(f"### {entry['name']}")
            if entry["description"]:
                lines.append(f"  {entry['description']}")
            if "handoff_prompt" in entry:
                lines.append(
                    f"  **Routes to this agent when:** {entry['handoff_prompt']}"
                )
            lines.append("")

        if sample_prompts:
            lines.append("## Sample Prompts")
            lines.append("")
            for prompt in sample_prompts:
                lines.append(f"- {prompt}")
            lines.append("")

        return "\n".join(lines)

    return app_info
