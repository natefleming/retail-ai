import pytest
from langchain_core.tools import BaseTool

from dao_ai.tools.app_info import create_app_info_tool


@pytest.mark.unit
def test_create_app_info_tool_basic() -> None:
    """Factory returns a callable BaseTool with expected name."""
    tool: BaseTool = create_app_info_tool(
        app_name="Test App",
        description="A test application.",
        agents=[
            {"name": "agent_a", "description": "Handles A tasks"},
        ],
    )
    assert isinstance(tool, BaseTool)
    assert tool.name == "app_info"


@pytest.mark.unit
def test_app_info_tool_includes_app_name() -> None:
    """Output contains the application name."""
    tool: BaseTool = create_app_info_tool(
        app_name="My App",
        description="Does useful things.",
        agents=[],
    )
    result: str = tool.invoke({})
    assert "# My App" in result
    assert "Does useful things." in result


@pytest.mark.unit
def test_app_info_tool_includes_agents() -> None:
    """Output lists agent names, descriptions, and handoff prompts."""
    tool: BaseTool = create_app_info_tool(
        app_name="Multi-Agent App",
        description="Orchestrates agents.",
        agents=[
            {
                "name": "planner",
                "description": "Plans things",
                "handoff_prompt": "Questions about planning",
            },
            {
                "name": "executor",
                "description": "Executes plans",
            },
        ],
    )
    result: str = tool.invoke({})
    assert "### planner" in result
    assert "Plans things" in result
    assert "Routes to this agent when:" in result
    assert "Questions about planning" in result
    assert "### executor" in result
    assert "Executes plans" in result


@pytest.mark.unit
def test_app_info_tool_includes_sample_prompts() -> None:
    """Output includes the sample prompts section when provided."""
    prompts: list[str] = ["What is X?", "Show me Y"]
    tool: BaseTool = create_app_info_tool(
        app_name="App",
        description="Desc",
        agents=[],
        sample_prompts=prompts,
    )
    result: str = tool.invoke({})
    assert "## Sample Prompts" in result
    assert "- What is X?" in result
    assert "- Show me Y" in result


@pytest.mark.unit
def test_app_info_tool_no_sample_prompts() -> None:
    """Output omits sample prompts section when none provided."""
    tool: BaseTool = create_app_info_tool(
        app_name="App",
        description="Desc",
        agents=[],
    )
    result: str = tool.invoke({})
    assert "Sample Prompts" not in result


@pytest.mark.unit
def test_app_info_tool_handles_full_agent_dict() -> None:
    """Factory gracefully handles full AgentModel-like dicts (extra keys ignored)."""
    agent_dict: dict = {
        "name": "forecasting",
        "description": "Demand forecasting",
        "model": {"provider": "databricks", "name": "gpt-4o"},
        "tools": [{"name": "some_tool"}],
        "handoff_prompt": "Forecast-related questions",
        "guardrails": [],
    }
    tool: BaseTool = create_app_info_tool(
        app_name="Store",
        description="Retail app",
        agents=[agent_dict],
    )
    result: str = tool.invoke({})
    assert "### forecasting" in result
    assert "Demand forecasting" in result
    assert "Forecast-related questions" in result
    assert "model" not in result.lower().split("forecasting")[0]
