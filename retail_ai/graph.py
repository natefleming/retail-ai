from typing import Any, Callable, Sequence

from databricks_langchain import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from loguru import logger

from retail_ai.config import AgentModel, AppConfig, OrchestrationModel
from retail_ai.guardrails import reflection_guardrail, with_guardrails
from retail_ai.messages import has_image
from retail_ai.nodes import (
    create_agent_node,
    message_validation_node,
    process_images_node,
    supervisor_node,
)
from retail_ai.state import AgentConfig, AgentState
from retail_ai.tools import create_tools


def route_message_validation(state: AgentState) -> str:
    if not state["is_valid_config"]:
        return END
    if has_image(state["messages"]):
        return "process_images"
    return "supervisor"


def _create_supervisor_graph(config: AppConfig) -> CompiledStateGraph:
    logger.debug("Creating supervisor graph")
    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    workflow.add_node("message_validation", message_validation_node(config=config))
    workflow.add_node("process_images", process_images_node(config=config))
    workflow.add_node("supervisor", supervisor_node(config=config))

    agents: Sequence[AgentModel] = config.app.agents
    for agent in agents:
        workflow.add_node(agent.name, create_agent_node(agent=agent))

    workflow.add_conditional_edges(
        "message_validation",
        route_message_validation,
        {
            "supervisor": "supervisor",
            "process_images": "process_images",
            END: END,
        },
    )

    workflow.add_edge("process_images", "supervisor")

    routes: dict[str, str] = {n: n for n in [agent.name for agent in agents]}
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        routes,
    )

    workflow.set_entry_point("message_validation")

    return workflow.compile()


def _create_swarm_graph(config: AppConfig) -> CompiledStateGraph:
    logger.debug("Creating swarm graph")
    agents: list[CompiledStateGraph] = []
    for registered_agent in config.app.agents:
        handoff_tools: list[Callable[..., Any]] = []
        for handoff_to_agent in config.app.agents:
            if registered_agent.name == handoff_to_agent.name:
                continue
            handoff_tools.append(
                create_handoff_tool(
                    agent_name=handoff_to_agent.name,
                    description=handoff_to_agent.description,
                )
            )
        # Create agent directly using create_react_agent instead of create_agent_node
        agents.append(
            _create_swarm_agent(agent=registered_agent, additional_tools=handoff_tools)
        )

    default_agent: AgentModel = config.app.orchestration.swarm.default_agent
    if isinstance(default_agent, AgentModel):
        default_agent = default_agent.name

    swarm_workflow: StateGraph = create_swarm(
        agents=agents,
        default_active_agent=default_agent,
        state_schema=AgentState,
        config_schema=AgentConfig,
    )

    checkpointer = None  # InMemorySaver()
    store = None  # InMemoryStore()
    swarm_node: CompiledStateGraph = swarm_workflow.compile(
        checkpointer=checkpointer, store=store
    )

    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    workflow.add_node("message_validation", message_validation_node(config=config))
    workflow.add_node("process_images", process_images_node(config=config))
    workflow.add_node("swarm", swarm_node)

    workflow.add_conditional_edges(
        "message_validation",
        route_message_validation,
        {
            "swarm": "swarm",
            "process_images": "process_images",
            END: END,
        },
    )

    workflow.add_edge("process_images", "swarm")

    workflow.set_entry_point("message_validation")

    return workflow.compile()


def _create_swarm_agent(
    agent: AgentModel, additional_tools: Sequence[BaseTool] = None
) -> CompiledStateGraph:
    """
    Create a compiled agent for swarm orchestration.

    Args:
        agent: Agent configuration
        additional_tools: Additional tools to include (handoff tools)

    Returns:
        CompiledStateGraph ready for swarm orchestration
    """
    logger.debug(f"Creating swarm agent for {agent.name}")

    # Get agent tools and combine with additional tools
    tools: Sequence[BaseTool] = create_tools(agent.tools)
    if additional_tools:
        tools = list(tools) + list(additional_tools)

    # Initialize model
    llm: LanguageModelLike = ChatDatabricks(
        model=agent.model.name, temperature=agent.model.temperature
    )

    # Format system prompt - for swarm, we use a simpler approach without state-based formatting
    prompt_template: PromptTemplate = PromptTemplate.from_template(agent.prompt)
    system_prompt: str = prompt_template.format(
        user_id="{user_id}",  # These will be filled at runtime
        store_num="{store_num}",
    )

    # Create the react agent
    compiled_agent: CompiledStateGraph = create_react_agent(
        model=llm,
        prompt=system_prompt,
        tools=tools,
        name=agent.name,  # Set the agent name
    )

    # Apply guardrails if specified
    for guardrail_definition in agent.guardrails:
        guardrail: CompiledStateGraph = reflection_guardrail(guardrail_definition)
        compiled_agent = with_guardrails(compiled_agent, guardrail)

    return compiled_agent


def create_retail_ai_graph(config: AppConfig) -> CompiledStateGraph:
    orchestration: OrchestrationModel = config.app.orchestration
    if orchestration.supervisor:
        return _create_supervisor_graph(config)

    if orchestration.swarm:
        return _create_swarm_graph(config)

    raise ValueError("No valid orchestration model found in the configuration.")
