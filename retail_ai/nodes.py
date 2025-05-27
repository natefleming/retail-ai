from typing import Any, Literal, Optional, Sequence

import mlflow
from databricks_langchain import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableSequence
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger
from mlflow.models import ModelConfig
from pydantic import BaseModel, Field

from retail_ai.guardrails import reflection_guardrail, with_guardrails
from retail_ai.messages import last_human_message
from retail_ai.state import AgentConfig, AgentState
from retail_ai.supervisor import Supervisor
from retail_ai.tools import (
    create_tools,
)
from retail_ai.types import AgentCallable


def create_agent_node(name: str, model_config: ModelConfig) -> AgentCallable:
    """
    Factory function that creates a LangGraph node for a specialized agent.

    This creates a node function that handles user requests using a specialized agent
    based on the provided agent_type. The function configures the agent with the
    appropriate model, prompt, tools, and guardrails from the model_config.

    Args:
        model_config: Configuration containing models, prompts, tools, and guardrails
        agent_type: Type of agent to create (e.g., "general", "product", "inventory")

    Returns:
        An agent callable function that processes state and returns responses
    """
    logger.debug(f"Creating agent node for {name}")
    try:
        agent_config = model_config.get("agents").get(name)
        if not agent_config:
            raise ValueError(f"No configuration found for agent name: {name}")

        model: str = agent_config.get("model").get("name")
        temperature: float = agent_config.get("model").get("temperature", 0.1)
        prompt: str = agent_config.get("prompt")
        guardrails: Sequence[dict[str, Any]] = agent_config.get("guardrails", [])
        tool_definitions: Sequence[dict[str, Any]] = agent_config.get("tools", [])
    except (AttributeError, KeyError) as e:
        logger.error(f"Error extracting configuration for {name}: {e}")
        raise ValueError(f"Invalid configuration for agent name: {name}")

    @mlflow.trace()
    def agent_node(
        state: AgentState, config: AgentConfig
    ) -> dict[str, BaseMessage] | CompiledStateGraph:
        """
        Process user messages using a specialized agent.

        Args:
            state: Current state containing messages and context
            config: Configuration parameters

        Returns:
            Either a dict with response messages or a CompiledStateGraph for further processing
        """
        logger.debug(f"Executing {name} agent node")

        # Initialize model with appropriate temperature
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=temperature)

        # Format system prompt with user context
        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = {
            "user_id": state.get("user_id", ""),
            "store_num": state.get("store_num", ""),
            # Add any additional context needed from state
        }
        system_prompt: str = prompt_template.format(**configurable)

        # Create tools for this agent
        tools: Sequence[BaseTool] = create_tools(tool_definitions)

        # Create the agent with ReAct framework
        agent: CompiledStateGraph = create_react_agent(
            model=llm,
            prompt=system_prompt,
            tools=tools,
        )

        # Apply guardrails if specified
        for guardrail_definition in guardrails:
            guardrail: CompiledStateGraph = reflection_guardrail(guardrail_definition)
            agent = with_guardrails(agent, guardrail)

        # Return the agent or its response
        return agent

    # Set function name dynamically for better debugging
    agent_node.__name__ = f"{name}_node_impl"

    return agent_node


def message_validation_node(model_config: ModelConfig) -> AgentCallable:
    @mlflow.trace()
    def message_validation(state: AgentState, config: AgentConfig) -> dict[str, Any]:
        logger.debug(f"state: {state}")

        configurable: dict[str, Any] = config.get("configurable", {})
        validation_errors: list[str] = []

        user_id: Optional[str] = configurable.get("user_id", "")
        if not user_id:
            validation_errors.append("user_id is required")

        store_num: Optional[str] = configurable.get("store_num", "")
        if not store_num:
            validation_errors.append("store_num is required")

        if validation_errors:
            logger.warning(f"Validation errors: {validation_errors}")

        return {"user_id": user_id, "store_num": store_num, "is_valid_config": True}

    return message_validation


def supervisor_node(model_config: ModelConfig) -> AgentCallable:
    """
    Create a node that routes questions to the appropriate specialized agent.

    This factory function returns a callable that uses a language model to analyze
    the latest user message and determine which agent should handle it based on content.
    The routing decision is structured through the Router Pydantic model.

    Args:
        model_name: Name of the language model to use for routing decisions

    Returns:
        An agent callable function that updates the state with the routing decision
    """
    logger.debug("Creating supervisor node")
    agents_config: dict[str, Any] = model_config.get("app").get("agents", [])
    supervisor_config: dict[str, Any] = (
        model_config.get("app").get("orchestration", {}).get("supervisor", {})
    )

    supervisor: Supervisor = Supervisor(agents=agents_config)

    prompt: str = supervisor.prompt
    allowed_routes: Sequence[str] = supervisor.allowed_routes
    model: str = supervisor_config.get("model").get("name")
    temperature: float = supervisor_config.get("temperature", 0.1)
    default_route: str | dict[str, Any] = supervisor_config.get("default_agent", None)
    if isinstance(default_route, dict):
        default_route = default_route.get("name", None)

    logger.debug(
        f"Creating supervisor node with model={model}, temperature={temperature}, "
        f"default_route={default_route}, allowed_routes={allowed_routes}"
    )

    @mlflow.trace()
    def supervisor(state: AgentState, config: AgentConfig) -> dict[str, str]:
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=temperature)

        class Router(BaseModel):
            route: Literal[tuple(allowed_routes)] = Field(
                default=default_route,
                description=f"The route to take. Must be one of {allowed_routes}",
            )

        Router.__doc__ = prompt

        chain: RunnableSequence = llm.with_structured_output(Router)

        # Extract all messages from the current state
        messages: Sequence[BaseMessage] = state["messages"]

        # Get the most recent message from the human user
        last_message: BaseMessage = last_human_message(messages)

        # Invoke the chain to determine the appropriate route
        response = chain.invoke([last_message])

        # Return the route decision to update the agent state
        return {"route": response.route}

    return supervisor


def process_images_node(model_config: ModelConfig) -> AgentCallable:
    process_image_config: dict[str, Any] = model_config.get("agents").get(
        "process_image", {}
    )
    model: str = process_image_config.get("model").get("name")
    temperature: float = process_image_config.get("model").get("temperature", 0.1)
    prompt: str = process_image_config.get("prompt")

    @mlflow.trace()
    def process_images(
        state: AgentState, config: AgentConfig
    ) -> dict[str, BaseMessage]:
        logger.debug("process_images")

        class ImageDetails(BaseModel):
            summary: str = Field(..., description="The summary of the image")
            product_names: Optional[Sequence[str]] = Field(
                ..., description="The name of the product", default_factory=list
            )
            upcs: Optional[Sequence[str]] = Field(
                ..., description="The UPC of the image", default_factory=list
            )

        class ImageProcessor(BaseModel):
            prompts: Sequence[str] = Field(
                ...,
                description="The prompts to use to process the image",
                default_factory=list,
            )
            image_details: Sequence[ImageDetails] = Field(
                ..., description="The details of the image", default_factory=list
            )

        ImageProcessor.__doc__ = prompt

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=temperature)

        last_message: HumanMessage = last_human_message(state["messages"])
        messages: Sequence[BaseMessage] = [last_message]

        llm_with_schema: LanguageModelLike = llm.with_structured_output(ImageProcessor)

        image_processor: ImageProcessor = llm_with_schema.invoke(input=messages)

        logger.debug(f"image_processor: {image_processor}")

        response_messages: Sequence[BaseMessage] = [
            RemoveMessage(last_message.id),
            HumanMessage(content=image_processor.model_dump_json()),
        ]

        return {"messages": response_messages}

    return process_images
