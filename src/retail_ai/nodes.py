from typing import Any, Callable, Literal, Optional, Sequence

import mlflow
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig, RunnableSequence
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from loguru import logger
from pydantic import BaseModel, Field

from retail_ai.config import (
    AgentModel,
    AppConfig,
    FactoryFunctionModel,
    PythonFunctionModel,
    SupervisorModel,
    ToolModel,
)
from retail_ai.guardrails import reflection_guardrail, with_guardrails
from retail_ai.messages import last_human_message
from retail_ai.state import AgentConfig, AgentState
from retail_ai.tools import create_tools
from retail_ai.types import AgentCallable


def make_prompt(base_system_prompt: str) -> Callable[[dict, RunnableConfig], list]:
    logger.debug(f"make_prompt: {base_system_prompt}")

    def prompt(state: AgentState, config: AgentConfig) -> list:
        prompt_template: PromptTemplate = PromptTemplate.from_template(
            base_system_prompt
        )

        params: dict[str, Any] = {
            input_variable: "" for input_variable in prompt_template.input_variables
        }
        params |= config.get("configurable", {})

        system_prompt: str = prompt_template.format(**params)

        messages: Sequence[BaseMessage] = state["messages"]
        messages = [SystemMessage(content=system_prompt)] + messages

        return messages

    return prompt


def create_hook(
    hook: PythonFunctionModel | FactoryFunctionModel | str,
) -> Callable[..., Any]:
    if isinstance(hook, str):
        hook = PythonFunctionModel(name=hook)
    if hook:
        hook = hook.as_tool()
    return hook


def create_agent_node(
    agent: AgentModel, additional_tools: Optional[Sequence[BaseTool]] = None
) -> AgentCallable:
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
    logger.debug(f"Creating agent node for {agent.name}")

    if agent.create_agent_hook:
        agent_hook = create_hook(agent.create_agent_hook)
        return agent_hook

    llm: LanguageModelLike = agent.model.as_chat_model()

    tool_models: Sequence[ToolModel] = agent.tools
    if not additional_tools:
        additional_tools = []
    tools: Sequence[BaseTool] = create_tools(tool_models) + additional_tools

    store: BaseStore = None
    if agent.memory and agent.memory.store:
        store = agent.memory.store.as_store()
        tools += [
            create_manage_memory_tool(namespace=("memory",)),
            create_search_memory_tool(namespace=("memory",)),
        ]

    checkpointer: BaseCheckpointSaver = None
    if agent.memory and agent.memory.checkpointer:
        checkpointer = agent.memory.checkpointer.as_checkpointer()

    pre_agent_hook: Callable[..., Any] = create_hook(agent.pre_agent_hook)
    post_agent_hook: Callable[..., Any] = create_hook(agent.post_agent_hook)

    compiled_agent: CompiledStateGraph = create_react_agent(
        model=llm,
        prompt=make_prompt(agent.prompt),
        tools=tools,
        store=store,
        checkpointer=checkpointer,
        pre_model_hook=pre_agent_hook,
        post_model_hook=post_agent_hook,
    )

    for guardrail_definition in agent.guardrails:
        guardrail: CompiledStateGraph = reflection_guardrail(guardrail_definition)
        compiled_agent = with_guardrails(compiled_agent, guardrail)

    compiled_agent.name = agent.name

    return compiled_agent


def message_validation_node(config: AppConfig) -> AgentCallable:
    message_validation_hook: Callable[..., Any] = create_hook(
        config.app.message_validation_hook
    )

    @mlflow.trace()
    def message_validation(state: AgentState, config: AgentConfig) -> dict[str, Any]:
        logger.debug("Running message validation")
        response: dict[str, Any] = {"is_valid": True, "error": None}
        if message_validation_hook:
            try:
                response |= message_validation_hook(
                    state=state,
                    config=config,
                )
            except Exception as e:
                logger.error(f"Message validation failed: {e}")
                return {"is_valid": False, "validation_error": str(e)}

        return response

    return message_validation


def _supervisor_prompt(agents: Sequence[AgentModel]) -> str:
    prompt_result: str = "Analyze the user question and select ONE specific route from the allowed options:\n\n"

    for agent in agents:
        route: str = agent.name
        handoff_prompt: str = agent.handoff_prompt
        prompt_result += f"  - Route to '{route}': {handoff_prompt}\n"

    prompt_result += (
        "\n Choose exactly ONE route that BEST matches the user's primary intent."
    )

    return prompt_result


def supervisor_node(config: AppConfig) -> AgentCallable:
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
    agents: Sequence[AgentModel] = config.app.agents

    supervisor_model: SupervisorModel = config.app.orchestration.supervisor

    prompt: str = _supervisor_prompt(agents=agents)
    logger.debug(f"Supervisor prompt: {prompt}")
    allowed_routes: Sequence[str] = list(set([a.name for a in agents]))
    model: str = supervisor_model.model.name
    temperature: float = supervisor_model.model.temperature
    default_route: str | AgentModel = supervisor_model.default_agent

    if isinstance(default_route, AgentModel):
        default_route = default_route.name

    logger.debug(
        f"Creating supervisor node with model={model}, temperature={temperature}, "
        f"default_route={default_route}, allowed_routes={allowed_routes}"
    )

    @mlflow.trace()
    def supervisor(state: AgentState, config: AgentConfig) -> dict[str, str]:
        llm: LanguageModelLike = supervisor_model.model.as_chat_model()

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


def process_images_node(config: AppConfig) -> AgentCallable:
    process_image_config: AgentModel = config.agents.get("process_image", {})
    prompt: str = process_image_config.prompt

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

        llm: LanguageModelLike = process_image_config.model.as_chat_model()

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
