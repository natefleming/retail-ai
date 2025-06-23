from typing import Any, Callable, Optional, Sequence

import mlflow
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from loguru import logger
from pydantic import BaseModel, Field

from dao_ai.config import (
    AgentModel,
    AppConfig,
    FactoryFunctionModel,
    FunctionHook,
    PythonFunctionModel,
    SummarizationModel,
    ToolModel,
)
from dao_ai.guardrails import reflection_guardrail, with_guardrails
from dao_ai.messages import last_human_message
from dao_ai.state import AgentConfig, AgentState
from dao_ai.tools import create_tools
from dao_ai.types import AgentCallable


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
        logger.debug(f"Using memory store: {store}")

        namespace: tuple[str, ...] = ("memory",)
        if agent.memory.store.namespace:
            namespace = namespace + (agent.memory.store.namespace,)
        logger.debug(f"Memory store namespace: {namespace}")

        tools += [
            create_manage_memory_tool(namespace=namespace, store=store),
            create_search_memory_tool(namespace=namespace, store=store),
        ]

    checkpointer: BaseCheckpointSaver = None
    if agent.memory and agent.memory.checkpointer:
        checkpointer = agent.memory.checkpointer.as_checkpointer()
        logger.debug(f"Using memory checkpointer: {checkpointer}")
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


def summarization_node(config: AppConfig) -> AgentCallable:
    summarization_model: SummarizationModel | None = config.app.summarization

    def summarization(state: AgentState, config: AgentConfig) -> AgentState:
        logger.debug("Running summarization node")

        if not summarization_model:
            logger.info("No summarization model configured, skipping summarization")
            return

        model: LanguageModelLike = summarization_model.model.as_chat_model()

        if summarization_model.retained_message_count:
            logger.debug("Trimming messages to retain only the last few messages")
            retained_message_count: int = summarization_model.retained_message_count

            summary = state.get("summary", "")
            summary_message: str
            if summary:
                summary_message = (
                    f"This is summary of the conversation to date: {summary}\n\n"
                    "Extend the summary by taking into account the new messages above:"
                )

            else:
                summary_message = "Create a summary of the conversation above:"

            messages: Sequence[BaseMessage] = state["messages"] + [
                HumanMessage(content=summary_message)
            ]
            response: AIMessage = model.invoke(
                messages,
            )

            delete_messages: Sequence[RemoveMessage] = [
                RemoveMessage(id=m.id)
                for m in state["messages"][:-retained_message_count]
            ]

            system_message: SystemMessage = SystemMessage(
                content=f"Summary of conversation earlier: {response.content}"
            )
            logger.debug(f"Summarization response: {response.content}")

            messages = system_message + state["messages"] + delete_messages

            return {"summary": response.content, "messages": delete_messages}

    return summarization


def message_hook_node(config: AppConfig) -> AgentCallable:
    message_hooks: Sequence[Callable[..., Any]] = [
        create_hook(hook) for hook in config.app.message_hooks
    ]

    @mlflow.trace()
    def message_hook(state: AgentState, config: AgentConfig) -> dict[str, Any]:
        logger.debug("Running message validation")
        response: dict[str, Any] = {"is_valid": True, "message_error": None}

        for message_hook in message_hooks:
            message_hook: FunctionHook
            if message_hook:
                try:
                    hook_response: dict[str, Any] = message_hook(
                        state=state,
                        config=config,
                    )
                    response.update(hook_response)
                    logger.debug(f"Hook response: {hook_response}")
                    if not response.get("is_valid", True):
                        break
                except Exception as e:
                    logger.error(f"Message validation failed: {e}")
                    response_messages: Sequence[BaseMessage] = [
                        AIMessage(content=str(e))
                    ]
                    return {
                        "is_valid": False,
                        "message_error": str(e),
                        "messages": response_messages,
                    }

        return response

    return message_hook


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
