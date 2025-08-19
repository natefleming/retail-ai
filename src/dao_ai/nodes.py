from typing import Any, Callable, Optional, Sequence

import mlflow
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime
from langmem import create_manage_memory_tool, create_search_memory_tool
from langmem.short_term import SummarizationNode
from langmem.short_term.summarization import TokenCounter
from loguru import logger

from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    ChatHistoryModel,
    FunctionHook,
    ToolModel,
)
from dao_ai.guardrails import reflection_guardrail, with_guardrails
from dao_ai.hooks.core import create_hooks
from dao_ai.prompts import make_prompt
from dao_ai.state import Context, IncomingState, SharedState
from dao_ai.tools import create_tools


def summarization_node(app_model: AppModel) -> RunnableLike:
    chat_history: ChatHistoryModel | None = app_model.chat_history
    if chat_history is None:
        raise ValueError(
            "AppModel must have chat_history configured to use summarization"
        )

    max_tokens: int = chat_history.max_tokens
    max_tokens_before_summary: int | None = chat_history.max_tokens_before_summary
    max_messages_before_summary: int | None = chat_history.max_messages_before_summary
    max_summary_tokens: int | None = chat_history.max_summary_tokens
    token_counter: TokenCounter = (
        count_tokens_approximately if max_tokens_before_summary else len
    )

    logger.debug(
        f"Creating summarization node with max_tokens: {max_tokens}, "
        f"max_tokens_before_summary: {max_tokens_before_summary}, "
        f"max_messages_before_summary: {max_messages_before_summary}, "
        f"max_summary_tokens: {max_summary_tokens}"
    )

    summarization_model: LanguageModelLike = chat_history.model.as_chat_model()

    node: RunnableLike = SummarizationNode(
        model=summarization_model,
        max_tokens=max_tokens,
        max_tokens_before_summary=max_tokens_before_summary
        or max_messages_before_summary,
        max_summary_tokens=max_summary_tokens,
        token_counter=token_counter,
        input_messages_key="messages",
        output_messages_key="summarized_messages",
    )
    return node


def call_agent_with_summarized_messages(agent: CompiledStateGraph) -> RunnableLike:
    def call_agent(state: SharedState, runtime: Runtime[Context]) -> SharedState:
        logger.debug(f"Calling agent {agent.name} with summarized messages")

        # Get the summarized messages from the summarization node
        messages: Sequence[AnyMessage] = state.get("summarized_messages", [])
        logger.debug(f"Found {len(messages)} summarized messages")
        logger.trace(f"Summarized messages: {[m.model_dump() for m in messages]}")

        input: dict[str, Any] = {
            "messages": messages,
        }

        response: dict[str, Any] = agent.invoke(input=input, context=runtime.context)
        response_messages = response.get("messages", [])
        logger.debug(f"Agent returned {len(response_messages)} messages")

        return {"messages": response_messages}

    return call_agent


def create_agent_node(
    app: AppModel,
    agent: AgentModel,
    additional_tools: Optional[Sequence[BaseTool]] = None,
) -> RunnableLike:
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
        agent_hook = next(iter(create_hooks(agent.create_agent_hook)), None)
        return agent_hook

    llm: LanguageModelLike = agent.model.as_chat_model()

    tool_models: Sequence[ToolModel] = agent.tools
    if not additional_tools:
        additional_tools = []
    tools: Sequence[BaseTool] = create_tools(tool_models) + additional_tools

    if app.orchestration.memory and app.orchestration.memory.store:
        namespace: tuple[str, ...] = ("memory",)
        if app.orchestration.memory.store.namespace:
            namespace = namespace + (app.orchestration.memory.store.namespace,)
        logger.debug(f"Memory store namespace: {namespace}")

        tools += [
            create_manage_memory_tool(namespace=namespace),
            create_search_memory_tool(namespace=namespace),
        ]

    pre_agent_hook: Callable[..., Any] = next(
        iter(create_hooks(agent.pre_agent_hook)), None
    )
    logger.debug(f"pre_agent_hook: {pre_agent_hook}")

    post_agent_hook: Callable[..., Any] = next(
        iter(create_hooks(agent.post_agent_hook)), None
    )
    logger.debug(f"post_agent_hook: {post_agent_hook}")

    compiled_agent: CompiledStateGraph = create_react_agent(
        name=agent.name,
        model=llm,
        prompt=make_prompt(agent.prompt),
        tools=tools,
        store=True,
        checkpointer=True,
        state_schema=SharedState,
        context_schema=Context,
        pre_model_hook=pre_agent_hook,
        post_model_hook=post_agent_hook,
    )

    for guardrail_definition in agent.guardrails:
        guardrail: CompiledStateGraph = reflection_guardrail(guardrail_definition)
        compiled_agent = with_guardrails(compiled_agent, guardrail)

    compiled_agent.name = agent.name

    agent_node: CompiledStateGraph

    chat_history: ChatHistoryModel = app.chat_history

    if chat_history is None:
        logger.debug("No chat history configured, using compiled agent directly")
        agent_node = compiled_agent
    else:
        logger.debug("Creating agent node with chat history summarization")
        workflow: StateGraph = StateGraph(
            SharedState,
            config_schema=RunnableConfig,
            input=SharedState,
            output=SharedState,
        )
        workflow.add_node("summarization", summarization_node(app))
        workflow.add_node(
            "agent",
            call_agent_with_summarized_messages(agent=compiled_agent),
        )
        workflow.add_edge("summarization", "agent")
        workflow.set_entry_point("summarization")
        agent_node = workflow.compile(name=agent.name)

    return agent_node


def message_hook_node(config: AppConfig) -> RunnableLike:
    message_hooks: Sequence[Callable[..., Any]] = create_hooks(config.app.message_hooks)

    @mlflow.trace()
    def message_hook(state: IncomingState, runtime: Runtime[Context]) -> SharedState:
        logger.debug("Running message validation")
        response: dict[str, Any] = {"is_valid": True, "message_error": None}

        for message_hook in message_hooks:
            message_hook: FunctionHook
            if message_hook:
                try:
                    hook_response: dict[str, Any] = message_hook(
                        state=state,
                        runtime=runtime,
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
