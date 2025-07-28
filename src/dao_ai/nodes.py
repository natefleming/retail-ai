import json
from typing import Any, Callable, Optional, Sequence

import mlflow
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    trim_messages,
)
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, messages_from_dict
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore, Item
from langmem import create_manage_memory_tool, create_search_memory_tool
from loguru import logger
from pydantic import BaseModel, Field

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
from dao_ai.messages import last_human_message
from dao_ai.prompts import make_prompt
from dao_ai.state import IncomingState, SharedState
from dao_ai.tools import create_tools


def _serialize_messages(
    messages: Sequence[BaseMessage],
) -> dict[str, Sequence[dict[str, Any]]]:
    """Convert LangChain messages to JSON-serializable dictionaries."""
    logger.debug(f"Serializing {len(messages)} messages to JSON format")
    serialized_messages: dict[str, Sequence[dict[str, Any]]] = {
        "messages": messages_to_dict(messages)
    }

    logger.trace(f"Serialized {len(serialized_messages)} messages to JSON format")
    logger.trace(f"Serialized messages: {serialized_messages}")

    return serialized_messages


def _deserialize_messages(
    serialized_messages: dict[str, Sequence[dict[str, Any]]],
) -> Sequence[BaseMessage]:
    """Convert JSON-serializable dictionaries back to LangChain messages."""
    serialized_messages = serialized_messages.get("messages", [])
    logger.debug(f"Deserializing {len(serialized_messages)} messages from JSON format")
    messages = messages_from_dict(serialized_messages)

    logger.trace(f"Deserialized {len(messages)} messages from JSON format")
    logger.trace(f"Deserialized messages: {[m.model_dump() for m in messages]}")

    return messages


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
        state_schema=SharedState,
        config_schema=RunnableConfig,
        checkpointer=True,
        pre_model_hook=pre_agent_hook,
        post_model_hook=post_agent_hook,
    )

    for guardrail_definition in agent.guardrails:
        guardrail: CompiledStateGraph = reflection_guardrail(guardrail_definition)
        compiled_agent = with_guardrails(compiled_agent, guardrail)

    compiled_agent.name = agent.name

    return compiled_agent


def load_conversation_history_node(
    store: BaseStore, app_config: AppConfig
) -> RunnableLike:
    """
    Create a node that loads the conversation history from the database.

    This node retrieves the conversation history for a given thread ID and returns it
    as a sequence of messages. Only loads history if enable_conversation_history is True.
    """

    @mlflow.trace()
    def load_conversation_history(
        state: IncomingState, config: RunnableConfig
    ) -> SharedState:
        logger.debug("Running load_conversation node")
        conversation_history_messages: Sequence[BaseMessage] = []

        configurable: dict[str, Any] = config.get("configurable", {})

        if store and "thread_id" in configurable and "user_id" in configurable:
            thread_id: str = configurable.get("thread_id")
            app_name: str = app_config.app.name or "default"
            user_id: str = configurable.get("user_id")
            logger.debug(
                f"Using store: {store} to load thread ID: {thread_id} for user ID: {app_name}/{user_id}"
            )

            namespace: tuple[str, ...] = ("conversations", app_name, user_id)
            conversation_history: Item | None = store.get(namespace, thread_id)

            if conversation_history:
                logger.debug(f"Loaded conversation history: {conversation_history}")
                # Deserialize the stored messages from JSON back to LangChain message objects
                serialized_messages: dict[str, Any] = conversation_history.value
                conversation_history_messages = _deserialize_messages(
                    serialized_messages
                )
                logger.debug(
                    f"Deserialized {len(conversation_history_messages)} messages from store"
                )

            else:
                logger.debug(
                    f"No conversation history found for thread ID: {thread_id}"
                )
        else:
            logger.debug(
                "No store available or missing thread_id/user_id, starting with empty conversation"
            )

        return {"conversation_history": conversation_history_messages}

    return load_conversation_history


def store_conversation_history_node(
    store: BaseStore, app_config: AppConfig
) -> RunnableLike:
    """
    Create a node that saves the conversation history to the database.

    This node saves the current conversation messages for a given thread ID and user ID
    to the store for future retrieval. Only stores history if enable_conversation_history is True.
    """

    @mlflow.trace()
    def store_conversation_history(
        state: SharedState, config: RunnableConfig
    ) -> SharedState:
        logger.debug("Running store_conversation node")

        configurable: dict[str, Any] = config.get("configurable", {})

        if store and "thread_id" in configurable and "user_id" in configurable:
            thread_id: str = configurable.get("thread_id")
            app_name: str = app_config.app.name or "default"
            user_id: str = configurable.get("user_id")
            messages: Sequence[BaseMessage] = state.get("messages", [])

            logger.debug(
                f"Saving conversation for thread ID: {thread_id} for user ID: {app_name}/{user_id}"
            )

            namespace: tuple[str, ...] = ("conversations", app_name, user_id)

            # Serialize the messages to JSON-safe format before storing

            serialized_messages: dict[str, Sequence[dict[str, Any]]] = (
                _serialize_messages(messages)
            )
            store.put(namespace, thread_id, serialized_messages)
            logger.debug(f"Saved {len(messages)} serialized messages to store")
        else:
            logger.debug(
                "No store available or missing thread_id/user_id, cannot save conversation"
            )

        return state

    return store_conversation_history


def summarization_node(config: AppConfig) -> RunnableLike:
    chat_history: ChatHistoryModel | None = config.app.chat_history

    def _create_summary(
        model: LanguageModelLike,
        messages_to_summarize: Sequence[BaseMessage],
        existing_summary: str,
    ) -> str:
        summary_message: str
        if existing_summary:
            summary_message = (
                f"This is a summary of the conversation so far: {existing_summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages: Sequence[BaseMessage] = messages_to_summarize + [
            HumanMessage(content=summary_message)
        ]
        response: AIMessage = model.invoke(input=messages)
        return response.content

    def _update_messages_with_summary(
        model: LanguageModelLike,
        state: SharedState,
        messages_to_summarize: Sequence[BaseMessage],
    ) -> dict[str, Any]:
        """Helper function to create summary and update messages."""
        existing_summary: str = state.get("summary", "")

        logger.trace(
            f"Creating summary for {len(messages_to_summarize)} messages with existing summary: {existing_summary}"
        )

        new_summary: str = _create_summary(
            model, messages_to_summarize, existing_summary
        )

        deleted_messages: Sequence[RemoveMessage] = [
            RemoveMessage(id=m.id) for m in messages_to_summarize
        ]

        logger.debug(
            f"Summarized {len(messages_to_summarize)} messages, created new summary"
        )

        logger.trace(
            f"New summary: {new_summary}\n"
            f"Deleted messages: {[m.id for m in messages_to_summarize]}"
        )

        return {
            "messages": deleted_messages,
            "summary": new_summary,
        }

    def summarization(state: SharedState, config: RunnableConfig) -> SharedState:
        logger.debug("=== CONVERSATION HISTORY MANAGEMENT START ===")

        if not chat_history:
            logger.debug(
                "No chat history configuration found - skipping conversation management"
            )
            return

        new_messages: Sequence[BaseMessage] = state.get("messages", [])
        conversation_history: Sequence[BaseMessage] = state.get("conversation_history", [])
        
        # Combine conversation history with new messages for processing
        all_messages: Sequence[BaseMessage] = list(conversation_history) + list(new_messages)

        logger.info(f"Processing conversation with {len(conversation_history)} history + {len(new_messages)} new = {len(all_messages)} total messages")

        model: LanguageModelLike = chat_history.model.as_chat_model()

        # Log configuration
        logger.debug(
            f"Configuration - summarize: {chat_history.summarize}, "
            f"max_message_count: {chat_history.max_message_count}, "
            f"retained_message_count: {chat_history.retained_message_count}, "
            f"max_tokens: {chat_history.max_tokens}"
        )

        # Check if summarization should occur based on max_message_count
        should_summarize: bool = False
        if chat_history.max_message_count is not None:
            current_message_count = len(all_messages)
            should_summarize = current_message_count > chat_history.max_message_count
            logger.debug(
                f"Summarization trigger check: {current_message_count} messages > "
                f"{chat_history.max_message_count} max_message_count = {should_summarize}"
            )
        else:
            logger.debug(
                "No max_message_count set - summarization trigger will depend on trimming results"
            )

        # Determine trimming parameters
        max_tokens: int
        token_counter: Callable[..., int]

        if chat_history.retained_message_count is not None:
            # For message count-based trimming, ensure we keep at least 1 message
            max_tokens = max(1, chat_history.retained_message_count)
            token_counter = len
            strategy_type = "message count"
            logger.debug(
                f"Using MESSAGE-COUNT trimming strategy: will keep last {max_tokens} messages"
            )
        else:
            max_tokens = chat_history.max_tokens
            token_counter = count_tokens_approximately
            strategy_type = "token count"
            logger.debug(
                f"Using TOKEN-COUNT trimming strategy: will keep ~{max_tokens} tokens worth of messages"
            )

        logger.trace(
            f"Original messages:\n{json.dumps([msg.model_dump() for msg in all_messages], indent=2)}"
        )

        # Always trim messages based on retained_message_count or max_tokens
        logger.debug(
            "Applying trimming strategy to determine which messages to keep..."
        )
        trimmed_messages: Sequence[BaseMessage] = trim_messages(
            all_messages,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=token_counter,
            allow_partial=False,
            include_system=True,
            start_on="human",
            #   end_on=("human", "tool"),
        )
        logger.trace(
            f"Messages after trimming:\n{json.dumps([msg.model_dump() for msg in trimmed_messages], indent=2)}"
        )

        messages_to_remove = len(all_messages) - len(trimmed_messages)
        logger.info(
            f"Trimming result: keeping {len(trimmed_messages)} messages, removing {messages_to_remove} messages"
        )

        # Handle case where trim_messages returns empty list - summarize all messages if summarization is enabled
        if len(trimmed_messages) == 0 and len(all_messages) > 0:
            logger.warning("EDGE CASE: Trimming strategy returned empty message list")
            if should_summarize and chat_history.summarize:
                logger.info(
                    "→ Creating summary of ALL messages due to empty trim result"
                )
                return _update_messages_with_summary(model, state, all_messages)
            else:
                logger.info(
                    "→ No action taken - summarization disabled or not triggered"
                )
                return None

        # Check if we need to remove messages (and potentially summarize them)
        if len(trimmed_messages) < len(all_messages):
            # Find messages that were removed by trimming
            trimmed_message_ids: set[str] = {m.id for m in trimmed_messages}
            messages_to_remove: Sequence[BaseMessage] = [
                m for m in all_messages if m.id not in trimmed_message_ids
            ]

            logger.info(
                f"Messages exceed {strategy_type} limit ({max_tokens}): "
                f"will remove {len(messages_to_remove)} oldest messages"
            )

            # Determine if we should create summaries for removed messages
            will_summarize = chat_history.summarize and (
                should_summarize or chat_history.max_message_count is None
            )

            if will_summarize:
                logger.info(
                    "SUMMARIZING removed messages (summarize=True and conditions met)"
                )
                logger.debug(
                    f"Summarization conditions: summarize={chat_history.summarize}, "
                    f"should_summarize={should_summarize}, "
                    f"max_message_count={chat_history.max_message_count}"
                )
                return _update_messages_with_summary(model, state, messages_to_remove)
            else:
                logger.info("→ REMOVING messages without summarization")
                logger.debug(
                    f"No summarization because: summarize={chat_history.summarize}, "
                    f"should_summarize={should_summarize}, "
                    f"max_message_count={chat_history.max_message_count}"
                )
                # Just remove the messages without summarizing
                deleted_messages: Sequence[RemoveMessage] = [
                    RemoveMessage(id=m.id) for m in messages_to_remove
                ]
                return {"messages": deleted_messages}
        else:
            logger.info(
                f"All {len(all_messages)} messages fit within {strategy_type} limit ({max_tokens}) - no action needed"
            )

        logger.debug("=== CONVERSATION HISTORY MANAGEMENT COMPLETE ===")
        return None

    return summarization


def message_hook_node(config: AppConfig) -> RunnableLike:
    message_hooks: Sequence[Callable[..., Any]] = create_hooks(config.app.message_hooks)

    @mlflow.trace()
    def message_hook(state: IncomingState, config: RunnableConfig) -> SharedState:
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


def process_images_node(config: AppConfig) -> RunnableLike:
    process_image_config: AgentModel = config.agents.get("process_image", {})
    prompt: str = process_image_config.prompt

    @mlflow.trace()
    def process_images(
        state: SharedState, config: RunnableConfig
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
