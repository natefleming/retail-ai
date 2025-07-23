import json
from typing import Any, Callable, Optional, Sequence

import mlflow
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, messages_from_dict
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore, Item
from langmem import create_manage_memory_tool, create_search_memory_tool
from loguru import logger
from pydantic import BaseModel, Field

from dao_ai.config import (
    AgentModel,
    AppConfig,
    FunctionHook,
    SummarizationModel,
    ToolModel,
)
from dao_ai.guardrails import reflection_guardrail, with_guardrails
from dao_ai.hooks.core import create_hooks
from dao_ai.messages import last_human_message
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


def make_prompt(base_system_prompt: str) -> Callable[[dict, RunnableConfig], list]:
    logger.debug(f"make_prompt: {base_system_prompt}")

    def prompt(state: SharedState, config: RunnableConfig) -> list:
        system_prompt: str = ""
        if base_system_prompt:
            prompt_template: PromptTemplate = PromptTemplate.from_template(
                base_system_prompt
            )

            params: dict[str, Any] = {
                input_variable: "" for input_variable in prompt_template.input_variables
            }
            params |= config.get("configurable", {})

            system_prompt: str = prompt_template.format(**params)

        summary: str = state.get("summary", "")
        if summary:
            system_prompt += (
                f"\n\n## Previous Conversation Summary\n\n{summary}\n\n---\n"
            )

        messages: Sequence[BaseMessage] = state["messages"]
        if system_prompt:
            messages = [SystemMessage(content=system_prompt)] + messages

        return messages

    return prompt


def create_agent_node(
    agent: AgentModel, additional_tools: Optional[Sequence[BaseTool]] = None
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
        store=store,
        state_schema=SharedState,
        config_schema=RunnableConfig,
        checkpointer=checkpointer,
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
    summarization_model: SummarizationModel | None = config.app.summarization

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
        logger.debug("Running summarization node")

        if not summarization_model:
            logger.debug("No summarization model configured, skipping summarization")
            return

        new_messages: Sequence[BaseMessage] = state.get("messages", [])
        all_messages: Sequence[BaseMessage] = new_messages

        model: LanguageModelLike = summarization_model.model.as_chat_model()

        max_tokens: int
        token_counter: Callable[..., int]

        if summarization_model.retained_message_count is not None:
            # For message count-based trimming, ensure we keep at least 1 message
            max_tokens = max(1, summarization_model.retained_message_count)
            token_counter = len
        else:
            max_tokens = summarization_model.max_tokens
            token_counter = count_tokens_approximately

        logger.debug(f"max_tokens: {max_tokens}, token_counter: {token_counter}")

        logger.trace(
            f"Original messages:\n{json.dumps([msg.model_dump() for msg in all_messages], indent=2)}"
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
            f"Trimmed messages:\n{json.dumps([msg.model_dump() for msg in trimmed_messages], indent=2)}"
        )

        logger.debug(
            f"Trimmed messages from {len(all_messages)} to {len(trimmed_messages)}"
        )

        # Handle case where trim_messages returns empty list - summarize all messages
        if len(trimmed_messages) == 0 and len(all_messages) > 0:
            logger.warning(
                "trim_messages returned empty list, summarizing all messages"
            )
            return _update_messages_with_summary(model, state, all_messages)

        if len(trimmed_messages) < len(all_messages):
            logger.debug(
                f"Trimmed {len(all_messages) - len(trimmed_messages)} messages due to limit: {max_tokens} "
                f"(using {'message count' if token_counter == len else 'token count'}). "
                f"Kept {len(trimmed_messages)} messages."
            )

            # Find messages that were removed by trimming
            trimmed_message_ids: set[str] = {m.id for m in trimmed_messages}
            messages_to_summarize: Sequence[BaseMessage] = [
                m for m in all_messages if m.id not in trimmed_message_ids
            ]

            return _update_messages_with_summary(model, state, messages_to_summarize)
        else:
            logger.debug(
                f"No messages trimmed ({len(all_messages)} messages fit within limit: {max_tokens}). "
                f"No summarization performed."
            )

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
