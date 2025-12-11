import uuid
from os import PathLike
from pathlib import Path
from typing import Any, Generator, Optional, Sequence, Union

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from loguru import logger
from mlflow import MlflowClient
from mlflow.pyfunc import ChatAgent, ChatModel, ResponsesAgent
from mlflow.types.agent import ChatContext
from mlflow.types.llm import (
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    ChatParams,
)
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.types.responses_helpers import (
    Message,
    ResponseInputTextParam,
)

from dao_ai.messages import (
    has_langchain_messages,
    has_mlflow_messages,
    has_mlflow_responses_messages,
)
from dao_ai.state import Context


def get_latest_model_version(model_name: str) -> int:
    """
    Retrieve the latest version number of a registered MLflow model.

    Queries the MLflow Model Registry to find the highest version number
    for a given model name, which is useful for ensuring we're using
    the most recent model version.

    Args:
        model_name: The name of the registered model in MLflow

    Returns:
        The latest version number as an integer
    """
    mlflow_client: MlflowClient = MlflowClient()
    latest_version: int = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


async def get_state_snapshot_async(
    graph: CompiledStateGraph, thread_id: str
) -> Optional[StateSnapshot]:
    """
    Retrieve the state snapshot from the graph for a given thread_id asynchronously.

    This utility function accesses the graph's checkpointer to retrieve the current
    state snapshot, which contains the full state values and metadata.

    Args:
        graph: The compiled LangGraph state machine
        thread_id: The thread/conversation ID to retrieve state for

    Returns:
        StateSnapshot if found, None otherwise
    """
    logger.debug(f"Retrieving state snapshot for thread_id: {thread_id}")
    try:
        # Check if graph has a checkpointer
        if graph.checkpointer is None:
            logger.debug("No checkpointer available in graph")
            return None

        # Get the current state from the checkpointer (use async version)
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        state_snapshot: Optional[StateSnapshot] = await graph.aget_state(config)

        if state_snapshot is None:
            logger.debug(f"No state found for thread_id: {thread_id}")
            return None

        return state_snapshot

    except Exception as e:
        logger.warning(f"Error retrieving state snapshot for thread {thread_id}: {e}")
        return None


def get_state_snapshot(
    graph: CompiledStateGraph, thread_id: str
) -> Optional[StateSnapshot]:
    """
    Retrieve the state snapshot from the graph for a given thread_id.

    This is a synchronous wrapper around get_state_snapshot_async.
    Use this for backward compatibility in synchronous contexts.

    Args:
        graph: The compiled LangGraph state machine
        thread_id: The thread/conversation ID to retrieve state for

    Returns:
        StateSnapshot if found, None otherwise
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(get_state_snapshot_async(graph, thread_id))
    except Exception as e:
        logger.warning(f"Error in synchronous state snapshot retrieval: {e}")
        return None


def get_genie_conversation_ids_from_state(
    state_snapshot: Optional[StateSnapshot],
) -> dict[str, str]:
    """
    Extract genie_conversation_ids from a state snapshot.

    This function extracts the genie_conversation_ids dictionary from the state
    snapshot values if present.

    Args:
        state_snapshot: The state snapshot to extract conversation IDs from

    Returns:
        A dictionary mapping genie space_id to conversation_id, or empty dict if not found
    """
    if state_snapshot is None:
        return {}

    try:
        # Extract state values - these contain the actual state data
        state_values: dict[str, Any] = state_snapshot.values

        # Extract genie_conversation_ids from state values
        genie_conversation_ids: dict[str, str] = state_values.get(
            "genie_conversation_ids", {}
        )

        if genie_conversation_ids:
            logger.debug(f"Retrieved genie_conversation_ids: {genie_conversation_ids}")
            return genie_conversation_ids

        return {}

    except Exception as e:
        logger.warning(f"Error extracting genie_conversation_ids from state: {e}")
        return {}


class LanggraphChatModel(ChatModel):
    """
    ChatModel that delegates requests to a LangGraph CompiledStateGraph.
    """

    def __init__(self, graph: CompiledStateGraph) -> None:
        self.graph = graph

    def predict(
        self, context, messages: list[ChatMessage], params: Optional[ChatParams] = None
    ) -> ChatCompletionResponse:
        logger.debug(f"messages: {messages}, params: {params}")
        if not messages:
            raise ValueError("Message list is empty.")

        request = {"messages": self._convert_messages_to_dict(messages)}

        context: Context = self._convert_to_context(params)
        custom_inputs: dict[str, Any] = {"configurable": context.model_dump()}

        # Use async ainvoke internally for parallel execution
        import asyncio

        async def _async_invoke():
            return await self.graph.ainvoke(
                request, context=context, config=custom_inputs
            )

        loop = asyncio.get_event_loop()
        response: dict[str, Sequence[BaseMessage]] = loop.run_until_complete(
            _async_invoke()
        )

        logger.trace(f"response: {response}")

        last_message: BaseMessage = response["messages"][-1]

        response_message = ChatMessage(role="assistant", content=last_message.content)
        return ChatCompletionResponse(choices=[ChatChoice(message=response_message)])

    def _convert_to_context(
        self, params: Optional[ChatParams | dict[str, Any]]
    ) -> Context:
        input_data = params
        if isinstance(params, ChatParams):
            input_data = params.to_dict()

        configurable: dict[str, Any] = {}
        if "configurable" in input_data:
            configurable: dict[str, Any] = input_data.pop("configurable")
        if "custom_inputs" in input_data:
            custom_inputs: dict[str, Any] = input_data.pop("custom_inputs")
            if "configurable" in custom_inputs:
                configurable: dict[str, Any] = custom_inputs.pop("configurable")

        if "user_id" in configurable:
            configurable["user_id"] = configurable["user_id"].replace(".", "_")

        if "conversation_id" in configurable and "thread_id" not in configurable:
            configurable["thread_id"] = configurable["conversation_id"]

        if "thread_id" not in configurable:
            configurable["thread_id"] = str(uuid.uuid4())

        context: Context = Context(**configurable)
        return context

    def predict_stream(
        self, context, messages: list[ChatMessage], params: ChatParams
    ) -> Generator[ChatCompletionChunk, None, None]:
        logger.debug(f"messages: {messages}, params: {params}")
        if not messages:
            raise ValueError("Message list is empty.")

        request = {"messages": self._convert_messages_to_dict(messages)}

        context: Context = self._convert_to_context(params)
        custom_inputs: dict[str, Any] = {"configurable": context.model_dump()}

        # Use async astream internally for parallel execution
        import asyncio

        async def _async_stream():
            async for nodes, stream_mode, messages_batch in self.graph.astream(
                request,
                context=context,
                config=custom_inputs,
                stream_mode=["messages", "custom"],
                subgraphs=True,
            ):
                nodes: tuple[str, ...]
                stream_mode: str
                messages_batch: Sequence[BaseMessage]
                logger.trace(
                    f"nodes: {nodes}, stream_mode: {stream_mode}, messages: {messages_batch}"
                )
                for message in messages_batch:
                    if (
                        isinstance(
                            message,
                            (
                                AIMessageChunk,
                                AIMessage,
                            ),
                        )
                        and message.content
                        and "summarization" not in nodes
                    ):
                        content = message.content
                        yield self._create_chat_completion_chunk(content)

        # Convert async generator to sync generator
        loop = asyncio.get_event_loop()
        async_gen = _async_stream()

        try:
            while True:
                try:
                    item = loop.run_until_complete(async_gen.__anext__())
                    yield item
                except StopAsyncIteration:
                    break
        finally:
            loop.run_until_complete(async_gen.aclose())

    def _create_chat_completion_chunk(self, content: str) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            choices=[
                ChatChunkChoice(
                    delta=ChatChoiceDelta(role="assistant", content=content)
                )
            ]
        )

    def _convert_messages_to_dict(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, Any]]:
        return [m.to_dict() for m in messages]


class LanggraphResponsesAgent(ResponsesAgent):
    """
    ResponsesAgent that delegates requests to a LangGraph CompiledStateGraph.

    This is the modern replacement for LanggraphChatModel, providing better
    support for streaming, tool calling, and async execution.
    """

    def __init__(self, graph: CompiledStateGraph) -> None:
        self.graph = graph

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Process a ResponsesAgentRequest and return a ResponsesAgentResponse.
        """
        logger.debug(f"ResponsesAgent request: {request}")

        # Convert ResponsesAgent input to LangChain messages
        messages = self._convert_request_to_langchain_messages(request)

        # Prepare context
        context: Context = self._convert_request_to_context(request)
        custom_inputs: dict[str, Any] = {"configurable": context.model_dump()}

        # Build the graph input state, including genie_conversation_ids if provided
        graph_input: dict[str, Any] = {"messages": messages}
        if request.custom_inputs and "genie_conversation_ids" in request.custom_inputs:
            graph_input["genie_conversation_ids"] = request.custom_inputs[
                "genie_conversation_ids"
            ]
            logger.debug(
                f"Including genie_conversation_ids in graph input: {graph_input['genie_conversation_ids']}"
            )

        # Use async ainvoke internally for parallel execution
        import asyncio

        async def _async_invoke():
            try:
                return await self.graph.ainvoke(
                    graph_input, context=context, config=custom_inputs
                )
            except Exception as e:
                logger.error(f"Error in graph.ainvoke: {e}")
                raise

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Handle case where no event loop exists (common in some deployment scenarios)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            response: dict[str, Sequence[BaseMessage]] = loop.run_until_complete(
                _async_invoke()
            )
        except Exception as e:
            logger.error(f"Error in async execution: {e}")
            raise

        # Convert response to ResponsesAgent format
        last_message: BaseMessage = response["messages"][-1]

        output_item = self.create_text_output_item(
            text=last_message.content, id=f"msg_{uuid.uuid4().hex[:8]}"
        )

        # Retrieve genie_conversation_ids from state if available
        custom_outputs: dict[str, Any] = custom_inputs.copy()
        thread_id: Optional[str] = context.thread_id
        if thread_id:
            state_snapshot: Optional[StateSnapshot] = loop.run_until_complete(
                get_state_snapshot_async(self.graph, thread_id)
            )
            genie_conversation_ids: dict[str, str] = (
                get_genie_conversation_ids_from_state(state_snapshot)
            )
            if genie_conversation_ids:
                custom_outputs["genie_conversation_ids"] = genie_conversation_ids

        return ResponsesAgentResponse(
            output=[output_item], custom_outputs=custom_outputs
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Process a ResponsesAgentRequest and yield ResponsesAgentStreamEvent objects.
        """
        logger.debug(f"ResponsesAgent stream request: {request}")

        # Convert ResponsesAgent input to LangChain messages
        messages: list[BaseMessage] = self._convert_request_to_langchain_messages(
            request
        )

        # Prepare context
        context: Context = self._convert_request_to_context(request)
        custom_inputs: dict[str, Any] = {"configurable": context.model_dump()}

        # Build the graph input state, including genie_conversation_ids if provided
        graph_input: dict[str, Any] = {"messages": messages}
        if request.custom_inputs and "genie_conversation_ids" in request.custom_inputs:
            graph_input["genie_conversation_ids"] = request.custom_inputs[
                "genie_conversation_ids"
            ]
            logger.debug(
                f"Including genie_conversation_ids in graph input: {graph_input['genie_conversation_ids']}"
            )

        # Use async astream internally for parallel execution
        import asyncio

        async def _async_stream():
            item_id = f"msg_{uuid.uuid4().hex[:8]}"
            accumulated_content = ""

            try:
                async for nodes, stream_mode, messages_batch in self.graph.astream(
                    graph_input,
                    context=context,
                    config=custom_inputs,
                    stream_mode=["messages", "custom"],
                    subgraphs=True,
                ):
                    nodes: tuple[str, ...]
                    stream_mode: str
                    messages_batch: Sequence[BaseMessage]

                    for message in messages_batch:
                        if (
                            isinstance(
                                message,
                                (
                                    AIMessageChunk,
                                    AIMessage,
                                ),
                            )
                            and message.content
                            and "summarization" not in nodes
                        ):
                            content = message.content
                            accumulated_content += content

                            # Yield streaming delta
                            yield ResponsesAgentStreamEvent(
                                **self.create_text_delta(delta=content, item_id=item_id)
                            )

                # Retrieve genie_conversation_ids from state if available
                custom_outputs: dict[str, Any] = custom_inputs.copy()
                thread_id: Optional[str] = context.thread_id

                if thread_id:
                    state_snapshot: Optional[
                        StateSnapshot
                    ] = await get_state_snapshot_async(self.graph, thread_id)
                    genie_conversation_ids: dict[str, str] = (
                        get_genie_conversation_ids_from_state(state_snapshot)
                    )
                    if genie_conversation_ids:
                        custom_outputs["genie_conversation_ids"] = (
                            genie_conversation_ids
                        )

                # Yield final output item
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        text=accumulated_content, id=item_id
                    ),
                    custom_outputs=custom_outputs,
                )
            except Exception as e:
                logger.error(f"Error in graph.astream: {e}")
                raise

        # Convert async generator to sync generator
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Handle case where no event loop exists (common in some deployment scenarios)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async_gen = _async_stream()

        try:
            while True:
                try:
                    item = loop.run_until_complete(async_gen.__anext__())
                    yield item
                except StopAsyncIteration:
                    break
                except Exception as e:
                    logger.error(f"Error in streaming: {e}")
                    raise
        finally:
            try:
                loop.run_until_complete(async_gen.aclose())
            except Exception as e:
                logger.warning(f"Error closing async generator: {e}")

    def _extract_text_from_content(
        self,
        content: Union[str, list[Union[ResponseInputTextParam, str, dict[str, Any]]]],
    ) -> str:
        """Extract text content from various MLflow content formats.

        MLflow ResponsesAgent supports multiple content formats:
        - str: Simple text content
        - list[ResponseInputTextParam]: Structured text objects with .text attribute
        - list[dict]: Dictionaries with "text" key
        - Mixed lists of the above types

        This method normalizes all formats to a single concatenated string.

        Args:
            content: The content to extract text from

        Returns:
            Concatenated text string from all content items
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for content_item in content:
                if isinstance(content_item, ResponseInputTextParam):
                    text_parts.append(content_item.text)
                elif isinstance(content_item, str):
                    text_parts.append(content_item)
                elif isinstance(content_item, dict) and "text" in content_item:
                    text_parts.append(content_item["text"])
            return "".join(text_parts)
        else:
            # Fallback for unknown types - try to extract text attribute
            return getattr(content, "text", str(content))

    def _convert_request_to_langchain_messages(
        self, request: ResponsesAgentRequest
    ) -> list[dict[str, Any]]:
        """Convert ResponsesAgent input to LangChain message format."""
        messages = []

        for input_item in request.input:
            if isinstance(input_item, Message):
                # Handle MLflow Message objects
                content = self._extract_text_from_content(input_item.content)
                messages.append({"role": input_item.role, "content": content})
            elif isinstance(input_item, dict):
                # Handle dict format
                if "role" in input_item and "content" in input_item:
                    content = self._extract_text_from_content(input_item["content"])
                    messages.append({"role": input_item["role"], "content": content})
            else:
                # Fallback for other object types with role/content attributes
                role = getattr(input_item, "role", "user")
                content = self._extract_text_from_content(
                    getattr(input_item, "content", "")
                )
                messages.append({"role": role, "content": content})

        return messages

    def _convert_request_to_context(self, request: ResponsesAgentRequest) -> Context:
        """Convert ResponsesAgent context to internal Context."""

        logger.debug(f"request.context: {request.context}")
        logger.debug(f"request.custom_inputs: {request.custom_inputs}")

        configurable: dict[str, Any] = {}

        # Process context values first (lower priority)
        # Use strong typing with forward-declared type hints instead of hasattr checks
        chat_context: Optional[ChatContext] = request.context
        if chat_context is not None:
            conversation_id: Optional[str] = chat_context.conversation_id
            user_id: Optional[str] = chat_context.user_id

            if conversation_id is not None:
                configurable["conversation_id"] = conversation_id
                configurable["thread_id"] = conversation_id

            if user_id is not None:
                configurable["user_id"] = user_id

        # Process custom_inputs after context so they can override context values (higher priority)
        if request.custom_inputs:
            if "configurable" in request.custom_inputs:
                configurable.update(request.custom_inputs.pop("configurable"))

            configurable.update(request.custom_inputs)

        if "user_id" in configurable:
            configurable["user_id"] = configurable["user_id"].replace(".", "_")

        if "thread_id" not in configurable:
            configurable["thread_id"] = str(uuid.uuid4())

        logger.debug(f"Creating context from: {configurable}")

        return Context(**configurable)


def create_agent(graph: CompiledStateGraph) -> ChatAgent:
    """
    Create an MLflow-compatible ChatAgent from a LangGraph state machine.

    Factory function that wraps a compiled LangGraph in the LangGraphChatAgent
    class to make it deployable through MLflow.

    Args:
        graph: A compiled LangGraph state machine

    Returns:
        An MLflow-compatible ChatAgent instance
    """
    return LanggraphChatModel(graph)


def create_responses_agent(graph: CompiledStateGraph) -> ResponsesAgent:
    """
    Create an MLflow-compatible ResponsesAgent from a LangGraph state machine.

    Factory function that wraps a compiled LangGraph in the LanggraphResponsesAgent
    class to make it deployable through MLflow.

    Args:
        graph: A compiled LangGraph state machine

    Returns:
        An MLflow-compatible ResponsesAgent instance
    """
    return LanggraphResponsesAgent(graph)


def _process_langchain_messages(
    app: LanggraphChatModel | CompiledStateGraph,
    messages: Sequence[BaseMessage],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> dict[str, Any] | Any:
    """Process LangChain messages using async LangGraph calls internally."""
    import asyncio

    if isinstance(app, LanggraphChatModel):
        app = app.graph

    # Use async ainvoke internally for parallel execution
    async def _async_invoke():
        return await app.ainvoke({"messages": messages}, config=custom_inputs)

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_async_invoke())


def _process_langchain_messages_stream(
    app: LanggraphChatModel | CompiledStateGraph,
    messages: Sequence[BaseMessage],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> Generator[AIMessageChunk, None, None]:
    """Process LangChain messages in streaming mode using async LangGraph calls internally."""
    import asyncio

    if isinstance(app, LanggraphChatModel):
        app = app.graph

    logger.debug(f"Processing messages: {messages}, custom_inputs: {custom_inputs}")

    custom_inputs = custom_inputs.get("configurable", custom_inputs or {})
    context: Context = Context(**custom_inputs)

    # Use async astream internally for parallel execution
    async def _async_stream():
        async for nodes, stream_mode, stream_messages in app.astream(
            {"messages": messages},
            context=context,
            config=custom_inputs,
            stream_mode=["messages", "custom"],
            subgraphs=True,
        ):
            nodes: tuple[str, ...]
            stream_mode: str
            stream_messages: Sequence[BaseMessage]
            logger.trace(
                f"nodes: {nodes}, stream_mode: {stream_mode}, messages: {stream_messages}"
            )
            for message in stream_messages:
                if (
                    isinstance(
                        message,
                        (
                            AIMessageChunk,
                            AIMessage,
                        ),
                    )
                    and message.content
                    and "summarization" not in nodes
                ):
                    yield message

    # Convert async generator to sync generator

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Handle case where no event loop exists (common in some deployment scenarios)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async_gen = _async_stream()

    try:
        while True:
            try:
                item = loop.run_until_complete(async_gen.__anext__())
                yield item
            except StopAsyncIteration:
                break
    finally:
        loop.run_until_complete(async_gen.aclose())


def _process_mlflow_messages(
    app: ChatModel,
    messages: Sequence[ChatMessage],
    custom_inputs: Optional[ChatParams] = None,
) -> ChatCompletionResponse:
    return app.predict(None, messages, custom_inputs)


def _process_mlflow_response_messages(
    app: ResponsesAgent,
    messages: ResponsesAgentRequest,
) -> ResponsesAgentResponse:
    """Process MLflow ResponsesAgent request in batch mode."""
    return app.predict(messages)


def _process_mlflow_messages_stream(
    app: ChatModel,
    messages: Sequence[ChatMessage],
    custom_inputs: Optional[ChatParams] = None,
) -> Generator[ChatCompletionChunk, None, None]:
    for event in app.predict_stream(None, messages, custom_inputs):
        event: ChatCompletionChunk
        yield event


def _process_mlflow_response_messages_stream(
    app: ResponsesAgent,
    messages: ResponsesAgentRequest,
) -> Generator[ResponsesAgentStreamEvent, None, None]:
    """Process MLflow ResponsesAgent request in streaming mode."""
    for event in app.predict_stream(messages):
        event: ResponsesAgentStreamEvent
        yield event


def _process_config_messages(
    app: LanggraphChatModel | LanggraphResponsesAgent,
    messages: dict[str, Any],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> ChatCompletionResponse | ResponsesAgentResponse:
    if isinstance(app, LanggraphChatModel):
        messages: Sequence[ChatMessage] = [ChatMessage(**m) for m in messages]
        params: ChatParams = ChatParams(**{"custom_inputs": custom_inputs})
        return _process_mlflow_messages(app, messages, params)

    elif isinstance(app, LanggraphResponsesAgent):
        input_messages: list[Message] = [Message(**m) for m in messages]
        request = ResponsesAgentRequest(
            input=input_messages, custom_inputs=custom_inputs
        )
        return _process_mlflow_response_messages(app, request)


def _process_config_messages_stream(
    app: LanggraphChatModel | LanggraphResponsesAgent,
    messages: dict[str, Any],
    custom_inputs: dict[str, Any],
) -> Generator[ChatCompletionChunk | ResponsesAgentStreamEvent, None, None]:
    if isinstance(app, LanggraphChatModel):
        messages: Sequence[ChatMessage] = [ChatMessage(**m) for m in messages]
        params: ChatParams = ChatParams(**{"custom_inputs": custom_inputs})

        for event in _process_mlflow_messages_stream(
            app, messages, custom_inputs=params
        ):
            yield event

    elif isinstance(app, LanggraphResponsesAgent):
        input_messages: list[Message] = [Message(**m) for m in messages]
        request = ResponsesAgentRequest(
            input=input_messages, custom_inputs=custom_inputs
        )

        for event in _process_mlflow_response_messages_stream(app, request):
            yield event


def process_messages_stream(
    app: LanggraphChatModel | LanggraphResponsesAgent,
    messages: Sequence[BaseMessage]
    | Sequence[ChatMessage]
    | ResponsesAgentRequest
    | dict[str, Any],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> Generator[
    ChatCompletionChunk | ResponsesAgentStreamEvent | AIMessageChunk, None, None
]:
    """
    Process messages through a ChatAgent in streaming mode.

    Utility function that normalizes message input formats and
    streams the agent's responses as they're generated using async LangGraph calls internally.

    Args:
        app: The ChatAgent to process messages with
        messages: Messages in various formats (list or dict)

    Yields:
        Individual message chunks from the streaming response
    """

    if has_mlflow_responses_messages(messages):
        for event in _process_mlflow_response_messages_stream(app, messages):
            yield event
    elif has_mlflow_messages(messages):
        for event in _process_mlflow_messages_stream(app, messages, custom_inputs):
            yield event
    elif has_langchain_messages(messages):
        for event in _process_langchain_messages_stream(app, messages, custom_inputs):
            yield event
    else:
        for event in _process_config_messages_stream(app, messages, custom_inputs):
            yield event


def process_messages(
    app: LanggraphChatModel | LanggraphResponsesAgent,
    messages: Sequence[BaseMessage]
    | Sequence[ChatMessage]
    | ResponsesAgentRequest
    | dict[str, Any],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> ChatCompletionResponse | ResponsesAgentResponse | dict[str, Any] | Any:
    """
    Process messages through a ChatAgent in batch mode.

    Utility function that normalizes message input formats and
    returns the complete response from the agent using async LangGraph calls internally.

    Args:
        app: The ChatAgent to process messages with
        messages: Messages in various formats (list or dict)

    Returns:
        Complete response from the agent
    """

    if has_mlflow_responses_messages(messages):
        return _process_mlflow_response_messages(app, messages)
    elif has_mlflow_messages(messages):
        return _process_mlflow_messages(app, messages, custom_inputs)
    elif has_langchain_messages(messages):
        return _process_langchain_messages(app, messages, custom_inputs)
    else:
        return _process_config_messages(app, messages, custom_inputs)


def display_graph(app: LanggraphChatModel | CompiledStateGraph) -> None:
    from IPython.display import HTML, Image, display

    if isinstance(app, LanggraphChatModel):
        app = app.graph

    try:
        content = Image(app.get_graph(xray=True).draw_mermaid_png())
    except Exception as e:
        print(e)
        ascii_graph: str = app.get_graph(xray=True).draw_ascii()
        html_content = f"""
    <pre style="font-family: monospace; line-height: 1.2; white-space: pre;">
    {ascii_graph}
    </pre>
    """
        content = HTML(html_content)

    display(content)


def save_image(app: LanggraphChatModel | CompiledStateGraph, path: PathLike) -> None:
    if isinstance(app, LanggraphChatModel):
        app = app.graph

    path = Path(path)
    content = app.get_graph(xray=True).draw_mermaid_png()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
