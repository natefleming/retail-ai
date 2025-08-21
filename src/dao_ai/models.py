import uuid
from os import PathLike
from pathlib import Path
from typing import Any, Generator, Optional, Sequence

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from mlflow import MlflowClient
from mlflow.pyfunc import ChatAgent, ChatModel
from mlflow.types.llm import (
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    ChatParams,
)

from dao_ai.messages import has_langchain_messages, has_mlflow_messages
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

        response: dict[str, Sequence[BaseMessage]] = self.graph.invoke(
            request, context=context, config=custom_inputs
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

        for nodes, stream_mode, messages_batch in self.graph.stream(
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


def _process_langchain_messages(
    app: LanggraphChatModel | CompiledStateGraph,
    messages: Sequence[BaseMessage],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> dict[str, Any] | Any:
    if isinstance(app, LanggraphChatModel):
        app = app.graph
    return app.invoke({"messages": messages}, config=custom_inputs)


def _process_langchain_messages_stream(
    app: LanggraphChatModel | CompiledStateGraph,
    messages: Sequence[BaseMessage],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> Generator[AIMessageChunk, None, None]:
    if isinstance(app, LanggraphChatModel):
        app = app.graph

    logger.debug(f"Processing messages: {messages}, custom_inputs: {custom_inputs}")

    custom_inputs = custom_inputs.get("configurable", custom_inputs or {})
    context: Context = Context(**custom_inputs)

    for nodes, stream_mode, messages in app.stream(
        {"messages": messages},
        context=context,
        config=custom_inputs,
        stream_mode=["messages", "custom"],
        subgraphs=True,
    ):
        nodes: tuple[str, ...]
        stream_mode: str
        messages: Sequence[BaseMessage]
        logger.trace(
            f"nodes: {nodes}, stream_mode: {stream_mode}, messages: {messages}"
        )
        for message in messages:
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


def _process_mlflow_messages(
    app: ChatModel,
    messages: Sequence[ChatMessage],
    custom_inputs: Optional[ChatParams] = None,
) -> ChatCompletionResponse:
    return app.predict(None, messages, custom_inputs)


def _process_mlflow_messages_stream(
    app: ChatModel,
    messages: Sequence[ChatMessage],
    custom_inputs: Optional[ChatParams] = None,
) -> Generator[ChatCompletionChunk, None, None]:
    for event in app.predict_stream(None, messages, custom_inputs):
        event: ChatCompletionChunk
        yield event


def _process_config_messages(
    app: ChatModel,
    messages: dict[str, Any],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> ChatCompletionResponse:
    messages: Sequence[ChatMessage] = [ChatMessage(**m) for m in messages]
    params: ChatParams = ChatParams(**{"custom_inputs": custom_inputs})

    return _process_mlflow_messages(app, messages, params)


def _process_config_messages_stream(
    app: ChatModel, messages: dict[str, Any], custom_inputs: dict[str, Any]
) -> Generator[ChatCompletionChunk, None, None]:
    messages: Sequence[ChatMessage] = [ChatMessage(**m) for m in messages]
    params: ChatParams = ChatParams(**{"custom_inputs": custom_inputs})

    for event in _process_mlflow_messages_stream(app, messages, custom_inputs=params):
        yield event


def process_messages_stream(
    app: LanggraphChatModel,
    messages: Sequence[BaseMessage] | Sequence[ChatMessage] | dict[str, Any],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> Generator[ChatCompletionChunk | AIMessageChunk, None, None]:
    """
    Process messages through a ChatAgent in streaming mode.

    Utility function that normalizes message input formats and
    streams the agent's responses as they're generated.

    Args:
        app: The ChatAgent to process messages with
        messages: Messages in various formats (list or dict)

    Yields:
        Individual message chunks from the streaming response
    """

    if has_mlflow_messages(messages):
        for event in _process_mlflow_messages_stream(app, messages, custom_inputs):
            yield event
    elif has_langchain_messages(messages):
        for event in _process_langchain_messages_stream(app, messages, custom_inputs):
            yield event
    else:
        for event in _process_config_messages_stream(app, messages, custom_inputs):
            yield event


def process_messages(
    app: LanggraphChatModel,
    messages: Sequence[BaseMessage] | Sequence[ChatMessage] | dict[str, Any],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> ChatCompletionResponse | dict[str, Any] | Any:
    """
    Process messages through a ChatAgent in batch mode.

    Utility function that normalizes message input formats and
    returns the complete response from the agent.

    Args:
        app: The ChatAgent to process messages with
        messages: Messages in various formats (list or dict)

    Returns:
        Complete response from the agent
    """

    if has_mlflow_messages(messages):
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
