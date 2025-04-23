from typing import (
    Any, 
    Generator, 
    Optional, 
    Union, 
    Sequence, 
    Iterator
)

from langchain_core.messages import BaseMessage, MessageLikeRepresentation
from langgraph.graph.state import CompiledStateGraph

import mlflow
from mlflow import MlflowClient
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode, parse_message
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)


def get_latest_model_version(model_name: str) -> int:
    mlflow_client: MlflowClient = MlflowClient()
    latest_version: int = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages: list[ChatAgentMessage] = []
        for event in self.agent.stream(request, stream_mode="updates", config=custom_inputs):
            for node_data in event.values():
                for msg in node_data.get("messages", []):
                    if isinstance(msg, BaseMessage):
                        msg = parse_message(msg)
                    messages.append(ChatAgentMessage(**msg))
              

        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}

        for event in self.agent.stream(request, stream_mode="updates", config=custom_inputs):
          for node_data in event.values():
              for msg in node_data.get("messages", []):
                if isinstance(msg, BaseMessage):
                    msg = parse_message(msg)
                    yield ChatAgentChunk(**{"delta": msg})


def create_agent(graph: CompiledStateGraph) -> ChatAgent:
    return LangGraphChatAgent(graph)
    

def process_messages_stream(
    app: ChatAgent,
    messages: Union[Sequence[MessageLikeRepresentation], dict[str, Any]]
) -> Iterator[BaseMessage]:
  if isinstance(messages, list):
    messages = {"messages": messages}
  for event in app.predict_stream(messages):
    yield event


def process_messages(
    app: ChatAgent,
    messages: Union[Sequence[MessageLikeRepresentation], dict[str, Any]]
) -> Iterator[BaseMessage]:
  if isinstance(messages, list):
    messages = {"messages": messages}
  return app.predict(messages)

