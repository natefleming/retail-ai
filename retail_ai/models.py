from typing import Any, Generator, Optional

from langgraph.graph.state import CompiledStateGraph

import mlflow
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

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