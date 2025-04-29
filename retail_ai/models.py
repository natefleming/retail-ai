from typing import Any, Generator, Iterator, Optional, Sequence, Union

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, MessageLikeRepresentation, convert_to_messages
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableLambda
from mlflow import MlflowClient
from mlflow.langchain.chat_agent_langgraph import parse_message
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (ChatAgentChunk, ChatAgentMessage,
                                ChatAgentResponse, ChatContext)

from langgraph.pregel.io import AddableValuesDict

from distutils.util import strtobool

from loguru import logger


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
        for event in self.agent.stream(request, config=custom_inputs, stream_mode="updates"):
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

        for message, metadata in self.agent.stream(request, config=custom_inputs, stream_mode="messages"):
            logger.debug("predict_stream: message={message}, type={type}, metadata={metadata}", message=message, type=type(message), metadata=metadata)
            if isinstance(message, BaseMessage):
                if not message.content or message.additional_kwargs.get("tool_calls"):
                    continue
                parsed_message = parse_message(message)
                logger.debug("predict_stream: parsed_message={parsed_message}", parsed_message=parsed_message)
                chunk: ChatAgentChunk = ChatAgentChunk(**{"delta": parsed_message})
                logger.debug("predict_stream: chunk={chunk}", chunk=chunk)
                yield chunk



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


def as_langgraph_chain(agent: CompiledStateGraph) -> RunnableLambda:

  def format_input(input_data: MessageLikeRepresentation) -> dict[str, Sequence[BaseMessage]]:
      match input_data:
          case str():
              return {"messages": [HumanMessage(content=input_data)]}
          case list() if all(isinstance(msg, BaseMessage) for msg in input_data):
              return {"messages": input_data}
          case dict() if "messages" in input_data:
              return input_data
          case _:
              return {"messages": [HumanMessage(content=str(input_data))]}


  def invoke_agent(
      agent: CompiledStateGraph, 
      input_data: dict[str, Any], 
      config: Optional[dict[str, Any]],
  ) -> Sequence[BaseMessage]:
    logger.debug(f"invoke_agent: input_data={input_data}, config={config}")
    result: AddableValuesDict = agent.invoke(input_data, config=config)
    messages: Sequence[BaseMessage] = result["messages"]
    logger.debug("invoke_agent: result={result}", result=result)
    return messages


  def stream_agent(
      agent: CompiledStateGraph, 
      input_data: dict[str, Any], 
      config: Optional[dict[str, Any]],
  ) -> Iterator[BaseMessage]:
    logger.debug(f"stream_agent: input_data={input_data}, config={config}")
    return agent.stream(
        input_data, 
        config=config, 
        stream_mode="messages"
    )

  def parse_configurable(input_data: MessageLikeRepresentation) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if isinstance(input_data, dict) and "configurable" in input_data:
        config = {
            "configurable": input_data.pop("configurable")
        } 
    return config

  def runnable_with_config(input_data: MessageLikeRepresentation, config: Optional[dict] = None) -> Sequence[BaseMessage] | Iterator[BaseMessage]:
      logger.debug(f"runnable_with_config: input_data={input_data}, config={config}")
      
      formatted_input: dict = format_input(input_data)
      config: dict[str, Any] = parse_configurable(input_data)

      should_stream: bool = strtobool(str(config.get("stream", "true")))
      if should_stream:
        for message, metadata in stream_agent(agent, formatted_input, config):
            if isinstance(message, BaseMessage):
                if not message.content or message.additional_kwargs.get("tool_calls") or isinstance(message, ToolMessage):
                    continue
                yield message
      else:
        return invoke_agent(agent, formatted_input, config)
  
  return RunnableLambda(runnable_with_config)






