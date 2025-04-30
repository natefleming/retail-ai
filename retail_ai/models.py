from distutils.util import strtobool
from typing import Any, Generator, Iterator, Optional, Sequence, Union

from langchain_core.messages import (BaseMessage, HumanMessage,
                                     MessageLikeRepresentation, ToolMessage)
from langchain_core.runnables import RunnableLambda
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel.io import AddableValuesDict
from loguru import logger
from mlflow import MlflowClient
from mlflow.langchain.chat_agent_langgraph import parse_message
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (ChatAgentChunk, ChatAgentMessage,
                                ChatAgentResponse, ChatContext)


from retail_ai.state import AgentState, AgentConfig

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


class LangGraphChatAgent(ChatAgent):
    """
    An MLflow ChatAgent implementation that wraps a LangGraph CompiledStateGraph.
    
    This class adapts a LangGraph agent to the MLflow ChatAgent interface,
    allowing it to be deployed through MLflow Model Serving with standard
    chat completion interfaces. It supports both batch and streaming inference.
    """

    def __init__(self, agent: CompiledStateGraph):
        """
        Initialize the LangGraphChatAgent with a compiled LangGraph state machine.
        
        Args:
            agent: A compiled LangGraph state machine representing the agent's workflow
        """
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """
        Process a batch inference request to generate a complete response.
        
        Executes the full agent workflow on the provided messages and returns
        the complete response, including all message objects generated during
        the execution.
        
        Args:
            messages: List of chat messages representing the conversation history
            context: Optional context information for the agent
            custom_inputs: Optional configuration parameters for the agent
            
        Returns:
            A ChatAgentResponse containing the generated messages
        """
        # Convert MLflow message format to the format expected by LangGraph
        request = {"messages": self._convert_messages_to_dict(messages)}

        # Process all messages from all nodes in the agent's graph
        messages: list[ChatAgentMessage] = []
        for event in self.agent.stream(request, config=custom_inputs, stream_mode="updates"):
            for node_data in event.values():
                for msg in node_data.get("messages", []):
                    if isinstance(msg, BaseMessage):
                        # Convert LangChain message to a dict representation
                        msg = parse_message(msg)
                    # Create a ChatAgentMessage from the parsed dict
                    messages.append(ChatAgentMessage(**msg))
              
        # Return all messages together in a single response
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """
        Process a streaming inference request to generate incremental responses.
        
        Executes the agent workflow and yields individual message chunks as they're
        generated, allowing for real-time streaming of responses to the client.
        Filters out tool messages and empty content to provide a clean stream.
        
        Args:
            messages: List of chat messages representing the conversation history
            context: Optional context information for the agent
            custom_inputs: Optional configuration parameters for the agent
            
        Yields:
            ChatAgentChunk objects containing delta updates to stream to the client
        """
        # Convert MLflow message format to the format expected by LangGraph
        request = {"messages": self._convert_messages_to_dict(messages)}

        # Stream messages from the agent's execution
        for message, metadata in self.agent.stream(request, config=custom_inputs, stream_mode="messages"):
            logger.debug(f"predict_stream: message={message}, type={type}, metadata={metadata}")
            
            if isinstance(message, BaseMessage):
                # Skip empty messages and tool calls to avoid streaming internal operations
                if not message.content or message.additional_kwargs.get("tool_calls"):
                    continue
                    
                # Parse the message into a format suitable for MLflow
                parsed_message = parse_message(message)
                logger.debug(f"predict_stream: parsed_message={parsed_message}")
                
                # Create and yield a chunk for streaming
                chunk: ChatAgentChunk = ChatAgentChunk(**{"delta": parsed_message})
                logger.debug(f"predict_stream: chunk={chunk}")
                yield chunk

    def _convert_messages_to_dict(self, messages: list[ChatAgentMessage]) -> list[dict[str, Any]]:
        """
        Convert ChatAgentMessage objects to dictionaries for LangGraph.
        
        Helper method to transform MLflow message objects into the dictionary
        format expected by LangGraph agents.
        
        Args:
            messages: List of ChatAgentMessage objects
            
        Returns:
            List of message dictionaries
        """
        return [message.model_dump() for message in messages]


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
    return LangGraphChatAgent(graph)
    

def process_messages_stream(
    app: ChatAgent,
    messages: Union[Sequence[MessageLikeRepresentation], dict[str, Any]]
) -> Iterator[BaseMessage]:
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
    if isinstance(messages, list):
        messages = {"messages": messages}
    for event in app.predict_stream(messages):
        yield event


def process_messages(
    app: ChatAgent,
    messages: Union[Sequence[MessageLikeRepresentation], dict[str, Any]]
) -> Iterator[BaseMessage]:
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
    if isinstance(messages, list):
        messages = {"messages": messages}
    return app.predict(messages)


def as_langgraph_chain(agent: CompiledStateGraph) -> RunnableLambda:
    """
    Convert a LangGraph agent into a LangChain runnable.
    
    Creates a LangChain-compatible runnable that wraps a LangGraph agent,
    handling input formatting, configurable parameters, and supporting
    both streaming and batch inference modes.
    
    Args:
        agent: The compiled LangGraph state machine
        
    Returns:
        A LangChain runnable that invokes the agent with proper formatting
    """

    def to_state(input_data: MessageLikeRepresentation) -> AgentState:
        """
        Format various input types into an AgentState object.
        
        Handles different input formats (string, list of messages, dictionary)
        and converts them to a proper AgentState instance for LangGraph agents.
        
        Args:
            input_data: Input in various formats
            
        Returns:
            AgentState object with properly formatted messages
        """
        match input_data:
            case str():
                return AgentState(messages=[HumanMessage(content=input_data)])
            case list() if all(isinstance(msg, BaseMessage) for msg in input_data):
                return AgentState(messages=input_data)
            case dict() if "messages" in input_data:
                # If input is already a dict with messages, convert to AgentState
                messages = input_data.get("messages", [])
                config = AgentConfig(**(input_data.get("config", {}) or {}))
                return AgentState(messages=messages, config=config)
            case _:
                return AgentState(messages=[HumanMessage(content=str(input_data))])


    def invoke_agent(
        agent: CompiledStateGraph, 
        state: AgentState, 
        config: Optional[AgentConfig],
    ) -> Sequence[BaseMessage]:
        """
        Invoke the agent in batch mode.
        
        Executes the full agent workflow and returns the complete results.
        
        Args:
            agent: The LangGraph agent to invoke
            input_data: Formatted input data
            config: Configuration parameters
            
        Returns:
            Sequence of messages from the completed execution
        """
        logger.debug(f"invoke_agent: state={state}, config={config}")
        result: AddableValuesDict = agent.invoke(state, config=config)
        messages: Sequence[BaseMessage] = result["messages"]
        logger.debug(f"invoke_agent: result={result}")
        return messages


    def stream_agent(
        agent: CompiledStateGraph, 
        state: AgentState, 
        config: Optional[AgentConfig],
    ) -> Iterator[BaseMessage]:
        """
        Invoke the agent in streaming mode.
        
        Executes the agent workflow and yields messages as they're generated.
        
        Args:
            agent: The LangGraph agent to stream from
            input_data: Formatted input data
            config: Configuration parameters
            
        Returns:
            Iterator of messages from the streaming execution
        """
        logger.debug(f"stream_agent: state={state}, config={config}")
        return agent.stream(
            state, 
            config=config, 
            stream_mode="messages"
        )


    def runnable_with_config(input_data: MessageLikeRepresentation, config: Optional[dict] = None) -> Sequence[BaseMessage] | Iterator[BaseMessage]:
        """
        Main runnable function that processes inputs with the agent.
        
        Handles input formatting, configuration extraction, and determines whether
        to use streaming or batch mode based on configuration.
        
        Args:
            input_data: The input to process (various formats supported)
            config: Additional configuration parameters
            
        Returns:
            Either a sequence of messages (batch) or an iterator of messages (streaming)
        """
        logger.debug(f"runnable_with_config: input_data={input_data}, config={config}")
        
        configurable: dict[str, Any] = {}
        if "configurable" in input_data:
            configurable: dict[str, Any] = input_data.pop("configurable")
        agent_state: AgentState = to_state(input_data)
        agent_config: AgentConfig = AgentConfig(**{"configurable": configurable})
        
        
        should_stream: bool = strtobool(str(configurable.get("stream", "true")))
        if should_stream:
            # Stream mode - yield messages one by one
            for message, metadata in stream_agent(agent, agent_state, agent_config):
                if isinstance(message, BaseMessage):
                    # Skip empty, tool call, or tool messages to clean the stream
                    if not message.content or message.additional_kwargs.get("tool_calls") or isinstance(message, ToolMessage):
                        continue
                    yield message
        else:
            # Batch mode - return all messages at once
            return invoke_agent(agent, agent_state, agent_config)
  
    # Create and return the runnable
    return RunnableLambda(runnable_with_config)






