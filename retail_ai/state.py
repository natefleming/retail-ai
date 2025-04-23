from typing import TypedDict, Literal, Sequence
from langgraph.graph import MessagesState
from langchain_core.documents.base import Document
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
 

class AgentConfig(TypedDict):
  ...

class AgentState(MessagesState):
  context: Sequence[Document]
  route: str
  remaining_steps: int