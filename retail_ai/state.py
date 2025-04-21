from typing import TypedDict, Literal, Sequence
from langgraph.graph import MessagesState
from langchain_core.documents.base import Document

 

class AgentConfig(TypedDict):
  ...

class AgentState(MessagesState):
  context: Sequence[Document]
  route: str