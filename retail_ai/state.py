from typing import Sequence, TypedDict

from langchain_core.documents.base import Document
from langgraph.graph import MessagesState


class AgentConfig(TypedDict):
  ...

class AgentState(MessagesState):
  context: Sequence[Document]
  route: str
  remaining_steps: int