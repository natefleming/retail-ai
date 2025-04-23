from typing import Callable, Optional, Sequence

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     ToolMessage)


def last_message(
  messages: Sequence[BaseMessage], 
  predicate: Optional[Callable[[BaseMessage], bool]] = None
) -> Optional[BaseMessage]:
  if predicate is None:
    def null_predicate(m: BaseMessage) -> bool:
      return True
    predicate = null_predicate
    
  return next(reversed([m for m in messages if predicate(m)]), None)


def last_human_message(messages: Sequence[BaseMessage]) -> Optional[HumanMessage]:
  return last_message(messages, lambda m: isinstance(m, HumanMessage))


def last_ai_message(messages: Sequence[BaseMessage]) -> Optional[AIMessage]:
  return last_message(messages, lambda m: isinstance(m, AIMessage))


def last_tool_message(messages: Sequence[BaseMessage]) -> Optional[ToolMessage]:
  return last_message(messages, lambda m: isinstance(m, ToolMessage))
