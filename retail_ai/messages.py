from typing import Callable, Optional, Sequence

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     ToolMessage)


def last_message(
  messages: Sequence[BaseMessage], 
  predicate: Optional[Callable[[BaseMessage], bool]] = None
) -> Optional[BaseMessage]:
  """
  Find the last message in a sequence that matches a given predicate.
  
  This function traverses the message history in reverse order to find
  the most recent message that satisfies the optional predicate function.
  If no predicate is provided, it returns the last message in the sequence.
  
  Args:
      messages: A sequence of LangChain message objects to search through
      predicate: Optional function that takes a message and returns True if it matches criteria
      
  Returns:
      The last message matching the predicate, or None if no matches found
  """
  if predicate is None:
    def null_predicate(m: BaseMessage) -> bool:
      return True
    predicate = null_predicate
    
  return next(reversed([m for m in messages if predicate(m)]), None)


def last_human_message(messages: Sequence[BaseMessage]) -> Optional[HumanMessage]:
  """
  Find the last message from a human user in the message history.
  
  This is a specialized wrapper around last_message that filters for HumanMessage objects.
  Used to retrieve the most recent user input for processing by the retail AI agent.
  
  Args:
      messages: A sequence of LangChain message objects to search through
      
  Returns:
      The last HumanMessage in the sequence, or None if no human messages found
  """
  return last_message(messages, lambda m: isinstance(m, HumanMessage))


def last_ai_message(messages: Sequence[BaseMessage]) -> Optional[AIMessage]:
  """
  Find the last message from the AI assistant in the message history.
  
  This is a specialized wrapper around last_message that filters for AIMessage objects.
  Used to retrieve the most recent AI response for context in multi-turn conversations.
  
  Args:
      messages: A sequence of LangChain message objects to search through
      
  Returns:
      The last AIMessage in the sequence, or None if no AI messages found
  """
  return last_message(messages, lambda m: isinstance(m, AIMessage))


def last_tool_message(messages: Sequence[BaseMessage]) -> Optional[ToolMessage]:
  """
  Find the last message from a tool in the message history.
  
  This is a specialized wrapper around last_message that filters for ToolMessage objects.
  Used to retrieve the most recent tool output, such as from vector search or Genie queries.
  
  Args:
      messages: A sequence of LangChain message objects to search through
      
  Returns:
      The last ToolMessage in the sequence, or None if no tool messages found
  """
  return last_message(messages, lambda m: isinstance(m, ToolMessage))
