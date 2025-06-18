import base64
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages.modifier import RemoveMessage
from mlflow.types.llm import ChatMessage


def remove_messages(
    messages: Sequence[BaseMessage], filter: Callable[[BaseMessage], bool] | None = None
) -> Sequence[RemoveMessage]:
    if filter:
        messages = [m for m in messages if filter(m)]
    return [RemoveMessage(m.id) for m in messages]


def message_with_images(
    message: HumanMessage, image_paths: Sequence[os.PathLike]
) -> BaseMessage:
    """
    Add an image to a LangChain message object.

    This function takes a LangChain message object and a path to an image file
    and returns a new message object with the image added. The image is added as a
    dictionary with the key "image" and the value being the path to the image file.

    Args:
        message: A LangChain message object to add the image to
        path: A Path object representing the path to the image file

    Returns:
        A new LangChain message object with the image added
    """

    if not image_paths:
        return message

    image_content: list[dict[str, Any]] = []
    for image_path in image_paths:
        image_path = Path(image_path)
        base64_image: str = base64.b64encode(Path(image_path).read_bytes()).decode(
            "utf-8"
        )
        image_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": message.content,
        }
    ] + image_content

    message_with_image: HumanMessage = HumanMessage(content=content)
    return message_with_image


def convert_to_langchain_messages(messages: dict[str, Any]) -> Sequence[BaseMessage]:
    langchain_messages: list[BaseMessage] = []
    for m in messages:
        image_paths: Sequence[str] = []
        if "image_paths" in m:
            image_paths = m.pop("image_paths")
        message: HumanMessage = HumanMessage(**m)
        message = message_with_images(message, image_paths)
        langchain_messages.append(message)
    return langchain_messages


def has_langchain_messages(messages: BaseMessage | Sequence[BaseMessage]) -> bool:
    if isinstance(messages, BaseMessage):
        messages = [messages]
    return any(isinstance(m, BaseMessage) for m in messages)


def has_mlflow_messages(messages: ChatMessage | Sequence[ChatMessage]) -> bool:
    if isinstance(messages, ChatMessage):
        messages = [messages]
    return any(isinstance(m, ChatMessage) for m in messages)


def has_image(messages: BaseMessage | Sequence[BaseMessage]) -> bool:
    """
    Check if a message contains an image.

    This function checks if the message content is a list of dictionaries
    containing an "image" key, or if the message has a "type" attribute equal to "image".

    Args:
        message: A LangChain message object to check for image content

    Returns:
        True if the message contains an image, False otherwise
    """

    def _has_image(message: BaseMessage) -> bool:
        if isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, dict) and item.get("type") in [
                    "image",
                    "image_url",
                ]:
                    return True
                if hasattr(item, "type") and item.type in ["image", "image_url"]:
                    return True
        return False

    if isinstance(messages, BaseMessage):
        messages = [messages]

    return any(_has_image(m) for m in messages)


def last_message(
    messages: Sequence[BaseMessage],
    predicate: Optional[Callable[[BaseMessage], bool]] = None,
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
    Used to retrieve the most recent user input for processing by the DAO AI agent.

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
