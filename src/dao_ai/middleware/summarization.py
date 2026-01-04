"""
Summarization middleware for DAO AI agents.

This module provides a LoggingSummarizationMiddleware that extends LangChain's
built-in SummarizationMiddleware with logging capabilities, and provides
helper utilities for creating summarization middleware from DAO AI configuration.

The middleware automatically:
- Summarizes older messages using a separate LLM call when thresholds are exceeded
- Replaces them with a summary message in State (permanently)
- Keeps recent messages intact for context
- Logs when summarization is triggered and completed

Example:
    from dao_ai.middleware import create_summarization_middleware
    from dao_ai.config import ChatHistoryModel, LLMModel

    chat_history = ChatHistoryModel(
        model=LLMModel(name="gpt-4o-mini"),
        max_tokens=256,
        max_tokens_before_summary=4000,
    )

    middleware = create_summarization_middleware(chat_history)
"""

from typing import Any, Tuple

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime
from loguru import logger

from dao_ai.config import ChatHistoryModel

__all__ = [
    "SummarizationMiddleware",
    "LoggingSummarizationMiddleware",
    "create_summarization_middleware",
]


class LoggingSummarizationMiddleware(SummarizationMiddleware):
    """
    SummarizationMiddleware with logging for when summarization occurs.

    This extends LangChain's SummarizationMiddleware to add logging at INFO level
    when summarization is triggered and completed, providing visibility into
    when conversation history is being summarized.

    Logs include:
    - Original message count and approximate token count (before summarization)
    - New message count and approximate token count (after summarization)
    - Number of messages that were summarized
    """

    def _log_summarization(
        self,
        original_message_count: int,
        original_token_count: int,
        result_messages: list[Any],
    ) -> None:
        """Log summarization details with before/after metrics."""
        # Result messages: [RemoveMessage, summary_message, ...preserved_messages]
        # New message count excludes RemoveMessage (index 0)
        new_messages = [
            msg for msg in result_messages if not self._is_remove_message(msg)
        ]
        new_message_count = len(new_messages)
        new_token_count = self.token_counter(new_messages) if new_messages else 0

        # Calculate how many messages were summarized
        # preserved = new_messages - 1 (the summary message)
        preserved_count = max(0, new_message_count - 1)
        summarized_count = original_message_count - preserved_count

        logger.info(
            "Conversation summarized",
            before_messages=original_message_count,
            before_tokens=original_token_count,
            after_messages=new_message_count,
            after_tokens=new_token_count,
            summarized_messages=summarized_count,
        )
        logger.debug(
            "Summarization details",
            trigger=self.trigger,
            keep=self.keep,
            preserved_messages=preserved_count,
            token_reduction=original_token_count - new_token_count,
        )

    def _is_remove_message(self, msg: Any) -> bool:
        """Check if a message is a RemoveMessage."""
        return type(msg).__name__ == "RemoveMessage"

    def before_model(
        self, state: dict[str, Any], runtime: Runtime
    ) -> dict[str, Any] | None:
        """Process messages before model invocation, logging when summarization occurs."""
        messages: list[BaseMessage] = state.get("messages", [])
        original_message_count = len(messages)
        original_token_count = self.token_counter(messages) if messages else 0

        result = super().before_model(state, runtime)

        if result is not None:
            result_messages = result.get("messages", [])
            self._log_summarization(
                original_message_count,
                original_token_count,
                result_messages,
            )

        return result

    async def abefore_model(
        self, state: dict[str, Any], runtime: Runtime
    ) -> dict[str, Any] | None:
        """Process messages before model invocation (async), logging when summarization occurs."""
        messages: list[BaseMessage] = state.get("messages", [])
        original_message_count = len(messages)
        original_token_count = self.token_counter(messages) if messages else 0

        result = await super().abefore_model(state, runtime)

        if result is not None:
            result_messages = result.get("messages", [])
            self._log_summarization(
                original_message_count,
                original_token_count,
                result_messages,
            )

        return result


def create_summarization_middleware(
    chat_history: ChatHistoryModel,
) -> LoggingSummarizationMiddleware:
    """
    Create a LoggingSummarizationMiddleware from DAO AI ChatHistoryModel configuration.

    This factory function creates a LoggingSummarizationMiddleware instance
    configured according to the DAO AI ChatHistoryModel settings. The middleware
    includes logging at INFO level when summarization is triggered.

    Args:
        chat_history: ChatHistoryModel configuration for summarization

    Returns:
        List containing LoggingSummarizationMiddleware configured with the specified parameters

    Example:
        from dao_ai.config import ChatHistoryModel, LLMModel

        chat_history = ChatHistoryModel(
            model=LLMModel(name="gpt-4o-mini"),
            max_tokens=256,
            max_tokens_before_summary=4000,
        )

        middleware = create_summarization_middleware(chat_history)
    """
    logger.debug(
        "Creating summarization middleware",
        max_tokens=chat_history.max_tokens,
        max_tokens_before_summary=chat_history.max_tokens_before_summary,
        max_messages_before_summary=chat_history.max_messages_before_summary,
    )

    # Get the LLM model
    model: LanguageModelLike = chat_history.model.as_chat_model()

    # Determine trigger condition
    # LangChain uses ("tokens", value) or ("messages", value) tuples
    trigger: Tuple[str, int]
    if chat_history.max_tokens_before_summary:
        trigger = ("tokens", chat_history.max_tokens_before_summary)
    elif chat_history.max_messages_before_summary:
        trigger = ("messages", chat_history.max_messages_before_summary)
    else:
        # Default to a reasonable token threshold
        trigger = ("tokens", chat_history.max_tokens * 10)

    # Determine keep condition - how many recent messages/tokens to preserve
    # Default to keeping enough for context
    keep: Tuple[str, int] = ("tokens", chat_history.max_tokens)

    logger.info("Summarization middleware configured", trigger=trigger, keep=keep)

    return LoggingSummarizationMiddleware(
        model=model,
        trigger=trigger,
        keep=keep,
    )
