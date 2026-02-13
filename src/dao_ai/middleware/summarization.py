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

Additionally, a Deep Agents variant is available via
``create_deep_summarization_middleware`` that adds:
- Backend offloading of conversation history before summarization
- Tool argument truncation for large write_file/edit_file args in old messages
- Fraction-based triggers using model profile data
- Thread-aware storage at configurable path prefix

Example:
    from dao_ai.middleware import create_summarization_middleware
    from dao_ai.config import ChatHistoryModel, LLMModel

    chat_history = ChatHistoryModel(
        model=LLMModel(name="gpt-4o-mini"),
        max_tokens=256,
        max_tokens_before_summary=4000,
    )

    middleware = create_summarization_middleware(chat_history)

    # Deep Agents variant with backend offloading:
    from dao_ai.middleware import create_deep_summarization_middleware

    middleware = create_deep_summarization_middleware(
        model="gpt-4o-mini",
        trigger=("tokens", 100000),
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents.middleware.summarization import (
    SummarizationMiddleware as DeepAgentsSummarizationMiddleware,
)
from deepagents.middleware.summarization import (
    TruncateArgsSettings,
)
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime
from loguru import logger

from dao_ai.config import ChatHistoryModel
from dao_ai.middleware._backends import resolve_backend

if TYPE_CHECKING:
    from dao_ai.config import VolumePathModel

__all__ = [
    "SummarizationMiddleware",
    "LoggingSummarizationMiddleware",
    "create_summarization_middleware",
    "create_deep_summarization_middleware",
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
    trigger: tuple[str, int]
    if chat_history.max_tokens_before_summary:
        trigger = ("tokens", chat_history.max_tokens_before_summary)
    elif chat_history.max_messages_before_summary:
        trigger = ("messages", chat_history.max_messages_before_summary)
    else:
        # Default to a reasonable token threshold
        trigger = ("tokens", chat_history.max_tokens * 10)

    # Determine keep condition - how many recent messages/tokens to preserve
    # Default to keeping enough for context
    keep: tuple[str, int] = ("tokens", chat_history.max_tokens)

    logger.info("Summarization middleware configured", trigger=trigger, keep=keep)

    return LoggingSummarizationMiddleware(
        model=model,
        trigger=trigger,
        keep=keep,
    )


def create_deep_summarization_middleware(
    model: str | LanguageModelLike,
    backend_type: str = "state",
    root_dir: str | None = None,
    volume_path: str | VolumePathModel | None = None,
    trigger: tuple[str, Any] | list[tuple[str, Any]] | None = None,
    keep: tuple[str, Any] = ("messages", 20),
    history_path_prefix: str = "/conversation_history",
    truncate_args_trigger: tuple[str, Any] | None = None,
    truncate_args_keep: tuple[str, Any] = ("messages", 20),
    truncate_args_max_length: int = 2000,
) -> DeepAgentsSummarizationMiddleware:
    """
    Create a Deep Agents SummarizationMiddleware with backend offloading.

    This factory function creates a SummarizationMiddleware from the Deep Agents
    library that extends LangChain's with:

    - **Backend offloading**: Persists full conversation history to files before
      summarization, enabling retrieval of full context if needed later.
    - **Tool argument truncation**: Truncates large ``write_file``/``edit_file``
      tool arguments in old messages to save context space.
    - **Fraction-based triggers**: Uses model profile data for intelligent
      trigger/keep thresholds.
    - **Thread-aware storage**: Stores history at
      ``{history_path_prefix}/{thread_id}.md``.

    Args:
        model: The language model for generating summaries. Can be a model
            identifier string (e.g., ``"openai:gpt-4o-mini"``) or a
            ``BaseChatModel`` instance.
        backend_type: Backend for storing offloaded history. One of
            ``"state"`` (ephemeral, default), ``"filesystem"`` (real
            disk), ``"store"`` (persistent), or ``"volume"``
            (Databricks Unity Catalog Volume).
        root_dir: Root directory for filesystem backend. Required when
            ``backend_type="filesystem"``.
        volume_path: Volume path for Databricks Volume backend.
            Required when ``backend_type="volume"``.
        trigger: Threshold(s) for triggering summarization. Can be a single
            tuple like ``("tokens", 100000)`` or ``("fraction", 0.85)``, or a
            list of tuples (any must match). If None, uses deepagents defaults
            based on model profile.
        keep: Context retention policy after summarization. Defaults to keeping
            last 20 messages.
        history_path_prefix: Path prefix for storing conversation history files.
            Defaults to ``"/conversation_history"``.
        truncate_args_trigger: Threshold for triggering argument truncation.
            E.g., ``("messages", 50)`` or ``("fraction", 0.5)``. If None,
            truncation is disabled.
        truncate_args_keep: Context retention for arg truncation. Messages after
            this threshold are preserved without truncation.
        truncate_args_max_length: Maximum character length for tool arguments
            before truncation. Defaults to 2000.

    Returns:
        A configured Deep Agents SummarizationMiddleware instance.

    Example:
        from dao_ai.middleware import create_deep_summarization_middleware

        # Basic usage with token trigger
        middleware = create_deep_summarization_middleware(
            model="gpt-4o-mini",
            trigger=("tokens", 100000),
        )

        # With filesystem backend and arg truncation
        middleware = create_deep_summarization_middleware(
            model="gpt-4o-mini",
            backend_type="filesystem",
            root_dir="/workspace",
            trigger=("fraction", 0.85),
            keep=("fraction", 0.10),
            truncate_args_trigger=("fraction", 0.5),
        )

    YAML Config:
        middleware:
          - name: dao_ai.middleware.summarization.create_deep_summarization_middleware
            args:
              model: "gpt-4o-mini"
              trigger: ["tokens", 100000]
              keep: ["messages", 20]
    """
    backend = resolve_backend(
        backend_type=backend_type,
        root_dir=root_dir,
        volume_path=volume_path,
    )

    # Build truncate_args_settings if trigger is specified
    truncate_args_settings: TruncateArgsSettings | None = None
    if truncate_args_trigger is not None:
        truncate_args_settings = TruncateArgsSettings(
            trigger=truncate_args_trigger,
            keep=truncate_args_keep,
            max_length=truncate_args_max_length,
        )

    logger.debug(
        "Creating Deep Agents SummarizationMiddleware",
        backend_type=backend_type,
        trigger=trigger,
        keep=keep,
        history_path_prefix=history_path_prefix,
        truncate_args_enabled=truncate_args_settings is not None,
    )

    middleware = DeepAgentsSummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=trigger,
        keep=keep,
        history_path_prefix=history_path_prefix,
        truncate_args_settings=truncate_args_settings,
    )

    logger.info(
        "Deep Agents SummarizationMiddleware created",
        backend_type=backend_type,
    )
    return middleware
