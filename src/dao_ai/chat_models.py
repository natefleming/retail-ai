import json
from typing import Any, Iterator, Optional, Sequence

from databricks_langchain import ChatDatabricks
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from loguru import logger


class ChatDatabricksFiltered(ChatDatabricks):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _preprocess_messages(
        self, messages: Sequence[BaseMessage]
    ) -> Sequence[BaseMessage]:
        logger.debug(f"Preprocessing {len(messages)} messages for filtering")

        logger.trace(
            f"Original messages:\n{json.dumps([msg.model_dump() for msg in messages], indent=2)}"
        )

        # Diagnostic logging to understand what types of messages we're getting
        message_types = {}
        remove_message_count = 0
        empty_content_count = 0

        for msg in messages:
            msg_type = msg.__class__.__name__
            message_types[msg_type] = message_types.get(msg_type, 0) + 1

            if msg_type == "RemoveMessage":
                remove_message_count += 1
            elif hasattr(msg, "content") and (msg.content == "" or msg.content is None):
                empty_content_count += 1

        logger.debug(f"Message type breakdown: {message_types}")
        logger.debug(
            f"RemoveMessage count: {remove_message_count}, Empty content count: {empty_content_count}"
        )

        filtered_messages = []
        for i, msg in enumerate(messages):
            # First, filter out RemoveMessage objects completely - they're LangGraph-specific
            # and should never be sent to an LLM
            if hasattr(msg, "__class__") and msg.__class__.__name__ == "RemoveMessage":
                logger.debug(f"Filtering out RemoveMessage at index {i}")
                continue

            # Be very conservative with filtering - only filter out messages that are:
            # 1. Have empty or None content AND
            # 2. Are not tool-related messages AND
            # 3. Don't break tool_use/tool_result pairing
            # 4. Are not the only remaining message (to avoid filtering everything)
            has_empty_content = hasattr(msg, "content") and (
                msg.content == "" or msg.content is None
            )

            # Check if this message has tool calls (non-empty list)
            has_tool_calls = (
                hasattr(msg, "tool_calls")
                and msg.tool_calls
                and len(msg.tool_calls) > 0
            )

            # Check if this is a tool result message
            is_tool_result = hasattr(msg, "tool_call_id") or isinstance(
                msg, ToolMessage
            )

            # Check if the previous message had tool calls (this message might be a tool result)
            prev_had_tool_calls = False
            if i > 0:
                prev_msg = messages[i - 1]
                prev_had_tool_calls = (
                    hasattr(prev_msg, "tool_calls")
                    and prev_msg.tool_calls
                    and len(prev_msg.tool_calls) > 0
                )

            # Check if the next message is a tool result (this message might be a tool use)
            next_is_tool_result = False
            if i < len(messages) - 1:
                next_msg = messages[i + 1]
                next_is_tool_result = hasattr(next_msg, "tool_call_id") or isinstance(
                    next_msg, ToolMessage
                )

            # Special handling for empty AIMessages - they might be placeholders or incomplete responses
            # Don't filter them if they're the only AI response or seem important to the conversation flow
            is_empty_ai_message = has_empty_content and isinstance(msg, AIMessage)

            # Only filter out messages with empty content that are definitely not needed
            should_filter = (
                has_empty_content
                and not has_tool_calls
                and not is_tool_result
                and not prev_had_tool_calls  # Don't filter if previous message had tool calls
                and not next_is_tool_result  # Don't filter if next message is a tool result
                and not (
                    is_empty_ai_message and len(messages) <= 2
                )  # Don't filter empty AI messages in short conversations
            )

            if should_filter:
                logger.debug(f"Filtering out message at index {i}: {msg.model_dump()}")
                continue
            else:
                filtered_messages.append(msg)

        logger.debug(
            f"Filtered {len(messages)} messages down to {len(filtered_messages)} messages"
        )

        # Log diagnostic information if all messages were filtered out
        if len(filtered_messages) == 0:
            logger.warning(
                f"All {len(messages)} messages were filtered out! This indicates a problem with the conversation state."
            )
            logger.debug(f"Original message types: {message_types}")

            if remove_message_count == len(messages):
                logger.warning(
                    "All messages were RemoveMessage objects - this suggests a bug in summarization logic"
                )
            elif empty_content_count > 0:
                logger.debug(f"{empty_content_count} messages had empty content")

        return filtered_messages

    def _postprocess_message(self, message: BaseMessage) -> BaseMessage:
        return message

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override _generate to apply message preprocessing and postprocessing."""
        # Apply message preprocessing
        processed_messages: Sequence[BaseMessage] = self._preprocess_messages(messages)

        if len(processed_messages) == 0:
            logger.error(
                "All messages were filtered out during preprocessing. This indicates a serious issue with the conversation state."
            )
            empty_generation = ChatGeneration(
                message=AIMessage(content="", id="empty-response")
            )
            return ChatResult(generations=[empty_generation])

        logger.trace(
            f"Processed messages:\n{json.dumps([msg.model_dump() for msg in processed_messages], indent=2)}"
        )

        result: ChatResult = super()._generate(
            processed_messages, stop, run_manager, **kwargs
        )

        if result.generations:
            for generation in result.generations:
                if isinstance(generation, ChatGeneration) and generation.message:
                    generation.message = self._postprocess_message(generation.message)

        return result

    def _stream(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """Override _stream to apply message preprocessing and postprocessing."""
        # Apply message preprocessing
        processed_messages: Sequence[BaseMessage] = self._preprocess_messages(messages)

        # Handle the edge case where all messages were filtered out
        if len(processed_messages) == 0:
            logger.error(
                "All messages were filtered out during preprocessing. This indicates a serious issue with the conversation state."
            )
            # Return an empty streaming result without calling the underlying API
            # This prevents API errors while making the issue visible through an empty response
            empty_chunk = ChatGenerationChunk(
                message=AIMessage(content="", id="empty-response")
            )
            yield empty_chunk
            return

        logger.trace(
            f"Processed messages:\n{json.dumps([msg.model_dump() for msg in processed_messages], indent=2)}"
        )

        # Call the parent ChatDatabricks implementation
        for chunk in super()._stream(processed_messages, stop, run_manager, **kwargs):
            chunk: ChatGenerationChunk
            # Apply message postprocessing to each chunk
            if isinstance(chunk, ChatGeneration) and chunk.message:
                chunk.message = self._postprocess_message(chunk.message)
            yield chunk
