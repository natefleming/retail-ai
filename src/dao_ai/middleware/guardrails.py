"""
Guardrail middleware for DAO AI agents.

This module provides middleware implementations for applying guardrails
to agent responses, including LLM-based judging and content validation.

Factory functions are provided for consistent configuration via the
DAO AI middleware factory pattern.
"""

from typing import Any, Optional

from langchain.agents.middleware import hook_config
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime
from loguru import logger
from openevals.llm import create_llm_as_judge

from dao_ai.messages import last_ai_message, last_human_message
from dao_ai.middleware.base import AgentMiddleware
from dao_ai.state import AgentState, Context


def _extract_text_content(message: BaseMessage) -> str:
    """
    Extract text content from a message, handling both string and list formats.

    Args:
        message: The message to extract text from

    Returns:
        The extracted text content as a string
    """
    content = message.content

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Extract text from content blocks (e.g., Claude's structured content)
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return " ".join(text_parts)
    else:
        return str(content)


__all__ = [
    "GuardrailMiddleware",
    "ContentFilterMiddleware",
    "SafetyGuardrailMiddleware",
    "create_guardrail_middleware",
    "create_content_filter_middleware",
    "create_safety_guardrail_middleware",
]


class GuardrailMiddleware(AgentMiddleware[AgentState, Context]):
    """
    Middleware that applies LLM-based guardrails to agent responses.

    Uses an LLM judge to evaluate responses against a prompt/criteria and
    can request improvements if the response doesn't meet the criteria.

    This is equivalent to the previous reflection_guardrail pattern but
    implemented as middleware for better composability.

    Args:
        guardrail_name: Name identifying this guardrail
        model: The LLM to use for evaluation
        prompt: The evaluation prompt/criteria
        num_retries: Maximum number of retry attempts (default: 3)
    """

    def __init__(
        self,
        name: str,
        model: LanguageModelLike,
        prompt: str,
        num_retries: int = 3,
    ):
        super().__init__()
        self.guardrail_name = name
        self.model = model
        self.prompt = prompt
        self.num_retries = num_retries
        self._retry_count = 0

    @property
    def name(self) -> str:
        """Return the guardrail name for middleware identification."""
        return self.guardrail_name

    def after_model(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """
        Evaluate the model's response using an LLM judge.

        If the response doesn't meet the guardrail criteria, returns a
        HumanMessage with feedback to trigger a retry.
        """
        messages: list[BaseMessage] = state.get("messages", [])

        if not messages:
            return None

        ai_message: AIMessage | None = last_ai_message(messages)
        human_message: HumanMessage | None = last_human_message(messages)

        if not ai_message or not human_message:
            return None

        # Skip evaluation if the AI message has tool calls (not the final response yet)
        if ai_message.tool_calls:
            logger.trace(
                "Guardrail skipping evaluation - AI message contains tool calls",
                guardrail_name=self.guardrail_name,
            )
            return None

        # Skip evaluation if the AI message has no content to evaluate
        if not ai_message.content:
            logger.trace(
                "Guardrail skipping evaluation - AI message has no content",
                guardrail_name=self.guardrail_name,
            )
            return None

        # Extract text content from messages (handles both string and structured content)
        human_content = _extract_text_content(human_message)
        ai_content = _extract_text_content(ai_message)

        logger.debug(
            "Evaluating response with guardrail",
            guardrail_name=self.guardrail_name,
            input_length=len(human_content),
            output_length=len(ai_content),
        )

        evaluator = create_llm_as_judge(
            prompt=self.prompt,
            judge=self.model,
        )

        eval_result = evaluator(inputs=human_content, outputs=ai_content)

        if eval_result["score"]:
            logger.debug(
                "Response approved by guardrail",
                guardrail_name=self.guardrail_name,
                comment=eval_result["comment"],
            )
            self._retry_count = 0
            return None
        else:
            self._retry_count += 1
            comment: str = eval_result["comment"]

            if self._retry_count >= self.num_retries:
                logger.warning(
                    "Guardrail failed - max retries reached",
                    guardrail_name=self.guardrail_name,
                    retry_count=self._retry_count,
                    max_retries=self.num_retries,
                    critique=comment,
                )
                self._retry_count = 0

                # Add system message to inform user of guardrail failure
                failure_message = (
                    f"⚠️ **Quality Check Failed**\n\n"
                    f"The response did not meet the '{self.guardrail_name}' quality standards "
                    f"after {self.num_retries} attempts.\n\n"
                    f"**Issue:** {comment}\n\n"
                    f"The best available response has been provided, but please be aware it may not fully meet quality expectations."
                )
                return {"messages": [AIMessage(content=failure_message)]}

            logger.warning(
                "Guardrail requested improvements",
                guardrail_name=self.guardrail_name,
                retry=self._retry_count,
                max_retries=self.num_retries,
                critique=comment,
            )

            content: str = "\n".join([str(human_message.content), comment])
            return {"messages": [HumanMessage(content=content)]}


class ContentFilterMiddleware(AgentMiddleware[AgentState, Context]):
    """
    Middleware that filters responses containing banned keywords.

    This is a deterministic guardrail that blocks responses containing
    specified keywords.

    Args:
        banned_keywords: List of keywords to block
        block_message: Message to return when content is blocked
    """

    def __init__(
        self,
        banned_keywords: list[str],
        block_message: str = "I cannot provide that response. Please rephrase your request.",
    ):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]
        self.block_message = block_message

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Block requests containing banned keywords."""
        messages: list[BaseMessage] = state.get("messages", [])

        if not messages:
            return None

        first_message = messages[0]
        if not isinstance(first_message, HumanMessage):
            return None

        content = str(first_message.content).lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                logger.warning(f"Content filter blocked request containing '{keyword}'")
                return {
                    "messages": [AIMessage(content=self.block_message)],
                    "jump_to": "end",
                }

        return None

    def after_model(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Block responses containing banned keywords."""
        messages: list[BaseMessage] = state.get("messages", [])

        if not messages:
            return None

        last_message: AIMessage | None = last_ai_message(messages)
        if not last_message:
            return None

        content = str(last_message.content).lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                logger.warning("Content filter blocked response", keyword=keyword)
                # Modify the last message content
                last_message.content = self.block_message
                return None

        return None


class SafetyGuardrailMiddleware(AgentMiddleware[AgentState, Context]):
    """
    Model-based safety guardrail middleware.

    Uses an LLM to evaluate whether responses are safe and appropriate.
    This is applied after the agent completes to validate the final output.

    Args:
        safety_model: The LLM to use for safety evaluation
    """

    def __init__(self, safety_model: Optional[LanguageModelLike] = None):
        super().__init__()
        self._safety_model = safety_model

    @property
    def safety_model(self) -> LanguageModelLike:
        """Lazily initialize the safety model."""
        if self._safety_model is None:
            from langchain.chat_models import init_chat_model

            self._safety_model = init_chat_model("gpt-4o-mini")
        return self._safety_model

    def after_agent(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Evaluate response safety before returning to user."""
        messages: list[BaseMessage] = state.get("messages", [])

        if not messages:
            return None

        last_message: AIMessage | None = last_ai_message(messages)
        if not last_message:
            return None

        safety_prompt = f"""Evaluate if this response is safe and appropriate.
        Respond with only 'SAFE' or 'UNSAFE'.
        
        Response: {last_message.content}"""

        result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])

        if "UNSAFE" in str(result.content):
            logger.warning("Safety guardrail blocked unsafe response")
            last_message.content = (
                "I cannot provide that response. Please rephrase your request."
            )

        return None


# =============================================================================
# Factory Functions
# =============================================================================


def create_guardrail_middleware(
    name: str,
    model: LanguageModelLike,
    prompt: str,
    num_retries: int = 3,
) -> GuardrailMiddleware:
    """
    Create a GuardrailMiddleware instance.

    Factory function for creating LLM-based guardrail middleware that evaluates
    agent responses against specified criteria using an LLM judge.

    Args:
        name: Name identifying this guardrail
        model: The LLM to use for evaluation
        prompt: The evaluation prompt/criteria
        num_retries: Maximum number of retry attempts (default: 3)

    Returns:
        List containing GuardrailMiddleware configured with the specified parameters

    Example:
        middleware = create_guardrail_middleware(
            name="tone_check",
            model=ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct"),
            prompt="Evaluate if the response is professional and helpful.",
            num_retries=2,
        )
    """
    logger.trace("Creating guardrail middleware", guardrail_name=name)
    return GuardrailMiddleware(
        name=name,
        model=model,
        prompt=prompt,
        num_retries=num_retries,
    )


def create_content_filter_middleware(
    banned_keywords: list[str],
    block_message: str = "I cannot provide that response. Please rephrase your request.",
) -> ContentFilterMiddleware:
    """
    Create a ContentFilterMiddleware instance.

    Factory function for creating deterministic content filter middleware
    that blocks requests/responses containing banned keywords.

    Args:
        banned_keywords: List of keywords to block
        block_message: Message to return when content is blocked

    Returns:
        List containing ContentFilterMiddleware configured with the specified parameters

    Example:
        middleware = create_content_filter_middleware(
            banned_keywords=["password", "secret", "api_key"],
            block_message="I cannot discuss sensitive credentials.",
        )
    """
    logger.trace(
        "Creating content filter middleware", keywords_count=len(banned_keywords)
    )
    return ContentFilterMiddleware(
        banned_keywords=banned_keywords,
        block_message=block_message,
    )


def create_safety_guardrail_middleware(
    safety_model: Optional[LanguageModelLike] = None,
) -> SafetyGuardrailMiddleware:
    """
    Create a SafetyGuardrailMiddleware instance.

    Factory function for creating model-based safety guardrail middleware
    that evaluates whether responses are safe and appropriate.

    Args:
        safety_model: The LLM to use for safety evaluation. If not provided,
            defaults to gpt-4o-mini.

    Returns:
        List containing SafetyGuardrailMiddleware configured with the specified model

    Example:
        from databricks_langchain import ChatDatabricks

        middleware = create_safety_guardrail_middleware(
            safety_model=ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct"),
        )
    """
    logger.trace("Creating safety guardrail middleware")
    return SafetyGuardrailMiddleware(safety_model=safety_model)
