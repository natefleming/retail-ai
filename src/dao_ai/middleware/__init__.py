# DAO AI Middleware Module
# This module provides middleware implementations compatible with LangChain v1's create_agent

# Re-export LangChain built-in middleware
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    SummarizationMiddleware,
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)

# DSPy-style assertion middleware
from dao_ai.middleware.assertions import (
    # Middleware classes
    AssertMiddleware,
    # Types
    Constraint,
    ConstraintResult,
    FunctionConstraint,
    KeywordConstraint,
    LengthConstraint,
    LLMConstraint,
    RefineMiddleware,
    SuggestMiddleware,
    # Factory functions
    create_assert_middleware,
    create_refine_middleware,
    create_suggest_middleware,
)
from dao_ai.middleware.base import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from dao_ai.middleware.core import create_factory_middleware
from dao_ai.middleware.guardrails import (
    ContentFilterMiddleware,
    GuardrailMiddleware,
    SafetyGuardrailMiddleware,
    create_content_filter_middleware,
    create_guardrail_middleware,
    create_safety_guardrail_middleware,
)
from dao_ai.middleware.human_in_the_loop import (
    create_hitl_middleware_from_tool_models,
    create_human_in_the_loop_middleware,
)
from dao_ai.middleware.message_validation import (
    CustomFieldValidationMiddleware,
    FilterLastHumanMessageMiddleware,
    MessageValidationMiddleware,
    RequiredField,
    ThreadIdValidationMiddleware,
    UserIdValidationMiddleware,
    create_custom_field_validation_middleware,
    create_filter_last_human_message_middleware,
    create_thread_id_validation_middleware,
    create_user_id_validation_middleware,
)
from dao_ai.middleware.summarization import (
    LoggingSummarizationMiddleware,
    create_summarization_middleware,
)

__all__ = [
    # Base class (from LangChain)
    "AgentMiddleware",
    # Types
    "ModelRequest",
    "ModelResponse",
    # LangChain decorators
    "before_agent",
    "before_model",
    "after_agent",
    "after_model",
    "wrap_model_call",
    "wrap_tool_call",
    "dynamic_prompt",
    # LangChain built-in middleware
    "SummarizationMiddleware",
    "LoggingSummarizationMiddleware",
    "HumanInTheLoopMiddleware",
    # Core factory function
    "create_factory_middleware",
    # DAO AI middleware implementations
    "GuardrailMiddleware",
    "ContentFilterMiddleware",
    "SafetyGuardrailMiddleware",
    "MessageValidationMiddleware",
    "UserIdValidationMiddleware",
    "ThreadIdValidationMiddleware",
    "CustomFieldValidationMiddleware",
    "RequiredField",
    "FilterLastHumanMessageMiddleware",
    # DSPy-style assertion middleware
    "Constraint",
    "ConstraintResult",
    "FunctionConstraint",
    "KeywordConstraint",
    "LengthConstraint",
    "LLMConstraint",
    "AssertMiddleware",
    "SuggestMiddleware",
    "RefineMiddleware",
    # DAO AI middleware factory functions
    "create_guardrail_middleware",
    "create_content_filter_middleware",
    "create_safety_guardrail_middleware",
    "create_user_id_validation_middleware",
    "create_thread_id_validation_middleware",
    "create_custom_field_validation_middleware",
    "create_filter_last_human_message_middleware",
    "create_summarization_middleware",
    "create_human_in_the_loop_middleware",
    "create_hitl_middleware_from_tool_models",
    # DSPy-style assertion factory functions
    "create_assert_middleware",
    "create_suggest_middleware",
    "create_refine_middleware",
]
