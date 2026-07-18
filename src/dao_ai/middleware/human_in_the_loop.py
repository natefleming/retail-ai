"""
Human-in-the-loop middleware for DAO AI agents.

This module provides utilities for creating HITL middleware from DAO AI configuration.
It re-exports LangChain's built-in HumanInTheLoopMiddleware.

LangChain's HumanInTheLoopMiddleware automatically:
- Pauses agent execution for human approval of tool calls
- Allows humans to approve, edit, reject, or respond to tool calls
- Uses LangGraph's interrupt mechanism for persistence across pauses

Example:
    from dao_ai.middleware import create_human_in_the_loop_middleware

    middleware = create_human_in_the_loop_middleware(
        interrupt_on={"send_email": True, "delete_record": True},
    )
"""

from typing import Any, Sequence

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import (
    Action,
    ActionRequest,
    ApproveDecision,
    Decision,
    DecisionType,
    EditDecision,
    HITLRequest,
    HITLResponse,
    InterruptOnConfig,
    RejectDecision,
    ReviewConfig,
)
from loguru import logger

from dao_ai.config import AuditModel, HumanInTheLoopModel, ToolModel

__all__ = [
    # LangChain middleware
    "HumanInTheLoopMiddleware",
    # LangChain HITL types
    "Action",
    "ActionRequest",
    "ApproveDecision",
    "Decision",
    "DecisionType",
    "EditDecision",
    "HITLRequest",
    "HITLResponse",
    "InterruptOnConfig",
    "RejectDecision",
    "ReviewConfig",
    # DAO AI helper functions and models
    "create_human_in_the_loop_middleware",
    "create_hitl_middleware_from_tool_models",
]


def _hitl_config_to_allowed_decisions(
    hitl_config: HumanInTheLoopModel,
) -> list[DecisionType]:
    """
    Extract allowed decisions from HumanInTheLoopModel.

    LangChain's HumanInTheLoopMiddleware supports 4 decision types:
    - "approve": Execute tool with original arguments
    - "edit": Modify arguments before execution
    - "reject": Skip execution with optional feedback message
    - "respond": Reply with a synthetic ToolMessage in place of executing the tool

    Args:
        hitl_config: HumanInTheLoopModel with allowed_decisions

    Returns:
        List of allowed decision types (e.g., ["approve", "edit", "reject", "respond"])
    """
    return hitl_config.allowed_decisions  # type: ignore


def _config_to_interrupt_on_entry(
    config: HumanInTheLoopModel | bool,
) -> dict[str, Any] | bool:
    """
    Convert a HITL config value to interrupt_on entry format.

    Args:
        config: HumanInTheLoopModel, True, or False

    Returns:
        dict with allowed_decisions and optional description, True, or False
    """
    if config is False:
        return False
    if config is True:
        return {"allowed_decisions": ["approve", "edit", "reject"]}
    if isinstance(config, HumanInTheLoopModel):
        interrupt_entry: dict[str, Any] = {
            "allowed_decisions": _hitl_config_to_allowed_decisions(config)
        }
        # If review_prompt is provided, use it as the description
        if config.review_prompt is not None:
            interrupt_entry["description"] = config.review_prompt
        return interrupt_entry

    logger.warning(
        "Unknown HITL config type, defaulting to True",
        config_type=type(config).__name__,
    )
    return True


def create_human_in_the_loop_middleware(
    interrupt_on: dict[str, HumanInTheLoopModel | bool | dict[str, Any]],
    description_prefix: str = "Tool execution pending approval",
) -> HumanInTheLoopMiddleware:
    """
    Create a HumanInTheLoopMiddleware instance.

    Factory function for creating LangChain's built-in HumanInTheLoopMiddleware.
    Accepts HumanInTheLoopModel, bool, or raw dict configurations per tool.

    Note: This middleware requires a checkpointer to be configured on the agent.

    Args:
        interrupt_on: Dictionary mapping tool names to HITL configuration.
            Each tool can be configured with:
            - HumanInTheLoopModel: Full configuration with custom settings
            - True: Enable HITL with default settings (approve, edit, reject; respond opt-in)
            - False: Disable HITL for this tool
            - dict: Raw interrupt_on config (e.g., {"allowed_decisions": [...]})
        description_prefix: Message prefix shown when pausing for review

    Returns:
        List containing HumanInTheLoopMiddleware configured with the specified parameters

    Example:
        from dao_ai.config import HumanInTheLoopModel

        middleware = create_human_in_the_loop_middleware(
            interrupt_on={
                "send_email": HumanInTheLoopModel(review_prompt="Review email"),
                "delete_record": True,
                "search": False,
            },
        )
    """
    # Convert HumanInTheLoopModel entries to dict format
    normalized_interrupt_on: dict[str, Any] = {}
    for tool_name, config in interrupt_on.items():
        if isinstance(config, (HumanInTheLoopModel, bool)):
            normalized_interrupt_on[tool_name] = _config_to_interrupt_on_entry(config)
        else:
            # Already in dict format
            normalized_interrupt_on[tool_name] = config

    logger.debug(
        "Creating HITL middleware",
        tools_count=len(normalized_interrupt_on),
        tools=list(normalized_interrupt_on.keys()),
    )

    return HumanInTheLoopMiddleware(
        interrupt_on=normalized_interrupt_on,
        description_prefix=description_prefix,
    )


def create_hitl_middleware_from_tool_models(
    tool_models: Sequence[ToolModel],
    description_prefix: str = "Tool execution pending approval",
) -> HumanInTheLoopMiddleware | None:
    """
    Create HumanInTheLoopMiddleware from ToolModel configurations.

    Scans tool_models for those with human_in_the_loop configured and
    creates the appropriate middleware. This is the primary entry point
    used by the agent node creation.

    When a HITL-gated tool ALSO has ``audit`` configured, the returned
    middleware is an ``AuditedHumanInTheLoopMiddleware`` which enriches
    the interrupt payload with a nonce + args_hash + displayed_summary
    (all stashed in ``AuditStash`` for the audit-receipt middleware to
    pick up at execution time). Non-audited HITL tools see identical
    behaviour to the vanilla LangChain middleware.

    Args:
        tool_models: List of ToolModel configurations from agent config
        description_prefix: Message prefix shown when pausing for review

    Returns:
        HumanInTheLoopMiddleware (or the AuditedHumanInTheLoopMiddleware
        subclass) if any tools require approval, otherwise ``None``.

    Example:
        from dao_ai.config import ToolModel, PythonFunctionModel, HumanInTheLoopModel

        tool_models = [
            ToolModel(
                name="email_tool",
                function=PythonFunctionModel(
                    name="send_email",
                    human_in_the_loop=HumanInTheLoopModel(
                        review_prompt="Review this email",
                    ),
                ),
            ),
        ]

        middleware = create_hitl_middleware_from_tool_models(tool_models)
    """
    from dao_ai.config import BaseFunctionModel
    from dao_ai.tools import resolve_tool_names

    interrupt_on: dict[str, HumanInTheLoopModel] = {}
    hitl_configs_by_tool: dict[str, HumanInTheLoopModel] = {}
    audited_hitl_tools: dict[str, AuditModel] = {}

    for tool_model in tool_models:
        function = tool_model.function

        if not isinstance(function, BaseFunctionModel):
            continue

        hitl_config: HumanInTheLoopModel | None = function.human_in_the_loop
        if not hitl_config:
            continue

        audit_config: AuditModel | None = function.audit

        # Get tool names created by this function (reuses the agent-build tool
        # registry when available; see dao_ai.tools.resolve_tool_names).
        for tool_name in resolve_tool_names(tool_model):
            if not tool_name:
                continue
            interrupt_on[tool_name] = hitl_config
            hitl_configs_by_tool[tool_name] = hitl_config
            if audit_config is not None:
                audited_hitl_tools[tool_name] = audit_config
                logger.trace(
                    "Tool configured for HITL with audit",
                    tool_name=tool_name,
                    audit_table=audit_config.table,
                )
            else:
                logger.trace("Tool configured for HITL", tool_name=tool_name)

    if not interrupt_on:
        logger.trace("No tools require HITL - returning None")
        return None

    if audited_hitl_tools:
        # Lazy import so the vanilla HITL path never pulls the audit code.
        from dao_ai.middleware.audit_hitl import AuditedHumanInTheLoopMiddleware

        normalized_interrupt_on: dict[str, Any] = {
            tool_name: _config_to_interrupt_on_entry(cfg)
            for tool_name, cfg in interrupt_on.items()
        }
        logger.debug(
            "Creating audited HITL middleware",
            hitl_tool_count=len(normalized_interrupt_on),
            audited_hitl_tool_count=len(audited_hitl_tools),
        )
        return AuditedHumanInTheLoopMiddleware(
            interrupt_on=normalized_interrupt_on,
            audited_tools=audited_hitl_tools,
            hitl_configs=hitl_configs_by_tool,
            description_prefix=description_prefix,
        )

    return create_human_in_the_loop_middleware(
        interrupt_on=interrupt_on,
        description_prefix=description_prefix,
    )
