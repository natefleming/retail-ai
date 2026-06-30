"""Context-arg-bind middleware for DAO AI agents.

Opt-in middleware that backfills missing tool-call arguments from
runtime ``Context`` fields, using the same ``{placeholder}`` template
syntax dao-ai already uses for system prompts (see
``dao_ai/prompts/__init__.py::make_prompt``).

Configuration is per-agent via YAML — no framework-baked argument or
context-field names. Example:

.. code-block:: yaml

    agents:
      - name: order_history
        middleware:
          - name: dao_ai.middleware.create_context_arg_bind_middleware
            args:
              bindings:
                customer_id: "{user_id}"
                # multi-placeholder template also works
                # idempotency_key: "{user_id}-{thread_id}"

Why this exists
---------------
LangGraph v3 streaming dispatches tool calls before all argument
chunks have accumulated. UC function tools then receive ``None`` for
required parameters and reject with errors like
``Parameter customer_id should be of type STRING ... got NoneType``.

The LLM is told via the system prompt what the value should be (the
prompt-template engine renders ``{user_id}`` -> the actual user_id
before the LLM ever sees the prompt). The middleware adds a second
layer of defense by filling the same value into the tool_call dict
just before the tool executes, so the LLM's emit-time omission can't
break the call.

Coverage guarantees
-------------------
The middleware is designed to work cleanly when:

- An agent has **multiple tools from different resources / types**
  (UC functions, Vector Search, MCP, REST, custom): the binding only
  applies to tools whose ``args_schema`` declares the bound parameter.
  Other tools pass through untouched.
- An agent fires **multiple tool calls in one turn**: ``wrap_tool_call``
  is invoked separately per call; each invocation operates on its own
  local args dict.
- The runtime executes **tool calls concurrently** (e.g. ``ToolNode``
  with ``asyncio.gather``): the middleware is stateless after
  construction — bindings live on ``self`` and are never mutated — so
  parallel invocations don't interfere.
- The LLM **does** emit a value for the bound arg: that value wins;
  the middleware only fills ``None`` / empty.
- A binding template references a **missing context field**: a warning
  is logged and that binding is skipped for the current call. The tool
  still runs with whatever the LLM emitted.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel

from dao_ai.state import AgentState, Context

__all__ = [
    "ContextArgBindMiddleware",
    "create_context_arg_bind_middleware",
]


def _accepted_args(tool: BaseTool | None) -> frozenset[str]:
    """Return the set of arg names the tool's args_schema declares.

    Returns an empty set if the tool has no schema (which means we
    can't safely add args) — the binding step will then skip every
    declared binding for this tool.
    """
    if tool is None:
        return frozenset()
    schema: type[BaseModel] | None = getattr(tool, "args_schema", None)
    if schema is None:
        return frozenset()
    fields = getattr(schema, "model_fields", None)
    if fields is None:
        return frozenset()
    return frozenset(fields.keys())


class ContextArgBindMiddleware(AgentMiddleware[AgentState, Context]):
    """Backfill tool-call args from runtime ``Context`` using
    config-declared bindings.

    Args:
        bindings: Mapping of ``tool-arg-name -> "{context-field}"`` template.
            The template uses Python ``str.format`` against
            ``runtime.context.model_dump()`` — any field on
            :class:`dao_ai.state.Context` (or extras since ``extra="allow"``)
            is referenceable. Multiple placeholders in one template work
            (e.g. ``"{user_id}-{thread_id}"``).
    """

    def __init__(self, bindings: dict[str, str]) -> None:
        super().__init__()
        # Frozen at construction — never mutated. Safe under concurrent
        # tool-call invocations.
        self._bindings: dict[str, str] = dict(bindings)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], "ToolMessage | Command[Any]"],
    ) -> "ToolMessage | Command[Any]":
        return handler(self._apply(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable["ToolMessage | Command[Any]"]],
    ) -> "ToolMessage | Command[Any]":
        return await handler(self._apply(request))

    def _apply(self, request: ToolCallRequest) -> ToolCallRequest:
        if not self._bindings:
            return request
        tool: BaseTool | None = request.tool
        accepted: frozenset[str] = _accepted_args(tool)
        # If the tool declares no schema, we can't know which args are
        # legal — skip rather than risk injecting an unknown kwarg.
        if not accepted:
            return request
        tool_call: dict[str, Any] = request.tool_call
        original_args: dict[str, Any] = tool_call.get("args") or {}
        args: dict[str, Any] = dict(original_args)
        context_dict: dict[str, Any] = {}
        runtime = request.runtime
        if runtime is not None and runtime.context is not None:
            context_dict = runtime.context.model_dump()
        patched: dict[str, Any] = {}
        for arg_name, template in self._bindings.items():
            if arg_name not in accepted:
                # Tool doesn't accept this arg — never touch it.
                continue
            if args.get(arg_name) not in (None, ""):
                # LLM provided a value — never override.
                continue
            try:
                args[arg_name] = template.format(**context_dict)
            except KeyError as missing:
                logger.warning(
                    "ContextArgBindMiddleware: skipped binding — "
                    "context field not available",
                    tool_name=tool_call.get("name"),
                    arg_name=arg_name,
                    template=template,
                    missing_field=str(missing).strip("'"),
                )
                continue
            patched[arg_name] = args[arg_name]
        if not patched:
            return request
        logger.info(
            "ContextArgBindMiddleware: filled tool-call args from context",
            tool_name=tool_call.get("name"),
            patched_keys=list(patched.keys()),
        )
        new_tool_call: dict[str, Any] = {**tool_call, "args": args}
        return request.override(tool_call=new_tool_call)


def create_context_arg_bind_middleware(
    bindings: dict[str, str],
) -> ContextArgBindMiddleware:
    """Factory matching the dao-ai middleware FQN pattern.

    Used from YAML:

    .. code-block:: yaml

        middleware:
          - name: dao_ai.middleware.create_context_arg_bind_middleware
            args:
              bindings:
                customer_id: "{user_id}"
    """
    return ContextArgBindMiddleware(bindings=bindings)
