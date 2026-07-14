"""
Audit-aware subclass of LangChain's ``HumanInTheLoopMiddleware``.

When a tool has both ``human_in_the_loop`` AND ``audit`` set on its
function block, this subclass runs in place of the vanilla HITL middleware
and enriches the interrupt payload with intent-verification material:

- ``args_hash_at_interrupt`` — SHA-256(JCS(args)) captured before the
  interrupt is raised.
- ``nonce`` + ``nonce_exp`` — server-issued single-use identifier bound
  to ``(thread_id, tool_call_id)``.
- ``displayed_summary`` — harness-rendered summary shown to the reviewer.

All three land on ``AuditStash`` keyed by ``(thread_id, tool_call_id)``
where the ``AuditReceiptMiddleware`` picks them up when the tool executes
(fail-closed args-hash recheck + full receipt fields). Tools with HITL but
without audit fall back to the parent implementation with zero behavioural
delta.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ReviewConfig,
)
from langchain_core.messages import ToolCall
from langgraph.runtime import Runtime
from loguru import logger

from dao_ai.audit import (
    AuditSinkManager,
    LakebaseAuditSink,
    args_hash_of,
    canonical_jcs,
)
from dao_ai.middleware.audit_receipt import AuditStash, AuditStashEntry
from dao_ai.state import Context

if TYPE_CHECKING:
    from dao_ai.config import AuditModel, HumanInTheLoopModel
    from dao_ai.state import AgentState


__all__ = ["AuditedHumanInTheLoopMiddleware"]


class AuditedHumanInTheLoopMiddleware(HumanInTheLoopMiddleware):
    """
    LangChain HITL middleware augmented with audit-stash enrichment.

    Parameters mirror ``HumanInTheLoopMiddleware`` plus two extras:

    - ``audited_tools``: mapping of tool name → resolved ``AuditModel``.
      Any tool present in both ``interrupt_on`` and ``audited_tools``
      triggers the enrichment path.
    - ``hitl_configs``: mapping of tool name → ``HumanInTheLoopModel`` used
      to look up ``review_prompt`` when rendering ``displayed_summary``.
    """

    def __init__(
        self,
        *,
        interrupt_on: dict[str, Any],
        audited_tools: dict[str, "AuditModel"],
        hitl_configs: dict[str, "HumanInTheLoopModel"],
        description_prefix: str = "Tool execution pending approval",
    ) -> None:
        super().__init__(
            interrupt_on=interrupt_on,
            description_prefix=description_prefix,
        )
        self._audited_tools: dict[str, "AuditModel"] = dict(audited_tools)
        self._hitl_configs: dict[str, "HumanInTheLoopModel"] = dict(hitl_configs)
        self._sinks_by_tool: dict[str, LakebaseAuditSink] = {
            tool_name: AuditSinkManager.for_config(audit_model)
            for tool_name, audit_model in audited_tools.items()
        }

    # ------------------------------------------------------------------
    # Override the exact per-tool-call hook LangChain exposes so both
    # sync `after_model` and async `aafter_model` benefit uniformly.
    # ------------------------------------------------------------------
    def _create_action_and_config(
        self,
        tool_call: ToolCall,
        config: InterruptOnConfig,
        state: "AgentState[Any]",  # type: ignore[type-arg]
        runtime: Runtime[Context],
    ) -> tuple[ActionRequest, ReviewConfig]:
        action_request, review_config = super()._create_action_and_config(
            tool_call, config, state, runtime
        )
        tool_name: str = tool_call["name"]
        if tool_name in self._audited_tools:
            self._enrich_and_stash(tool_call, action_request, runtime)
        return action_request, review_config

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _enrich_and_stash(
        self,
        tool_call: ToolCall,
        action_request: ActionRequest,
        runtime: Runtime[Context],
    ) -> None:
        tool_name: str = tool_call["name"]
        tool_call_id: Optional[str] = tool_call.get("id")
        if tool_call_id is None:
            logger.warning(
                "Audit-HITL enrichment skipped — no tool_call_id",
                tool_name=tool_name,
            )
            return

        args: dict[str, Any] = tool_call.get("args") or {}
        args_jcs: str = canonical_jcs(args)
        args_hash: str = args_hash_of(args)

        thread_id: str = self._thread_id_for(runtime)
        sink: LakebaseAuditSink = self._sinks_by_tool[tool_name]

        nonce, nonce_exp = self._issue_nonce_sync(
            sink=sink, thread_id=thread_id, tool_call_id=tool_call_id
        )

        hitl_config: Optional["HumanInTheLoopModel"] = self._hitl_configs.get(tool_name)
        displayed_summary: str = self._render_displayed_summary(
            tool_name=tool_name,
            args_jcs=args_jcs,
            review_prompt=(
                hitl_config.review_prompt if hitl_config is not None else None
            ),
        )

        AuditStash.put(
            thread_id,
            tool_call_id,
            AuditStashEntry(
                args_hash_at_interrupt=args_hash,
                nonce=nonce,
                nonce_exp=nonce_exp,
                displayed_summary=displayed_summary,
            ),
        )
        # Give the reviewer a stable hash prefix in the visible payload so
        # the receipt and the UI reference the same intent identifier.
        prefix: str = args_hash[:8]
        base_description: str = action_request.get("description", "")
        action_request["description"] = (
            f"{base_description}\n\nintent-hash: {prefix} nonce-prefix: {nonce[:8]}"
        )
        logger.debug(
            "Audit-HITL stash populated",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args_hash_prefix=prefix,
        )

    def _render_displayed_summary(
        self,
        *,
        tool_name: str,
        args_jcs: str,
        review_prompt: Optional[str],
    ) -> str:
        """
        Build the summary string that will be persisted verbatim on the
        receipt. This is harness-generated (never model-generated) per the
        intent-verification research: the model must not be able to lie
        to the user about what will execute.
        """
        prompt_line: str = review_prompt or f"Review tool call to `{tool_name}`."
        # Show the reviewer-friendly rendering (indented JSON) but bind the
        # audit-relevant hash over the byte-exact JCS form.
        pretty_args: str = json.dumps(json.loads(args_jcs), indent=2)
        return f"{prompt_line}\n\nTool: {tool_name}\nArgs:\n{pretty_args}"

    def _thread_id_for(self, runtime: Runtime[Context]) -> str:
        context: Context = runtime.context
        if isinstance(context.thread_id, str) and context.thread_id:
            return context.thread_id
        # Fall back to the RunnableConfig — HITL is always driven with one.
        configurable_any: Any = runtime.config.get("configurable")
        if isinstance(configurable_any, dict):
            candidate: Any = configurable_any.get("thread_id")
            if isinstance(candidate, str) and candidate:
                return candidate
        return "unknown-thread"

    def _issue_nonce_sync(
        self,
        *,
        sink: LakebaseAuditSink,
        thread_id: str,
        tool_call_id: str,
    ) -> tuple[str, datetime]:
        """
        Issue a nonce from inside the sync ``after_model`` path.

        LangChain calls this middleware synchronously even under an async
        graph run. nest_asyncio is already installed in dao-ai's serving
        entry points, which makes ``loop.run_until_complete`` on an active
        loop safe. If we're outside a loop (e.g. in unit tests) we fall
        back to ``asyncio.run``.
        """
        coro = sink.nonces.issue(thread_id=thread_id, tool_call_id=tool_call_id)
        try:
            loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:  # pragma: no cover — nest_asyncio is a hard dep
            logger.warning(
                "nest_asyncio not available; nonce issuance may block the event loop"
            )
        return loop.run_until_complete(coro)
