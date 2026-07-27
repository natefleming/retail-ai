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

import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    Decision,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ReviewConfig,
)
from langchain_core.messages import ToolCall, ToolMessage
from langgraph.runtime import Runtime
from loguru import logger

from dao_ai.audit import (
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
        self._nonce_ttl_by_tool: dict[str, int] = {
            name: cfg.nonce_ttl_seconds for name, cfg in audited_tools.items()
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
    # Override to update the audit stash with the decision + edited-args
    # hash BEFORE the tool executes. LangChain calls this per pending
    # tool_call after the human resume payload is applied. Overriding
    # (non-static) here works because Python's method resolution picks up
    # the subclass method when the base class calls
    # ``self._process_decision(...)``.
    # ------------------------------------------------------------------
    def _process_decision(  # type: ignore[override]
        self,
        decision: Decision,
        tool_call: ToolCall,
        config: InterruptOnConfig,
    ) -> tuple[ToolCall | None, ToolMessage | None]:
        result: tuple[ToolCall | None, ToolMessage | None] = (
            HumanInTheLoopMiddleware._process_decision(decision, tool_call, config)
        )
        tool_name: str = tool_call["name"]
        if tool_name in self._audited_tools:
            self._decorate_stash_with_decision(decision, tool_call)
        return result

    def _decorate_stash_with_decision(
        self,
        decision: Decision,
        tool_call: ToolCall,
    ) -> None:
        """
        Push the resume-time decision + edited-args hash onto the pending
        AuditStash entry so ``AuditReceiptMiddleware`` sees a fully-populated
        approval receipt (and knows to compare against ``edited_args_hash``
        instead of ``args_hash_at_interrupt`` when the decision was
        ``edit``).
        """
        tool_call_id: Optional[str] = tool_call.get("id")
        if tool_call_id is None:
            return
        thread_id: str = self._thread_id_from_stash(tool_call_id)
        if thread_id == "unknown-thread":
            # No stash entry exists (e.g. process restart between interrupt
            # and resume) — nothing to decorate.
            return
        entry: Optional[AuditStashEntry] = AuditStash.take(thread_id, tool_call_id)
        if entry is None:
            return
        decision_type: Any = (
            decision.get("type") if isinstance(decision, dict) else None
        )
        if isinstance(decision_type, str):
            entry.decision = decision_type
        # Copy every field except ``type`` into decision_detail so the
        # receipt records the edited action, reject message, respond text,
        # etc. verbatim.
        detail: dict[str, Any] = {
            k: v for k, v in (decision or {}).items() if k != "type"
        }
        entry.decision_detail = detail or None
        if entry.confirmed_via is None:
            entry.confirmed_via = "chat_ui"
        # For ``edit`` decisions, capture the edited canonical args + hash
        # so the middleware knows the legitimate new hash to expect.
        if decision_type == "edit" and isinstance(detail.get("edited_action"), dict):
            edited: dict[str, Any] = detail["edited_action"]
            edited_args: Any = edited.get("args") if isinstance(edited, dict) else None
            if isinstance(edited_args, dict):
                edited_jcs: str = canonical_jcs(edited_args)
                entry.edited_args_jcs = edited_jcs
                entry.edited_args_hash = args_hash_of(edited_args)
        AuditStash.put(thread_id, tool_call_id, entry)

    @staticmethod
    def _thread_id_from_stash(tool_call_id: str) -> str:
        """
        Recover the thread_id under which the interrupt-time stash was
        stored, delegating to
        :meth:`AuditStash.find_thread_id_by_tool_call_id` for the
        encapsulated lookup. Returns ``"unknown-thread"`` when no entry
        matches (e.g. process restart between interrupt and resume).
        """
        found: Optional[str] = AuditStash.find_thread_id_by_tool_call_id(tool_call_id)
        return found if found is not None else "unknown-thread"

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
        nonce, nonce_exp = self._issue_local_nonce(tool_name)

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
                tool_name=tool_name,
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

    def _issue_local_nonce(self, tool_name: str) -> tuple[str, datetime]:
        """
        Generate a nonce entirely in-process — no DB write.

        The interrupt-raising call site is synchronous (LangChain's
        ``after_model`` runs sync even under an async graph). Attempting to
        block on a Lakebase INSERT here would need ``nest_asyncio``, which
        is incompatible with ``uvloop`` (raises
        ``Can't patch loop of type <class 'uvloop.Loop'>``). Uvicorn uses
        uvloop by default under Databricks Apps, so we keep the nonce
        server-local for v1.

        The single-use guarantee is enforced by :class:`AuditStash.take`
        (per-process, per ``(thread_id, tool_call_id)``). Cross-process
        persistence + atomic DB single-use lands in v1.5 as documented in
        ``docs/audit.md`` under "Known limitations".
        """
        ttl_seconds: int = self._nonce_ttl_by_tool.get(tool_name, 300)
        nonce: str = secrets.token_urlsafe(32)
        exp: datetime = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        return nonce, exp
