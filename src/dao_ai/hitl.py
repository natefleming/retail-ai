"""Shared human-in-the-loop (HITL) decision logic for dao-ai agents.

Both the OpenAI Responses path (:class:`dao_ai.models.LanggraphResponsesAgent`)
and the Google A2A path (:class:`dao_ai.apps.a2a.executor.A2AAgentExecutor`)
need to decide, for each incoming request, whether the next graph turn is a
fresh invocation or a resume from a prior :func:`langgraph.types.interrupt`.
This module owns that decision so the protocols stay in lock-step.

The three call sites (``apredict``, ``apredict_stream``, A2A executor) used
to duplicate ~75 lines of decision logic each; they now share
:func:`decide_graph_turn`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from langchain_community.adapters.openai import convert_openai_messages
from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from loguru import logger

from dao_ai.config import AuditModel, BaseFunctionModel, ToolModel


@dataclass
class GraphTurn:
    """Result of deciding how to drive the graph for one request.

    Exactly one of the three fields is populated:

    * ``resume_command`` — caller invokes ``graph.ainvoke(resume_command, ...)``
      or ``graph.astream(resume_command, ...)``.
    * ``graph_input`` — caller invokes ``graph.ainvoke(graph_input, ...)``
      or ``graph.astream(graph_input, ...)``.
    * ``validation_error_message`` — caller MUST NOT invoke the graph;
      surface this error message to the user instead.
    """

    resume_command: Optional[Command] = None
    graph_input: Optional[dict[str, Any]] = None
    validation_error_message: Optional[str] = None

    @property
    def should_skip_graph(self) -> bool:
        """True when the caller should short-circuit without driving the graph."""
        return self.validation_error_message is not None

    @property
    def stream_input(self) -> Command | dict[str, Any]:
        """The first positional argument for ``graph.astream`` / ``graph.ainvoke``.

        Raises if called when :attr:`should_skip_graph` is True.
        """
        if self.resume_command is not None:
            return self.resume_command
        if self.graph_input is not None:
            return self.graph_input
        raise RuntimeError(
            "GraphTurn has no graph input; check should_skip_graph first."
        )


async def decide_graph_turn(
    *,
    graph: CompiledStateGraph,
    messages: list[dict[str, Any]],
    custom_inputs: Optional[dict[str, Any]],
    runtime_config: dict[str, Any],
    session_input: Optional[dict[str, Any]] = None,
    tool_models: Optional[Sequence[ToolModel]] = None,
) -> GraphTurn:
    """Decide how to drive the next graph turn for a single request.

    Three branches, in priority order:

    1. **Explicit structured decisions.** When ``custom_inputs['decisions']``
       is set, resume the graph with
       ``Command(resume={'decisions': decisions})``. This is the canonical
       machine-to-machine HITL contract used by both Responses
       (``custom_inputs.decisions``) and A2A (DataPart payload merged into
       ``custom_inputs``).

    2. **Free-text resume of an interrupted graph.** When a checkpointer
       exists and the current snapshot is interrupted, use the LLM-based
       :func:`dao_ai.models.handle_interrupt_response` parser to extract
       decisions from the incoming user message(s). On invalid input,
       returns a ``validation_error_message`` so the caller can
       short-circuit.

    3. **Fresh invocation.** Build a graph input from the incoming messages
       (plus optional ``genie_conversation_ids`` carried in ``session_input``).

    Args:
        graph: The compiled LangGraph.
        messages: OpenAI-format message dicts extracted from the incoming
            request.
        custom_inputs: ``request.custom_inputs`` (Responses) or the merged
            DataPart payload (A2A). May be ``None``.
        runtime_config: The ``config={"configurable": {...}}`` dict the
            caller will pass to ``graph.ainvoke`` / ``graph.astream``; used
            here to load the current state snapshot.
        session_input: Session dict carrying things like
            ``genie_conversation_ids``. Defaults to ``{}``.

    Returns:
        A :class:`GraphTurn` describing the next step.
    """
    # Lazy import to avoid circular dependency: dao_ai.models imports this
    # module from within method bodies of LanggraphResponsesAgent.
    from dao_ai.models import handle_interrupt_response, is_interrupted

    session_input = session_input or {}

    audited_hitl_tools: dict[str, AuditModel] = _audited_hitl_tools_from(tool_models)

    # 1. Explicit structured decisions.
    if custom_inputs and "decisions" in custom_inputs:
        decisions = custom_inputs["decisions"]
        logger.info(
            "HITL: explicit decisions in custom_inputs",
            decisions_count=len(decisions) if hasattr(decisions, "__len__") else None,
        )
        if audited_hitl_tools:
            # `reject` and `respond` short-circuit the tool call — the audit
            # middleware never fires for them, so this is the only site where
            # a receipt can be written. Decoration for approve/edit happens
            # inside AuditedHumanInTheLoopMiddleware._process_decision.
            await _record_hitl_non_executions(
                graph=graph,
                decisions=decisions,
                runtime_config=runtime_config,
                audited_hitl_tools=audited_hitl_tools,
            )
        return GraphTurn(resume_command=Command(resume={"decisions": decisions}))

    # 2. Snapshot-based resume from an interrupted graph.
    if graph.checkpointer is not None:
        snapshot = await graph.aget_state(config=runtime_config)
        if is_interrupted(snapshot):
            logger.info("HITL: graph is interrupted, parsing user response")
            message_objects: list[BaseMessage] = convert_openai_messages(messages)
            parsed: dict[str, Any] = handle_interrupt_response(
                snapshot=snapshot,
                messages=message_objects,
                model=None,
            )
            if not parsed.get("is_valid", False):
                validation = parsed.get(
                    "validation_message",
                    "Your response was unclear. Please provide a clear decision "
                    "for each action.",
                )
                logger.warning("HITL: invalid resume input", validation=validation)
                return GraphTurn(validation_error_message=validation)
            decisions = parsed.get("decisions", [])
            logger.info(
                "HITL: LLM parsed decisions from free-text resume",
                decisions_count=len(decisions),
            )
            if audited_hitl_tools:
                await _record_hitl_non_executions(
                    graph=graph,
                    decisions=decisions,
                    runtime_config=runtime_config,
                    audited_hitl_tools=audited_hitl_tools,
                    snapshot=snapshot,
                )
            return GraphTurn(resume_command=Command(resume={"decisions": decisions}))

    # 3. Fresh invocation.
    graph_input: dict[str, Any] = {"messages": messages}
    if "genie_conversation_ids" in session_input:
        graph_input["genie_conversation_ids"] = session_input["genie_conversation_ids"]
    logger.trace(
        "HITL: fresh graph invocation",
        message_count=len(messages),
        has_genie_session=("genie_conversation_ids" in session_input),
    )
    return GraphTurn(graph_input=graph_input)


# ----------------------------------------------------------------------
# Audit-rejection tap
# ----------------------------------------------------------------------


def _audited_hitl_tools_from(
    tool_models: Optional[Sequence[ToolModel]],
) -> dict[str, AuditModel]:
    """Return ``{tool_name: audit_config}`` for tools with both HITL and audit set."""
    if not tool_models:
        return {}
    audited: dict[str, AuditModel] = {}
    for tool_model in tool_models:
        function = tool_model.function
        if not isinstance(function, BaseFunctionModel):
            continue
        audit_config: Optional[AuditModel] = function.audit
        hitl_config = function.human_in_the_loop
        if audit_config is None or hitl_config is None:
            continue
        for func_tool in function.as_tools():
            tool_name: Optional[str] = getattr(func_tool, "name", None)
            if isinstance(tool_name, str) and tool_name:
                audited[tool_name] = audit_config
    return audited


async def _record_hitl_non_executions(
    *,
    graph: CompiledStateGraph,
    decisions: Any,
    runtime_config: dict[str, Any],
    audited_hitl_tools: dict[str, AuditModel],
    snapshot: Any = None,
) -> None:
    """
    Write receipts for HITL decisions that short-circuit tool execution.

    ``reject`` and ``respond`` both cause LangChain's HITL middleware to
    inject a synthetic ``ToolMessage`` in place of running the tool, so
    the audit middleware's ``awrap_tool_call`` never fires — this tap is
    the only site where such non-executions surface. ``approve`` and
    ``edit`` still route through the middleware; their receipts are
    written from there.

    Best-effort — sink I/O failures are logged but do not block the
    resume.
    """
    try:
        from dao_ai.audit import (
            AuditReceipt,
            AuditSinkManager,
            ExecutionStatus,
            LakebaseAuditSink,
            ReceiptKind,
        )
        from dao_ai.audit.base import canonical_jcs, sha256_hex
        from dao_ai.middleware.audit_receipt import AuditStash, AuditStashEntry
        from dao_ai.models import _extract_interrupt_value
    except ImportError as exc:  # pragma: no cover — audit is bundled
        logger.warning(
            "Audit imports unavailable — skipping non-execution tap",
            error=repr(exc),
        )
        return

    from langchain.agents.middleware.human_in_the_loop import (
        ActionRequest,
        HITLRequest,
    )

    if not isinstance(decisions, list) or not decisions:
        return

    if snapshot is None:
        snapshot = await graph.aget_state(config=runtime_config)
    if not getattr(snapshot, "interrupts", ()):
        return

    interrupt_data: list[HITLRequest] = [
        _extract_interrupt_value(interrupt) for interrupt in snapshot.interrupts
    ]
    all_actions: list[ActionRequest] = []
    for hitl_request in interrupt_data:
        all_actions.extend(hitl_request.get("action_requests", []))
    if not all_actions:
        return

    thread_id: str = _thread_id_from_config(runtime_config)
    approver_sub: Optional[str] = _approver_sub_from_config(runtime_config)
    import uuid

    _SHORT_CIRCUITING: set[str] = {"reject", "respond"}

    for idx, decision in enumerate(decisions):
        if idx >= len(all_actions):
            break
        if not isinstance(decision, dict):
            continue
        decision_type: Any = decision.get("type")
        if decision_type not in _SHORT_CIRCUITING:
            continue
        action: ActionRequest = all_actions[idx]
        tool_name: str = action.get("name", "")
        audit_config: Optional[AuditModel] = audited_hitl_tools.get(tool_name)
        if audit_config is None:
            continue

        sink: LakebaseAuditSink = AuditSinkManager.for_config(audit_config)
        # Recover the interrupt-time stash by (thread_id, tool_name). This
        # avoids depending on ``snapshot.values["messages"]`` being
        # rehydrated to typed AIMessage objects, which the checkpointer
        # does not always guarantee on resume paths.
        stash_lookup: Optional[tuple[str, AuditStashEntry]] = (
            AuditStash.take_by_tool_name(thread_id, tool_name)
        )
        tool_call_id: Optional[str] = None
        stash_entry: Optional[AuditStashEntry] = None
        if stash_lookup is not None:
            tool_call_id, stash_entry = stash_lookup

        args_dict: dict[str, Any] = action.get("args") or {}
        args_jcs: str = canonical_jcs(args_dict)
        args_hash: str = sha256_hex(args_jcs)

        decision_detail: Optional[dict[str, Any]] = {
            k: v for k, v in decision.items() if k != "type"
        } or None

        receipt = AuditReceipt(
            receipt_id=uuid.uuid4().hex,
            receipt_kind=ReceiptKind.REJECTION,
            thread_id=thread_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args_jcs=args_jcs,
            args_hash=args_hash,
            args_hash_at_interrupt=(
                stash_entry.args_hash_at_interrupt if stash_entry is not None else None
            ),
            displayed_summary=(
                stash_entry.displayed_summary if stash_entry is not None else None
            ),
            decision=decision_type if isinstance(decision_type, str) else None,
            decision_detail=decision_detail,
            approver_sub=(
                stash_entry.approver_sub if stash_entry is not None else None
            ) or approver_sub,
            confirmed_via=(
                stash_entry.confirmed_via if stash_entry is not None else None
            ) or "chat_ui",
            nonce=stash_entry.nonce if stash_entry is not None else None,
            nonce_exp=stash_entry.nonce_exp if stash_entry is not None else None,
            execution_status=ExecutionStatus.NOT_EXECUTED_REJECTED,
        )
        try:
            await sink.record(receipt)
            logger.info(
                "HITL non-execution receipt recorded",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                decision=decision_type,
                receipt_id=receipt.receipt_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to write HITL non-execution receipt",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                decision=decision_type,
                error=repr(exc),
            )


def _approver_sub_from_config(runtime_config: dict[str, Any]) -> Optional[str]:
    configurable: Any = runtime_config.get("configurable")
    if isinstance(configurable, dict):
        user_id: Any = configurable.get("user_id")
        if isinstance(user_id, str) and user_id:
            return user_id
    return None


def _thread_id_from_config(runtime_config: dict[str, Any]) -> str:
    configurable: Any = runtime_config.get("configurable")
    if isinstance(configurable, dict):
        candidate: Any = configurable.get("thread_id")
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown-thread"


