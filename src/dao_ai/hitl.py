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
from typing import Any, Optional

from langchain_community.adapters.openai import convert_openai_messages
from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from loguru import logger


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

    # 1. Explicit structured decisions.
    if custom_inputs and "decisions" in custom_inputs:
        decisions = custom_inputs["decisions"]
        logger.info(
            "HITL: explicit decisions in custom_inputs",
            decisions_count=len(decisions) if hasattr(decisions, "__len__") else None,
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
