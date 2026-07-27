"""A2A :class:`AgentExecutor` that delegates to a dao-ai LangGraph.

This is the heart of the A2A↔dao-ai bridge. Each incoming A2A request
arrives here as a :class:`RequestContext`; we translate it to dao-ai's
LangGraph input shape, drive the graph, and republish the result as A2A
:class:`TaskStatusUpdateEvent` / :class:`TaskArtifactUpdateEvent` on the
event queue.

The HITL decision (fresh invocation vs. resume of an interrupt) is
delegated to :func:`dao_ai.hitl.decide_graph_turn`, so the A2A executor
and the OpenAI Responses path stay in lock-step on what constitutes a
resume.

Semantic mappings:

* A2A ``Message.parts`` → LangGraph ``state.messages``. ``TextPart``s
  become a single user message (joined by ``\\n``). ``DataPart``s are
  merged into ``custom_inputs`` so the canonical
  ``custom_inputs.decisions`` HITL resume contract Just Works™.
* ``RequestContext.context_id`` → LangGraph ``thread_id`` (the dao-ai
  ``conversation_id`` alias).
* ``ServerCallContext.state['headers']`` (populated by a2a-sdk's
  :class:`DefaultCallContextBuilder`) → ``configurable.headers`` so OBO
  flows the same way it does in the Responses path.
* LangGraph terminal ``AIMessage`` → A2A ``TaskArtifactUpdateEvent`` with
  ``TextPart`` (and optional ``DataPart`` when ``structured_response``
  is present).
* LangGraph ``interrupt()`` → A2A ``TaskState.input_required`` with the
  interrupt payload as a ``DataPart`` so machine clients can act on it
  programmatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TextPart
from langgraph.errors import GraphInterrupt
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from dao_ai.hitl import decide_graph_turn
from dao_ai.state import Context as DaoContext

if TYPE_CHECKING:
    from dao_ai.config import AppConfig, ToolModel


class A2AAgentExecutor(AgentExecutor):
    """Bridges A2A requests to a dao-ai compiled graph.

    The executor is constructed once per app process and reused across all
    requests. The underlying :class:`CompiledStateGraph` is created lazily
    on first use so that test fixtures can inject a pre-built graph.
    """

    def __init__(
        self,
        config: "AppConfig",
        *,
        graph: Optional[CompiledStateGraph] = None,
    ) -> None:
        self.config = config
        self._graph: Optional[CompiledStateGraph] = graph

    @property
    def graph(self) -> CompiledStateGraph:
        if self._graph is None:
            self._graph = self.config.as_graph()
        return self._graph

    @property
    def _tool_models(self) -> list["ToolModel"]:
        """Flattened tool_models across every agent — used by the HITL audit tap."""
        return [tool for agent in self.config.agents.values() for tool in agent.tools]

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id
        is_new_task = context.current_task is None
        message_parts_count = (
            len(context.message.parts)
            if context.message and context.message.parts
            else 0
        )
        logger.info(
            "A2A: execute start",
            task_id=task_id,
            context_id=context_id,
            new_task=is_new_task,
            message_parts=message_parts_count,
        )

        updater = TaskUpdater(
            event_queue,
            task_id=task_id,
            context_id=context_id,
        )

        # Surface a 'submitted' event for brand-new tasks so streaming
        # clients see the lifecycle start.
        if is_new_task:
            await updater.submit()
        await updater.start_work()

        try:
            messages, custom_inputs = self._extract_request(context)
            dao_context = self._build_dao_context(context, custom_inputs)
            runtime_config: dict[str, Any] = {"configurable": dao_context.model_dump()}
            logger.trace(
                "A2A: request translated to graph input",
                task_id=task_id,
                context_id=context_id,
                message_count=len(messages),
                custom_input_keys=sorted(custom_inputs.keys()),
                has_obo_headers=bool(dao_context.headers),
            )

            turn = await decide_graph_turn(
                graph=self.graph,
                messages=messages,
                custom_inputs=custom_inputs,
                runtime_config=runtime_config,
                tool_models=self._tool_models,
            )

            if turn.should_skip_graph:
                # User's resume input couldn't be parsed; surface the
                # validation message and complete without re-driving.
                logger.warning(
                    "A2A: validation_error short-circuit",
                    task_id=task_id,
                    context_id=context_id,
                    message=turn.validation_error_message,
                )
                await updater.complete(
                    message=updater.new_agent_message(
                        parts=[
                            Part(
                                root=TextPart(
                                    text=(
                                        "❌ Invalid Response\n\n"
                                        f"{turn.validation_error_message}"
                                    )
                                )
                            )
                        ]
                    )
                )
                return

            try:
                response = await self.graph.ainvoke(
                    turn.stream_input,
                    context=dao_context,
                    config=runtime_config,
                )
            except GraphInterrupt:
                logger.info(
                    "A2A: GraphInterrupt raised mid-invocation",
                    task_id=task_id,
                    context_id=context_id,
                )
                if self.graph.checkpointer is not None:
                    snapshot = await self.graph.aget_state(config=runtime_config)
                    await self._emit_input_required(
                        updater, list(snapshot.interrupts or [])
                    )
                    return
                raise

            interrupts = self._detect_interrupts_post_invoke(response, runtime_config)
            if interrupts:
                logger.info(
                    "A2A: interrupts detected post-invoke",
                    task_id=task_id,
                    context_id=context_id,
                    interrupts_count=len(interrupts),
                )
                await self._emit_input_required(updater, interrupts)
                return

            await self._emit_completed(updater, response)
            logger.success(
                "A2A: execute completed",
                task_id=task_id,
                context_id=context_id,
            )

        except Exception as exc:
            logger.exception(
                "A2A executor error",
                task_id=task_id,
                context_id=context_id,
                error=str(exc),
            )
            await updater.failed(
                message=updater.new_agent_message(
                    parts=[Part(root=TextPart(text=f"Agent error: {exc}"))]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info(
            "A2A: cancel requested",
            task_id=context.task_id,
            context_id=context.context_id,
        )
        updater = TaskUpdater(
            event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.cancel()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_request(
        self, context: RequestContext
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Translate the A2A request into a LangGraph messages list and
        a merged ``custom_inputs`` dict.

        TextPart parts are concatenated into a single user message; DataPart
        payloads are merged into ``custom_inputs``. Any
        ``configurable``/``decisions`` keys in the DataPart are preserved
        verbatim so the dao-ai HITL contract works.
        """
        message = context.message
        if message is None:
            return [], {}

        text_chunks: list[str] = []
        custom_inputs: dict[str, Any] = {}
        for part in message.parts or []:
            inner = part.root if hasattr(part, "root") else part
            if isinstance(inner, TextPart):
                if inner.text:
                    text_chunks.append(inner.text)
            elif isinstance(inner, DataPart):
                if inner.data:
                    custom_inputs.update(inner.data)
            # FilePart is intentionally not handled in Phase 1.

        messages: list[dict[str, Any]] = []
        if text_chunks:
            messages.append({"role": "user", "content": "\n".join(text_chunks)})
        return messages, custom_inputs

    def _build_dao_context(
        self,
        context: RequestContext,
        custom_inputs: dict[str, Any],
    ) -> DaoContext:
        """Build the dao-ai :class:`Context` for this turn.

        ``context_id`` is the canonical ``thread_id``. OBO headers come from
        the A2A call context (populated by a2a-sdk's
        :class:`DefaultCallContextBuilder`), with any explicit
        ``custom_inputs['configurable']['headers']`` taking precedence.
        """
        thread_id = context.context_id

        headers: dict[str, str] = {}
        call_context = getattr(context, "call_context", None)
        if call_context is not None:
            state_headers = (
                call_context.state.get("headers") if call_context.state else None
            )
            if isinstance(state_headers, dict):
                headers.update(state_headers)
        configurable = (
            custom_inputs.get("configurable")
            if isinstance(custom_inputs, dict)
            else None
        )
        if isinstance(configurable, dict):
            user_headers = configurable.get("headers")
            if isinstance(user_headers, dict):
                headers.update(user_headers)

        user_id: str | None = None
        if isinstance(configurable, dict):
            user_id = configurable.get("user_id")

        return DaoContext(
            user_id=user_id,
            thread_id=thread_id,
            headers=headers or None,
        )

    def _detect_interrupts_post_invoke(
        self,
        response: Any,
        runtime_config: dict[str, Any],
    ) -> list[Any]:
        """Pull interrupts off the response or the post-invoke snapshot."""
        interrupts: list[Any] = []
        if isinstance(response, dict) and "__interrupt__" in response:
            interrupts = list(response["__interrupt__"])
        # NOTE: A second pass via aget_state would catch interrupts that
        # ainvoke swallowed without setting __interrupt__. We rely on the
        # caller observing the GraphInterrupt exception path for that case.
        return interrupts

    async def _emit_input_required(
        self, updater: TaskUpdater, interrupts: list[Any]
    ) -> None:
        """Emit a terminal ``input-required`` status with the interrupt
        payload as a ``DataPart`` for machine clients."""
        payload = [self._interrupt_to_dict(i) for i in interrupts]
        logger.info(
            "A2A: emitting input-required",
            task_id=updater.task_id,
            context_id=updater.context_id,
            interrupts_count=len(payload),
        )
        await updater.requires_input(
            message=updater.new_agent_message(
                parts=[
                    Part(
                        root=DataPart(
                            data={"interrupts": payload},
                            metadata={"reason": "human-in-the-loop"},
                        )
                    )
                ]
            ),
            final=True,
        )

    def _interrupt_to_dict(self, interrupt: Any) -> dict[str, Any]:
        """Serialize a LangGraph :class:`Interrupt` into a JSON-able dict.

        Falls back to ``str(interrupt)`` when the value is not naturally
        serializable.
        """
        if hasattr(interrupt, "model_dump"):
            try:
                return interrupt.model_dump(mode="json")
            except Exception:  # pragma: no cover
                pass
        value = getattr(interrupt, "value", interrupt)
        if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            return {"value": value}
        return {"value": str(value)}

    async def _emit_completed(
        self,
        updater: TaskUpdater,
        response: Any,
    ) -> None:
        """Emit the terminal artifact + completed status."""
        from langchain_core.messages import BaseMessage  # local import

        parts: list[Part] = []

        # Pull text from the final AIMessage.
        if isinstance(response, dict):
            messages: list[BaseMessage] = response.get("messages", []) or []
            if messages:
                last = messages[-1]
                content = getattr(last, "content", None)
                text = self._coerce_message_text(content)
                if text:
                    parts.append(Part(root=TextPart(text=text)))

            # Surface structured_response as a DataPart so machine clients
            # can consume the typed output without re-parsing the text.
            structured = response.get("structured_response")
            if structured is not None:
                data = self._coerce_structured(structured)
                if data is not None:
                    parts.append(Part(root=DataPart(data=data)))

        if not parts:
            parts.append(Part(root=TextPart(text="")))

        logger.debug(
            "A2A: emitting terminal artifact",
            task_id=updater.task_id,
            context_id=updater.context_id,
            part_kinds=[type(p.root).__name__ for p in parts],
        )
        await updater.add_artifact(
            parts=parts,
            name="response",
            last_chunk=True,
        )
        await updater.complete()

    @staticmethod
    def _coerce_message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for block in content:
                if isinstance(block, str):
                    chunks.append(block)
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "".join(chunks)
        return ""

    @staticmethod
    def _coerce_structured(value: Any) -> Optional[dict[str, Any]]:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump(mode="json")
            except Exception:  # pragma: no cover
                pass
        if isinstance(value, dict):
            return value
        from dataclasses import asdict, is_dataclass

        if is_dataclass(value):
            try:
                return asdict(value)
            except Exception:  # pragma: no cover
                return None
        return None
