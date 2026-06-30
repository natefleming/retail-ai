"""Streaming primitives for the dao-ai inference layer.

The dao-ai streaming consumer reads ``CompiledStateGraph.astream`` with
``stream_mode=["messages","updates"]``. The ``messages`` channel emits
``(AIMessageChunk, metadata)`` tuples; the ``updates`` channel emits
per-node state diffs. These two channels together carry everything the
runtime needs:

* token-by-token streaming text — ``messages`` channel
* tool-call dispatch and ToolMessage results — ``updates`` channel
* per-node attribution — ``metadata.langgraph_node`` and ``chunk.name``
* structured responses, interrupts — ``updates`` channel state diffs

Why not v3?
-----------
``CompiledStateGraph.astream_events(version="v3")`` is the experimental
typed-projection API LangGraph 1.2.x ships. It looked attractive (typed
``stream.messages.text``, ``stream.values``, etc.) but the v3 compat
bridge ``LangChain._compat_bridge.chunks_to_events`` mishandles tool-call
chunk merging from ``databricks-langchain`` for certain models
(verified: ``databricks-gpt-oss-120b`` never dispatches tools under v3;
``databricks-claude-sonnet-4-5`` does). Per
``tests/dao_ai/test_streaming_tool_dispatch.py``, the stable
``astream(stream_mode=...)`` API dispatches tools for every model
combination, so we stay on it.

Per-agent visibility
--------------------
The runtime asks ``AgentFilter`` per-chunk whether to surface its
deltas. The filter is constructed from the
``Context.agent_visibility`` map published by the orchestration
factories. Each ``AIMessageChunk`` carries either ``chunk.name``
(set by ``create_agent(name=…)``) or ``metadata["langgraph_node"]`` —
the listener tries both, then falls back to ``allow`` for
graph-level events with no attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Mapping, Protocol, runtime_checkable

from langchain_core.messages import AIMessage, AIMessageChunk
from loguru import logger
from mlflow.types.responses import ResponsesAgentStreamEvent

from dao_ai.orchestration.core import _flatten_message_content


# ---------------------------------------------------------------------------
# Visibility filter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentFilter:
    """Per-agent visibility predicate consumed by listeners."""

    include_agents: frozenset[str] | None = None
    exclude_agents: frozenset[str] = field(default_factory=frozenset)
    exclude_node_substrings: frozenset[str] = field(
        default_factory=lambda: frozenset({"summarization", "Middleware"})
    )

    @classmethod
    def from_visibility(cls, visibility: dict[str, bool]) -> "AgentFilter":
        """Build a filter that excludes every agent with
        ``surface_to_user=False``. Agents marked True and unconfigured
        agents both default to visible."""
        excluded: frozenset[str] = frozenset(
            name for name, surfaced in visibility.items() if surfaced is False
        )
        return cls(exclude_agents=excluded)

    def allows(self, agent: str | None) -> bool:
        """Return True if events attributed to ``agent`` should surface."""
        if agent is None:
            return True
        if self.include_agents is not None and agent not in self.include_agents:
            return False
        if agent in self.exclude_agents:
            return False
        return True

    def allows_node(self, node: str | None) -> bool:
        """Pre-filter non-agent nodes (memory middleware, summarization)
        before agent attribution runs."""
        if node is None:
            return True
        return not any(s in node for s in self.exclude_node_substrings)


# ---------------------------------------------------------------------------
# Agent attribution
# ---------------------------------------------------------------------------


class AgentResolver:
    """Resolve ``AIMessage.id -> agent name`` by recording ``updates``
    channel state diffs. Listeners pass ``(chunk, metadata)`` tuples to
    :meth:`attribute`; the resolver tries ``chunk.name`` (set by
    ``create_agent(name=…)``) first, then ``metadata["langgraph_node"]``,
    then a cached id→name map populated from ``record_update``.
    """

    def __init__(self) -> None:
        self._id_to_agent: dict[str, str] = {}

    def record_update(self, update: Mapping[str, Any] | None) -> None:
        """Capture ``AIMessage.name`` from any agent's state diff."""
        if not update:
            return
        for _node, payload in update.items():
            if not isinstance(payload, Mapping):
                continue
            messages = payload.get("messages") or []
            for msg in messages:
                if (
                    isinstance(msg, AIMessage)
                    and msg.id is not None
                    and msg.name
                ):
                    self._id_to_agent[msg.id] = msg.name

    def attribute(
        self,
        chunk: AIMessageChunk,
        metadata: Mapping[str, Any] | None,
    ) -> str | None:
        """Return the agent name a chunk belongs to, or ``None`` if
        unattributable."""
        name: str | None = chunk.name
        if name:
            return name
        if chunk.id is not None and chunk.id in self._id_to_agent:
            return self._id_to_agent[chunk.id]
        if metadata is not None:
            node: Any = metadata.get("langgraph_node")
            if isinstance(node, str) and node:
                return node
        return None


# ---------------------------------------------------------------------------
# Listener interface
# ---------------------------------------------------------------------------


@runtime_checkable
class StreamListener(Protocol):
    """Listener for a single chunk class (text, reasoning, tool_calls).

    Each listener owns the end-to-end transform from a single
    ``(AIMessageChunk, metadata)`` tuple to zero or more
    :class:`ResponsesAgentStreamEvent` s. Stage 1 ships one concrete
    listener (``TextDeltaListener``); stage 2 will register additional
    listeners for reasoning + tool calls under user control.
    """

    @property
    def channel(self) -> str:
        """Channel name (``"text"``, ``"reasoning"``, ``"tool_calls"``)."""

    async def consume(
        self,
        chunk: AIMessageChunk,
        metadata: Mapping[str, Any] | None,
        resolver: AgentResolver,
    ) -> AsyncIterator[ResponsesAgentStreamEvent]:
        """Process a chunk and yield user-stream events."""


# ---------------------------------------------------------------------------
# Concrete: text deltas
# ---------------------------------------------------------------------------


@dataclass
class TextDeltaListener:
    """Emit ``response.output_text.delta`` events for the user-facing
    SSE stream. Suppresses chunks attributed to agents whose
    ``surface_to_user`` is False, and skips middleware/summarization
    chunks entirely.
    """

    filter: AgentFilter
    item_id: str
    create_text_delta: Callable[..., dict[str, Any]]
    on_token: Callable[[str], None] | None = None

    @property
    def channel(self) -> str:
        return "text"

    async def consume(
        self,
        chunk: AIMessageChunk,
        metadata: Mapping[str, Any] | None,
        resolver: AgentResolver,
    ) -> AsyncIterator[ResponsesAgentStreamEvent]:
        node: str | None = (
            metadata.get("langgraph_node") if metadata else None
        )
        if isinstance(node, str) and not self.filter.allows_node(node):
            return
        agent: str | None = resolver.attribute(chunk, metadata)
        if not self.filter.allows(agent):
            logger.info(
                "Suppressing text delta for silent agent",
                agent=agent,
                node=node,
            )
            return
        content: object = chunk.content
        flat: object = _flatten_message_content(content)
        text: str = flat if isinstance(flat, str) else ""
        if not text:
            return
        if self.on_token is not None:
            self.on_token(text)
        yield ResponsesAgentStreamEvent(
            **self.create_text_delta(delta=text, item_id=self.item_id)
        )


# ---------------------------------------------------------------------------
# Skeleton listeners — stage 2 surfaces these to user code
# ---------------------------------------------------------------------------


@dataclass
class ReasoningDeltaListener:
    """Reserved skeleton — surfaces reasoning chunks in stage 2."""

    filter: AgentFilter

    @property
    def channel(self) -> str:
        return "reasoning"

    async def consume(
        self,
        chunk: AIMessageChunk,
        metadata: Mapping[str, Any] | None,
        resolver: AgentResolver,
    ) -> AsyncIterator[ResponsesAgentStreamEvent]:
        # Stage 1: no-op. Stage 2: inspect chunk.additional_kwargs for
        # reasoning blocks (provider-specific) and yield events.
        if False:
            yield  # pragma: no cover — makes this an async generator


@dataclass
class ToolCallListener:
    """Reserved skeleton — surfaces tool-call deltas in stage 2."""

    filter: AgentFilter

    @property
    def channel(self) -> str:
        return "tool_calls"

    async def consume(
        self,
        chunk: AIMessageChunk,
        metadata: Mapping[str, Any] | None,
        resolver: AgentResolver,
    ) -> AsyncIterator[ResponsesAgentStreamEvent]:
        if False:
            yield  # pragma: no cover
