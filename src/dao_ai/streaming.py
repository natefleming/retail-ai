"""v3 streaming primitives for the dao-ai inference layer.

Decomposes the v3 ``astream_events`` consumer into small, testable
listener objects that each own one typed projection (text, reasoning,
tool calls, lifecycle) and accept an :class:`AgentFilter` controlling
which agents' events surface to the user stream.

Why these shapes exist
----------------------
- ``AgentFilter`` keeps the visibility decision in one place. Today it's
  populated from :attr:`AgentModel.surface_to_user`; in stage 2 the same
  filter shape can be constructed from per-request configurable so user
  code can subscribe to specific channels without touching dao-ai source.

- ``AgentResolver`` solves the chat-event-to-agent attribution problem
  inherent to dao-ai's swarm pattern: each agent is a compiled subgraph
  invoked via ``ainvoke``, so v3 reports ``chat.node`` as the inner node
  name (``"model"``, ``"MemoryContextMiddleware.before_model"``) rather
  than the parent agent's name. The agent identity only becomes
  available when the final ``AIMessage`` flows back to the parent state
  with ``msg.name`` set by :func:`create_agent_node_handler`. The
  resolver bridges that gap via ``message_id -> agent name`` correlation
  with per-id :class:`asyncio.Event` signalling, so the messages
  consumer waits on an actual signal rather than polling.

- Concrete listeners (``TextDeltaListener``) own a single projection
  end-to-end: consume a chat object, resolve its agent, apply the
  filter, and yield the corresponding :class:`ResponsesAgentStreamEvent`
  s for the channel. Reasoning and tool-call listeners are reserved as
  skeletons for stage 2; their interface is identical so adding them
  later doesn't reshape ``apredict_stream``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol, runtime_checkable

from langchain_core.language_models.chat_model_stream import AsyncChatModelStream
from langchain_core.messages import AIMessage
from loguru import logger
from mlflow.types.responses import ResponsesAgentStreamEvent


# ---------------------------------------------------------------------------
# Visibility filter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentFilter:
    """Per-agent visibility predicate consumed by listeners.

    A listener calls :meth:`allows` with the resolved source-agent name
    (or ``None`` for graph-level events with no agent attribution) and
    surfaces the event iff this returns True.

    Stage 1 populates the filter from
    :attr:`dao_ai.config.AgentModel.surface_to_user` via
    :meth:`from_visibility`. Stage 2 (per-request listeners exposed to
    user code) builds the same shape directly from request configurable.
    """

    include_agents: frozenset[str] | None = None
    exclude_agents: frozenset[str] = field(default_factory=frozenset)
    """Names of LangGraph nodes that emit chat events but are not real
    agents (memory middleware, summarization, etc.). Always suppressed
    regardless of ``include_agents`` / ``exclude_agents``."""
    exclude_node_substrings: frozenset[str] = field(
        default_factory=lambda: frozenset({"summarization", "Middleware"})
    )

    @classmethod
    def from_visibility(cls, visibility: dict[str, bool]) -> "AgentFilter":
        """Build a filter that excludes every agent with
        ``surface_to_user=False``. Agents with ``surface_to_user=True`` and
        unconfigured agents both default to visible."""
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
        """Return True if a chat event with the given ``chat.node`` should
        be considered at all. Pre-filters non-agent nodes (middleware,
        summarization) before agent attribution runs."""
        if node is None:
            return True
        return not any(s in node for s in self.exclude_node_substrings)


# ---------------------------------------------------------------------------
# Agent attribution
# ---------------------------------------------------------------------------


class AgentResolver:
    """Resolve ``chat.output_message.id -> agent name`` by tailing
    ``stream.values`` for ``AIMessage(name=<agent>)`` entries.

    Listeners call :meth:`resolve` with a message id and ``await`` an
    :class:`asyncio.Event` populated by :meth:`record_state`. There is no
    polling: the values consumer signals when the agent name lands, the
    listener wakes immediately. A per-call timeout prevents deadlock if
    a state snapshot for that message never arrives.
    """

    def __init__(self) -> None:
        self._id_to_agent: dict[str, str] = {}
        self._events: dict[str, asyncio.Event] = {}

    def record_state(self, state: dict[str, Any]) -> None:
        """Walk a state snapshot and capture agent-tagged AIMessages."""
        messages: list[Any] = state.get("messages") or []
        for msg in messages:
            if (
                isinstance(msg, AIMessage)
                and msg.id is not None
                and msg.name
            ):
                self._id_to_agent[msg.id] = msg.name
                event: asyncio.Event | None = self._events.get(msg.id)
                if event is not None:
                    event.set()

    async def resolve(
        self,
        message_id: str | None,
        *,
        timeout: float = 1.0,
    ) -> str | None:
        """Return the agent name for ``message_id``, or ``None`` if the
        values consumer didn't populate it within ``timeout``."""
        if message_id is None:
            return None
        if message_id in self._id_to_agent:
            return self._id_to_agent[message_id]
        event: asyncio.Event = self._events.setdefault(
            message_id, asyncio.Event()
        )
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug(
                "Agent attribution timeout",
                message_id=message_id,
                timeout=timeout,
            )
        return self._id_to_agent.get(message_id)


# ---------------------------------------------------------------------------
# Listener interface
# ---------------------------------------------------------------------------


@runtime_checkable
class StreamListener(Protocol):
    """Listener for a single v3 typed projection.

    A listener owns the end-to-end transform from one
    :class:`AsyncChatModelStream` (one LLM call) to zero or more
    :class:`ResponsesAgentStreamEvent` s. Stage 1 ships one concrete
    implementation per channel; stage 2 will let user code register
    additional listeners alongside.
    """

    @property
    def channel(self) -> str:
        """A short name identifying which projection this listener owns
        (``"text"``, ``"reasoning"``, ``"tool_calls"``)."""

    async def consume(
        self,
        chat: AsyncChatModelStream,
        resolver: AgentResolver,
    ) -> AsyncIterator[ResponsesAgentStreamEvent]:
        """Process a single chat object and yield user-stream events."""


# ---------------------------------------------------------------------------
# Concrete: text deltas
# ---------------------------------------------------------------------------


@dataclass
class TextDeltaListener:
    """Buffers ``chat.text`` tokens for one LLM call, resolves the source
    agent via the message-id correlation channel, then either flushes
    the buffer as ``response.output_text.delta`` events or drops it
    based on :class:`AgentFilter`.

    The buffer-and-flush trade-off is forced by the swarm architecture
    (see module docstring). For a single-agent app where ``chat.node``
    directly identifies the agent, attribution resolves to the fast
    path and visible agents stream live.
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
        chat: AsyncChatModelStream,
        resolver: AgentResolver,
    ) -> AsyncIterator[ResponsesAgentStreamEvent]:
        node: str | None = chat.node
        # Pre-filter non-agent nodes (memory middleware, summarization).
        if not self.filter.allows_node(node):
            return
        # Fast path: chat.node directly names a configured agent.
        if node is not None and not self.filter.allows(node):
            # The agent is known-silent at this stage; consume the
            # async iterator to avoid back-pressure, then drop.
            async for _ in chat.text:
                pass
            return
        buffered: list[str] = []
        async for token in chat.text:
            buffered.append(token)
        # Slow path: resolve via output_message.id <-> stream.values.
        resolved: str | None = node
        out_msg: AIMessage | None = chat.output_message
        if out_msg is not None and out_msg.id is not None:
            mapped: str | None = await resolver.resolve(out_msg.id)
            if mapped is not None:
                resolved = mapped
        if not self.filter.allows(resolved):
            logger.info(
                "Suppressing text deltas for silent agent",
                agent=resolved,
                chat_node=node,
                buffered_chars=sum(len(t) for t in buffered),
            )
            return
        for token in buffered:
            if self.on_token is not None:
                self.on_token(token)
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=token, item_id=self.item_id)
            )


# ---------------------------------------------------------------------------
# Skeleton listeners (stage 2 surfaces these to the user as opt-in)
# ---------------------------------------------------------------------------


@dataclass
class ReasoningDeltaListener:
    """Reserved skeleton — exposes ``chat.reasoning`` deltas. Stage 1
    does not surface reasoning to the user stream."""

    filter: AgentFilter

    @property
    def channel(self) -> str:
        return "reasoning"

    async def consume(
        self,
        chat: AsyncChatModelStream,
        resolver: AgentResolver,
    ) -> AsyncIterator[ResponsesAgentStreamEvent]:
        # Drain to avoid back-pressure; stage 2 will yield events here
        # for user-registered reasoning listeners.
        async for _ in chat.reasoning:
            pass
        if False:
            yield  # pragma: no cover  # makes this an async generator


@dataclass
class ToolCallListener:
    """Reserved skeleton — exposes ``chat.tool_calls`` deltas. Stage 1
    does not surface tool calls to the user stream."""

    filter: AgentFilter

    @property
    def channel(self) -> str:
        return "tool_calls"

    async def consume(
        self,
        chat: AsyncChatModelStream,
        resolver: AgentResolver,
    ) -> AsyncIterator[ResponsesAgentStreamEvent]:
        async for _ in chat.tool_calls:
            pass
        if False:
            yield  # pragma: no cover
