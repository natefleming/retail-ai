"""A ``BaseChatModel`` backed by the Databricks Genie Agent Mode API.

This is the streaming, model-resource counterpart to the legacy
``type: genie`` tool. It calls
``POST /api/2.0/genie/agents/{agent_id}/responses`` — an SSE-streaming,
OpenAI-Responses-*style* endpoint (Beta) — and surfaces Genie's output as
``AIMessageChunk``s so a plain :class:`~dao_ai.config.AgentModel` with no
tools becomes a natively-streaming "Genie specialist" node.

Why a chat model instead of a tool
-----------------------------------
A LangGraph tool node is atomic: it returns one ``ToolMessage`` after the
whole stream completes, so Genie output never reaches the end user through
the agent's own response stream. A model node, by contrast, streams
``AIMessageChunk``s to the outer stream (``stream_mode="messages"``) and is
routable by a supervisor like any other sub-agent.

Statelessness and multi-turn
-----------------------------
``BaseChatModel`` has no access to graph state or ``ToolRuntime``. The Genie
server owns conversation history keyed by its own ``conversation_id`` (issued
on the first turn), which is **independent** of the LangGraph ``thread_id``
used for graph-state persistence. This model is pure with respect to that id:
:attr:`conversation_id` is passed in as a field (or ``None`` to start fresh),
and the Genie-issued id for the turn is stamped on the returned message's
``response_metadata['genie_conversation_id']``.

Ownership of the cross-turn value lives in
:class:`dao_ai.middleware.genie_agent.GenieAgentMiddleware`, which reads the
prior id from ``session.genie.spaces[agent_id]`` before the call and writes the
newly-issued id back to the same channel afterward — exactly the mechanism the
legacy ``type: genie`` tool uses. The model itself keeps no state and never
scans the message history.

Authentication / OBO
--------------------
The SSE call authenticates via :class:`dao_ai.auth.WorkspaceBearerAuth`
wrapping a :class:`~databricks.sdk.WorkspaceClient` injected at construction.
:class:`dao_ai.middleware.genie_agent.GenieAgentMiddleware` rebuilds this chat
model per request with a user-scoped ``WorkspaceClient`` (OBO) *and* the prior
``conversation_id`` in one step — the two per-request concerns are handled
together (see ``GenieAgentModel.chat_model_for_workspace_client``).
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, AsyncIterator, Iterator, Optional

import httpx
import mlflow
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from loguru import logger

from dao_ai.auth import WorkspaceBearerAuth

# Key under which the Genie conversation_id is stamped on the AIMessage's
# response_metadata so the next turn can continue the same server-side
# conversation.
CONVERSATION_ID_METADATA_KEY: str = "genie_conversation_id"

# MLflow span attribute keys.
ATTR_AGENT_ID = "dao_ai.genie_agent.agent_id"
ATTR_CONVERSATION_ID = "dao_ai.genie_agent.conversation_id"
ATTR_RESPONSE_ID = "dao_ai.genie_agent.response_id"
ATTR_TERMINAL_STATE = "dao_ai.genie_agent.stream.terminal_state"
ATTR_TOTAL_EVENTS = "dao_ai.genie_agent.stream.total_events"


class GenieAgentError(RuntimeError):
    """Raised when the Genie Agent Mode API returns a terminal error event
    or a non-2xx HTTP status."""


def _parse_sse_lines(lines: list[str]) -> Iterator[tuple[Optional[str], dict[str, Any]]]:
    """Yield ``(event, data)`` pairs from buffered SSE lines.

    Parses the standard ``event:`` / ``data:`` line format; a blank line
    terminates a record. ``data:`` is expected to be JSON; malformed data is
    skipped with a debug log.
    """
    event_type: Optional[str] = None
    data_lines: list[str] = []
    for raw_line in lines:
        line: str = raw_line.rstrip("\r")
        if line == "":
            if data_lines:
                data_str: str = "\n".join(data_lines)
                try:
                    payload: dict[str, Any] = json.loads(data_str)
                except json.JSONDecodeError as exc:
                    logger.debug(f"genie_agent: skipping non-JSON SSE data: {exc}")
                else:
                    yield event_type, payload
            event_type = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue  # SSE comment / keepalive
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
        # Unknown SSE field — ignore per spec.


def _format_function_call_block(item: dict[str, Any]) -> str:
    """Render a ``function_call`` item (execute_sql) as a fenced markdown block."""
    name: str = item.get("name") or "function_call"
    arguments_raw: Any = item.get("arguments")
    arguments: dict[str, Any] = {}
    if isinstance(arguments_raw, str):
        try:
            arguments = json.loads(arguments_raw)
        except json.JSONDecodeError:
            arguments = {"raw": arguments_raw}
    elif isinstance(arguments_raw, dict):
        arguments = arguments_raw

    title: Optional[str] = arguments.get("title") if isinstance(arguments, dict) else None
    sql: Optional[str] = arguments.get("sql") if isinstance(arguments, dict) else None

    if name == "execute_sql" and sql:
        heading: str = f"**{title}**\n" if title else ""
        return f"{heading}```sql\n{sql}\n```"
    return f"**{name}**\n```json\n{json.dumps(arguments, indent=2)}\n```"


def _format_function_call_output(item: dict[str, Any]) -> str:
    """Render a ``function_call_output`` item's markdown table string."""
    output: Any = item.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return json.dumps(output, indent=2)
    return str(output) if output is not None else ""


def _format_message_item(item: dict[str, Any]) -> str:
    """Concatenate ``output_text`` chunks from a ``message`` item."""
    parts: list[str] = []
    for content in item.get("content") or []:
        if not isinstance(content, dict):
            continue
        if content.get("type") == "output_text":
            text: Optional[str] = content.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts)


class _StreamState:
    """Accumulates SSE events into ordered content segments + terminal status.

    Feeds both the sync (``_generate``) and async (``_astream``) code paths so
    the event → text mapping lives in one place. Each accepted event that
    produces visible text returns that text (with a trailing separator) from
    :meth:`handle` so a streaming caller can emit it as a chunk; the same text
    is also appended to :attr:`segments` for the aggregated final message.
    """

    def __init__(self) -> None:
        self.conversation_id: Optional[str] = None
        self.response_id: Optional[str] = None
        self.segments: list[str] = []
        self.event_count: int = 0
        self.terminal_status: Optional[str] = None
        self.terminal_error: Optional[str] = None

    def handle(self, event_type: Optional[str], payload: dict[str, Any]) -> Optional[str]:
        """Process one event. Returns text to stream, or ``None``."""
        self.event_count += 1
        kind: str = event_type or payload.get("type") or ""

        if kind == "response.created":
            response_obj: dict[str, Any] = payload.get("response") or {}
            self.conversation_id = response_obj.get("conversation_id") or self.conversation_id
            self.response_id = response_obj.get("id") or self.response_id
            return None

        if kind == "response.output_item.done":
            item: dict[str, Any] = payload.get("item") or {}
            item_type: str = item.get("type") or ""
            text: Optional[str] = None
            if item_type == "function_call":
                text = _format_function_call_block(item)
            elif item_type == "function_call_output":
                text = _format_function_call_output(item) or None
            elif item_type == "message":
                text = _format_message_item(item) or None
            # reasoning items are intentionally not surfaced.
            if text:
                self.segments.append(text)
                return text + "\n\n"
            return None

        if kind in {"response.completed", "response.failed"}:
            response_obj = payload.get("response") or {}
            self.conversation_id = response_obj.get("conversation_id") or self.conversation_id
            self.response_id = response_obj.get("id") or self.response_id
            self.terminal_status = response_obj.get("status") or kind
            error: Optional[dict[str, Any]] = response_obj.get("error")
            if error:
                code: str = error.get("code") or "error"
                message: str = error.get("message") or ""
                self.terminal_error = f"[{code}] {message}"
            return None

        return None

    def aggregated_content(self) -> str:
        content: str = "\n\n".join(seg for seg in self.segments if seg)
        return content or "(Genie Agent returned no output.)"

    def raise_on_error(self) -> None:
        if self.terminal_error:
            raise GenieAgentError(
                f"Genie Agent Mode API failed (status={self.terminal_status}): "
                f"{self.terminal_error}"
            )


def _latest_human_text(messages: list[BaseMessage]) -> str:
    """Return the text of the most recent ``HumanMessage``.

    Genie takes exactly one user turn; prior turns and system prompts are not
    replayed (the server owns history via ``conversation_id``).
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content: Any = message.content
            if isinstance(content, str):
                return content
            # Content-block form: concatenate text blocks.
            if isinstance(content, list):
                parts: list[str] = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") in {"text", "input_text"}
                ]
                return "\n".join(p for p in parts if p)
    raise GenieAgentError("GenieAgentChatModel: no HumanMessage found in the input.")


class GenieAgentChatModel(BaseChatModel):
    """LangChain chat model that streams the Databricks Genie Agent Mode API.

    Not registered as a serving endpoint — it targets
    ``/api/2.0/genie/agents/{agent_id}/responses`` directly. Bind a specific
    :class:`~databricks.sdk.WorkspaceClient` (SP or forwarded-user) at
    construction so OBO is honored per request.
    """

    agent_id: str
    """32-char hex Genie agent/space id (the renamed ``space_id``)."""

    workspace_client: Any
    """The :class:`~databricks.sdk.WorkspaceClient` whose auth identity Genie
    sees. Swapped per request for OBO. Typed ``Any`` to avoid a strict pydantic
    ``isinstance`` gate — only ``.config.host`` and ``.config.authenticate()``
    (via :class:`~dao_ai.auth.WorkspaceBearerAuth`) are used."""

    conversation_id: Optional[str] = None
    """Genie-issued conversation id to continue on this call, or ``None`` to
    start a new Genie conversation. The Genie service owns this value; the model
    only replays it. Set per request by
    :class:`dao_ai.middleware.genie_agent.GenieAgentMiddleware`, which reads it
    from ``session.genie.spaces[agent_id]`` and writes the newly-issued id back
    to the same channel. Kept independent of the LangGraph ``thread_id`` used
    for graph-state persistence."""

    timeout_seconds: int = 300
    """httpx client timeout. The Genie server-side response cap is 90 minutes."""

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "databricks-genie-agent"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "timeout_seconds": self.timeout_seconds}

    # -- request helpers ------------------------------------------------

    def _url(self) -> str:
        host: str = self.workspace_client.config.host.rstrip("/")
        return f"{host}/api/2.0/genie/agents/{self.agent_id}/responses"

    def _body(self, messages: list[BaseMessage]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": _latest_human_text(messages)}
                    ],
                }
            ]
        }
        if self.conversation_id:
            body["conversation_id"] = self.conversation_id
        return body

    def _set_span_attributes(self, state: _StreamState) -> None:
        span = mlflow.get_current_active_span()
        if span is None:
            return
        span.set_attribute(ATTR_AGENT_ID, self.agent_id)
        span.set_attribute(ATTR_TOTAL_EVENTS, state.event_count)
        if state.conversation_id:
            span.set_attribute(ATTR_CONVERSATION_ID, state.conversation_id)
        if state.response_id:
            span.set_attribute(ATTR_RESPONSE_ID, state.response_id)
        if state.terminal_status:
            span.set_attribute(ATTR_TERMINAL_STATE, state.terminal_status)

    def _final_message(self, state: _StreamState) -> AIMessage:
        response_metadata: dict[str, Any] = {}
        if state.conversation_id:
            response_metadata[CONVERSATION_ID_METADATA_KEY] = state.conversation_id
        if state.response_id:
            response_metadata["genie_response_id"] = state.response_id
        return AIMessage(
            content=state.aggregated_content(),
            response_metadata=response_metadata,
        )

    @staticmethod
    def _raise_for_http(status_code: int, body: bytes) -> None:
        message: str = body.decode("utf-8", errors="replace") if body else ""
        raise GenieAgentError(
            f"Genie Agent Mode API returned HTTP {status_code}: {message}"
        )

    # -- async streaming (primary path) --------------------------------

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        state = _StreamState()
        url: str = self._url()
        body: dict[str, Any] = self._body(messages)

        async with httpx.AsyncClient(
            auth=WorkspaceBearerAuth(self.workspace_client),
            timeout=self.timeout_seconds,
        ) as client:
            async with client.stream(
                "POST", url, json=body, headers={"Accept": "text/event-stream"}
            ) as response:
                if response.status_code >= 400:
                    self._raise_for_http(response.status_code, await response.aread())

                buffer: list[str] = []
                async for raw_line in response.aiter_lines():
                    buffer.append(raw_line)
                    if raw_line.rstrip("\r") != "":
                        continue
                    # Complete record boundary — parse and drain.
                    for event_type, payload in _parse_sse_lines(buffer):
                        text: Optional[str] = state.handle(event_type, payload)
                        if text:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=text)
                            )
                            if run_manager:
                                await run_manager.on_llm_new_token(text, chunk=chunk)
                            yield chunk
                    buffer = []

        self._set_span_attributes(state)
        state.raise_on_error()

        # Emit a terminal chunk carrying the conversation_id metadata so the
        # accumulated AIMessage the caller assembles can be continued next turn.
        final_meta: dict[str, Any] = {}
        if state.conversation_id:
            final_meta[CONVERSATION_ID_METADATA_KEY] = state.conversation_id
        if state.response_id:
            final_meta["genie_response_id"] = state.response_id
        if final_meta:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", response_metadata=final_meta)
            )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        state = _StreamState()
        url: str = self._url()
        body: dict[str, Any] = self._body(messages)

        async with httpx.AsyncClient(
            auth=WorkspaceBearerAuth(self.workspace_client),
            timeout=self.timeout_seconds,
        ) as client:
            async with client.stream(
                "POST", url, json=body, headers={"Accept": "text/event-stream"}
            ) as response:
                if response.status_code >= 400:
                    self._raise_for_http(response.status_code, await response.aread())
                lines: list[str] = [line async for line in response.aiter_lines()]

        for event_type, payload in _parse_sse_lines(lines):
            state.handle(event_type, payload)
        self._set_span_attributes(state)
        state.raise_on_error()
        return ChatResult(generations=[ChatGeneration(message=self._final_message(state))])

    # -- sync path (aggregating) ---------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        state = _StreamState()
        url: str = self._url()
        body: dict[str, Any] = self._body(messages)

        with httpx.Client(
            auth=WorkspaceBearerAuth(self.workspace_client),
            timeout=self.timeout_seconds,
        ) as client:
            with client.stream(
                "POST", url, json=body, headers={"Accept": "text/event-stream"}
            ) as response:
                if response.status_code >= 400:
                    self._raise_for_http(response.status_code, response.read())
                lines: list[str] = list(response.iter_lines())

        for event_type, payload in _parse_sse_lines(lines):
            state.handle(event_type, payload)
        self._set_span_attributes(state)
        state.raise_on_error()
        return ChatResult(generations=[ChatGeneration(message=self._final_message(state))])
