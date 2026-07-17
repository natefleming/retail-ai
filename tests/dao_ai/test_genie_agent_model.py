"""Unit tests for the Genie Agent *model* resource.

Covers :class:`dao_ai.genie.agent_chat_model.GenieAgentChatModel` and
:class:`dao_ai.config.GenieAgentModel`:

1. SSE stream aggregation (sync ``_generate`` + async ``_astream``).
2. Multi-turn: the prior ``conversation_id`` is recovered from the most
   recent ``AIMessage.response_metadata`` and sent back in the body; a new
   id is stamped on the returned message.
3. ``_astream`` yields ``AIMessageChunk``s in stream order.
4. HTTP >=400 and ``response.failed`` both raise :class:`GenieAgentError`.
5. Config: ``GenieAgentModel`` exposes the ``InferenceEndpointModel``-duck-
   typed surface (``name``, ``on_behalf_of_user``, ``workspace_client_from``,
   ``chat_model_for_workspace_client``, ``as_chat_model``) and parses through
   the ``AgentModel.model`` union.

httpx is driven with a real ``MockTransport`` so the SSE parsing exercises the
genuine ``iter_lines``/``aiter_lines`` code paths. The WorkspaceClient is a
stub exposing only ``config.host`` and ``config.authenticate``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Iterable
from unittest.mock import MagicMock

import httpx
import pytest

from dao_ai.config import AgentModel, GenieAgentModel, GenieRoomModel, InferenceEndpointModel
from dao_ai.genie.agent_chat_model import (
    CONVERSATION_ID_METADATA_KEY,
    GenieAgentChatModel,
    GenieAgentError,
)
from langchain_core.messages import AIMessage, HumanMessage

AGENT_ID: str = "01f05dd06c421ad6b522bf7a517cf6d2"
HOST: str = "https://example.cloud.databricks.com"


# ---------------------------------------------------------------------------
# SSE fixtures
# ---------------------------------------------------------------------------


def _sse(events: Iterable[tuple[str, dict[str, Any]]]) -> bytes:
    lines: list[str] = []
    for event_type, payload in events:
        lines.append(f"event: {event_type}")
        lines.append(f"data: {json.dumps(payload)}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _happy_events(conversation_id: str = "conv-1") -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "response.created",
            {"response": {"id": "resp-1", "conversation_id": conversation_id, "status": "in_progress"}},
        ),
        (
            "response.output_item.done",
            {
                "item": {
                    "id": "call-a",
                    "type": "function_call",
                    "name": "execute_sql",
                    "arguments": json.dumps(
                        {"title": "Store count", "sql": "SELECT state, COUNT(*) FROM stores GROUP BY state"}
                    ),
                }
            },
        ),
        (
            "response.output_item.done",
            {"item": {"id": "out-a", "type": "function_call_output", "output": "| state | count |\n|---|---|\n| CA | 42 |\n"}},
        ),
        (
            "response.output_item.done",
            {"item": {"id": "msg-a", "type": "message", "content": [{"type": "output_text", "text": "California leads with 42 stores."}]}},
        ),
        (
            "response.completed",
            {"response": {"id": "resp-1", "conversation_id": conversation_id, "status": "completed"}},
        ),
    ]


def _fake_workspace_client() -> Any:
    """A duck-typed WorkspaceClient: ``config.host`` + ``config.authenticate``.

    ``GenieAgentChatModel.workspace_client`` is typed ``Any`` and only touches
    ``.config.host`` and ``.config.authenticate()`` (via ``WorkspaceBearerAuth``),
    so a plain stub suffices — no need to construct the real SDK client.
    """
    ws = MagicMock(name="WorkspaceClient")
    ws.config.host = HOST
    ws.config.authenticate.return_value = {"Authorization": "Bearer stub-token"}
    return ws


def _model_with_transport(
    handler: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    record: dict[str, Any] | None = None,
) -> GenieAgentChatModel:
    """Build a GenieAgentChatModel whose httpx clients use a MockTransport."""

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record["url"] = str(request.url)
            record["body"] = json.loads(request.content.decode("utf-8"))
            record["auth"] = request.headers.get("Authorization")
        return handler(request)

    transport = httpx.MockTransport(_mock_handler)

    real_async_init = httpx.AsyncClient.__init__
    real_sync_init = httpx.Client.__init__

    def _async_init(self: httpx.AsyncClient, **kwargs: Any) -> None:
        kwargs["transport"] = transport
        real_async_init(self, **kwargs)

    def _sync_init(self: httpx.Client, **kwargs: Any) -> None:
        kwargs["transport"] = transport
        real_sync_init(self, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", _async_init)
    monkeypatch.setattr(httpx.Client, "__init__", _sync_init)

    return GenieAgentChatModel(
        agent_id=AGENT_ID,
        workspace_client=_fake_workspace_client(),
    )


# ---------------------------------------------------------------------------
# Streaming aggregation
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_generate_aggregates_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record: dict[str, Any] = {}
        model = _model_with_transport(
            lambda req: httpx.Response(200, content=_sse(_happy_events())),
            monkeypatch,
            record=record,
        )
        result = model.invoke([HumanMessage("How many stores by state?")])
        content = result.content
        assert "```sql" in content
        assert "SELECT state, COUNT(*) FROM stores" in content
        assert "California leads with 42 stores." in content
        assert "| state | count |" in content
        # conversation_id stamped on the returned message.
        assert result.response_metadata[CONVERSATION_ID_METADATA_KEY] == "conv-1"
        # Request shape: single user turn, no conversation_id on first turn.
        assert record["url"].endswith(f"/api/2.0/genie/agents/{AGENT_ID}/responses")
        assert record["body"]["input"][0]["content"][0]["text"] == "How many stores by state?"
        assert "conversation_id" not in record["body"]
        assert record["auth"] == "Bearer stub-token"

    def test_astream_yields_chunks_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _model_with_transport(
            lambda req: httpx.Response(200, content=_sse(_happy_events())),
            monkeypatch,
        )

        async def _collect() -> list[str]:
            chunks: list[str] = []
            async for chunk in model.astream([HumanMessage("q")]):
                chunks.append(chunk.content)
            return chunks

        chunks = asyncio.run(_collect())
        joined = "".join(chunks)
        # SQL precedes table precedes narrative.
        assert joined.index("```sql") < joined.index("| state | count |") < joined.index("California leads")

    def test_astream_final_chunk_carries_conversation_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _model_with_transport(
            lambda req: httpx.Response(200, content=_sse(_happy_events())),
            monkeypatch,
        )

        async def _accumulate() -> AIMessage:
            acc = None
            async for chunk in model.astream([HumanMessage("q")]):
                acc = chunk if acc is None else acc + chunk
            return acc

        acc = asyncio.run(_accumulate())
        assert acc.response_metadata.get(CONVERSATION_ID_METADATA_KEY) == "conv-1"


# ---------------------------------------------------------------------------
# conversation_id is a pure field on the model (no message scanning)
# ---------------------------------------------------------------------------


class TestConversationIdField:
    def test_conversation_id_field_sent_in_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record: dict[str, Any] = {}
        model = _model_with_transport(
            lambda req: httpx.Response(200, content=_sse(_happy_events("conv-2"))),
            monkeypatch,
            record=record,
        )
        # The field — not any message metadata — drives the body.
        model.conversation_id = "conv-existing"
        model.invoke(
            [
                HumanMessage("first"),
                AIMessage(
                    "prev answer",
                    response_metadata={CONVERSATION_ID_METADATA_KEY: "ignored-metadata"},
                ),
                HumanMessage("follow-up"),
            ]
        )
        assert record["body"]["conversation_id"] == "conv-existing"
        # Latest human turn is the one sent.
        assert record["body"]["input"][0]["content"][0]["text"] == "follow-up"

    def test_no_conversation_id_omits_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record: dict[str, Any] = {}
        model = _model_with_transport(
            lambda req: httpx.Response(200, content=_sse(_happy_events())),
            monkeypatch,
            record=record,
        )
        # Even with a prior AIMessage carrying metadata, the model does not
        # scan it — with conversation_id unset, no conversation_id is sent.
        model.invoke(
            [
                AIMessage(
                    "prev", response_metadata={CONVERSATION_ID_METADATA_KEY: "should-not-be-used"}
                ),
                HumanMessage("q"),
            ]
        )
        assert "conversation_id" not in record["body"]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_http_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _model_with_transport(
            lambda req: httpx.Response(404, content=b'{"error_code":"FEATURE_DISABLED"}'),
            monkeypatch,
        )
        with pytest.raises(GenieAgentError, match="404"):
            model.invoke([HumanMessage("q")])

    def test_response_failed_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            ("response.created", {"response": {"id": "r", "conversation_id": "c", "status": "in_progress"}}),
            (
                "response.failed",
                {"response": {"id": "r", "conversation_id": "c", "status": "failed", "error": {"code": "sql_execution_error", "message": "Table not found"}}},
            ),
        ]
        model = _model_with_transport(
            lambda req: httpx.Response(200, content=_sse(events)),
            monkeypatch,
        )
        with pytest.raises(GenieAgentError, match="sql_execution_error"):
            model.invoke([HumanMessage("q")])

    def test_no_human_message_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _model_with_transport(
            lambda req: httpx.Response(200, content=_sse(_happy_events())),
            monkeypatch,
        )
        with pytest.raises(GenieAgentError, match="no HumanMessage"):
            model.invoke([AIMessage("only assistant")])


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestConfigModel:
    def test_duck_typed_surface(self) -> None:
        m = GenieAgentModel(genie_room=GenieRoomModel(space_id=AGENT_ID))
        assert m.name == AGENT_ID
        assert m.on_behalf_of_user is False
        # Methods OBOModelMiddleware / create_agent_node rely on exist.
        assert hasattr(m, "workspace_client_from")
        assert hasattr(m, "chat_model_for_workspace_client")
        assert hasattr(m, "as_chat_model")

    def test_agent_id_alias(self) -> None:
        m = GenieAgentModel.model_validate({"genie_room": {"agent_id": AGENT_ID}})
        assert m.name == AGENT_ID

    def test_obo_delegates_to_room(self) -> None:
        m = GenieAgentModel(genie_room=GenieRoomModel(space_id=AGENT_ID, on_behalf_of_user=True))
        assert m.on_behalf_of_user is True

    def test_parses_through_agent_model_union(self) -> None:
        agent = AgentModel.model_validate(
            {"name": "genie_specialist", "model": {"genie_room": {"agent_id": AGENT_ID}}, "tools": []}
        )
        assert isinstance(agent.model, GenieAgentModel)

    def test_serving_endpoint_still_parses(self) -> None:
        agent = AgentModel.model_validate(
            {"name": "normal", "model": {"name": "databricks-claude-sonnet-4"}, "tools": []}
        )
        assert isinstance(agent.model, InferenceEndpointModel)

    def test_unresolvable_agent_id_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A name-only room (no space_id) with no env fallback cannot resolve
        # an agent_id without a live lookup.
        monkeypatch.delenv("DATABRICKS_GENIE_SPACE_ID", raising=False)
        m = GenieAgentModel(genie_room=GenieRoomModel(name="Some Room"))
        with pytest.raises(ValueError, match="unable to resolve agent_id"):
            _ = m.name


# ---------------------------------------------------------------------------
# AppConfig-level: GenieAgentModel room must be registered under genie_rooms
# ---------------------------------------------------------------------------


class TestGenieRoomRegistrationValidator:
    def test_unregistered_room_rejected(self) -> None:
        from dao_ai.config import AppConfig

        cfg = {
            "agents": {
                "g": {"name": "g", "model": {"genie_room": {"agent_id": AGENT_ID}}, "tools": []}
            }
        }
        with pytest.raises(ValueError, match="not registered under resources.genie_rooms"):
            AppConfig(**cfg)

    def test_registered_room_passes(self) -> None:
        from dao_ai.config import AppConfig

        cfg = {
            "resources": {"genie_rooms": {"retail": {"agent_id": AGENT_ID}}},
            "agents": {
                "g": {"name": "g", "model": {"genie_room": {"agent_id": AGENT_ID}}, "tools": []}
            },
        }
        # Should not raise.
        AppConfig(**cfg)

    def test_serving_endpoint_agent_unaffected(self) -> None:
        from dao_ai.config import AppConfig

        cfg = {
            "agents": {
                "x": {"name": "x", "model": {"name": "databricks-claude-sonnet-4"}, "tools": []}
            }
        }
        # A non-Genie model is never subject to the genie_rooms check.
        AppConfig(**cfg)
