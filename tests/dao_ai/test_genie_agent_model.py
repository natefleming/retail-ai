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
6. ``bind_tools`` is a no-op returning the model itself, so a langchain agent
   loop that hands the brain a tool (supervisor handoff, swarm handoff) runs
   instead of dying in ``BaseChatModel.bind_tools``.

httpx is driven with a real ``MockTransport`` so the SSE parsing exercises the
genuine ``iter_lines``/``aiter_lines`` code paths. The WorkspaceClient is a
stub exposing only ``config.host`` and ``config.authenticate``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Iterable
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from dao_ai.config import (
    AgentModel,
    GenieAgentModel,
    GenieRoomModel,
    InferenceEndpointModel,
)
from dao_ai.genie.agent_chat_model import (
    CONVERSATION_ID_METADATA_KEY,
    GenieAgentChatModel,
    GenieAgentError,
)

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
            {
                "response": {
                    "id": "resp-1",
                    "conversation_id": conversation_id,
                    "status": "in_progress",
                }
            },
        ),
        (
            "response.output_item.done",
            {
                "item": {
                    "id": "call-a",
                    "type": "function_call",
                    "name": "execute_sql",
                    "arguments": json.dumps(
                        {
                            "title": "Store count",
                            "sql": "SELECT state, COUNT(*) FROM stores GROUP BY state",
                        }
                    ),
                }
            },
        ),
        (
            "response.output_item.done",
            {
                "item": {
                    "id": "out-a",
                    "type": "function_call_output",
                    "output": "| state | count |\n|---|---|\n| CA | 42 |\n",
                }
            },
        ),
        (
            "response.output_item.done",
            {
                "item": {
                    "id": "msg-a",
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "California leads with 42 stores.",
                        }
                    ],
                }
            },
        ),
        (
            "response.completed",
            {
                "response": {
                    "id": "resp-1",
                    "conversation_id": conversation_id,
                    "status": "completed",
                }
            },
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
        assert (
            record["body"]["input"][0]["content"][0]["text"]
            == "How many stores by state?"
        )
        assert "conversation_id" not in record["body"]
        assert record["auth"] == "Bearer stub-token"

    def test_astream_yields_chunks_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        assert (
            joined.index("```sql")
            < joined.index("| state | count |")
            < joined.index("California leads")
        )

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
    def test_conversation_id_field_sent_in_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
                    response_metadata={
                        CONVERSATION_ID_METADATA_KEY: "ignored-metadata"
                    },
                ),
                HumanMessage("follow-up"),
            ]
        )
        assert record["body"]["conversation_id"] == "conv-existing"
        # Latest human turn is the one sent.
        assert record["body"]["input"][0]["content"][0]["text"] == "follow-up"

    def test_no_conversation_id_omits_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
                    "prev",
                    response_metadata={
                        CONVERSATION_ID_METADATA_KEY: "should-not-be-used"
                    },
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
            lambda req: httpx.Response(
                404, content=b'{"error_code":"FEATURE_DISABLED"}'
            ),
            monkeypatch,
        )
        with pytest.raises(GenieAgentError, match="404"):
            model.invoke([HumanMessage("q")])

    def test_response_failed_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            (
                "response.created",
                {
                    "response": {
                        "id": "r",
                        "conversation_id": "c",
                        "status": "in_progress",
                    }
                },
            ),
            (
                "response.failed",
                {
                    "response": {
                        "id": "r",
                        "conversation_id": "c",
                        "status": "failed",
                        "error": {
                            "code": "sql_execution_error",
                            "message": "Table not found",
                        },
                    }
                },
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
        m = GenieAgentModel(
            genie_room=GenieRoomModel(space_id=AGENT_ID, on_behalf_of_user=True)
        )
        assert m.on_behalf_of_user is True

    def test_parses_through_agent_model_union(self) -> None:
        agent = AgentModel.model_validate(
            {
                "name": "genie_specialist",
                "model": {"genie_room": {"agent_id": AGENT_ID}},
                "tools": [],
            }
        )
        assert isinstance(agent.model, GenieAgentModel)

    def test_serving_endpoint_still_parses(self) -> None:
        agent = AgentModel.model_validate(
            {
                "name": "normal",
                "model": {"name": "databricks-claude-sonnet-4"},
                "tools": [],
            }
        )
        assert isinstance(agent.model, InferenceEndpointModel)

    def test_bare_room_dict_autowrapped(self) -> None:
        # A bare room dict (agent_id) assigned to model is wrapped.
        agent = AgentModel.model_validate(
            {"name": "g", "model": {"agent_id": AGENT_ID}, "tools": []}
        )
        assert isinstance(agent.model, GenieAgentModel)
        assert agent.model.name == AGENT_ID

    def test_bare_room_instance_autowrapped(self) -> None:
        agent = AgentModel.model_validate(
            {"name": "g", "model": GenieRoomModel(space_id=AGENT_ID), "tools": []}
        )
        assert isinstance(agent.model, GenieAgentModel)

    def test_bare_room_uses_default_timeout(self) -> None:
        agent = AgentModel.model_validate(
            {"name": "g", "model": {"space_id": AGENT_ID}, "tools": []}
        )
        assert agent.model.timeout_seconds == 300

    def test_explicit_wrapper_keeps_custom_timeout(self) -> None:
        agent = AgentModel.model_validate(
            {
                "name": "g",
                "model": {"genie_room": {"agent_id": AGENT_ID}, "timeout_seconds": 600},
                "tools": [],
            }
        )
        assert isinstance(agent.model, GenieAgentModel)
        assert agent.model.timeout_seconds == 600

    def test_unresolvable_agent_id_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
                "g": {
                    "name": "g",
                    "model": {"genie_room": {"agent_id": AGENT_ID}},
                    "tools": [],
                }
            }
        }
        with pytest.raises(
            ValueError, match="not registered under resources.genie_rooms"
        ):
            AppConfig(**cfg)

    def test_registered_room_passes(self) -> None:
        from dao_ai.config import AppConfig

        cfg = {
            "resources": {"genie_rooms": {"retail": {"agent_id": AGENT_ID}}},
            "agents": {
                "g": {
                    "name": "g",
                    "model": {"genie_room": {"agent_id": AGENT_ID}},
                    "tools": [],
                }
            },
        }
        # Should not raise.
        AppConfig(**cfg)

    def test_serving_endpoint_agent_unaffected(self) -> None:
        from dao_ai.config import AppConfig

        cfg = {
            "agents": {
                "x": {
                    "name": "x",
                    "model": {"name": "databricks-claude-sonnet-4"},
                    "tools": [],
                }
            }
        }
        # A non-Genie model is never subject to the genie_rooms check.
        AppConfig(**cfg)


# ---------------------------------------------------------------------------
# Bare-room coercion: shape test breadth
# ---------------------------------------------------------------------------


class TestBareRoomCoercion:
    """``AgentModel.model`` coerces a bare Genie room into a ``GenieAgentModel``.

    The coercion is by *shape*: a dict carrying any key only a room can have.
    Testing ``space_id``/``agent_id`` alone is not enough — a managed room is
    declared with provisioning fields and may legitimately have neither, and a
    YAML anchor expands to a plain dict, so those rooms take the dict path.
    """

    @staticmethod
    def _model_for(shape: object) -> object:
        return AgentModel.model_validate(
            {"name": "a", "model": shape, "tools": []}
        ).model

    def test_provisioning_room_without_space_id_is_wrapped(self) -> None:
        """A managed room declared by table_sources/warehouse — no space_id."""
        model = self._model_for(
            {
                "name": "Retail Genie",
                "warehouse": {"name": "shared-wh"},
                "table_sources": [{"table": {"name": "cat.sch.sales"}}],
            }
        )
        assert isinstance(model, GenieAgentModel)

    def test_text_instructions_only_room_is_wrapped(self) -> None:
        model = self._model_for(
            {"name": "Retail Genie", "text_instructions": ["Prefer aggregates."]}
        )
        assert isinstance(model, GenieAgentModel)

    def test_sample_questions_room_is_wrapped(self) -> None:
        model = self._model_for(
            {"name": "Retail Genie", "sample_questions": ["How many stores?"]}
        )
        assert isinstance(model, GenieAgentModel)

    def test_plain_endpoint_name_still_resolves_to_endpoint(self) -> None:
        """The shared-key shape must not be stolen from serving endpoints."""
        model = self._model_for({"name": "databricks-claude-sonnet-4"})
        assert isinstance(model, InferenceEndpointModel)

    def test_endpoint_with_inference_knobs_still_resolves_to_endpoint(self) -> None:
        model = self._model_for(
            {
                "name": "databricks-claude-sonnet-4",
                "temperature": 0.1,
                "max_tokens": 100,
            }
        )
        assert isinstance(model, InferenceEndpointModel)

    def test_room_only_keys_exclude_shared_keys(self) -> None:
        """The derived key set must never contain a key both classes accept.

        Deriving it from ``model_fields`` keeps it from drifting as fields are
        added; this pins the property that makes it safe to test against.
        """
        from dao_ai.config import _genie_room_only_keys

        room_only = _genie_room_only_keys()
        for shared in (
            "name",
            "description",
            "on_behalf_of_user",
            "pat",
            "service_principal",
            "client_id",
            "client_secret",
            "workspace_host",
        ):
            assert shared not in room_only
        # And the keys the coercion depends on are present.
        for room_key in ("space_id", "agent_id", "table_sources", "warehouse"):
            assert room_key in room_only


# ---------------------------------------------------------------------------
# Chat-model surface lives on GenieRoomModel; the wrapper only overrides timeout
# ---------------------------------------------------------------------------


class TestRoomChatModelSurface:
    def test_room_builds_chat_model_with_default_timeout(self) -> None:
        from dao_ai.config import GENIE_AGENT_DEFAULT_TIMEOUT_SECONDS

        room = GenieRoomModel(space_id=AGENT_ID)
        chat = room.chat_model_for_workspace_client(_fake_workspace_client())
        assert isinstance(chat, GenieAgentChatModel)
        assert chat.agent_id == AGENT_ID
        assert chat.timeout_seconds == GENIE_AGENT_DEFAULT_TIMEOUT_SECONDS

    def test_as_chat_model_uses_room_client(self) -> None:
        room = GenieRoomModel(space_id=AGENT_ID)
        ws = _fake_workspace_client()
        with patch.object(
            GenieRoomModel,
            "workspace_client",
            new_callable=lambda: property(lambda s: ws),
        ):
            chat = room.as_chat_model()
        assert isinstance(chat, GenieAgentChatModel)
        assert chat.workspace_client is ws

    def test_wrapper_forwards_its_timeout_override(self) -> None:
        agent = AgentModel.model_validate(
            {
                "name": "g",
                "model": {"genie_room": {"agent_id": AGENT_ID}, "timeout_seconds": 600},
                "tools": [],
            }
        )
        chat = agent.model.chat_model_for_workspace_client(_fake_workspace_client())
        assert chat.timeout_seconds == 600

    def test_wrapper_name_is_agent_id_not_room_title(self) -> None:
        """``nodes.py`` and trace ``ResourceInfo`` log this — it must stay the id."""
        m = GenieAgentModel(
            genie_room=GenieRoomModel(name="Retail Sales", space_id=AGENT_ID)
        )
        assert m.name == AGENT_ID

    def test_room_agent_id_does_not_fall_back_to_name_lookup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolution is static-only: no Genie API call on the model-build path.

        A name-only room therefore cannot serve as a model, which is why
        ``AppConfig`` reports the bare-assignment case loudly instead.
        """
        monkeypatch.delenv("DATABRICKS_GENIE_SPACE_ID", raising=False)
        room = GenieRoomModel(name="Some Room")
        with patch.object(
            GenieRoomModel,
            "_resolve_space_id_by_name",
            side_effect=AssertionError("live lookup on the model-build path"),
        ):
            with pytest.raises(ValueError, match="unable to resolve agent_id"):
                _ = room._agent_id


# ---------------------------------------------------------------------------
# AppConfig catches the shape coercion cannot decide
# ---------------------------------------------------------------------------


class TestBareNameOnlyRoomRejected:
    """``{name: X}`` is valid for both union members, so shape cannot decide it.

    It silently resolves to an ``InferenceEndpointModel`` — the config loads
    clean and the agent then points at a serving endpoint that does not exist.
    ``AppConfig`` sees the room registry, so it can catch this.
    """

    def test_name_only_room_bare_assigned_raises(self) -> None:
        from dao_ai.config import AppConfig

        cfg = {
            "resources": {"genie_rooms": {"retail": {"name": "Retail Genie"}}},
            "agents": {
                "g": {"name": "g", "model": {"name": "Retail Genie"}, "tools": []}
            },
        }
        with pytest.raises(ValueError, match="indistinguishable from a serving"):
            AppConfig(**cfg)

    def test_explicit_wrapper_is_not_flagged(self) -> None:
        from dao_ai.config import AppConfig

        cfg = {
            "resources": {"genie_rooms": {"retail": {"name": "Retail Genie"}}},
            "agents": {
                "g": {
                    "name": "g",
                    "model": {"genie_room": {"name": "Retail Genie"}},
                    "tools": [],
                }
            },
        }
        # Name-only rooms cannot be id-matched, so the registration check skips.
        AppConfig(**cfg)

    def test_unrelated_endpoint_name_is_untouched(self) -> None:
        from dao_ai.config import AppConfig

        cfg = {
            "resources": {"genie_rooms": {"retail": {"name": "Retail Genie"}}},
            "agents": {
                "g": {
                    "name": "g",
                    "model": {"name": "databricks-claude-sonnet-4"},
                    "tools": [],
                }
            },
        }
        AppConfig(**cfg)

    def test_endpoint_with_knobs_sharing_a_room_name_is_untouched(self) -> None:
        """Any endpoint-specific key means the user meant an endpoint."""
        from dao_ai.config import AppConfig

        cfg = {
            "resources": {"genie_rooms": {"retail": {"name": "Retail Genie"}}},
            "agents": {
                "g": {
                    "name": "g",
                    "model": {"name": "Retail Genie", "temperature": 0.1},
                    "tools": [],
                }
            },
        }
        AppConfig(**cfg)


# ---------------------------------------------------------------------------
# bind_tools: Genie owns its tool loop, client tools are ignored
# ---------------------------------------------------------------------------


class TestBindTools:
    def test_returns_self(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _model_with_transport(
            lambda req: httpx.Response(200, content=_sse(_happy_events())),
            monkeypatch,
        )

        @tool
        def handoff_to_supervisor() -> str:
            """Hand control back to the supervisor."""
            return "ok"

        assert model.bind_tools([handoff_to_supervisor]) is model
        assert model.bind_tools([]) is model
        assert model.bind_tools([handoff_to_supervisor], tool_choice="any") is model

    def test_agent_loop_with_a_tool_answers_from_genie(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure this guards: ``create_agent(model, tools=[...])`` calls
        ``bind_tools`` on the first model call whenever the tool list is
        non-empty. The brain must take that turn, answer with Genie's text and
        emit no tool call, so the loop ends after one model call."""
        record: dict[str, Any] = {}
        model = _model_with_transport(
            lambda req: httpx.Response(200, content=_sse(_happy_events())),
            monkeypatch,
            record=record,
        )
        calls: list[str] = []

        @tool
        def handoff_to_supervisor() -> str:
            """Hand control back to the supervisor."""
            calls.append("handoff")
            return "ok"

        agent = create_agent(model=model, tools=[handoff_to_supervisor])
        result = agent.invoke({"messages": [HumanMessage("How many stores by state?")]})

        last = result["messages"][-1]
        assert isinstance(last, AIMessage)
        assert "California leads with 42 stores." in last.content
        assert not last.tool_calls
        assert calls == []
        assert record["url"].endswith(f"/api/2.0/genie/agents/{AGENT_ID}/responses")
