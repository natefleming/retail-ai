"""Unit tests for :class:`dao_ai.middleware.genie_agent.GenieAgentMiddleware`.

The middleware owns Genie conversation continuity (independent of the LangGraph
thread_id). It must:

1. Read the prior conversation_id from ``session.genie.spaces[agent_id]`` and
   build the per-request model with it (and an OBO-aware workspace client).
2. Read the Genie-issued conversation_id off the response and persist it back
   to ``session`` via an ``ExtendedModelResponse`` Command — keyed by agent_id.
3. Not emit a session update when the id is unchanged.

The model build + handler are stubbed; this tests the middleware's read/write
wiring, not the SSE call (covered by test_genie_agent_model.py).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from langchain.agents.middleware.types import ExtendedModelResponse, ModelResponse
from langchain_core.messages import AIMessage

from dao_ai.config import GenieAgentModel, GenieRoomModel
from dao_ai.genie.agent_chat_model import CONVERSATION_ID_METADATA_KEY
from dao_ai.middleware.genie_agent import GenieAgentMiddleware
from dao_ai.state import Context, SessionState

AGENT_ID = "01f05dd06c421ad6b522bf7a517cf6d2"


def _middleware(monkeypatch: Any) -> tuple[GenieAgentMiddleware, dict[str, Any]]:
    """Build a middleware whose model-build is stubbed to record its args."""
    model = GenieAgentModel(genie_room=GenieRoomModel(space_id=AGENT_ID))
    built: dict[str, Any] = {}

    # Stub OBO client resolution + the per-request model build so we don't hit
    # the network or need a real WorkspaceClient.
    monkeypatch.setattr(
        GenieRoomModel, "workspace_client_from", lambda self, ctx, *, strict=False: MagicMock()
    )

    def _fake_build(self: GenieAgentModel, ws: Any, *, conversation_id: str | None = None) -> Any:
        built["conversation_id"] = conversation_id
        return MagicMock(name="GenieAgentChatModel")

    monkeypatch.setattr(GenieAgentModel, "chat_model_for_workspace_client", _fake_build)
    return GenieAgentMiddleware(genie_model=model), built


def _request(session: SessionState | None) -> MagicMock:
    request = MagicMock(name="ModelRequest")
    request.state = {"session": session} if session is not None else {}
    request.runtime.context = Context(user_id="u", thread_id="thr-1")
    request.override.return_value = request
    return request


def _handler_returning(conversation_id: str | None):
    def _handler(_req: Any) -> ModelResponse:
        meta = {CONVERSATION_ID_METADATA_KEY: conversation_id} if conversation_id else {}
        return ModelResponse(result=[AIMessage("answer", response_metadata=meta)])

    return _handler


class TestReadPriorId:
    def test_prior_id_passed_to_model_build(self, monkeypatch: Any) -> None:
        mw, built = _middleware(monkeypatch)
        session = SessionState()
        session.genie.update_space(space_id=AGENT_ID, conversation_id="conv-prior")
        request = _request(session)

        mw.wrap_model_call(request, _handler_returning("conv-prior"))
        assert built["conversation_id"] == "conv-prior"

    def test_no_session_means_no_prior_id(self, monkeypatch: Any) -> None:
        mw, built = _middleware(monkeypatch)
        request = _request(None)
        mw.wrap_model_call(request, _handler_returning("conv-new"))
        assert built["conversation_id"] is None


class TestPersistIssuedId:
    def test_new_id_persisted_to_session(self, monkeypatch: Any) -> None:
        mw, _ = _middleware(monkeypatch)
        request = _request(None)
        result = mw.wrap_model_call(request, _handler_returning("conv-new"))
        assert isinstance(result, ExtendedModelResponse)
        session: SessionState = result.command.update["session"]
        assert session.genie.get_conversation_id(AGENT_ID) == "conv-new"

    def test_unchanged_id_no_command(self, monkeypatch: Any) -> None:
        mw, _ = _middleware(monkeypatch)
        session = SessionState()
        session.genie.update_space(space_id=AGENT_ID, conversation_id="conv-same")
        request = _request(session)
        result = mw.wrap_model_call(request, _handler_returning("conv-same"))
        # Unchanged id -> plain ModelResponse, no session write.
        assert not isinstance(result, ExtendedModelResponse)

    def test_no_issued_id_no_command(self, monkeypatch: Any) -> None:
        mw, _ = _middleware(monkeypatch)
        request = _request(None)
        result = mw.wrap_model_call(request, _handler_returning(None))
        assert not isinstance(result, ExtendedModelResponse)

    def test_id_keyed_by_agent_id(self, monkeypatch: Any) -> None:
        mw, _ = _middleware(monkeypatch)
        # A different agent's conversation must not be clobbered.
        session = SessionState()
        session.genie.update_space(space_id="other-agent", conversation_id="other-conv")
        request = _request(session)
        result = mw.wrap_model_call(request, _handler_returning("conv-new"))
        merged: SessionState = result.command.update["session"]
        assert merged.genie.get_conversation_id(AGENT_ID) == "conv-new"
        assert merged.genie.get_conversation_id("other-agent") == "other-conv"


class TestAsync:
    def test_awrap_persists(self, monkeypatch: Any) -> None:
        mw, built = _middleware(monkeypatch)
        session = SessionState()
        session.genie.update_space(space_id=AGENT_ID, conversation_id="conv-prior")
        request = _request(session)

        async def _ahandler(_req: Any) -> ModelResponse:
            return ModelResponse(
                result=[AIMessage("a", response_metadata={CONVERSATION_ID_METADATA_KEY: "conv-next"})]
            )

        result = asyncio.run(mw.awrap_model_call(request, _ahandler))
        assert built["conversation_id"] == "conv-prior"
        assert isinstance(result, ExtendedModelResponse)
        assert result.command.update["session"].genie.get_conversation_id(AGENT_ID) == "conv-next"
