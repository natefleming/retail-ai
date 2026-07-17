"""Per-request middleware for :class:`~dao_ai.config.GenieAgentModel` agents.

A Genie-Agent-backed agent has two per-request concerns that a plain
``BaseChatModel`` cannot handle on its own:

1. **OBO** — the SSE call must run under the calling user's identity when
   ``genie_room.on_behalf_of_user`` is set, so the model is rebuilt per request
   with a user-scoped ``WorkspaceClient`` (same idea as
   :class:`dao_ai.middleware.obo.OBOModelMiddleware`).
2. **Genie conversation continuity** — the Genie service issues a
   ``conversation_id`` on the first turn that must be replayed on later turns to
   continue the same server-side conversation. This id is **independent** of the
   LangGraph ``thread_id`` used for graph-state persistence; it is cached in
   ``session.genie.spaces[agent_id]`` — exactly where the legacy ``type: genie``
   tool keeps it — and merged through the ``merge_session`` reducer.

Both are inseparable per-request rebuilds of the model, so one middleware owns
both. This replaces the generic ``OBOModelMiddleware`` for Genie models (a
second, generic model-swap middleware would clobber the conversation-bound
model this one installs).

Flow per call:

* read the prior ``conversation_id`` from ``request.state['session']``;
* build a :class:`GenieAgentChatModel` bound to the (OBO-aware) workspace client
  and that prior id;
* run the model via ``handler(request.override(model=...))``;
* read the Genie-issued id off the returned ``AIMessage.response_metadata``
  (reliable here — read within the same run, before the ResponsesAgent
  serialization strips metadata);
* persist it back to ``session`` via
  ``ExtendedModelResponse(command=Command(update={"session": ...}))``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage
from langgraph.types import Command
from loguru import logger

from dao_ai.config import GenieAgentModel
from dao_ai.genie.agent_chat_model import CONVERSATION_ID_METADATA_KEY
from dao_ai.middleware.base import AgentMiddleware
from dao_ai.state import AgentState, Context, SessionState
from dao_ai.tools.tracing import ResourceInfo, set_resource_attributes


class GenieAgentMiddleware(AgentMiddleware[AgentState, Context]):
    """Rebuild a Genie chat model per request with OBO + prior conversation_id,
    and persist the newly-issued conversation_id back to ``session``."""

    def __init__(self, genie_model: GenieAgentModel) -> None:
        self.genie_model = genie_model

    # -- helpers -------------------------------------------------------

    def _prior_conversation_id(self, state: AgentState) -> Optional[str]:
        """Read the last Genie conversation_id for this agent from session."""
        session: Optional[SessionState] = (state or {}).get("session")
        if session is None:
            return None
        return session.genie.get_conversation_id(self.genie_model.name)

    def _build_model(
        self, context: Context | None, prior_conversation_id: Optional[str]
    ) -> LanguageModelLike:
        workspace_client = self.genie_model.workspace_client_from(context)
        set_resource_attributes(
            ResourceInfo(
                "genie_agent",
                self.genie_model.on_behalf_of_user,
                self.genie_model.name,
            )
        )
        return self.genie_model.chat_model_for_workspace_client(
            workspace_client, conversation_id=prior_conversation_id
        )

    @staticmethod
    def _issued_conversation_id(response: ModelResponse) -> Optional[str]:
        """Pull the Genie-issued conversation_id off the response messages."""
        for message in reversed(response.result or []):
            if isinstance(message, AIMessage):
                conv_id: Any = (message.response_metadata or {}).get(
                    CONVERSATION_ID_METADATA_KEY
                )
                if conv_id:
                    return str(conv_id)
        return None

    def _session_command(
        self,
        state: AgentState,
        issued_conversation_id: Optional[str],
        prior_conversation_id: Optional[str],
    ) -> Command | None:
        """Build a session-update Command when the conversation_id changed."""
        if not issued_conversation_id:
            return None
        if issued_conversation_id == prior_conversation_id:
            return None  # nothing new to persist
        session: SessionState = (state or {}).get("session") or SessionState()
        session.genie.update_space(
            space_id=self.genie_model.name,
            conversation_id=issued_conversation_id,
        )
        logger.debug(
            "GenieAgentMiddleware persisted conversation_id",
            agent_id=self.genie_model.name,
            conversation_id=issued_conversation_id,
        )
        return Command(update={"session": session})

    # -- sync ----------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        context: Context | None = request.runtime.context if request.runtime else None
        prior: Optional[str] = self._prior_conversation_id(request.state)
        model: LanguageModelLike = self._build_model(context, prior)
        response: ModelResponse = handler(request.override(model=model))
        command: Command | None = self._session_command(
            request.state, self._issued_conversation_id(response), prior
        )
        if command is None:
            return response
        return ExtendedModelResponse(model_response=response, command=command)

    # -- async ---------------------------------------------------------

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        context: Context | None = request.runtime.context if request.runtime else None
        prior: Optional[str] = self._prior_conversation_id(request.state)
        model: LanguageModelLike = self._build_model(context, prior)
        response: ModelResponse = await handler(request.override(model=model))
        command: Command | None = self._session_command(
            request.state, self._issued_conversation_id(response), prior
        )
        if command is None:
            return response
        return ExtendedModelResponse(model_response=response, command=command)
