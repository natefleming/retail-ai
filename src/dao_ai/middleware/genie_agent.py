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
from uuid import uuid4

from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from loguru import logger

from dao_ai.config import GenieAgentModel
from dao_ai.genie.agent_chat_model import CONVERSATION_ID_METADATA_KEY
from dao_ai.middleware.base import AgentMiddleware
from dao_ai.state import AgentState, Context, SessionState
from dao_ai.tools.tracing import ResourceInfo, set_resource_attributes

# Prefix shared by every handoff tool created by ``create_handoff_tool`` /
# ``_create_handoff_back_to_supervisor_tool`` (orchestration). The deterministic
# handback discovers the bound handback tool by this prefix rather than
# hardcoding its name, so a rename in orchestration cannot silently break it.
_HANDOFF_TOOL_PREFIX: str = "handoff_to_"

# Fallback summary when Genie's answer has no usable text to summarize.
_DEFAULT_HANDBACK_SUMMARY: str = "Genie provided a data analysis above."

# Longest summary string synthesized for the injected handback tool call.
_MAX_HANDBACK_SUMMARY_CHARS: int = 500


class GenieAgentMiddleware(AgentMiddleware[AgentState, Context]):
    """Rebuild a Genie chat model per request with OBO + prior conversation_id,
    and persist the newly-issued conversation_id back to ``session``."""

    def __init__(self, genie_model: GenieAgentModel, handback: bool = False) -> None:
        self.genie_model = genie_model
        # When True (set from ``AgentModel.handoff`` for a Genie brain under a
        # supervisor), inject a ``handoff_to_supervisor`` tool call into Genie's
        # answer so the worker returns control to the supervisor instead of
        # being a graph sink. LLM-free.
        self.handback = handback

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

    # -- deterministic handback ----------------------------------------

    @staticmethod
    def _handback_tool_name(request: ModelRequest) -> Optional[str]:
        """Return the name of the bound handoff/handback tool, if any.

        A Genie worker opted into handback is given exactly one such tool by the
        supervisor (``_create_handoff_back_to_supervisor_tool``); we discover it
        by the shared ``handoff_to_`` prefix rather than hardcoding the name.
        Returns ``None`` when no handoff tool is bound (e.g. the Genie agent is a
        single-agent app), in which case injection is skipped.
        """
        for tool in request.tools or []:
            if isinstance(tool, BaseTool) and tool.name.startswith(
                _HANDOFF_TOOL_PREFIX
            ):
                return tool.name
        return None

    @staticmethod
    def _synthesize_summary(message: AIMessage) -> str:
        """Build a non-empty ``summary`` arg from Genie's answer text.

        ``handoff_to_supervisor(summary: str)`` rejects an empty summary at
        ToolNode schema validation, so always return something.
        """
        content: Any = message.content
        text: str = ""
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts: list[str] = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
                and block.get("type") in {"text", "output_text"}
            ]
            text = "\n".join(p for p in parts if p).strip()
        if not text:
            return _DEFAULT_HANDBACK_SUMMARY
        if len(text) > _MAX_HANDBACK_SUMMARY_CHARS:
            return text[: _MAX_HANDBACK_SUMMARY_CHARS - 1].rstrip() + "…"
        return text

    def _maybe_inject_handback(
        self, request: ModelRequest, response: ModelResponse
    ) -> None:
        """Attach a handback tool call to Genie's answer, in place.

        Genie never emits a client tool call, so under a supervisor its turn ends
        unless we route control back. We rewrite the final ``AIMessage`` to carry
        a ``handoff_to_supervisor`` tool call; langchain's agent loop then routes
        to the ToolNode, which runs the handback tool → ``Command(goto=...,
        graph=PARENT)`` — the same return path every normal worker uses.
        """
        tool_name: Optional[str] = self._handback_tool_name(request)
        if tool_name is None:
            logger.warning(
                "Genie handback enabled but no handoff tool is bound; "
                "skipping injection (the worker will remain a graph sink)",
                agent_id=self.genie_model.name,
            )
            return

        result: list[Any] = response.result or []
        for index in range(len(result) - 1, -1, -1):
            message: Any = result[index]
            if not isinstance(message, AIMessage):
                continue
            if message.tool_calls:
                return  # Genie unexpectedly emitted a tool call; leave it be.
            tool_call: dict[str, Any] = {
                "name": tool_name,
                "args": {"summary": self._synthesize_summary(message)},
                "id": f"genie_handback_{uuid4().hex}",
                "type": "tool_call",
            }
            result[index] = message.model_copy(update={"tool_calls": [tool_call]})
            logger.debug(
                "Injected deterministic Genie handback tool call",
                agent_id=self.genie_model.name,
                tool=tool_name,
            )
            return

        # Reached only if the response carried no AIMessage to attach to (e.g.
        # an empty result). Handback was requested and a tool was bound, so the
        # worker unexpectedly stays a graph sink this turn — say so, mirroring
        # the no-tool-bound branch above rather than failing silently.
        logger.warning(
            "Genie handback enabled and a handoff tool is bound, but the response "
            "contained no AIMessage to attach the handback to; the worker remains "
            "a graph sink for this turn",
            agent_id=self.genie_model.name,
        )

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
        issued: Optional[str] = self._issued_conversation_id(response)
        if self.handback:
            self._maybe_inject_handback(request, response)
        command: Command | None = self._session_command(request.state, issued, prior)
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
        issued: Optional[str] = self._issued_conversation_id(response)
        if self.handback:
            self._maybe_inject_handback(request, response)
        command: Command | None = self._session_command(request.state, issued, prior)
        if command is None:
            return response
        return ExtendedModelResponse(model_response=response, command=command)
