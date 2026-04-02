"""
OBO (On-Behalf-Of) model middleware for LLM inference.

Enforces per-user authentication on LLM serving endpoint calls by
intercepting each model call and replacing the model with a
``ChatDatabricks`` instance backed by an OBO-authenticated
``WorkspaceClient``.

This mirrors the tool-level OBO pattern used by vector search, SQL,
Genie, UC functions, and other resources -- all of which call
``resource.workspace_client_from(context)`` at invocation time to
obtain a ``WorkspaceClient`` authenticated with the calling user's
forwarded token.

Deployment targets:
- **Databricks Apps**: uses ``x-forwarded-access-token`` from
  ``Context.headers`` (captured by ``_inject_headers_into_request``).
- **Model Serving**: falls back to ``ModelServingUserCredentials``
  (per-request via thread-local in the serving environment).
"""

from typing import Awaitable, Callable

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from loguru import logger

from dao_ai.config import LLMModel
from dao_ai.middleware.base import AgentMiddleware
from dao_ai.state import AgentState, Context


class OBOModelMiddleware(AgentMiddleware[AgentState, Context]):
    """Replace the agent's model with an OBO-authenticated ``ChatDatabricks`` per request.

    On each model call the middleware:
    1. Extracts ``Context`` from ``request.runtime``
    2. Calls ``llm_model.workspace_client_from(context)`` to obtain
       a ``WorkspaceClient`` carrying the calling user's token
    3. Creates a fresh ``ChatDatabricks`` with that workspace client
    4. Swaps the model via ``request.override(model=...)``
    """

    def __init__(self, llm_model: LLMModel) -> None:
        self.llm_model = llm_model

    def _create_obo_model(self, context: Context | None) -> ChatDatabricks:
        workspace_client: WorkspaceClient = self.llm_model.workspace_client_from(
            context
        )
        logger.debug(
            "OBOModelMiddleware: created OBO ChatDatabricks",
            model=self.llm_model.name,
            auth_type=workspace_client.config.auth_type,
        )
        return ChatDatabricks(
            model=self.llm_model.name,
            temperature=self.llm_model.temperature,
            max_tokens=self.llm_model.max_tokens,
            use_responses_api=self.llm_model.use_responses_api,
            workspace_client=workspace_client,
        )

    # -- sync ----------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        context: Context | None = request.runtime.context if request.runtime else None
        obo_model: ChatDatabricks = self._create_obo_model(context)
        return handler(request.override(model=obo_model))

    # -- async ---------------------------------------------------------

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        context: Context | None = request.runtime.context if request.runtime else None
        obo_model: ChatDatabricks = self._create_obo_model(context)
        return await handler(request.override(model=obo_model))
