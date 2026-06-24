"""Dispatcher for ``type: serving_endpoint`` tools.

Constructs a single :class:`StructuredTool` that, on first invocation,
resolves the OpenAI API contract (Responses vs Chat Completions) and
caches the result. Subsequent invocations skip the resolution step.

Resolution precedence (via :func:`resolve_api`):
1. Explicit ``api:`` value from the YAML config (no probe runs).
2. Lazy probe of ``WorkspaceClient.serving_endpoints.get(name).task``
   via :func:`discover_serving_endpoint_api`.
3. Per-type default (``"completions"`` for ``type: serving_endpoint``).

The actual inference call is delegated to
:func:`create_agent_endpoint_tool` with the appropriate
``use_responses_api`` set on the :class:`InferenceEndpointModel`.
``create_agent_endpoint_tool`` builds a ``ChatDatabricks`` chat model
which handles OBO, AI Gateway routing, and OpenAI-compatible wire
shapes.
"""

from __future__ import annotations

from typing import Annotated, Optional

from databricks.sdk import WorkspaceClient
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import InjectedToolArg, StructuredTool
from loguru import logger

from dao_ai.config import InferenceEndpointModel
from dao_ai.state import Context
from dao_ai.tools._api_discovery import (
    ApiContract,
    discover_serving_endpoint_api,
    resolve_api,
)


def create_serving_endpoint_dispatcher(
    llm: InferenceEndpointModel,
    *,
    api: Optional[ApiContract] = None,
    default_api: ApiContract = "completions",
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> StructuredTool:
    """Create a tool that calls a Model Serving endpoint via
    ``ChatDatabricks``, picking the wire shape lazily on first call.

    Args:
        llm: The :class:`InferenceEndpointModel` for the target endpoint.
            ``use_responses_api`` on this model is overridden by the
            resolved api (explicit > discovered > default) at invocation.
        api: Explicit OpenAI API contract. When provided, discovery is
            skipped entirely. When ``None`` (the default), the dispatcher
            probes ``WorkspaceClient.serving_endpoints.get(name).task``
            on first invocation.
        default_api: Fallback contract if discovery returns no signal.
            Defaults to ``"completions"`` (FMAPI is the most common
            Model Serving target).
        name: Tool name shown to the LLM. Defaults to the endpoint name.
        description: Tool description shown to the LLM during function
            calling.

    Returns:
        A :class:`StructuredTool` that sends one user message to the
        endpoint and returns the assistant's reply.
    """
    tool_name: str = name if name is not None else llm.name

    if api is not None:
        logger.debug(
            "serving_endpoint_dispatcher: api={!r} (explicit); no probe will "
            "run | endpoint={!r}",
            api,
            llm.name,
        )
    else:
        logger.debug(
            "serving_endpoint_dispatcher: api=None, default={!r}, probe will "
            "run on first invoke | endpoint={!r}",
            default_api,
            llm.name,
        )

    # Closure cache for the lazily-resolved api contract.
    cache: dict[str, Optional[ApiContract]] = {"resolved": None}

    default_description: str = (
        "Delegate a prompt to a Databricks Model Serving endpoint and "
        "return the assistant's reply."
    )
    if description is None:
        description = default_description

    doc_signature: str = (
        "\nArgs:\n"
        "    prompt (str): The prompt to send to the endpoint.\n"
        "\nReturns:\n"
        "    The assistant's reply.\n"
    )
    doc: str = description + doc_signature

    async def serving_endpoint_dispatcher(
        prompt: str,
        runtime: Annotated[ToolRuntime[Context], InjectedToolArg] = None,
    ) -> AIMessage:
        context: Context | None = runtime.context if runtime else None

        # Build the WorkspaceClient once per call using the model's own
        # OBO-aware resolver. Used for both discovery and inference.
        ws: WorkspaceClient
        if llm.on_behalf_of_user:
            ws = llm.workspace_client_from(context)
        else:
            ws = WorkspaceClient()

        # Resolve api once per tool instance and cache. Probe runs only
        # if api was unset at config time.
        resolved: Optional[ApiContract] = cache["resolved"]
        if resolved is None:
            resolved_api = resolve_api(
                explicit=api,
                discover=lambda: discover_serving_endpoint_api(llm.name, ws),
                default=default_api,
            )
            resolved = resolved_api.value
            cache["resolved"] = resolved
            logger.info(
                "serving_endpoint_dispatcher resolved api={!r} ({}) | endpoint={!r}",
                resolved,
                resolved_api.origin,
                llm.name,
            )

        # Build the effective InferenceEndpointModel with the resolved
        # use_responses_api flag. ChatDatabricks consumes this directly.
        effective_llm: InferenceEndpointModel = llm.model_copy(
            update={"use_responses_api": resolved == "responses"}
        )

        # Delegate to create_agent_endpoint_tool's underlying chat model.
        # We invoke the chat model directly instead of building the
        # StructuredTool wrapper because (a) we already have the
        # WorkspaceClient resolved here and (b) it avoids double tool
        # wrapping. The behavior matches create_agent_endpoint_tool's
        # core: build chat model (OBO if needed), invoke with the prompt.
        if effective_llm.on_behalf_of_user:
            model = effective_llm.chat_model_for_workspace_client(ws)
        else:
            model = effective_llm.as_chat_model()

        response: AIMessage = await model.ainvoke([HumanMessage(content=prompt)])
        return response

    return StructuredTool.from_function(
        coroutine=serving_endpoint_dispatcher,
        name=tool_name,
        description=doc,
        parse_docstring=False,
    )
