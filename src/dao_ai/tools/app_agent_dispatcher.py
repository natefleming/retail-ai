"""Dispatcher for ``type: app`` tools.

Constructs a single :class:`StructuredTool` that, on first invocation,
resolves the OpenAI API contract (Responses vs Chat Completions) and
caches the result. Subsequent invocations skip the resolution step.

Resolution precedence (via :func:`resolve_api`):
1. Explicit ``api:`` value from the YAML config (no probe runs).
2. Lazy probe of ``<app_url>/agent/info`` via
   :func:`discover_app_agent_api`.
3. Per-type default (``"responses"`` for ``type: app``).

The HTTP call to the App is inlined here rather than delegated to the
``create_responses_agent_tool`` / ``create_chat_completions_agent_tool``
factories — those factories still ship for direct ``type: factory`` use,
but inter-tool delegation would have required threading
LangChain-injected runtime args through two ``ainvoke`` calls. Inline is
simpler.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Annotated, Any, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI
from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, StructuredTool
from loguru import logger

from dao_ai.config import DatabricksAppModel
from dao_ai.state import Context
from dao_ai.tools._api_discovery import (
    ApiContract,
    discover_app_agent_api,
    resolve_api,
)
from dao_ai.tools.tracing import (
    ATTR_APP_AGENT_API,
    ATTR_APP_AGENT_APP_NAME,
    ATTR_APP_AGENT_MODEL,
    ATTR_APP_AGENT_OBO,
    ATTR_APP_AGENT_PROMPT_CHARS,
    ATTR_APP_AGENT_RESPONSE_CHARS,
    ResourceInfo,
    set_resource_attributes,
)

_DEFAULT_DESCRIPTION: str = dedent(
    """\
    Delegate a prompt to a Databricks App and return the assistant's
    reply as a single string.
    """
).strip()


def _coerce_app(value: DatabricksAppModel | dict[str, Any]) -> DatabricksAppModel:
    """Coerce a raw dict (from YAML factory args) into a DatabricksAppModel."""
    if isinstance(value, DatabricksAppModel):
        return value
    if isinstance(value, dict):
        return DatabricksAppModel(**value)
    raise TypeError(
        f"create_app_dispatcher: 'app' must be a DatabricksAppModel or "
        f"dict, got {type(value).__name__}."
    )


def create_app_dispatcher(
    app: DatabricksAppModel | dict[str, Any],
    *,
    api: Optional[ApiContract] = None,
    default_api: ApiContract = "responses",
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> StructuredTool:
    """Create a tool that calls a Databricks App via the OpenAI Responses
    or Chat Completions API, picking the wire shape lazily on first call.

    Args:
        app: The :class:`DatabricksAppModel` resource (or a dict that
            Pydantic will validate into one). The App may host an agent
            (ResponsesAgent or otherwise) or any other HTTP service that
            speaks OpenAI Responses / Chat Completions.
        api: Explicit OpenAI API contract. When provided, discovery is
            skipped entirely. When ``None`` (the default), the dispatcher
            probes ``<app_url>/agent/info`` on first invocation.
        default_api: Fallback contract if discovery returns no signal.
            Defaults to ``"responses"`` (canonical for
            ``mlflow.agents`` ResponsesAgent deployments).
        name: Tool name shown to the LLM. Defaults to the app's name.
        description: Tool description shown to the LLM during function
            calling.

    Returns:
        A :class:`StructuredTool` that sends one user message to the App
        and returns the assistant's reply as a single string.
    """
    coerced_app: DatabricksAppModel = _coerce_app(app)
    tool_name: str = name if name is not None else coerced_app.name
    tool_description: str = (
        description if description is not None else _DEFAULT_DESCRIPTION
    )
    model_id: str = f"apps/{coerced_app.name}"

    if api is not None:
        logger.debug(
            "app_dispatcher: api={!r} (explicit); no probe will run | app={!r}",
            api,
            coerced_app.name,
        )
    else:
        logger.debug(
            "app_dispatcher: api=None, default={!r}, probe will run on first "
            "invoke | app={!r}",
            default_api,
            coerced_app.name,
        )

    # Closure cache for the lazily-resolved api contract. Stores the
    # resolved value only — the origin is logged at resolution time and
    # not needed on cache hits.
    cache: dict[str, Optional[ApiContract]] = {"resolved": None}

    doc_signature: str = dedent(
        """
        Args:
            prompt (str): Message to send to the App.

        Returns:
            str: The assistant's reply text.
        """
    )
    doc: str = tool_description + "\n" + doc_signature

    async def app_dispatcher(
        prompt: str,
        runtime: Annotated[ToolRuntime[Context], InjectedToolArg] = None,
    ) -> str:
        context: Context | None = runtime.context if runtime else None
        ws: WorkspaceClient = coerced_app.workspace_client_from(context)

        # Resolve api once per tool instance and cache the result. Probe
        # is invoked only if api was unset at config time.
        resolved: Optional[ApiContract] = cache["resolved"]
        if resolved is None:
            resolved_api = resolve_api(
                explicit=api,
                discover=lambda: discover_app_agent_api(coerced_app.url, ws),
                default=default_api,
            )
            resolved = resolved_api.value
            cache["resolved"] = resolved
            logger.info(
                "app_dispatcher resolved api={!r} ({}) | app={!r}",
                resolved,
                resolved_api.origin,
                coerced_app.name,
            )

        set_resource_attributes(
            ResourceInfo(
                resource_type="app_agent",
                on_behalf_of_user=bool(coerced_app.on_behalf_of_user),
                name=coerced_app.name,
            )
        )

        tool_span = mlflow.get_current_active_span()
        if tool_span is not None:
            tool_span.set_attribute(ATTR_APP_AGENT_APP_NAME, coerced_app.name)
            tool_span.set_attribute(ATTR_APP_AGENT_API, resolved)
            tool_span.set_attribute(ATTR_APP_AGENT_MODEL, model_id)
            tool_span.set_attribute(
                ATTR_APP_AGENT_OBO, bool(coerced_app.on_behalf_of_user)
            )
            tool_span.set_attribute(ATTR_APP_AGENT_PROMPT_CHARS, len(prompt))

        client: DatabricksOpenAI = DatabricksOpenAI(workspace_client=ws)
        output_text: str
        if resolved == "responses":
            response = client.responses.create(
                model=model_id,
                input=[{"role": "user", "content": prompt}],
            )
            output_text = response.output_text
        else:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
            )
            output_text = response.choices[0].message.content or ""

        if tool_span is not None:
            tool_span.set_attribute(ATTR_APP_AGENT_RESPONSE_CHARS, len(output_text))

        return output_text

    return StructuredTool.from_function(
        coroutine=app_dispatcher,
        name=tool_name,
        description=doc,
        parse_docstring=False,
    )
