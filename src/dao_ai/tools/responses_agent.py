"""Tool factory for calling a Databricks App via the MLflow Responses API.

The factory wraps :class:`databricks_openai.DatabricksOpenAI` — the official
OpenAI client subclass that pulls auth from a
:class:`databricks.sdk.WorkspaceClient`. Routing to the App happens via the
``model='apps/<name>'`` prefix (the workspace's serving-endpoint proxy
forwards to the App's ``POST /v1/responses`` route). OBO is auto-derived
from :attr:`DatabricksAppModel.on_behalf_of_user` via
:meth:`DatabricksAppModel.workspace_client_from` — no manual header minting,
no bespoke ``httpx`` code.

For external (non-Databricks) targets or explicit A2A protocol use, see
:mod:`dao_ai.tools.a2a_agent`. For Model Serving endpoints, see
:func:`dao_ai.tools.create_agent_endpoint_tool`.
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
    Delegate a prompt to a Databricks App hosting a ResponsesAgent and
    return the assistant's reply as a single string.
    """
).strip()


def _coerce_app(value: DatabricksAppModel | dict[str, Any]) -> DatabricksAppModel:
    """Coerce a raw dict (from YAML factory args) into a DatabricksAppModel.

    Pydantic validates the dict against the full model so misconfigured apps
    fail fast at factory time.
    """
    if isinstance(value, DatabricksAppModel):
        return value
    if isinstance(value, dict):
        return DatabricksAppModel(**value)
    raise TypeError(
        f"create_responses_agent_tool: 'app' must be a DatabricksAppModel or "
        f"dict, got {type(value).__name__}."
    )


def create_responses_agent_tool(
    app: DatabricksAppModel | dict[str, Any],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> StructuredTool:
    """Create a tool that calls a Databricks App via the OpenAI Responses API.

    Args:
        app: The :class:`DatabricksAppModel` resource (or a dict that
            Pydantic will validate into one — factory args arrive as dicts
            when delivered via YAML). The App must expose the
            ``POST /v1/responses`` route — i.e. it must be deployed with the
            MLflow ResponsesAgent interface (the default for ``dao-ai``
            apps and ``mlflow.agents`` deployments).
        name: Tool name shown to the LLM. Defaults to the app's ``name``.
        description: Tool description shown to the LLM during function
            calling. Defaults to a generic delegation prompt.

    Returns:
        A :class:`StructuredTool` that sends one user message and returns
        the assistant's reply as a single string.

    Notes:
        - OBO: if ``app.on_behalf_of_user`` is ``True``, the tool reads the
          calling user's ``x-forwarded-access-token`` from
          ``runtime.context.headers`` per call and forwards it to the App.
          Otherwise the dao-ai service principal calls the App.
        - Routing: the ``apps/<name>`` model prefix tells the workspace
          serving-endpoint proxy to forward the request to the App's
          ``/v1/responses`` route. No explicit ``base_url`` override
          required.
    """
    coerced_app: DatabricksAppModel = _coerce_app(app)
    tool_name: str = name if name is not None else coerced_app.name
    tool_description: str = (
        description if description is not None else _DEFAULT_DESCRIPTION
    )
    model_id: str = f"apps/{coerced_app.name}"

    logger.debug(
        "Creating Responses agent tool",
        tool_name=tool_name,
        app_name=coerced_app.name,
        on_behalf_of_user=coerced_app.on_behalf_of_user,
        model=model_id,
    )

    doc_signature: str = dedent(
        """
        Args:
            prompt (str): Message to send to the App.

        Returns:
            str: The assistant's reply text.
        """
    )
    doc: str = tool_description + "\n" + doc_signature

    async def responses_agent(
        prompt: str,
        runtime: Annotated[ToolRuntime[Context], InjectedToolArg] = None,
    ) -> str:
        context: Context | None = runtime.context if runtime else None

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
            tool_span.set_attribute(ATTR_APP_AGENT_API, "responses")
            tool_span.set_attribute(ATTR_APP_AGENT_MODEL, model_id)
            tool_span.set_attribute(
                ATTR_APP_AGENT_OBO, bool(coerced_app.on_behalf_of_user)
            )
            tool_span.set_attribute(ATTR_APP_AGENT_PROMPT_CHARS, len(prompt))

        logger.info(
            "Invoking Responses agent",
            app_name=coerced_app.name,
            model=model_id,
            prompt_chars=len(prompt),
            on_behalf_of_user=coerced_app.on_behalf_of_user,
        )

        ws: WorkspaceClient = coerced_app.workspace_client_from(context)
        client: DatabricksOpenAI = DatabricksOpenAI(workspace_client=ws)
        response = client.responses.create(
            model=model_id,
            input=[{"role": "user", "content": prompt}],
        )
        output_text: str = response.output_text

        if tool_span is not None:
            tool_span.set_attribute(ATTR_APP_AGENT_RESPONSE_CHARS, len(output_text))

        return output_text

    return StructuredTool.from_function(
        coroutine=responses_agent,
        name=tool_name,
        description=doc,
        parse_docstring=False,
    )
