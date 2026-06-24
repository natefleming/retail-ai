from textwrap import dedent
from typing import Annotated, Any, Callable, Optional, Sequence

from langchain.tools import ToolRuntime
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import InjectedToolArg, StructuredTool
from loguru import logger

from dao_ai.config import InferenceEndpointModel
from dao_ai.state import Context
from dao_ai.tools.tracing import ResourceInfo, set_resource_attributes


def create_agent_endpoint_tool(
    llm: InferenceEndpointModel | dict[str, Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
    auto_detect_responses_api: bool = False,
) -> Callable[..., Any]:
    """Create a tool that calls a Model Serving endpoint via ChatDatabricks.

    Args:
        llm: The :class:`InferenceEndpointModel` config for the endpoint.
        name: Tool name shown to the LLM. Defaults to ``agent_endpoint``.
        description: Tool description. Defaults to a generic prompt.
        auto_detect_responses_api: When True, lazily probe
            ``serving_endpoints.get(name).task`` on first invocation and
            set ``use_responses_api`` based on whether the endpoint is an
            ``agent/v1/responses`` ResponsesAgent (UC-registered agent)
            or an ``llm/v1/chat`` chat-completions endpoint (FMAPI).
            Cached after the first detection. Falls back to ``False`` on
            lookup failure.
    """
    logger.debug(
        "Creating agent endpoint tool",
        name=name,
        description=description,
        auto_detect_responses_api=auto_detect_responses_api,
    )

    default_description: str = dedent("""
    This tool allows you to interact with a language model endpoint to answer questions.
    You can ask questions about various topics, and the model will respond with relevant information.
    Please ask clear and concise questions to get the best responses.
    """)

    if isinstance(llm, dict):
        llm = InferenceEndpointModel(**llm)

    if description is None:
        description = default_description

    doc_signature: str = dedent("""
    Args:
        prompt (str):  The prompt to send to the language model endpoint for generating a response.

    Returns:
        response (AIMessage):  An AIMessage object containing the response from the language model.
    """)

    doc: str = description + "\n" + doc_signature

    # Closure cache for the lazily-resolved use_responses_api flag.
    # None = unresolved; True/False = resolved value.
    auto_detect_cache: dict[str, Optional[bool]] = {"resolved": None}

    def _resolve_effective_llm(
        workspace_client: "WorkspaceClient | None",
    ) -> InferenceEndpointModel:
        """Return llm, overriding use_responses_api from the auto-detect probe."""
        if not auto_detect_responses_api:
            return llm
        if auto_detect_cache["resolved"] is not None:
            return llm.model_copy(
                update={"use_responses_api": auto_detect_cache["resolved"]}
            )
        # First call — probe the endpoint task to choose the right wire shape.
        from databricks.sdk import WorkspaceClient as _WS

        ws_for_lookup: _WS = workspace_client or _WS()
        detected: bool = False
        try:
            ep_info = ws_for_lookup.serving_endpoints.get(llm.name)
            task_str: str = (getattr(ep_info, "task", None) or "").lower()
            detected = "responses" in task_str
            logger.info(
                "Auto-detected endpoint task",
                endpoint=llm.name,
                task=task_str or "<unknown>",
                use_responses_api=detected,
            )
        except Exception as exc:
            logger.warning(
                f"serving_endpoints.get({llm.name!r}) failed during "
                f"auto-detect: {type(exc).__name__}: {exc}. "
                f"Defaulting to use_responses_api=False (chat completions)."
            )
        auto_detect_cache["resolved"] = detected
        return llm.model_copy(update={"use_responses_api": detected})

    async def agent_endpoint(
        prompt: str,
        runtime: Annotated[ToolRuntime[Context], InjectedToolArg] = None,
    ) -> AIMessage:
        context: Context | None = runtime.context if runtime else None

        set_resource_attributes(
            ResourceInfo("agent_endpoint", llm.on_behalf_of_user, llm.name)
        )

        if llm.on_behalf_of_user:
            from databricks.sdk import WorkspaceClient

            workspace_client: WorkspaceClient = llm.workspace_client_from(context)
            effective_llm: InferenceEndpointModel = _resolve_effective_llm(
                workspace_client
            )
            logger.debug(
                "Creating OBO chat client for agent endpoint tool",
                model=effective_llm.name,
                auth_type=workspace_client.config.auth_type,
                ai_gateway=effective_llm.ai_gateway,
                use_responses_api=effective_llm.use_responses_api,
            )
            model: LanguageModelLike = effective_llm.chat_model_for_workspace_client(
                workspace_client
            )
        else:
            effective_llm = _resolve_effective_llm(None)
            model = effective_llm.as_chat_model()

        messages: Sequence[BaseMessage] = [HumanMessage(content=prompt)]
        response: AIMessage = await model.ainvoke(messages)
        return response

    name: str = name if name else agent_endpoint.__name__

    structured_tool: StructuredTool = StructuredTool.from_function(
        coroutine=agent_endpoint, name=name, description=doc, parse_docstring=False
    )

    return structured_tool
