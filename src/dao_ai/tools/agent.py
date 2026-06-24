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
) -> Callable[..., Any]:
    """Create a tool that calls a Model Serving endpoint via ChatDatabricks.

    The wire shape (Responses vs Chat Completions) is read from
    ``llm.use_responses_api`` — set it to ``True`` for UC-registered
    agent endpoints (``agent/v1/responses`` task) and ``False`` for
    FMAPI / chat-completions endpoints (``llm/v1/chat`` task).

    For automatic task-based selection at runtime, use the higher-level
    :func:`dao_ai.tools.create_serving_endpoint_dispatcher` (used by
    ``type: serving_endpoint`` in YAML) which probes the endpoint's
    task field via the Databricks SDK and resolves ``use_responses_api``
    accordingly.

    Args:
        llm: The :class:`InferenceEndpointModel` config for the endpoint.
            ``llm.use_responses_api`` (default False) selects the wire
            shape.
        name: Tool name shown to the LLM. Defaults to ``agent_endpoint``.
        description: Tool description. Defaults to a generic prompt.
    """
    logger.debug(
        "Creating agent endpoint tool",
        name=name,
        description=description,
        use_responses_api=(
            llm.use_responses_api if isinstance(llm, InferenceEndpointModel) else None
        ),
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

            workspace_client: WorkspaceClient = llm.workspace_client_from(
                context, strict=True
            )
            logger.debug(
                "Creating OBO chat client for agent endpoint tool",
                model=llm.name,
                auth_type=workspace_client.config.auth_type,
                ai_gateway=llm.ai_gateway,
                use_responses_api=llm.use_responses_api,
            )
            model: LanguageModelLike = llm.chat_model_for_workspace_client(
                workspace_client
            )
        else:
            model = llm.as_chat_model()

        messages: Sequence[BaseMessage] = [HumanMessage(content=prompt)]
        response: AIMessage = await model.ainvoke(messages)
        return response

    name: str = name if name else agent_endpoint.__name__

    structured_tool: StructuredTool = StructuredTool.from_function(
        coroutine=agent_endpoint, name=name, description=doc, parse_docstring=False
    )

    return structured_tool
