from textwrap import dedent
from typing import Any, Callable, Optional, Sequence

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from loguru import logger

from dao_ai.config import LLMModel


def create_agent_endpoint_tool(
    llm: LLMModel | dict[str, Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[..., Any]:
    logger.debug(
        f"Creating agent endpoint tool with name: {name} and description: {description}"
    )

    default_description: str = dedent("""
    This tool allows you to interact with a language model endpoint to answer questions.
    You can ask questions about various topics, and the model will respond with relevant information.
    Please ask clear and concise questions to get the best responses.
    """)

    if isinstance(llm, dict):
        llm = LLMModel(**llm)

    if description is None:
        description = default_description

    doc_signature: str = dedent("""
    Args:
        prompt (str):  The prompt to send to the language model endpoint for generating a response.

    Returns:
        response (AIMessage):  An AIMessage object containing the response from the language model.
    """)

    doc: str = description + "\n" + doc_signature

    def agent_endpoint(prompt: str) -> AIMessage:
        model: LanguageModelLike = llm.as_chat_model()
        messages: Sequence[BaseMessage] = [HumanMessage(content=prompt)]
        response: AIMessage = model.invoke(messages)
        return response

    name: str = name if name else agent_endpoint.__name__

    structured_tool: StructuredTool = StructuredTool.from_function(
        func=agent_endpoint, name=name, description=doc, parse_docstring=False
    )

    return structured_tool
