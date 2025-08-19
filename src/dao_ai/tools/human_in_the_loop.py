from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanInterruptConfig
from langgraph.types import interrupt
from loguru import logger

from dao_ai.config import (
    BaseFunctionModel,
    HumanInTheLoopModel,
)


def add_human_in_the_loop(
    tool: RunnableLike,
    *,
    interrupt_config: HumanInterruptConfig | None = None,
    review_prompt: Optional[str] = "Please review the tool call",
) -> BaseTool:
    """
    Wrap a tool with human-in-the-loop functionality.
    This function takes a tool (either a callable or a BaseTool instance) and wraps it
    with a human-in-the-loop mechanism. When the tool is invoked, it will first
    request human review before executing the tool's logic. The human can choose to
    accept, edit the input, or provide a custom response.

    Args:
        tool (Callable[..., Any] | BaseTool): _description_
        interrupt_config (HumanInterruptConfig | None, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        BaseTool: _description_
    """
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }

    logger.debug(f"Wrapping tool {tool} with human-in-the-loop functionality")

    @create_tool(tool.name, description=tool.description, args_schema=tool.args_schema)
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input) -> Any:
        logger.debug(f"call_tool_with_interrupt: {tool.name} with input: {tool_input}")
        request: HumanInterrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input,
            },
            "config": interrupt_config,
            "description": review_prompt,
        }

        logger.debug(f"Human interrupt request: {request}")
        response: dict[str, Any] = interrupt([request])[0]
        logger.debug(f"Human interrupt response: {response}")

        if response["type"] == "accept":
            tool_response = tool.invoke(tool_input, config=config)
        elif response["type"] == "edit":
            tool_input = response["args"]["args"]
            tool_response = tool.invoke(tool_input, config=config)
        elif response["type"] == "response":
            user_feedback = response["args"]
            tool_response = user_feedback
        else:
            raise ValueError(f"Unknown interrupt response type: {response['type']}")

        return tool_response

    return call_tool_with_interrupt


def as_human_in_the_loop(
    tool: RunnableLike, function: BaseFunctionModel | str
) -> RunnableLike:
    if isinstance(function, BaseFunctionModel):
        human_in_the_loop: HumanInTheLoopModel | None = function.human_in_the_loop
        if human_in_the_loop:
            logger.debug(f"Adding human-in-the-loop to tool: {tool.name}")
            tool = add_human_in_the_loop(
                tool=tool,
                interrupt_config=human_in_the_loop.interupt_config,
                review_prompt=human_in_the_loop.review_prompt,
            )
    return tool
