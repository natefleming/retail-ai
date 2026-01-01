from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables.base import RunnableLike
from loguru import logger


def create_search_tool() -> RunnableLike:
    """
    Create a DuckDuckGo search tool.

    Returns:
        RunnableLike: A DuckDuckGo search tool that returns results as a list
    """
    logger.trace("Creating DuckDuckGo search tool")
    return DuckDuckGoSearchRun(output_format="list")
