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
    from dao_ai._extras import require_extra

    # DuckDuckGoSearchRun imports the ``ddgs`` backend lazily on first use;
    # surface a friendly missing-extra error instead of an opaque one.
    require_extra("search", feature="DuckDuckGo web search tool", package="ddgs")
    return DuckDuckGoSearchRun(output_format="list")
