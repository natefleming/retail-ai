"""Memory tools for DAO AI."""

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langmem import create_search_memory_tool as langmem_create_search_memory_tool
from pydantic import BaseModel, Field


def create_search_memory_tool(namespace: tuple[str, ...]) -> BaseTool:
    """
    Create a Databricks-compatible search_memory tool.

    The langmem search_memory tool has a 'filter' field with additionalProperties: true
    in its schema, which Databricks LLM endpoints reject. This function creates a
    wrapper tool that omits the problematic filter field.

    Args:
        namespace: The memory namespace tuple

    Returns:
        A StructuredTool compatible with Databricks
    """
    # Get the original tool
    original_tool = langmem_create_search_memory_tool(namespace=namespace)

    # Create a schema without the problematic filter field
    class SearchMemoryInput(BaseModel):
        """Input for search_memory tool."""

        query: str = Field(..., description="The search query")
        limit: int = Field(default=10, description="Maximum number of results")
        offset: int = Field(default=0, description="Offset for pagination")

    # Create a wrapper function
    async def search_memory_wrapper(
        query: str, limit: int = 10, offset: int = 0
    ) -> Any:
        """Search your long-term memories for information relevant to your current context."""
        return await original_tool.ainvoke(
            {"query": query, "limit": limit, "offset": offset}
        )

    # Create the new tool
    return StructuredTool.from_function(
        coroutine=search_memory_wrapper,
        name="search_memory",
        description="Search your long-term memories for information relevant to your current context.",
        args_schema=SearchMemoryInput,
    )
