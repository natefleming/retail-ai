"""
Base classes and types for Genie cache implementations.

This module provides the foundational types used across different cache
implementations (LRU, Semantic, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementResponse, StatementState
from databricks_ai_bridge.genie import GenieResponse
from loguru import logger

from dao_ai.config import WarehouseModel


class GenieServiceBase(ABC):
    """Abstract base class for Genie service implementations."""

    @abstractmethod
    def ask_question(
        self, question: str, conversation_id: str | None = None
    ) -> GenieResponse:
        """Ask a question to Genie and return the response."""
        pass


@dataclass
class SQLCacheEntry:
    """
    A cache entry storing the SQL query metadata for re-execution.

    Instead of caching the full result, we cache the SQL query so that
    on cache hit we can re-execute it to get fresh data.
    """

    query: str
    description: str
    conversation_id: str
    created_at: datetime


@dataclass
class CacheResult:
    """
    Result of a cache-aware query with metadata about cache behavior.

    Attributes:
        response: The GenieResponse (fresh data, possibly from cached SQL)
        cache_hit: Whether the SQL query came from cache
        served_by: Name of the layer that served the cached SQL (None if from origin)
    """

    response: GenieResponse
    cache_hit: bool
    served_by: str | None = None


def execute_sql_via_warehouse(
    warehouse: WarehouseModel,
    sql: str,
    layer_name: str = "cache",
) -> pd.DataFrame | str:
    """
    Execute SQL using a Databricks warehouse and return results as DataFrame.

    This is a shared utility for cache implementations that need to re-execute
    cached SQL queries.

    Args:
        warehouse: The warehouse configuration for SQL execution
        sql: The SQL query to execute
        layer_name: Name of the cache layer (for logging)

    Returns:
        DataFrame with results, or error message string
    """
    w: WorkspaceClient = warehouse.workspace_client
    warehouse_id: str = str(warehouse.warehouse_id)

    logger.debug(f"[{layer_name}] Executing cached SQL: {sql[:100]}...")

    statement_response: StatementResponse = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="30s",
    )

    # Poll for completion if still running
    while statement_response.status.state in [
        StatementState.PENDING,
        StatementState.RUNNING,
    ]:
        statement_response = w.statement_execution.get_statement(
            statement_response.statement_id
        )

    if statement_response.status.state != StatementState.SUCCEEDED:
        error_msg: str = f"SQL execution failed: {statement_response.status}"
        logger.error(f"[{layer_name}] {error_msg}")
        return error_msg

    # Convert to DataFrame
    if statement_response.result and statement_response.result.data_array:
        columns: list[str] = []
        if statement_response.manifest and statement_response.manifest.schema:
            columns = [col.name for col in statement_response.manifest.schema.columns]

        data: list[list[Any]] = statement_response.result.data_array
        if columns:
            return pd.DataFrame(data, columns=columns)
        else:
            return pd.DataFrame(data)

    return pd.DataFrame()
