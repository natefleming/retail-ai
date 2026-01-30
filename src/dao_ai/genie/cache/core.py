"""
Core utilities for Genie cache implementations.

This module provides shared utility functions used by different cache
implementations (LRU, Semantic, etc.). These are concrete implementations
of common operations needed across cache types.
"""

from typing import Any

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementResponse, StatementState
from loguru import logger

from dao_ai.config import WarehouseModel


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

    logger.trace("Executing cached SQL", layer=layer_name, sql=sql[:100])

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
        logger.error(
            "SQL execution failed",
            layer=layer_name,
            status=str(statement_response.status),
        )
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
