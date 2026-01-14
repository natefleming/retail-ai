"""
SQL execution tool for running SQL statements against Databricks SQL warehouses.

This module provides a factory function for creating tools that execute
pre-configured SQL statements against a Databricks SQL warehouse.
"""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementResponse, StatementState
from langchain.tools import ToolRuntime, tool
from loguru import logger

from dao_ai.config import WarehouseModel, value_of
from dao_ai.state import Context


def create_execute_statement_tool(
    warehouse: WarehouseModel,
    statement: str,
    name: str = "execute_sql_tool",
    description: str | None = None,
) -> tool:
    """
    Create a tool for executing a pre-configured SQL statement against a Databricks SQL warehouse.

    This factory function generates a tool that executes a specific SQL statement
    (defined at configuration time) against a Databricks SQL warehouse. The SQL is
    fixed and cannot be modified by the LLM at runtime, making this suitable for
    providing agents with specific, pre-defined queries.

    Args:
        warehouse: WarehouseModel containing warehouse configuration and credentials
        sql: The SQL statement to execute (configured at tool creation time)
        name: Optional custom name for the tool. Defaults to "execute_sql_tool"
        description: Optional custom description for the tool. If None, uses default description

    Returns:
        A LangChain tool that executes the pre-configured SQL statement and returns results

    Example:
        ```python
        from dao_ai.config import WarehouseModel
        from dao_ai.tools.sql import create_execute_sql_tool

        # Create warehouse model
        warehouse = WarehouseModel(
            name="analytics_warehouse",
            warehouse_id="abc123def456",
        )

        # Create SQL execution tool with pre-configured query
        customer_count_tool = create_execute_sql_tool(
            warehouse=warehouse,
            sql="SELECT COUNT(*) as customer_count FROM catalog.schema.customers",
            name="get_customer_count",
            description="Get the total number of customers in the database"
        )

        # Use the tool (no parameters needed - SQL is pre-configured)
        result = customer_count_tool.invoke({})
        ```
    """
    if description is None:
        description = f"Execute a pre-configured SQL query against the {warehouse.name} warehouse and return the results."

    warehouse_id: str = value_of(warehouse.warehouse_id)

    logger.debug(
        "Creating SQL execution tool",
        tool_name=name,
        warehouse_name=warehouse.name,
        warehouse_id=warehouse_id,
        sql_preview=statement[:100] + "..." if len(statement) > 100 else statement,
    )

    @tool(name_or_callable=name, description=description)
    def execute_statement_tool(runtime: ToolRuntime[Context] = None) -> str:
        """
        Execute the pre-configured SQL statement against the Databricks SQL warehouse.

        Returns:
            A string containing the query results or execution status
        """
        logger.info(
            "Executing SQL statement",
            tool_name=name,
            warehouse_id=warehouse_id,
            sql_preview=statement[:100] + "..." if len(statement) > 100 else statement,
        )

        # Get workspace client with OBO support via context
        context: Context | None = runtime.context if runtime else None
        workspace_client: WorkspaceClient = warehouse.workspace_client_from(context)

        try:
            # Execute the SQL statement
            statement_response: StatementResponse = (
                workspace_client.statement_execution.execute_statement(
                    warehouse_id=warehouse_id,
                    statement=statement,
                    wait_timeout="30s",
                )
            )

            # Poll for completion if still pending
            while statement_response.status.state in [
                StatementState.PENDING,
                StatementState.RUNNING,
            ]:
                logger.trace(
                    "SQL statement still executing, polling...",
                    statement_id=statement_response.statement_id,
                    state=statement_response.status.state,
                )
                statement_response = workspace_client.statement_execution.get_statement(
                    statement_response.statement_id
                )

            # Check execution status
            if statement_response.status.state != StatementState.SUCCEEDED:
                error_msg: str = (
                    f"SQL execution failed with state {statement_response.status.state}"
                )
                if statement_response.status.error:
                    error_msg += f": {statement_response.status.error.message}"
                logger.error(
                    "SQL execution failed",
                    tool_name=name,
                    statement_id=statement_response.statement_id,
                    error=error_msg,
                )
                return f"Error: {error_msg}"

            # Extract results
            result = statement_response.result
            if result is None:
                logger.debug(
                    "SQL statement executed successfully with no results",
                    tool_name=name,
                    statement_id=statement_response.statement_id,
                )
                return "SQL statement executed successfully (no results returned)"

            # Format results
            if result.data_array:
                rows = result.data_array
                row_count = len(rows)

                # Get column names if available
                columns = []
                if (
                    statement_response.manifest
                    and statement_response.manifest.schema
                    and statement_response.manifest.schema.columns
                ):
                    columns = [
                        col.name for col in statement_response.manifest.schema.columns
                    ]

                logger.info(
                    "SQL query returned results",
                    tool_name=name,
                    row_count=row_count,
                    column_count=len(columns),
                )

                # Format as a simple text table
                result_lines = []
                if columns:
                    result_lines.append(" | ".join(columns))
                    result_lines.append("-" * (len(" | ".join(columns))))

                for row in rows:
                    result_lines.append(
                        " | ".join(
                            str(cell) if cell is not None else "NULL" for cell in row
                        )
                    )

                # Add summary
                result_lines.append("")
                result_lines.append(
                    f"({row_count} row{'s' if row_count != 1 else ''} returned)"
                )

                return "\n".join(result_lines)
            else:
                logger.debug(
                    "SQL statement executed successfully with empty result set",
                    tool_name=name,
                    statement_id=statement_response.statement_id,
                )
                return "SQL statement executed successfully (empty result set)"

        except Exception as e:
            error_msg = f"Failed to execute SQL: {str(e)}"
            logger.error(
                "SQL execution failed with exception",
                tool_name=name,
                warehouse_id=warehouse_id,
                error=str(e),
                exc_info=True,
            )
            return f"Error: {error_msg}"

    return execute_statement_tool
