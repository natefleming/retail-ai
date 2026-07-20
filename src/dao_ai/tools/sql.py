"""
SQL execution tool for running SQL statements against Databricks SQL warehouses
and Lakebase / Postgres databases.

This module provides factory functions for creating tools that execute a
pre-configured SQL statement — optionally with bound parameters whose values
come from the LLM or the runtime ``Context`` — against either a Databricks SQL
warehouse (``WarehouseModel``) or a Lakebase / Postgres database
(``DatabaseModel``).
"""

from typing import Annotated, Any, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    StatementParameterListItem,
    StatementResponse,
    StatementState,
)
from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, StructuredTool
from loguru import logger
from pydantic import Field, create_model

from dao_ai.config import (
    DatabaseModel,
    ParamSource,
    StatementParam,
    WarehouseModel,
    value_of,
)
from dao_ai.state import Context
from dao_ai.tools.tracing import ResourceInfo, set_resource_attributes

# Map the declared param ``type`` to a Python type used for the LLM-facing
# ``args_schema``. Values bound to the warehouse SDK are stringified regardless
# (the SDK takes string values); psycopg binds native Python values directly.
_PARAM_PY_TYPES: dict[str, type] = {
    "string": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def _preview(statement: str, limit: int = 100) -> str:
    """Return a truncated statement preview safe for logging."""
    return statement[:limit] + "..." if len(statement) > limit else statement


def _llm_params(params: list[StatementParam] | None) -> list[StatementParam]:
    return [p for p in (params or []) if p.source == ParamSource.LLM]


def _build_args_schema(name: str, params: list[StatementParam] | None):
    """Build the LLM-facing pydantic ``args_schema`` from LLM-sourced params.

    Context-sourced params are intentionally omitted so the model never sees or
    supplies them. When there are no LLM params the returned schema has no
    fields, which yields a zero-argument tool (preserving legacy behavior).
    """
    fields: dict[str, tuple[Any, Any]] = {}
    for p in _llm_params(params):
        py_type = _PARAM_PY_TYPES.get(p.type, str)
        if p.required and p.default is None:
            annotation: Any = py_type
            field = Field(description=p.description or f"Value for '{p.name}'.")
        else:
            annotation = Optional[py_type]
            field = Field(
                default=p.default,
                description=p.description or f"Value for '{p.name}'.",
            )
        fields[p.name] = (annotation, field)
    return create_model(f"{name}_Args", **fields)


def _resolve_values(
    params: list[StatementParam] | None,
    llm_kwargs: dict[str, Any],
    context: Context | None,
) -> dict[str, Any]:
    """Assemble the ``{marker_name: value}`` map used for native binding.

    LLM params are read from the model-supplied kwargs (with ``default`` applied
    when absent); context params are read from ``Context`` attributes via
    ``context_key`` (falling back to the param name). Raises ``ValueError`` for a
    required value that is missing, so the caller can surface a clean error.
    """
    values: dict[str, Any] = {}
    for p in params or []:
        if p.source == ParamSource.LLM:
            if p.name in llm_kwargs and llm_kwargs[p.name] is not None:
                values[p.name] = llm_kwargs[p.name]
            elif p.default is not None:
                values[p.name] = p.default
            elif p.required:
                raise ValueError(f"Missing required parameter '{p.name}'.")
        else:  # CONTEXT
            key = p.context_key or p.name
            value = getattr(context, key, None) if context is not None else None
            if value is None:
                value = p.default
            if value is None and p.required:
                raise ValueError(
                    f"Missing required context parameter '{p.name}' "
                    f"(context key '{key}')."
                )
            if value is not None:
                values[p.name] = value
    return values


def _format_result_table(columns: list[str], rows: list[list[Any]]) -> str:
    """Format columns + rows into the shared text-table representation."""
    row_count = len(rows)
    result_lines: list[str] = []
    if columns:
        header = " | ".join(columns)
        result_lines.append(header)
        result_lines.append("-" * len(header))
    for row in rows:
        result_lines.append(
            " | ".join(str(cell) if cell is not None else "NULL" for cell in row)
        )
    result_lines.append("")
    result_lines.append(f"({row_count} row{'s' if row_count != 1 else ''} returned)")
    return "\n".join(result_lines)


def create_execute_statement_tool(
    target: WarehouseModel | DatabaseModel | dict | None = None,
    statement: str | None = None,
    params: list[StatementParam] | None = None,
    name: str = "execute_sql_tool",
    description: str | None = None,
    *,
    warehouse: WarehouseModel | dict | None = None,
    database: DatabaseModel | dict | None = None,
) -> StructuredTool:
    """Create a tool that executes a pre-configured SQL statement.

    Dispatches on the backend type: a ``WarehouseModel`` runs the statement
    against a Databricks SQL warehouse (``:name`` bind markers); a
    ``DatabaseModel`` runs it against a Lakebase / Postgres database
    (``%(name)s`` bind markers). Dicts are coerced to the matching model — a
    ``warehouse_id`` key selects ``WarehouseModel``, otherwise ``DatabaseModel``.

    The backend may be supplied positionally as ``target`` or via the
    ``warehouse=`` / ``database=`` keywords (the latter kept for backward
    compatibility with existing ``type: factory`` configs). Exactly one backend
    must be provided.

    The statement is fixed at configuration time; only declared ``params`` values
    are supplied at runtime — either by the LLM (surfaced in the tool schema) or
    from the runtime ``Context``. Values are bound natively (never interpolated
    into the SQL string), so both backends are injection-safe.

    Args:
        target: ``WarehouseModel`` / ``DatabaseModel`` (or a dict coerced to one).
        statement: The SQL statement. Use ``:name`` markers for a warehouse
            target, ``%(name)s`` markers for a Lakebase / Postgres target.
        params: Optional bound-parameter declarations.
        name: Tool name visible to the LLM. Defaults to ``"execute_sql_tool"``.
        description: Optional tool description; a sensible default is generated.
        warehouse: Warehouse backend (keyword alias for ``target``).
        database: Lakebase / Postgres backend (keyword alias for ``target``).

    Returns:
        A LangChain ``StructuredTool`` that executes the statement and returns a
        text-table string of results.
    """
    provided = [t for t in (target, warehouse, database) if t is not None]
    if len(provided) != 1:
        raise ValueError(
            "create_execute_statement_tool requires exactly one backend via "
            "'target', 'warehouse', or 'database'."
        )
    backend = provided[0]
    if statement is None:
        raise ValueError("create_execute_statement_tool requires a 'statement'.")

    if isinstance(backend, dict):
        if "warehouse_id" in backend:
            backend = WarehouseModel.model_validate(backend)
        else:
            backend = DatabaseModel.model_validate(backend)

    if isinstance(backend, WarehouseModel):
        return create_warehouse_statement_tool(
            backend, statement, params=params, name=name, description=description
        )
    if isinstance(backend, DatabaseModel):
        return create_lakebase_statement_tool(
            backend, statement, params=params, name=name, description=description
        )
    raise ValueError(
        "create_execute_statement_tool requires a WarehouseModel or DatabaseModel "
        f"backend; got {type(backend).__name__}."
    )


def create_warehouse_statement_tool(
    warehouse: WarehouseModel | dict,
    statement: str,
    params: list[StatementParam] | None = None,
    name: str = "execute_sql_tool",
    description: str | None = None,
) -> StructuredTool:
    """Create a SQL tool that runs a statement against a Databricks SQL warehouse.

    Parameters use ``:name`` bind markers and are passed as
    ``StatementParameterListItem`` entries to the Statement Execution API — never
    interpolated into the SQL. Obtains the workspace client per request via
    ``warehouse.workspace_client_from(context)`` so OBO (Apps forwarded token or
    Model Serving user credentials), service principal, PAT, and ambient auth all
    work transparently.

    Args:
        warehouse: ``WarehouseModel`` or dict (auto-coerced) with warehouse config.
        statement: SQL statement using ``:name`` markers for any parameters.
        params: Optional bound-parameter declarations.
        name: Tool name visible to the LLM. Defaults to ``"execute_sql_tool"``.
        description: Optional custom tool description.
    """
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel.model_validate(warehouse)

    if description is None:
        description = (
            f"Execute a pre-configured SQL query against the {warehouse.name} "
            "warehouse and return the results."
        )

    warehouse_id: str = value_of(warehouse.warehouse_id)
    args_schema = _build_args_schema(name, params)

    logger.debug(
        "Creating SQL execution tool",
        tool_name=name,
        backend="sql_warehouse",
        warehouse_name=warehouse.name,
        warehouse_id=warehouse_id,
        param_count=len(params or []),
        sql_preview=_preview(statement),
    )

    def execute_statement_tool(
        runtime: Annotated[ToolRuntime[Context], InjectedToolArg] = None,
        **llm_kwargs: Any,
    ) -> str:
        set_resource_attributes(
            ResourceInfo("sql_warehouse", warehouse.on_behalf_of_user, warehouse.name)
        )

        context: Context | None = runtime.context if runtime else None

        try:
            values = _resolve_values(params, llm_kwargs, context)
        except ValueError as e:
            logger.error(
                "SQL parameter resolution failed", tool_name=name, error=str(e)
            )
            return f"Error: {e}"

        logger.info(
            "Executing SQL statement",
            tool_name=name,
            warehouse_id=warehouse_id,
            param_names=list(values.keys()),
            sql_preview=_preview(statement),
        )

        # Get workspace client with OBO support via context
        workspace_client: WorkspaceClient = warehouse.workspace_client_from(context)

        sdk_params: list[StatementParameterListItem] | None = [
            StatementParameterListItem(name=k, value=str(v)) for k, v in values.items()
        ] or None

        try:
            statement_response: StatementResponse = (
                workspace_client.statement_execution.execute_statement(
                    warehouse_id=warehouse_id,
                    statement=statement,
                    parameters=sdk_params,
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

            if result.data_array:
                rows = result.data_array
                columns: list[str] = []
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
                    row_count=len(rows),
                    column_count=len(columns),
                )
                return _format_result_table(columns, rows)

            logger.debug(
                "SQL statement executed successfully with empty result set",
                tool_name=name,
                statement_id=statement_response.statement_id,
            )
            return "SQL statement executed successfully (empty result set)"

        except Exception as e:
            logger.error(
                "SQL execution failed with exception",
                tool_name=name,
                warehouse_id=warehouse_id,
                error=str(e),
                exc_info=True,
            )
            return f"Error: Failed to execute SQL: {str(e)}"

    return StructuredTool.from_function(
        func=execute_statement_tool,
        name=name,
        description=description,
        args_schema=args_schema,
    )


def create_lakebase_statement_tool(
    database: DatabaseModel | dict,
    statement: str,
    params: list[StatementParam] | None = None,
    name: str = "execute_sql_tool",
    description: str | None = None,
) -> StructuredTool:
    """Create a SQL tool that runs a statement against a Lakebase / Postgres database.

    Parameters use ``%(name)s`` bind markers and are passed as a mapping to
    psycopg via ``DatabaseModel.aexecute_query`` — never interpolated into the
    SQL. Results are formatted into the same text table the warehouse path emits.

    OBO note: Lakebase credentials are minted from the workspace client the shared
    connection pool was built with. Model Serving OBO works today (the
    ``workspace_client`` property uses ``ModelServingUserCredentials``). Databricks
    Apps OBO (forwarded user token) is not yet wired through the shared pool — see
    the ``TODO(lakebase-obo)`` below for the one-line seam.

    Args:
        database: ``DatabaseModel`` or dict (auto-coerced) with database config.
        statement: SQL statement using ``%(name)s`` markers for any parameters.
        params: Optional bound-parameter declarations.
        name: Tool name visible to the LLM. Defaults to ``"execute_sql_tool"``.
        description: Optional custom tool description.
    """
    if isinstance(database, dict):
        database = DatabaseModel.model_validate(database)

    if description is None:
        description = (
            f"Execute a pre-configured SQL query against the {database.name} "
            "Lakebase database and return the results."
        )

    args_schema = _build_args_schema(name, params)

    logger.debug(
        "Creating SQL execution tool",
        tool_name=name,
        backend="lakebase",
        database_name=database.name,
        param_count=len(params or []),
        sql_preview=_preview(statement),
    )

    async def execute_statement_tool(
        runtime: Annotated[ToolRuntime[Context], InjectedToolArg] = None,
        **llm_kwargs: Any,
    ) -> str:
        set_resource_attributes(
            ResourceInfo("lakebase", database.on_behalf_of_user, database.name)
        )

        context: Context | None = runtime.context if runtime else None

        # TODO(lakebase-obo): Databricks Apps OBO for Lakebase needs the
        # forwarded user token to reach credential minting. Once the platform
        # supports it, thread `context` into the pool acquisition so
        # `memory/databricks.py:_lakebase_pool_kwargs` can call
        # `database.workspace_client_from(context)` instead of the
        # `database.workspace_client` property. Model Serving OBO already works
        # via that property today. We surface the OBO intent on the trace above.
        if database.on_behalf_of_user and context is not None and context.headers:
            logger.trace(
                "Lakebase OBO requested; Apps forwarded-token path not yet wired "
                "through the shared pool (Model Serving OBO is honored)",
                tool_name=name,
                database_name=database.name,
            )

        try:
            values = _resolve_values(params, llm_kwargs, context)
        except ValueError as e:
            logger.error(
                "SQL parameter resolution failed", tool_name=name, error=str(e)
            )
            return f"Error: {e}"

        logger.info(
            "Executing SQL statement",
            tool_name=name,
            database_name=database.name,
            param_names=list(values.keys()),
            sql_preview=_preview(statement),
        )

        try:
            rows: list[dict[str, Any]] = await database.aexecute_query(
                statement, values or None
            )
        except Exception as e:
            logger.error(
                "SQL execution failed with exception",
                tool_name=name,
                database_name=database.name,
                error=str(e),
                exc_info=True,
            )
            return f"Error: Failed to execute SQL: {str(e)}"

        if not rows:
            logger.debug(
                "SQL statement executed successfully with no rows",
                tool_name=name,
                database_name=database.name,
            )
            return "SQL statement executed successfully (no results returned)"

        columns: list[str] = list(rows[0].keys())
        data_rows: list[list[Any]] = [[row.get(c) for c in columns] for row in rows]
        logger.info(
            "SQL query returned results",
            tool_name=name,
            row_count=len(data_rows),
            column_count=len(columns),
        )
        return _format_result_table(columns, data_rows)

    return StructuredTool.from_function(
        coroutine=execute_statement_tool,
        name=name,
        description=description,
        args_schema=args_schema,
    )
