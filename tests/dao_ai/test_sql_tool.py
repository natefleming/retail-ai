"""Tests for SQL execution tool."""

from unittest.mock import MagicMock, Mock

import pytest
from databricks.sdk.service.sql import (
    ColumnInfo,
    ResultData,
    ResultManifest,
    ResultSchema,
    StatementResponse,
    StatementState,
    StatementStatus,
)

from dao_ai.config import WarehouseModel
from dao_ai.tools.sql import create_execute_statement_tool


@pytest.fixture
def mock_warehouse() -> WarehouseModel:
    """Create a mock warehouse model."""
    warehouse = WarehouseModel(
        name="test_warehouse",
        warehouse_id="test_warehouse_id",
    )
    return warehouse


@pytest.mark.unit
def test_create_execute_sql_tool(mock_warehouse: WarehouseModel) -> None:
    """Test that the factory function creates a tool with correct attributes."""
    test_sql = "SELECT * FROM test_table"
    tool = create_execute_statement_tool(mock_warehouse, statement=test_sql)

    assert tool is not None
    assert tool.name == "execute_sql_tool"
    assert "Execute a pre-configured SQL query" in tool.description
    assert hasattr(tool, "invoke")


@pytest.mark.unit
def test_create_execute_sql_tool_with_custom_name(
    mock_warehouse: WarehouseModel,
) -> None:
    """Test creating a tool with custom name and description."""
    custom_name = "my_sql_tool"
    custom_description = "Custom SQL execution tool"
    test_sql = "SELECT COUNT(*) FROM customers"

    tool = create_execute_statement_tool(
        warehouse=mock_warehouse,
        statement=test_sql,
        name=custom_name,
        description=custom_description,
    )

    assert tool.name == custom_name
    assert tool.description == custom_description


@pytest.mark.unit
def test_execute_sql_tool_success(mock_warehouse: WarehouseModel) -> None:
    """Test successful SQL execution with results."""
    from unittest.mock import PropertyMock, patch

    test_sql = "SELECT * FROM test_table"

    # Create mock response
    mock_response = StatementResponse(
        statement_id="test_statement_id",
        status=StatementStatus(state=StatementState.SUCCEEDED),
        result=ResultData(
            data_array=[
                ["value1", "value2"],
                ["value3", "value4"],
            ],
        ),
        manifest=ResultManifest(
            schema=ResultSchema(
                columns=[
                    ColumnInfo(name="col1"),
                    ColumnInfo(name="col2"),
                ]
            )
        ),
    )

    # Create mock workspace client
    mock_ws = MagicMock()
    mock_ws.statement_execution.execute_statement.return_value = mock_response

    # Mock the workspace_client property
    with patch.object(
        type(mock_warehouse),
        "workspace_client",
        new_callable=PropertyMock,
        return_value=mock_ws,
    ):
        # Create tool and execute (no parameters needed - SQL is pre-configured)
        tool = create_execute_statement_tool(mock_warehouse, statement=test_sql)
        result = tool.invoke({})

        # Verify result format
        assert isinstance(result, str)
        assert "col1" in result
        assert "col2" in result
        assert "value1" in result
        assert "value2" in result
        assert "(2 rows returned)" in result


@pytest.mark.unit
def test_execute_sql_tool_no_results(mock_warehouse: WarehouseModel) -> None:
    """Test SQL execution with no results (e.g., INSERT statement)."""
    from unittest.mock import PropertyMock, patch

    test_sql = "INSERT INTO test_table VALUES (1, 2)"

    mock_response = StatementResponse(
        statement_id="test_statement_id",
        status=StatementStatus(state=StatementState.SUCCEEDED),
        result=None,
    )

    # Create mock workspace client
    mock_ws = MagicMock()
    mock_ws.statement_execution.execute_statement.return_value = mock_response

    # Mock the workspace_client property
    with patch.object(
        type(mock_warehouse),
        "workspace_client",
        new_callable=PropertyMock,
        return_value=mock_ws,
    ):
        tool = create_execute_statement_tool(mock_warehouse, statement=test_sql)
        result = tool.invoke({})

        assert "executed successfully" in result.lower()


@pytest.mark.unit
def test_execute_sql_tool_error(mock_warehouse: WarehouseModel) -> None:
    """Test SQL execution with error."""
    from unittest.mock import PropertyMock, patch

    test_sql = "SELECT * FROM nonexistent_table"

    mock_error = Mock()
    mock_error.message = "Table not found"

    mock_response = StatementResponse(
        statement_id="test_statement_id",
        status=StatementStatus(
            state=StatementState.FAILED,
            error=mock_error,
        ),
    )

    # Create mock workspace client
    mock_ws = MagicMock()
    mock_ws.statement_execution.execute_statement.return_value = mock_response

    # Mock the workspace_client property
    with patch.object(
        type(mock_warehouse),
        "workspace_client",
        new_callable=PropertyMock,
        return_value=mock_ws,
    ):
        tool = create_execute_statement_tool(mock_warehouse, statement=test_sql)
        result = tool.invoke({})

        assert "Error" in result
        assert "Table not found" in result


@pytest.mark.unit
def test_execute_sql_tool_exception(mock_warehouse: WarehouseModel) -> None:
    """Test SQL execution with exception."""
    from unittest.mock import PropertyMock, patch

    test_sql = "SELECT * FROM test_table"

    # Create mock workspace client
    mock_ws = MagicMock()
    mock_ws.statement_execution.execute_statement.side_effect = Exception(
        "Connection failed"
    )

    # Mock the workspace_client property
    with patch.object(
        type(mock_warehouse),
        "workspace_client",
        new_callable=PropertyMock,
        return_value=mock_ws,
    ):
        tool = create_execute_statement_tool(mock_warehouse, statement=test_sql)
        result = tool.invoke({})

        assert "Error" in result
        assert "Connection failed" in result


# ---------------------------------------------------------------------------
# Parameterized statements + backend dispatch (warehouse)
# ---------------------------------------------------------------------------


def _runtime(context=None):
    """Minimal ToolRuntime stand-in exposing a ``.context`` attribute."""
    return type("_Runtime", (), {"context": context})()


@pytest.mark.unit
def test_warehouse_llm_param_in_schema(mock_warehouse: WarehouseModel) -> None:
    """LLM-sourced params appear in the tool's args schema; context params don't."""
    from dao_ai.config import ParamSource, StatementParam

    tool = create_execute_statement_tool(
        mock_warehouse,
        statement="SELECT * FROM t WHERE id = :id AND store = :store",
        params=[
            StatementParam(name="id", type="int", description="row id"),
            StatementParam(name="store", source=ParamSource.CONTEXT, type="int"),
        ],
    )
    assert "id" in tool.args
    assert "store" not in tool.args  # context param hidden from the model


@pytest.mark.unit
def test_warehouse_binds_parameters(mock_warehouse: WarehouseModel) -> None:
    """LLM + context values are passed as StatementParameterListItem entries."""
    from unittest.mock import PropertyMock, patch

    from databricks.sdk.service.sql import StatementParameterListItem

    from dao_ai.config import ParamSource, StatementParam
    from dao_ai.state import Context

    mock_response = StatementResponse(
        statement_id="sid",
        status=StatementStatus(state=StatementState.SUCCEEDED),
        result=ResultData(data_array=[["ok"]]),
        manifest=ResultManifest(schema=ResultSchema(columns=[ColumnInfo(name="c")])),
    )
    mock_ws = MagicMock()
    mock_ws.statement_execution.execute_statement.return_value = mock_response

    with patch.object(
        type(mock_warehouse),
        "workspace_client",
        new_callable=PropertyMock,
        return_value=mock_ws,
    ):
        tool = create_execute_statement_tool(
            mock_warehouse,
            statement="SELECT * FROM t WHERE id = :id AND store = :store",
            params=[
                StatementParam(name="id", type="int"),
                StatementParam(name="store", source=ParamSource.CONTEXT, type="int"),
            ],
        )
        result = tool.invoke(
            {"id": 7, "runtime": _runtime(Context(user_id="u", store=42))}
        )

    assert "(1 row returned)" in result
    _, kwargs = mock_ws.statement_execution.execute_statement.call_args
    sent = {p.name: p.value for p in kwargs["parameters"]}
    assert sent == {"id": "7", "store": "42"}
    assert all(isinstance(p, StatementParameterListItem) for p in kwargs["parameters"])


@pytest.mark.unit
def test_warehouse_optional_default_applied(mock_warehouse: WarehouseModel) -> None:
    """An omitted optional param falls back to its default."""
    from unittest.mock import PropertyMock, patch

    from dao_ai.config import StatementParam

    mock_response = StatementResponse(
        statement_id="sid",
        status=StatementStatus(state=StatementState.SUCCEEDED),
        result=ResultData(data_array=[["ok"]]),
        manifest=ResultManifest(schema=ResultSchema(columns=[ColumnInfo(name="c")])),
    )
    mock_ws = MagicMock()
    mock_ws.statement_execution.execute_statement.return_value = mock_response

    with patch.object(
        type(mock_warehouse),
        "workspace_client",
        new_callable=PropertyMock,
        return_value=mock_ws,
    ):
        tool = create_execute_statement_tool(
            mock_warehouse,
            statement="SELECT * FROM t LIMIT :limit",
            params=[
                StatementParam(name="limit", type="int", required=False, default=25)
            ],
        )
        tool.invoke({"runtime": _runtime()})

    _, kwargs = mock_ws.statement_execution.execute_statement.call_args
    sent = {p.name: p.value for p in kwargs["parameters"]}
    assert sent == {"limit": "25"}


@pytest.mark.unit
def test_missing_required_context_param_errors(
    mock_warehouse: WarehouseModel,
) -> None:
    """A required context param that isn't on Context yields an Error string."""
    from dao_ai.config import ParamSource, StatementParam
    from dao_ai.state import Context

    tool = create_execute_statement_tool(
        mock_warehouse,
        statement="SELECT * FROM t WHERE store = :store",
        params=[StatementParam(name="store", source=ParamSource.CONTEXT, type="int")],
    )
    result = tool.invoke({"runtime": _runtime(Context(user_id="u"))})
    assert result.startswith("Error:")
    assert "store" in result


@pytest.mark.unit
def test_zero_param_tool_takes_no_args(mock_warehouse: WarehouseModel) -> None:
    """params=None preserves the legacy zero-argument tool shape."""
    tool = create_execute_statement_tool(mock_warehouse, statement="SELECT 1")
    assert tool.args == {}
    assert tool.name == "execute_sql_tool"


# ---------------------------------------------------------------------------
# Dispatch + Lakebase backend
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dispatch_unknown_target_raises() -> None:
    """An unsupported backend type raises ValueError."""
    with pytest.raises(ValueError, match="WarehouseModel or DatabaseModel"):
        create_execute_statement_tool("not-a-model", statement="SELECT 1")


@pytest.mark.unit
def test_dispatch_dict_coercion() -> None:
    """Dicts coerce by shape: warehouse_id -> warehouse, else database."""
    wh_tool = create_execute_statement_tool(
        {"name": "wh", "warehouse_id": "abc"}, statement="SELECT 1"
    )
    assert wh_tool.coroutine is None  # warehouse path is sync
    db_tool = create_execute_statement_tool(
        {"name": "db", "project": "proj"}, statement="SELECT 1"
    )
    assert db_tool.coroutine is not None  # lakebase path is async


@pytest.mark.unit
def test_lakebase_binds_mapping_and_formats() -> None:
    """Lakebase path binds a {name: value} mapping and formats a text table."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from dao_ai.config import DatabaseModel, ParamSource, StatementParam
    from dao_ai.state import Context

    database = DatabaseModel(name="retail", project="retail-consumer-goods")
    tool = create_execute_statement_tool(
        database,
        statement=(
            "SELECT product_id, on_hand FROM inventory "
            "WHERE store = %(store)s AND category = %(category)s"
        ),
        params=[
            StatementParam(name="category", type="string"),
            StatementParam(name="store", source=ParamSource.CONTEXT, type="int"),
        ],
    )
    # LLM schema exposes only the llm param
    assert "category" in tool.args
    assert "store" not in tool.args

    rows = [{"product_id": 1, "on_hand": 5}, {"product_id": 2, "on_hand": 0}]
    with patch.object(
        DatabaseModel, "aexecute_query", new=AsyncMock(return_value=rows)
    ) as mock_q:
        result = asyncio.run(
            tool.ainvoke(
                {
                    "category": "paint",
                    "runtime": _runtime(Context(user_id="u", store=42)),
                }
            )
        )

    args, _ = mock_q.call_args
    assert args[1] == {"category": "paint", "store": 42}
    assert "product_id | on_hand" in result
    assert "(2 rows returned)" in result


@pytest.mark.unit
def test_lakebase_no_rows() -> None:
    """Lakebase path reports the no-results message for an empty result set."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from dao_ai.config import DatabaseModel

    database = DatabaseModel(name="retail", project="retail-consumer-goods")
    tool = create_execute_statement_tool(database, statement="SELECT 1 WHERE false")
    with patch.object(DatabaseModel, "aexecute_query", new=AsyncMock(return_value=[])):
        result = asyncio.run(tool.ainvoke({"runtime": _runtime()}))
    assert "no results returned" in result
