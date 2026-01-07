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
