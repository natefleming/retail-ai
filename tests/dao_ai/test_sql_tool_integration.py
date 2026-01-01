"""Integration tests for SQL execution tool.

These tests require actual Databricks credentials and a warehouse.
They are marked with @pytest.mark.integration and will be skipped
if credentials are not available.
"""

import os

import pytest

from dao_ai.config import WarehouseModel
from dao_ai.tools.sql import create_execute_statement_tool


@pytest.fixture
def warehouse() -> WarehouseModel:
    """Create a warehouse model using environment variables."""
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        pytest.skip("DATABRICKS_WAREHOUSE_ID not set")

    return WarehouseModel(
        name="test_warehouse",
        warehouse_id=warehouse_id,
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_execute_simple_select_statement(warehouse: WarehouseModel) -> None:
    """Test executing a simple SELECT statement."""
    # Create a tool with a simple query
    tool = create_execute_statement_tool(
        warehouse=warehouse,
        statement="SELECT 1 as test_column",
        name="test_select",
        description="Test simple SELECT statement",
    )

    # Execute the tool
    result = tool.invoke({})

    # Verify result
    assert isinstance(result, str)
    assert "test_column" in result
    assert "1" in result
    assert "1 row" in result


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_execute_arithmetic_query(warehouse: WarehouseModel) -> None:
    """Test executing a query with arithmetic operations."""
    tool = create_execute_statement_tool(
        warehouse=warehouse,
        statement="""
            SELECT 
                1 + 1 as addition,
                10 - 3 as subtraction,
                5 * 4 as multiplication,
                20 / 4 as division
        """,
        name="test_arithmetic",
        description="Test arithmetic operations",
    )

    result = tool.invoke({})

    assert "addition" in result
    assert "subtraction" in result
    assert "multiplication" in result
    assert "division" in result
    assert "2" in result  # 1 + 1
    assert "7" in result  # 10 - 3
    assert "20" in result  # 5 * 4
    assert "5" in result  # 20 / 4


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_execute_multiple_rows(warehouse: WarehouseModel) -> None:
    """Test executing a query that returns multiple rows."""
    tool = create_execute_statement_tool(
        warehouse=warehouse,
        statement="""
            SELECT number, number * 2 as doubled
            FROM (VALUES (1), (2), (3), (4), (5)) AS t(number)
        """,
        name="test_multiple_rows",
        description="Test query with multiple rows",
    )

    result = tool.invoke({})

    assert "number" in result
    assert "doubled" in result
    assert "5 rows returned" in result


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_execute_aggregate_query(warehouse: WarehouseModel) -> None:
    """Test executing an aggregate query."""
    tool = create_execute_statement_tool(
        warehouse=warehouse,
        statement="""
            SELECT 
                COUNT(*) as row_count,
                SUM(value) as total,
                AVG(value) as average,
                MIN(value) as minimum,
                MAX(value) as maximum
            FROM (VALUES (10), (20), (30), (40), (50)) AS t(value)
        """,
        name="test_aggregates",
        description="Test aggregate functions",
    )

    result = tool.invoke({})

    assert "row_count" in result
    assert "total" in result
    assert "average" in result
    assert "minimum" in result
    assert "maximum" in result
    assert "5" in result  # COUNT(*)
    assert "150" in result  # SUM
    assert "30" in result  # AVG
    assert "10" in result  # MIN
    assert "50" in result  # MAX


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_execute_date_functions(warehouse: WarehouseModel) -> None:
    """Test executing a query with date functions."""
    tool = create_execute_statement_tool(
        warehouse=warehouse,
        statement="""
            SELECT 
                CURRENT_DATE() as today,
                CURRENT_TIMESTAMP() as now,
                DATE_ADD(CURRENT_DATE(), 7) as next_week
        """,
        name="test_dates",
        description="Test date functions",
    )

    result = tool.invoke({})

    assert "today" in result
    assert "now" in result
    assert "next_week" in result


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_execute_string_functions(warehouse: WarehouseModel) -> None:
    """Test executing a query with string functions."""
    tool = create_execute_statement_tool(
        warehouse=warehouse,
        statement="""
            SELECT 
                UPPER('hello') as uppercase,
                LOWER('WORLD') as lowercase,
                CONCAT('Hello', ' ', 'World') as concatenated,
                LENGTH('test') as string_length
        """,
        name="test_strings",
        description="Test string functions",
    )

    result = tool.invoke({})

    assert "uppercase" in result
    assert "HELLO" in result
    assert "lowercase" in result
    assert "world" in result
    assert "concatenated" in result
    assert "Hello World" in result
    assert "string_length" in result
    assert "4" in result


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_execute_with_null_values(warehouse: WarehouseModel) -> None:
    """Test executing a query that includes NULL values."""
    tool = create_execute_statement_tool(
        warehouse=warehouse,
        statement="""
            SELECT 
                CAST(NULL AS INT) as null_value,
                COALESCE(CAST(NULL AS INT), 42) as with_default,
                CASE WHEN 1 = 1 THEN 'yes' ELSE NULL END as conditional
        """,
        name="test_nulls",
        description="Test NULL handling",
    )

    result = tool.invoke({})

    assert "null_value" in result
    assert "NULL" in result
    assert "with_default" in result
    assert "42" in result
    assert "conditional" in result
    assert "yes" in result


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_execute_invalid_sql(warehouse: WarehouseModel) -> None:
    """Test executing invalid SQL to ensure proper error handling."""
    tool = create_execute_statement_tool(
        warehouse=warehouse,
        statement="SELECT * FROM this_table_definitely_does_not_exist_12345",
        name="test_invalid_sql",
        description="Test error handling",
    )

    result = tool.invoke({})

    # Should return an error message, not raise an exception
    assert "Error" in result or "error" in result.lower()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_execute_empty_result_set(warehouse: WarehouseModel) -> None:
    """Test executing a query that returns no rows."""
    tool = create_execute_statement_tool(
        warehouse=warehouse,
        statement="""
            SELECT * FROM (VALUES (1), (2), (3)) AS t(value)
            WHERE value > 10
        """,
        name="test_empty_result",
        description="Test empty result set",
    )

    result = tool.invoke({})

    assert "0 rows" in result or "empty" in result.lower()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_tool_can_be_invoked_multiple_times(warehouse: WarehouseModel) -> None:
    """Test that a tool can be invoked multiple times successfully."""
    tool = create_execute_statement_tool(
        warehouse=warehouse,
        statement="SELECT RAND() as random_value",
        name="test_reusable",
        description="Test tool reusability",
    )

    # Execute multiple times
    result1 = tool.invoke({})
    result2 = tool.invoke({})
    result3 = tool.invoke({})

    # All should succeed
    assert "random_value" in result1
    assert "random_value" in result2
    assert "random_value" in result3

    # Results may differ (random values)
    # But all should be valid


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_tool_with_custom_name_and_description(warehouse: WarehouseModel) -> None:
    """Test creating a tool with custom name and description."""
    custom_name = "my_custom_sql_tool"
    custom_description = "This is my custom SQL tool for testing"

    tool = create_execute_statement_tool(
        warehouse=warehouse,
        statement="SELECT 'success' as status",
        name=custom_name,
        description=custom_description,
    )

    # Verify tool attributes
    assert tool.name == custom_name
    assert tool.description == custom_description

    # Verify it still executes correctly
    result = tool.invoke({})
    assert "status" in result
    assert "success" in result
