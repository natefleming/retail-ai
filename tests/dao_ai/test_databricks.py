from unittest.mock import Mock

import pytest
from conftest import has_databricks_env
from databricks.sdk.service.catalog import FunctionInfo, TableInfo
from mlflow.models.resources import DatabricksFunction, DatabricksTable

from dao_ai.config import AppConfig, FunctionModel, SchemaModel, TableModel
from dao_ai.providers.databricks import DatabricksProvider


@pytest.mark.unit
def test_table_model_validation():
    """Test TableModel validation logic."""
    # Should fail when neither name nor schema is provided
    with pytest.raises(
        ValueError, match="Either 'name' or 'schema_model' must be provided"
    ):
        TableModel()

    # Should succeed with name only
    table = TableModel(name="my_table")
    assert table.name == "my_table"
    assert table.schema_model is None

    # Should succeed with schema only
    schema = SchemaModel(catalog_name="main", schema_name="default")
    table = TableModel(schema=schema)
    assert table.name is None
    assert table.schema_model is not None

    # Should succeed with both
    table = TableModel(name="my_table", schema=schema)
    assert table.name == "my_table"
    assert table.schema_model is not None


@pytest.mark.unit
def test_table_model_full_name():
    """Test TableModel full_name property."""
    # Name only
    table = TableModel(name="my_table")
    assert table.full_name == "my_table"

    # Schema only
    schema = SchemaModel(catalog_name="main", schema_name="default")
    table = TableModel(schema=schema)
    assert table.full_name == "main.default"

    # Both name and schema
    table = TableModel(name="my_table", schema=schema)
    assert table.full_name == "main.default.my_table"


@pytest.mark.unit
def test_table_model_as_resources_single_table():
    """Test TableModel.as_resources with specific table name."""
    schema = SchemaModel(catalog_name="main", schema_name="default")
    table = TableModel(name="my_table", schema=schema)

    resources = table.as_resources()

    assert len(resources) == 1
    assert isinstance(resources[0], DatabricksTable)
    assert resources[0].name == "main.default.my_table"
    assert not resources[0].on_behalf_of_user


@pytest.mark.unit
def test_table_model_as_resources_discovery_mode(monkeypatch):
    """Test TableModel.as_resources in discovery mode (schema only)."""
    # Mock the workspace client and table listing
    mock_workspace_client = Mock()
    mock_table_info_1 = Mock(spec=TableInfo)
    mock_table_info_1.name = "table1"
    mock_table_info_2 = Mock(spec=TableInfo)
    mock_table_info_2.name = "table2"

    mock_workspace_client.tables.list.return_value = iter(
        [mock_table_info_1, mock_table_info_2]
    )

    schema = SchemaModel(catalog_name="main", schema_name="default")
    table = TableModel(schema=schema)

    # Mock the WorkspaceClient constructor
    with monkeypatch.context() as m:
        m.setattr(
            "dao_ai.config.WorkspaceClient", lambda **kwargs: mock_workspace_client
        )

        resources = table.as_resources()

        assert len(resources) == 2
        assert all(isinstance(r, DatabricksTable) for r in resources)
        assert resources[0].name == "main.default.table1"
        assert resources[1].name == "main.default.table2"

        # Verify the workspace client was called correctly
        mock_workspace_client.tables.list.assert_called_once_with(
            catalog_name="main", schema_name="default"
        )


@pytest.mark.unit
def test_table_model_as_resources_discovery_mode_with_filtering(monkeypatch):
    """Test TableModel.as_resources discovery mode with excluded suffixes and prefixes filtering."""
    # Mock the workspace client and table listing with tables that should be filtered
    mock_workspace_client = Mock()
    
    # Create mock tables - some should be filtered out
    mock_tables = []
    table_names = [
        "valid_table1",  # Should be included
        "valid_table2",  # Should be included  
        "data_payload",  # Should be excluded (ends with _payload)
        "test_assessment_logs",  # Should be excluded (ends with _assessment_logs)
        "app_request_logs",  # Should be excluded (ends with _request_logs)
        "trace_logs_daily",  # Should be excluded (starts with trace_logs_)
        "trace_logs_hourly",  # Should be excluded (starts with trace_logs_)
        "normal_trace_table",  # Should be included (contains trace but doesn't start with trace_logs_)
    ]
    
    for name in table_names:
        mock_table = Mock(spec=TableInfo)
        mock_table.name = name
        mock_tables.append(mock_table)

    mock_workspace_client.tables.list.return_value = iter(mock_tables)

    schema = SchemaModel(catalog_name="main", schema_name="default")
    table = TableModel(schema=schema)

    # Mock the WorkspaceClient constructor
    with monkeypatch.context() as m:
        m.setattr(
            "dao_ai.config.WorkspaceClient", lambda **kwargs: mock_workspace_client
        )

        resources = table.as_resources()

        # Should only have 3 tables (the valid ones that weren't filtered)
        assert len(resources) == 3
        assert all(isinstance(r, DatabricksTable) for r in resources)
        
        # Check that only the expected tables are included
        resource_names = [r.name for r in resources]
        expected_names = [
            "main.default.valid_table1",
            "main.default.valid_table2", 
            "main.default.normal_trace_table"
        ]
        assert sorted(resource_names) == sorted(expected_names)
        
        # Verify that filtered tables are not included
        filtered_out_names = [
            "main.default.data_payload",
            "main.default.test_assessment_logs",
            "main.default.app_request_logs",
            "main.default.trace_logs_daily",
            "main.default.trace_logs_hourly"
        ]
        for filtered_name in filtered_out_names:
            assert filtered_name not in resource_names

        # Verify the workspace client was called correctly
        mock_workspace_client.tables.list.assert_called_once_with(
            catalog_name="main", schema_name="default"
        )


@pytest.mark.unit
def test_function_model_validation():
    """Test FunctionModel validation logic."""
    # Should fail when neither name nor schema is provided
    with pytest.raises(
        ValueError, match="Either 'name' or 'schema_model' must be provided"
    ):
        FunctionModel()

    # Should succeed with name only
    function = FunctionModel(name="my_function")
    assert function.name == "my_function"
    assert function.schema_model is None

    # Should succeed with schema only
    schema = SchemaModel(catalog_name="main", schema_name="default")
    function = FunctionModel(schema=schema)
    assert function.name is None
    assert function.schema_model is not None

    # Should succeed with both
    function = FunctionModel(name="my_function", schema=schema)
    assert function.name == "my_function"
    assert function.schema_model is not None


@pytest.mark.unit
def test_function_model_full_name():
    """Test FunctionModel full_name property."""
    # Name only
    function = FunctionModel(name="my_function")
    assert function.full_name == "my_function"

    # Schema only
    schema = SchemaModel(catalog_name="main", schema_name="default")
    function = FunctionModel(schema=schema)
    assert function.full_name == "main.default"

    # Both name and schema
    function = FunctionModel(name="my_function", schema=schema)
    assert function.full_name == "main.default.my_function"


@pytest.mark.unit
def test_function_model_as_resources_single_function():
    """Test FunctionModel.as_resources with specific function name."""
    schema = SchemaModel(catalog_name="main", schema_name="default")
    function = FunctionModel(name="my_function", schema=schema)

    resources = function.as_resources()

    assert len(resources) == 1
    assert isinstance(resources[0], DatabricksFunction)
    assert resources[0].name == "main.default.my_function"
    assert not resources[0].on_behalf_of_user


@pytest.mark.unit
def test_function_model_as_resources_discovery_mode(monkeypatch):
    """Test FunctionModel.as_resources in discovery mode (schema only)."""
    # Mock the workspace client and function listing
    mock_workspace_client = Mock()
    mock_function_info_1 = Mock(spec=FunctionInfo)
    mock_function_info_1.name = "function1"
    mock_function_info_2 = Mock(spec=FunctionInfo)
    mock_function_info_2.name = "function2"

    mock_workspace_client.functions.list.return_value = iter(
        [mock_function_info_1, mock_function_info_2]
    )

    schema = SchemaModel(catalog_name="main", schema_name="default")
    function = FunctionModel(schema=schema)

    # Mock the WorkspaceClient constructor
    with monkeypatch.context() as m:
        m.setattr(
            "dao_ai.config.WorkspaceClient", lambda **kwargs: mock_workspace_client
        )

        resources = function.as_resources()

        assert len(resources) == 2
        assert all(isinstance(r, DatabricksFunction) for r in resources)
        assert resources[0].name == "main.default.function1"
        assert resources[1].name == "main.default.function2"

        # Verify the workspace client was called correctly
        mock_workspace_client.functions.list.assert_called_once_with(
            catalog_name="main", schema_name="default"
        )


@pytest.mark.unit
def test_resource_models_on_behalf_of_user():
    """Test that resources respect on_behalf_of_user flag."""
    schema = SchemaModel(catalog_name="main", schema_name="default")

    # Test TableModel
    table = TableModel(name="my_table", schema=schema)
    table.on_behalf_of_user = True

    table_resources = table.as_resources()
    assert table_resources[0].on_behalf_of_user

    # Test FunctionModel
    function = FunctionModel(name="my_function", schema=schema)
    function.on_behalf_of_user = True

    function_resources = function.as_resources()
    assert function_resources[0].on_behalf_of_user


@pytest.mark.unit
def test_table_model_api_scopes():
    """Test TableModel API scopes."""
    table = TableModel(name="my_table")
    assert table.api_scopes == []


@pytest.mark.unit
def test_function_model_api_scopes():
    """Test FunctionModel API scopes."""
    function = FunctionModel(name="my_function")
    assert function.api_scopes == ["sql.statement-execution"]


@pytest.mark.system
@pytest.mark.slow
@pytest.mark.skipif(
    not has_databricks_env(), reason="Missing Databricks environment variables"
)
@pytest.mark.skip("Skipping Databricks agent creation test")
def test_databricks_create_agent(config: AppConfig) -> None:
    provider: DatabricksProvider = DatabricksProvider()
    provider.create_agent(config=config)
    assert True
