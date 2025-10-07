from unittest.mock import MagicMock, Mock, patch

import pytest
from conftest import has_databricks_env
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.catalog import FunctionInfo, TableInfo
from mlflow.models.resources import DatabricksFunction, DatabricksTable

from dao_ai.config import (
    AppConfig,
    DatabaseModel,
    FunctionModel,
    SchemaModel,
    TableModel,
)
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
            "main.default.normal_trace_table",
        ]
        assert sorted(resource_names) == sorted(expected_names)

        # Verify that filtered tables are not included
        filtered_out_names = [
            "main.default.data_payload",
            "main.default.test_assessment_logs",
            "main.default.app_request_logs",
            "main.default.trace_logs_daily",
            "main.default.trace_logs_hourly",
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


# ==================== create_lakebase Tests ====================


@pytest.mark.unit
def test_database_model_capacity_validation():
    """Test DatabaseModel capacity field validation."""
    # Valid capacity values
    db1 = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        capacity="CU_1",
        user="test_user",
        password="test_password",
    )
    assert db1.capacity == "CU_1"

    db2 = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        capacity="CU_2",
        user="test_user",
        password="test_password",
    )
    assert db2.capacity == "CU_2"

    # Default capacity should be CU_2
    db3 = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )
    assert db3.capacity == "CU_2"


@pytest.mark.unit
def test_create_lakebase_new_database():
    """Test create_lakebase when database doesn't exist."""
    # Mock workspace client
    mock_workspace_client = MagicMock()
    mock_workspace_client.database.get_database_instance.side_effect = NotFound()
    mock_workspace_client.database.create_database_instance.return_value = None

    # Create database model
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        description="Test database",
        host="localhost",
        database="test_database",
        port=5432,
        capacity="CU_2",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Mock workspace_client property on database
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        # Call create_lakebase
        provider.create_lakebase(database)

    # Verify create was called with correct parameters
    mock_workspace_client.database.create_database_instance.assert_called_once()
    call_args = mock_workspace_client.database.create_database_instance.call_args
    database_instance = call_args.kwargs["database_instance"]
    assert database_instance.name == "test_db"
    assert database_instance.capacity == "CU_2"


@pytest.mark.unit
def test_create_lakebase_database_already_exists_available():
    """Test create_lakebase when database already exists and is AVAILABLE."""
    # Mock workspace client
    mock_workspace_client = MagicMock()

    # Mock existing database instance
    mock_instance = MagicMock()
    mock_instance.state = "AVAILABLE"
    mock_workspace_client.database.get_database_instance.return_value = mock_instance

    # Create database model
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Mock workspace_client property on database
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        # Call create_lakebase
        provider.create_lakebase(database)

    # Verify get was called but create was not
    mock_workspace_client.database.get_database_instance.assert_called_once_with(
        name="test_db"
    )
    mock_workspace_client.database.create_database_instance.assert_not_called()


@pytest.mark.unit
def test_create_lakebase_database_starting_state():
    """Test create_lakebase when database is in STARTING state."""
    # Mock workspace client
    mock_workspace_client = MagicMock()

    # Mock initial instance in STARTING state
    mock_instance_starting = MagicMock()
    mock_instance_starting.state = "STARTING"

    # Mock instance that becomes AVAILABLE
    mock_instance_available = MagicMock()
    mock_instance_available.state = "AVAILABLE"

    # First call returns STARTING, second returns AVAILABLE
    mock_workspace_client.database.get_database_instance.side_effect = [
        mock_instance_starting,
        mock_instance_available,
    ]

    # Create database model
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Mock workspace_client property and time.sleep
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        with patch("time.sleep"):
            # Call create_lakebase
            provider.create_lakebase(database)

    # Verify get was called twice (initial check + one in loop)
    assert mock_workspace_client.database.get_database_instance.call_count == 2
    mock_workspace_client.database.create_database_instance.assert_not_called()


@pytest.mark.unit
def test_create_lakebase_database_updating_state():
    """Test create_lakebase when database is in UPDATING state."""
    # Mock workspace client
    mock_workspace_client = MagicMock()

    # Mock initial instance in UPDATING state
    mock_instance_updating = MagicMock()
    mock_instance_updating.state = "UPDATING"

    # Mock instance that becomes AVAILABLE
    mock_instance_available = MagicMock()
    mock_instance_available.state = "AVAILABLE"

    # First call returns UPDATING, second returns AVAILABLE
    mock_workspace_client.database.get_database_instance.side_effect = [
        mock_instance_updating,
        mock_instance_available,
    ]

    # Create database model
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Mock workspace_client property and time.sleep
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        with patch("time.sleep"):
            # Call create_lakebase
            provider.create_lakebase(database)

    # Verify get was called twice
    assert mock_workspace_client.database.get_database_instance.call_count == 2
    mock_workspace_client.database.create_database_instance.assert_not_called()


@pytest.mark.unit
def test_create_lakebase_database_stopped_state():
    """Test create_lakebase when database is in STOPPED state."""
    # Mock workspace client
    mock_workspace_client = MagicMock()

    # Mock existing database instance in STOPPED state
    mock_instance = MagicMock()
    mock_instance.state = "STOPPED"
    mock_workspace_client.database.get_database_instance.return_value = mock_instance

    # Create database model
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Mock workspace_client property
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        # Call create_lakebase - should return without error
        provider.create_lakebase(database)

    # Verify get was called but create was not
    mock_workspace_client.database.get_database_instance.assert_called_once()
    mock_workspace_client.database.create_database_instance.assert_not_called()


@pytest.mark.unit
def test_create_lakebase_database_deleting_state():
    """Test create_lakebase when database is in DELETING state."""
    # Mock workspace client
    mock_workspace_client = MagicMock()

    # Mock existing database instance in DELETING state
    mock_instance = MagicMock()
    mock_instance.state = "DELETING"
    mock_workspace_client.database.get_database_instance.return_value = mock_instance

    # Create database model
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Mock workspace_client property
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        # Call create_lakebase - should return without error
        provider.create_lakebase(database)

    # Verify get was called but create was not
    mock_workspace_client.database.get_database_instance.assert_called_once()
    mock_workspace_client.database.create_database_instance.assert_not_called()


@pytest.mark.unit
def test_create_lakebase_concurrent_creation():
    """Test create_lakebase when database is created concurrently by another process."""
    # Mock workspace client
    mock_workspace_client = MagicMock()
    mock_workspace_client.database.get_database_instance.side_effect = NotFound()

    # Simulate concurrent creation - create raises "already exists" error
    mock_workspace_client.database.create_database_instance.side_effect = Exception(
        "RESOURCE_ALREADY_EXISTS: Database already exists"
    )

    # Create database model
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Mock workspace_client property
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        # Should not raise exception
        provider.create_lakebase(database)

    # Verify both get and create were called
    mock_workspace_client.database.get_database_instance.assert_called_once()
    mock_workspace_client.database.create_database_instance.assert_called_once()


@pytest.mark.unit
def test_create_lakebase_unexpected_error():
    """Test create_lakebase handles unexpected errors appropriately."""
    # Mock workspace client
    mock_workspace_client = MagicMock()
    mock_workspace_client.database.get_database_instance.side_effect = NotFound()

    # Simulate unexpected error during creation
    mock_workspace_client.database.create_database_instance.side_effect = Exception(
        "Unexpected error occurred"
    )

    # Create database model
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Mock workspace_client property
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        # Should raise the exception
        with pytest.raises(Exception, match="Unexpected error occurred"):
            provider.create_lakebase(database)


@pytest.mark.unit
def test_create_lakebase_timeout_waiting_for_available():
    """Test create_lakebase handles timeout when waiting for AVAILABLE state."""
    # Mock workspace client
    mock_workspace_client = MagicMock()

    # Mock instance that stays in STARTING state
    mock_instance_starting = MagicMock()
    mock_instance_starting.state = "STARTING"
    mock_workspace_client.database.get_database_instance.return_value = (
        mock_instance_starting
    )

    # Create database model
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Mock workspace_client property and time.sleep
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        with patch("time.sleep") as mock_sleep:
            # Call create_lakebase - should handle timeout gracefully
            provider.create_lakebase(database)

            # Verify sleep was called multiple times (waiting in loop)
            assert mock_sleep.call_count > 0


@pytest.mark.unit
def test_create_lakebase_default_values():
    """Test create_lakebase uses correct default values."""
    # Mock workspace client
    mock_workspace_client = MagicMock()
    mock_workspace_client.database.get_database_instance.side_effect = NotFound()
    mock_workspace_client.database.create_database_instance.return_value = None

    # Create database model with minimal parameters
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Mock workspace_client property
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        # Call create_lakebase
        provider.create_lakebase(database)

    # Verify create was called with default values
    call_args = mock_workspace_client.database.create_database_instance.call_args
    database_instance = call_args.kwargs["database_instance"]
    assert database_instance.capacity == "CU_2"  # Default capacity


@pytest.mark.unit
def test_create_lakebase_custom_capacity_cu1():
    """Test create_lakebase with custom capacity CU_1."""
    # Mock workspace client
    mock_workspace_client = MagicMock()
    mock_workspace_client.database.get_database_instance.side_effect = NotFound()
    mock_workspace_client.database.create_database_instance.return_value = None

    # Create database model with CU_1 capacity
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        capacity="CU_1",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Mock workspace_client property
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        # Call create_lakebase
        provider.create_lakebase(database)

    # Verify create was called with CU_1 capacity
    call_args = mock_workspace_client.database.create_database_instance.call_args
    database_instance = call_args.kwargs["database_instance"]
    assert database_instance.capacity == "CU_1"


@pytest.mark.unit
def test_create_lakebase_database_disappears_during_wait():
    """Test create_lakebase when database disappears while waiting for AVAILABLE state."""
    # Mock workspace client
    mock_workspace_client = MagicMock()

    # Mock initial instance in STARTING state
    mock_instance_starting = MagicMock()
    mock_instance_starting.state = "STARTING"

    # First call returns STARTING, second raises NotFound (database disappeared)
    mock_workspace_client.database.get_database_instance.side_effect = [
        mock_instance_starting,
        NotFound(),
    ]

    # Create database model
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Mock workspace_client property and time.sleep
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        with patch("time.sleep"):
            # Should not raise exception
            provider.create_lakebase(database)

    # Verify get was called twice
    assert mock_workspace_client.database.get_database_instance.call_count == 2


# ==================== create_lakebase_instance_role Tests ====================


@pytest.mark.unit
def test_create_lakebase_instance_role_success():
    """Test creating a lakebase instance role successfully."""
    # Mock workspace client
    mock_workspace_client = MagicMock()
    mock_workspace_client.database.get_database_instance_role.side_effect = NotFound()
    mock_workspace_client.database.create_database_instance_role.return_value = None

    # Create database model with client_id
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        client_id="test-client-id-123",
        client_secret="test-secret",
        workspace_host="https://test.databricks.com",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Call create_lakebase_instance_role
    provider.create_lakebase_instance_role(database)

    # Verify get was called to check if role exists
    mock_workspace_client.database.get_database_instance_role.assert_called_once_with(
        instance_name="test_db",
        name="test-client-id-123",
    )

    # Verify create was called with correct parameters
    mock_workspace_client.database.create_database_instance_role.assert_called_once()
    call_args = mock_workspace_client.database.create_database_instance_role.call_args
    assert call_args.kwargs["instance_name"] == "test_db"

    role = call_args.kwargs["database_instance_role"]
    assert role.name == "test-client-id-123"
    assert role.identity_type.value == "SERVICE_PRINCIPAL"
    assert role.membership_role.value == "DATABRICKS_SUPERUSER"


@pytest.mark.unit
def test_create_lakebase_instance_role_already_exists():
    """Test when instance role already exists."""
    # Mock workspace client
    mock_workspace_client = MagicMock()
    mock_existing_role = MagicMock()
    mock_existing_role.name = "test-client-id-123"
    mock_workspace_client.database.get_database_instance_role.return_value = (
        mock_existing_role
    )

    # Create database model with client_id
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        client_id="test-client-id-123",
        client_secret="test-secret",
        workspace_host="https://test.databricks.com",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Call create_lakebase_instance_role
    provider.create_lakebase_instance_role(database)

    # Verify get was called
    mock_workspace_client.database.get_database_instance_role.assert_called_once()

    # Verify create was NOT called since role already exists
    mock_workspace_client.database.create_database_instance_role.assert_not_called()


@pytest.mark.unit
def test_create_lakebase_instance_role_missing_client_id():
    """Test that a warning is logged and method returns early when client_id is not provided."""
    # Mock workspace client
    mock_workspace_client = MagicMock()

    # Create database model WITHOUT client_id
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Call create_lakebase_instance_role - should log warning and return early
    provider.create_lakebase_instance_role(database)

    # Verify no API calls were made
    mock_workspace_client.database.get_database_instance_role.assert_not_called()
    mock_workspace_client.database.create_database_instance_role.assert_not_called()


@pytest.mark.unit
def test_create_lakebase_instance_role_concurrent_creation():
    """Test when role is created concurrently by another process."""
    # Mock workspace client
    mock_workspace_client = MagicMock()
    mock_workspace_client.database.get_database_instance_role.side_effect = NotFound()

    # Simulate concurrent creation - create raises "already exists" error
    mock_workspace_client.database.create_database_instance_role.side_effect = (
        Exception("RESOURCE_ALREADY_EXISTS: Role already exists")
    )

    # Create database model with client_id
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        client_id="test-client-id-123",
        client_secret="test-secret",
        workspace_host="https://test.databricks.com",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Should not raise exception
    provider.create_lakebase_instance_role(database)

    # Verify both get and create were called
    mock_workspace_client.database.get_database_instance_role.assert_called_once()
    mock_workspace_client.database.create_database_instance_role.assert_called_once()


@pytest.mark.unit
def test_create_lakebase_instance_role_unexpected_error():
    """Test that unexpected errors are raised."""
    # Mock workspace client
    mock_workspace_client = MagicMock()
    mock_workspace_client.database.get_database_instance_role.side_effect = NotFound()

    # Simulate unexpected error during creation
    mock_workspace_client.database.create_database_instance_role.side_effect = (
        Exception("Unexpected error occurred")
    )

    # Create database model with client_id
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        client_id="test-client-id-123",
        client_secret="test-secret",
        workspace_host="https://test.databricks.com",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Should raise the exception
    with pytest.raises(Exception, match="Unexpected error occurred"):
        provider.create_lakebase_instance_role(database)


@pytest.mark.unit
def test_create_lakebase_instance_role_with_composite_variable():
    """Test creating role when client_id is a CompositeVariableModel."""
    from dao_ai.config import CompositeVariableModel, EnvironmentVariableModel

    # Mock workspace client
    mock_workspace_client = MagicMock()
    mock_workspace_client.database.get_database_instance_role.side_effect = NotFound()
    mock_workspace_client.database.create_database_instance_role.return_value = None

    # Create database model with CompositeVariableModel for client_id
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        client_id=CompositeVariableModel(
            default_value="test-client-id-456",
            options=[EnvironmentVariableModel(env="TEST_CLIENT_ID")],
        ),
        client_secret="test-secret",
        workspace_host="https://test.databricks.com",
    )

    # Create provider
    provider = DatabricksProvider(w=mock_workspace_client)

    # Call create_lakebase_instance_role
    provider.create_lakebase_instance_role(database)

    # Verify get was called with resolved client_id
    mock_workspace_client.database.get_database_instance_role.assert_called_once()
    call_args = mock_workspace_client.database.get_database_instance_role.call_args
    assert call_args.kwargs["name"] == "test-client-id-456"
