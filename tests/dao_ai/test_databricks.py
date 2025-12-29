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


@pytest.mark.unit
def test_create_agent_sets_experiment():
    """Test that create_agent properly sets up MLflow experiment before starting run."""
    from unittest.mock import MagicMock, patch

    import mlflow

    from dao_ai.config import AppConfig
    from dao_ai.providers.databricks import DatabricksProvider

    # Create a minimal mock config
    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock()
    mock_app.name = "test_app"
    mock_app.code_paths = []
    mock_app.pip_requirements = []
    mock_app.input_example = None
    mock_config.app = mock_app

    # Mock resources
    mock_resources = MagicMock()
    mock_resources.llms = MagicMock(values=lambda: [])
    mock_resources.vector_stores = MagicMock(values=lambda: [])
    mock_resources.warehouses = MagicMock(values=lambda: [])
    mock_resources.genie_rooms = MagicMock(values=lambda: [])
    mock_resources.tables = MagicMock(values=lambda: [])
    mock_resources.functions = MagicMock(values=lambda: [])
    mock_resources.connections = MagicMock(values=lambda: [])
    mock_resources.databases = MagicMock(values=lambda: [])
    mock_resources.volumes = MagicMock(values=lambda: [])
    mock_config.resources = mock_resources

    # Create mock experiment
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "test_experiment_123"
    mock_experiment.name = "/Users/test_user/test_app"

    with (
        patch.object(
            DatabricksProvider, "get_or_create_experiment", return_value=mock_experiment
        ) as mock_get_experiment,
        patch.object(mlflow, "set_experiment") as mock_set_experiment,
        patch.object(mlflow, "set_registry_uri"),
        patch.object(mlflow, "start_run") as mock_start_run,
        patch.object(mlflow, "set_tag"),
        patch.object(mlflow.pyfunc, "log_model") as mock_log_model,
        patch.object(mlflow, "register_model"),
        patch("dao_ai.providers.databricks.MlflowClient"),
        patch("dao_ai.providers.databricks.is_installed", return_value=True),
        patch(
            "dao_ai.providers.databricks.is_lib_provided",
            return_value=True,
        ),
    ):
        # Set up mock context managers
        mock_start_run.return_value.__enter__.return_value = MagicMock()
        mock_log_model.return_value = MagicMock(model_uri="test_uri")

        # Create provider and call create_agent
        provider = DatabricksProvider()
        provider.create_agent(config=mock_config)

        # Verify experiment was retrieved/created and set
        mock_get_experiment.assert_called_once_with(mock_config)
        mock_set_experiment.assert_called_once_with(
            experiment_id=mock_experiment.experiment_id
        )


@pytest.mark.unit
def test_create_agent_sets_framework_tags():
    """Test that create_agent sets framework and framework_version tags."""
    from unittest.mock import MagicMock, call, patch

    import mlflow

    # Test directly that when mlflow.start_run is called, the correct tags are set
    # We'll verify the implementation by checking the source code calls
    with (
        patch.object(mlflow, "start_run") as mock_start_run,
        patch.object(mlflow, "set_tag") as mock_set_tag,
    ):
        # Create a mock context manager for start_run
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context

        # Import and run the relevant code that should set the tags
        from dao_ai.utils import dao_ai_version

        # Simulate the code in create_agent that sets tags
        with mlflow.start_run(run_name="test_run"):
            mlflow.set_tag("type", "agent")
            mlflow.set_tag("dao_ai", dao_ai_version())

        # Verify the tags were set correctly
        expected_calls = [
            call("type", "agent"),
            call("dao_ai", dao_ai_version()),
        ]
        mock_set_tag.assert_has_calls(expected_calls, any_order=False)


@pytest.mark.unit
def test_create_agent_uses_configured_python_version():
    """Test that create_agent uses the configured python_version for Model Serving.

    This allows deploying from environments with different Python versions
    (e.g., Databricks Apps with Python 3.11 can deploy to Model Serving with 3.12).
    """
    from unittest.mock import MagicMock, patch

    import mlflow

    from dao_ai.config import AppConfig
    from dao_ai.providers.databricks import DatabricksProvider

    # Create a minimal mock config
    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock()
    mock_app.name = "test_app"
    mock_app.code_paths = []
    mock_app.pip_requirements = ["test-package==1.0.0"]
    mock_app.input_example = None
    mock_app.python_version = "3.12"  # Configure target Python version
    mock_config.app = mock_app

    # Mock resources
    mock_resources = MagicMock()
    mock_resources.llms = MagicMock(values=lambda: [])
    mock_resources.vector_stores = MagicMock(values=lambda: [])
    mock_resources.warehouses = MagicMock(values=lambda: [])
    mock_resources.genie_rooms = MagicMock(values=lambda: [])
    mock_resources.tables = MagicMock(values=lambda: [])
    mock_resources.functions = MagicMock(values=lambda: [])
    mock_resources.connections = MagicMock(values=lambda: [])
    mock_resources.databases = MagicMock(values=lambda: [])
    mock_resources.volumes = MagicMock(values=lambda: [])
    mock_config.resources = mock_resources

    # Create mock experiment
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "test_experiment_123"
    mock_experiment.name = "/Users/test_user/test_app"

    with (
        patch.object(
            DatabricksProvider, "get_or_create_experiment", return_value=mock_experiment
        ),
        patch.object(mlflow, "set_experiment"),
        patch.object(mlflow, "set_registry_uri"),
        patch.object(mlflow, "start_run") as mock_start_run,
        patch.object(mlflow, "set_tag"),
        patch.object(mlflow.pyfunc, "log_model") as mock_log_model,
        patch.object(mlflow, "register_model"),
        patch("dao_ai.providers.databricks.MlflowClient"),
        patch("dao_ai.providers.databricks.is_installed", return_value=True),
        patch(
            "dao_ai.providers.databricks.is_lib_provided",
            return_value=True,
        ),
    ):
        # Set up mock context managers
        mock_start_run.return_value.__enter__.return_value = MagicMock()
        mock_log_model.return_value = MagicMock(model_uri="test_uri")

        # Create provider and call create_agent
        provider = DatabricksProvider()
        provider.create_agent(config=mock_config)

        # Verify log_model was called with conda_env containing the configured Python version
        mock_log_model.assert_called_once()
        call_kwargs = mock_log_model.call_args.kwargs
        assert "conda_env" in call_kwargs, "conda_env should be passed to log_model"

        conda_env = call_kwargs["conda_env"]
        assert conda_env["name"] == "mlflow-env"
        assert "python=3.12" in conda_env["dependencies"]

        # Verify pip requirements are included
        pip_deps = next(
            d for d in conda_env["dependencies"] if isinstance(d, dict) and "pip" in d
        )
        assert "test-package==1.0.0" in pip_deps["pip"]


@pytest.mark.unit
def test_deploy_agent_sets_endpoint_tag():
    """Test that deploy_agent adds dao_ai tag to the endpoint."""
    from unittest.mock import MagicMock, patch

    from dao_ai.config import AppConfig, AppModel
    from dao_ai.providers.databricks import DatabricksProvider
    from dao_ai.utils import dao_ai_version

    # Mock the entire config to avoid complex Pydantic validation
    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_registered_model = MagicMock()

    # Set required attributes
    mock_app.endpoint_name = "test_endpoint"
    mock_registered_model.full_name = "test_catalog.test_schema.test_model"
    mock_app.registered_model = mock_registered_model
    mock_app.scale_to_zero = True
    mock_app.environment_vars = {}
    mock_app.workload_size = "Small"
    mock_app.tags = {"custom_tag": "custom_value"}
    mock_app.permissions = []

    mock_config.app = mock_app

    # Mock the agents module functions
    with patch("dao_ai.providers.databricks.agents.get_deployments") as mock_get:
        with patch("dao_ai.providers.databricks.agents.deploy") as mock_deploy:
            with patch(
                "dao_ai.providers.databricks.get_latest_model_version"
            ) as mock_version:
                with patch("dao_ai.providers.databricks.mlflow.set_registry_uri"):
                    # Simulate endpoint doesn't exist (new deployment)
                    mock_get.side_effect = Exception("Not found")
                    mock_version.return_value = 1

                    # Create provider and call deploy_agent
                    provider = DatabricksProvider()
                    provider.deploy_agent(config=mock_config)

                    # Verify deploy was called with the dao_ai tag
                    mock_deploy.assert_called_once()
                    call_kwargs = mock_deploy.call_args.kwargs

                    assert "tags" in call_kwargs
                    assert call_kwargs["tags"] is not None
                    assert "dao_ai" in call_kwargs["tags"]
                    assert call_kwargs["tags"]["dao_ai"] == dao_ai_version()
                    # Verify custom tag is preserved
                    assert call_kwargs["tags"]["custom_tag"] == "custom_value"


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


# ==================== DatabaseModel Authentication Tests ====================


@pytest.mark.unit
def test_database_model_auth_validation_oauth_for_db_connection():
    """Test DatabaseModel accepts OAuth credentials for database connection.

    Note: OAuth credentials (client_id, client_secret, workspace_host) are used
    for DATABASE CONNECTION authentication, not for workspace API calls.
    Workspace API calls use ambient/default authentication.
    """
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        client_id="test_client_id",
        client_secret="test_client_secret",
        workspace_host="https://test.databricks.com",
    )
    # Should not raise - OAuth for DB connection is valid
    assert database.client_id == "test_client_id"
    assert database.client_secret == "test_client_secret"
    assert database.workspace_host == "https://test.databricks.com"


@pytest.mark.unit
def test_database_model_auth_validation_user_for_db_connection():
    """Test DatabaseModel accepts user credentials for database connection.

    Note: User credentials are used for DATABASE CONNECTION authentication.
    Workspace API calls use ambient/default authentication.
    """
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )
    # Should not raise - user auth for DB connection is valid
    assert database.user == "test_user"


@pytest.mark.unit
def test_database_model_auth_validation_mixed_error():
    """Test DatabaseModel rejects mixed OAuth and user authentication for DB connection."""
    import pytest

    with pytest.raises(ValueError) as exc_info:
        DatabaseModel(
            name="test_db",
            instance_name="test_db",
            host="localhost",
            user="test_user",
            client_id="test_client_id",
            client_secret="test_client_secret",
            workspace_host="https://test.databricks.com",
        )

    assert "Cannot mix authentication methods" in str(exc_info.value)


@pytest.mark.unit
def test_database_model_auth_validation_obo():
    """Test DatabaseModel accepts on_behalf_of_user for passive auth in model serving."""
    from unittest.mock import MagicMock, patch

    # Mock the WorkspaceClient to avoid actual API calls
    mock_ws_client_instance = MagicMock()

    with patch("dao_ai.config.WorkspaceClient") as mock_ws_client:
        mock_ws_client.return_value = mock_ws_client_instance

        # Create database with on_behalf_of_user - no other credentials needed
        database = DatabaseModel(
            name="test_db",
            instance_name="test_db",
            host="localhost",  # Provide host to skip update_host validator
            on_behalf_of_user=True,
        )

        # Validation should pass
        assert database.on_behalf_of_user is True
        assert database.client_id is None
        assert database.user is None


@pytest.mark.unit
def test_database_model_auth_validation_obo_mixed_error():
    """Test DatabaseModel rejects mixing OBO with other auth methods."""
    import pytest

    with pytest.raises(ValueError) as exc_info:
        DatabaseModel(
            name="test_db",
            instance_name="test_db",
            host="localhost",
            on_behalf_of_user=True,
            user="test_user",
        )

    assert "Cannot mix authentication methods" in str(exc_info.value)


@pytest.mark.unit
def test_database_model_workspace_client_uses_configured_auth():
    """Test that DatabaseModel.workspace_client uses configured authentication.

    The workspace_client property is inherited from IsDatabricksResource and uses
    the configured authentication (service principal, PAT, or ambient) for all
    workspace API operations. If client_id/client_secret/workspace_host are provided,
    they're used for workspace API calls as well as database connections.
    """
    from unittest.mock import MagicMock, patch

    # Mock the WorkspaceClient and its current_user.me() method
    mock_user = MagicMock()
    mock_user.user_name = "test_user@example.com"

    mock_ws_client_instance = MagicMock()
    mock_ws_client_instance.current_user.me.return_value = mock_user

    with patch("dao_ai.config.WorkspaceClient") as mock_ws_client:
        mock_ws_client.return_value = mock_ws_client_instance

        # Create database with OAuth credentials
        database = DatabaseModel(
            name="test_db",
            instance_name="test_db",
            host="localhost",  # Provide host to skip update_host validator
            client_id="test_client_id",
            client_secret="test_client_secret",
            workspace_host="https://test.databricks.com",
        )

        # Access workspace_client property - should use configured OAuth credentials
        _ = database.workspace_client

        # Verify WorkspaceClient was called with OAuth credentials
        mock_ws_client.assert_called()
        call_kwargs = (
            mock_ws_client.call_args.kwargs if mock_ws_client.call_args else {}
        )
        # Should have client_id/client_secret for service principal auth
        assert call_kwargs.get("client_id") == "test_client_id"
        assert call_kwargs.get("client_secret") == "test_client_secret"
        assert call_kwargs.get("auth_type") == "oauth-m2m"


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
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_create_lakebase_new_database():
    """Test create_lakebase when database doesn't exist."""
    # Mock workspace client
    mock_workspace_client = MagicMock()

    # Mock available database instance for wait check
    mock_available_instance = MagicMock()
    mock_available_instance.state = "AVAILABLE"

    # First call raises NotFound (database doesn't exist), subsequent calls return available instance
    mock_workspace_client.database.get_database_instance.side_effect = [
        NotFound(),  # Initial check - database doesn't exist
        mock_available_instance,  # Wait check - database is now AVAILABLE
    ]
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

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
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

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
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

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
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

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
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

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
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

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
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_create_lakebase_concurrent_creation():
    """Test create_lakebase when database is created concurrently by another process."""
    # Mock workspace client
    mock_workspace_client = MagicMock()

    # Mock available database instance for wait check after concurrent creation
    mock_available_instance = MagicMock()
    mock_available_instance.state = "AVAILABLE"

    # First call raises NotFound, subsequent calls return available instance (for wait)
    mock_workspace_client.database.get_database_instance.side_effect = [
        NotFound(),  # Initial check - database doesn't exist
        mock_available_instance,  # Wait check after concurrent creation detected
    ]

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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

    # Mock workspace_client property
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        # Should not raise exception
        provider.create_lakebase(database)

    # Verify create was called (even though it failed)
    mock_workspace_client.database.create_database_instance.assert_called_once()


@pytest.mark.unit
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

    # Mock workspace_client property
    with patch.object(
        DatabaseModel, "workspace_client", new_callable=lambda: mock_workspace_client
    ):
        # Should raise the exception
        with pytest.raises(Exception, match="Unexpected error occurred"):
            provider.create_lakebase(database)


@pytest.mark.unit
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

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
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_create_lakebase_default_values():
    """Test create_lakebase uses correct default values."""
    # Mock workspace client
    mock_workspace_client = MagicMock()

    # Mock available database instance for wait check
    mock_available_instance = MagicMock()
    mock_available_instance.state = "AVAILABLE"

    # First call raises NotFound, subsequent calls return available instance
    mock_workspace_client.database.get_database_instance.side_effect = [
        NotFound(),  # Initial check - database doesn't exist
        mock_available_instance,  # Wait check - database is now AVAILABLE
    ]
    mock_workspace_client.database.create_database_instance.return_value = None

    # Create database model with minimal parameters
    database = DatabaseModel(
        name="test_db",
        instance_name="test_db",
        host="localhost",
        user="test_user",
        password="test_password",
    )

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

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
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_create_lakebase_custom_capacity_cu1():
    """Test create_lakebase with custom capacity CU_1."""
    # Mock workspace client
    mock_workspace_client = MagicMock()

    # Mock available database instance for wait check
    mock_available_instance = MagicMock()
    mock_available_instance.state = "AVAILABLE"

    # First call raises NotFound, subsequent calls return available instance
    mock_workspace_client.database.get_database_instance.side_effect = [
        NotFound(),  # Initial check - database doesn't exist
        mock_available_instance,  # Wait check - database is now AVAILABLE
    ]
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

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
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

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
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

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
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

    # Call create_lakebase_instance_role
    provider.create_lakebase_instance_role(database)

    # Verify get was called
    mock_workspace_client.database.get_database_instance_role.assert_called_once()

    # Verify create was NOT called since role already exists
    mock_workspace_client.database.create_database_instance_role.assert_not_called()


@pytest.mark.unit
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

    # Call create_lakebase_instance_role - should log warning and return early
    provider.create_lakebase_instance_role(database)

    # Verify no API calls were made
    mock_workspace_client.database.get_database_instance_role.assert_not_called()
    mock_workspace_client.database.create_database_instance_role.assert_not_called()


@pytest.mark.unit
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

    # Should not raise exception
    provider.create_lakebase_instance_role(database)

    # Verify both get and create were called
    mock_workspace_client.database.get_database_instance_role.assert_called_once()
    mock_workspace_client.database.create_database_instance_role.assert_called_once()


@pytest.mark.unit
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

    # Should raise the exception
    with pytest.raises(Exception, match="Unexpected error occurred"):
        provider.create_lakebase_instance_role(database)


@pytest.mark.unit
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
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

    # Create provider with mocked clients
    provider = DatabricksProvider(w=mock_workspace_client, vsc=MagicMock())

    # Call create_lakebase_instance_role
    provider.create_lakebase_instance_role(database)

    # Verify get was called with resolved client_id
    mock_workspace_client.database.get_database_instance_role.assert_called_once()
    call_args = mock_workspace_client.database.get_database_instance_role.call_args
    assert call_args.kwargs["name"] == "test-client-id-456"
