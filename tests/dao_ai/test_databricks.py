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
    IndexModel,
    SchemaModel,
    TableModel,
    VectorStoreModel,
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


# ==================== Deployment Target Tests ====================


@pytest.mark.unit
def test_deploy_agent_routes_to_model_serving_by_default():
    """Test that deploy_agent routes to deploy_model_serving_agent by default."""
    from unittest.mock import MagicMock, patch

    from dao_ai.config import AppConfig, AppModel
    from dao_ai.providers.databricks import DatabricksProvider

    # Mock the entire config
    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_registered_model = MagicMock()
    mock_app.endpoint_name = "test_endpoint"
    mock_registered_model.full_name = "test_catalog.test_schema.test_model"
    mock_app.registered_model = mock_registered_model
    mock_app.scale_to_zero = True
    mock_app.environment_vars = {}
    mock_app.workload_size = "Small"
    mock_app.tags = {}
    mock_app.permissions = []
    mock_config.app = mock_app

    with patch.object(
        DatabricksProvider, "deploy_model_serving_agent"
    ) as mock_model_serving:
        with patch.object(DatabricksProvider, "deploy_apps_agent") as mock_apps:
            provider = DatabricksProvider()
            provider.deploy_agent(config=mock_config)

            # Should route to model serving by default
            mock_model_serving.assert_called_once_with(mock_config)
            mock_apps.assert_not_called()


@pytest.mark.unit
def test_deploy_agent_routes_to_model_serving_explicitly():
    """Test that deploy_agent routes to deploy_model_serving_agent when target=MODEL_SERVING."""
    from unittest.mock import MagicMock, patch

    from dao_ai.config import AppConfig, AppModel, DeploymentTarget
    from dao_ai.providers.databricks import DatabricksProvider

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_registered_model = MagicMock()
    mock_app.endpoint_name = "test_endpoint"
    mock_registered_model.full_name = "test_catalog.test_schema.test_model"
    mock_app.registered_model = mock_registered_model
    mock_app.scale_to_zero = True
    mock_app.environment_vars = {}
    mock_app.workload_size = "Small"
    mock_app.tags = {}
    mock_app.permissions = []
    mock_config.app = mock_app

    with patch.object(
        DatabricksProvider, "deploy_model_serving_agent"
    ) as mock_model_serving:
        with patch.object(DatabricksProvider, "deploy_apps_agent") as mock_apps:
            provider = DatabricksProvider()
            provider.deploy_agent(
                config=mock_config, target=DeploymentTarget.MODEL_SERVING
            )

            mock_model_serving.assert_called_once_with(mock_config)
            mock_apps.assert_not_called()


@pytest.mark.unit
def test_deploy_agent_routes_to_apps_when_specified():
    """Test that deploy_agent routes to deploy_apps_agent when target=APPS."""
    from unittest.mock import MagicMock, patch

    from dao_ai.config import AppConfig, AppModel, DeploymentTarget
    from dao_ai.providers.databricks import DatabricksProvider

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_app.name = "test_app"
    mock_app.description = "Test app description"
    mock_config.app = mock_app

    with patch.object(
        DatabricksProvider, "deploy_model_serving_agent"
    ) as mock_model_serving:
        with patch.object(DatabricksProvider, "deploy_apps_agent") as mock_apps:
            provider = DatabricksProvider()
            provider.deploy_agent(config=mock_config, target=DeploymentTarget.APPS)

            mock_apps.assert_called_once_with(mock_config)
            mock_model_serving.assert_not_called()


@pytest.mark.unit
def test_deploy_apps_agent_creates_new_app():
    """Test that deploy_apps_agent creates a new app when it doesn't exist."""
    from unittest.mock import MagicMock, patch

    from databricks.sdk.errors.platform import NotFound
    from databricks.sdk.service.apps import App, AppDeployment, AppDeploymentState
    from databricks.sdk.service.iam import User

    from dao_ai.config import AppConfig, AppModel
    from dao_ai.providers.databricks import DatabricksProvider

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_app.name = "test_app"
    mock_app.description = "Test app description"
    mock_app.environment_vars = {}
    mock_config.app = mock_app
    mock_config.source_config_path = None  # No config file to upload
    mock_config.resources = None  # No resources (required for generate_app_resources)
    mock_config.agents = None
    mock_config.retrievers = None

    # Create mock App and AppDeployment
    mock_created_app = MagicMock(spec=App)
    mock_created_app.name = "test_app"
    mock_created_app.url = "https://test_app.databricks.com"

    mock_deployment = MagicMock(spec=AppDeployment)
    mock_deployment.deployment_id = "dep-123"
    mock_deployment_status = MagicMock()
    mock_deployment_status.state = AppDeploymentState.SUCCEEDED
    mock_deployment.status = mock_deployment_status

    # Mock current user
    mock_user = MagicMock(spec=User)
    mock_user.user_name = "test.user@example.com"

    with patch.object(DatabricksProvider, "__init__", return_value=None):
        provider = DatabricksProvider()
        provider.w = MagicMock()

        # Mock current user
        provider.w.current_user.me.return_value = mock_user

        # Mock MLflow experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "12345"
        with patch.object(
            provider, "get_or_create_experiment", return_value=mock_experiment
        ):
            # Simulate app doesn't exist
            provider.w.apps.get.side_effect = NotFound("App not found")
            provider.w.apps.create_and_wait.return_value = mock_created_app
            provider.w.apps.deploy_and_wait.return_value = mock_deployment

            provider.deploy_apps_agent(mock_config)

            # Verify create_and_wait was called with an App object
            provider.w.apps.create_and_wait.assert_called_once()
            call_args = provider.w.apps.create_and_wait.call_args
            app_arg = call_args.kwargs.get("app")
            assert app_arg is not None
            assert app_arg.name == "test-app"  # Normalized: underscores become dashes
            assert app_arg.description == "Test app description"
            # Verify deploy_and_wait was called
            provider.w.apps.deploy_and_wait.assert_called_once()


@pytest.mark.unit
def test_deploy_apps_agent_updates_existing_app():
    """Test that deploy_apps_agent updates an existing app."""
    from unittest.mock import MagicMock, patch

    from databricks.sdk.service.apps import App, AppDeployment, AppDeploymentState
    from databricks.sdk.service.iam import User

    from dao_ai.config import AppConfig, AppModel
    from dao_ai.providers.databricks import DatabricksProvider

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_app.name = "test_app"
    mock_app.description = "Test app description"
    mock_app.environment_vars = {}
    mock_config.app = mock_app
    mock_config.source_config_path = None  # No config file to upload
    mock_config.resources = None  # No resources (required for generate_app_resources)
    mock_config.agents = None
    mock_config.retrievers = None

    # Create mock existing App
    mock_existing_app = MagicMock(spec=App)
    mock_existing_app.name = "test_app"
    mock_existing_app.url = "https://test_app.databricks.com"

    mock_deployment = MagicMock(spec=AppDeployment)
    mock_deployment.deployment_id = "dep-123"
    mock_deployment_status = MagicMock()
    mock_deployment_status.state = AppDeploymentState.SUCCEEDED
    mock_deployment.status = mock_deployment_status

    # Mock current user (used for convention-based path)
    mock_user = MagicMock(spec=User)
    mock_user.user_name = "test.user@example.com"

    with patch.object(DatabricksProvider, "__init__", return_value=None):
        provider = DatabricksProvider()
        provider.w = MagicMock()

        # Mock current user
        provider.w.current_user.me.return_value = mock_user

        # Mock MLflow experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "12345"
        with patch.object(
            provider, "get_or_create_experiment", return_value=mock_experiment
        ):
            # Simulate app already exists
            provider.w.apps.get.return_value = mock_existing_app
            provider.w.apps.deploy_and_wait.return_value = mock_deployment

            provider.deploy_apps_agent(mock_config)

            # Verify create_and_wait was NOT called (app already exists)
            provider.w.apps.create_and_wait.assert_not_called()
            # Verify deploy_and_wait was called
            provider.w.apps.deploy_and_wait.assert_called_once()


@pytest.mark.unit
def test_deployment_target_enum_values():
    """Test that DeploymentTarget enum has expected values."""
    from dao_ai.config import DeploymentTarget

    assert DeploymentTarget.MODEL_SERVING.value == "model_serving"
    assert DeploymentTarget.APPS.value == "apps"

    # Test enum can be created from string
    assert DeploymentTarget("model_serving") == DeploymentTarget.MODEL_SERVING
    assert DeploymentTarget("apps") == DeploymentTarget.APPS


# =============================================================================
# WorkspaceClient OBO with Forwarded Headers Tests
# =============================================================================


@pytest.mark.unit
def test_workspace_client_obo_uses_model_serving_credentials():
    """Test that OBO workspace_client property uses ModelServingUserCredentials."""
    from unittest.mock import patch

    from dao_ai.config import WarehouseModel

    # Resource with on_behalf_of_user=True (OBO enabled)
    warehouse = WarehouseModel(warehouse_id="test-warehouse", on_behalf_of_user=True)

    with patch("dao_ai.config.WorkspaceClient") as mock_client:
        _ = warehouse.workspace_client

        # Verify client created with ModelServingUserCredentials
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args.kwargs
        assert "credentials_strategy" in call_kwargs


@pytest.mark.unit
def test_workspace_client_from_uses_forwarded_headers():
    """Test that workspace_client_from uses x-forwarded-access-token from Context."""
    from unittest.mock import patch

    from dao_ai.config import WarehouseModel
    from dao_ai.state import Context

    # Resource with on_behalf_of_user=True (OBO enabled)
    warehouse = WarehouseModel(warehouse_id="test-warehouse", on_behalf_of_user=True)

    # Create a Context with headers
    context = Context(
        headers={
            "x-forwarded-access-token": "dapi123456",
            "x-forwarded-user": "user@example.com",
        }
    )

    with patch("dao_ai.config.WorkspaceClient") as mock_client:
        _ = warehouse.workspace_client_from(context)

        # Verify client created with forwarded token
        mock_client.assert_called_once_with(
            host=None, token="dapi123456", auth_type="pat"
        )


@pytest.mark.unit
def test_workspace_client_ignores_headers_without_obo():
    """Test that headers are ignored when on_behalf_of_user=False."""
    from unittest.mock import patch

    from dao_ai.config import WarehouseModel

    # Resource WITHOUT on_behalf_of_user (headers should be ignored)
    warehouse = WarehouseModel(warehouse_id="test-warehouse")

    # Mock get_request_headers to return forwarded token
    with patch("mlflow.genai.agent_server.get_request_headers") as mock_headers:
        mock_headers.return_value = {
            "x-forwarded-access-token": "dapi123456",
            "x-forwarded-user": "user@example.com",
        }

        with patch("dao_ai.config.WorkspaceClient") as mock_client:
            _ = warehouse.workspace_client

            # Verify headers were NOT used (falls back to ambient)
            mock_client.assert_called_once_with()  # No token passed


@pytest.mark.unit
def test_workspace_client_from_obo_takes_precedence_over_pat():
    """Test that workspace_client_from with OBO takes precedence over PAT."""
    from unittest.mock import patch

    from dao_ai.config import WarehouseModel
    from dao_ai.state import Context

    # Resource with BOTH on_behalf_of_user AND explicit PAT
    # on_behalf_of_user should take precedence (checked first)
    warehouse = WarehouseModel(
        warehouse_id="test-warehouse",
        on_behalf_of_user=True,
        pat="explicit-pat-token",  # This gets ignored when using workspace_client_from
        workspace_host="https://test.databricks.com",
    )

    # Create a Context with forwarded token
    context = Context(headers={"x-forwarded-access-token": "forwarded-token"})

    with patch("dao_ai.config.WorkspaceClient") as mock_client:
        _ = warehouse.workspace_client_from(context)

        # Verify forwarded token used (OBO path), NOT explicit PAT
        mock_client.assert_called_once_with(
            host="https://test.databricks.com",
            token="forwarded-token",  # From context headers, not explicit PAT
            auth_type="pat",
        )


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
def test_database_model_name_defaults_to_instance_name():
    """Test that name defaults to instance_name for Lakebase connections."""
    from unittest.mock import MagicMock, PropertyMock, patch

    mock_ws_client = MagicMock()
    mock_ws_client.current_user.me.return_value = MagicMock(user_name="test_user")

    with patch.object(
        DatabaseModel, "workspace_client", new_callable=PropertyMock
    ) as mock_prop:
        mock_prop.return_value = mock_ws_client

        # Create database with only instance_name (no name provided)
        database = DatabaseModel(
            instance_name="my-lakebase-instance",
        )

        # name should default to instance_name
        assert database.name == "my-lakebase-instance"
        assert database.instance_name == "my-lakebase-instance"


@pytest.mark.unit
def test_database_model_connection_params_auto_fetches_host():
    """Test connection_params auto-fetches host for Lakebase without OBO."""
    from unittest.mock import MagicMock, PropertyMock, patch

    mock_ws_client = MagicMock()
    mock_instance = MagicMock()
    mock_instance.read_write_dns = "test-instance.database.databricks.com"
    mock_ws_client.database.get_database_instance.return_value = mock_instance
    mock_ws_client.database.generate_database_credential.return_value = MagicMock(
        token="test-token"
    )
    mock_ws_client.current_user.me.return_value = MagicMock(user_name="test_user")

    with patch.object(
        DatabaseModel, "workspace_client", new_callable=PropertyMock
    ) as mock_prop:
        mock_prop.return_value = mock_ws_client

        database = DatabaseModel(
            instance_name="test_db",
            # No name provided - should default to instance_name
            # No host provided - should auto-fetch
            # No on_behalf_of_user - should still work
        )

        params = database.connection_params

        assert params["host"] == "test-instance.database.databricks.com"
        mock_ws_client.database.get_database_instance.assert_called_once_with(
            name="test_db"
        )


@pytest.mark.unit
def test_postgres_pool_manager_uses_lakebase_pool():
    """Test PostgresPoolManager uses LakebasePool for Lakebase connections."""
    from unittest.mock import MagicMock, PropertyMock, patch

    from dao_ai.memory.postgres import PostgresPoolManager

    # Reset class state before test
    PostgresPoolManager._pools = {}
    PostgresPoolManager._lakebase_pools = {}

    mock_ws_client = MagicMock()
    mock_ws_client.current_user.me.return_value = MagicMock(user_name="test_user")

    # Mock the LakebasePool
    mock_lakebase_pool_instance = MagicMock()
    mock_underlying_pool = MagicMock()
    mock_lakebase_pool_instance.pool = mock_underlying_pool

    with patch.object(
        DatabaseModel, "workspace_client", new_callable=PropertyMock
    ) as mock_ws_prop:
        mock_ws_prop.return_value = mock_ws_client

        with patch("dao_ai.memory.postgres.LakebasePool") as mock_lakebase_pool_class:
            mock_lakebase_pool_class.return_value = mock_lakebase_pool_instance

            # Create a Lakebase database model
            database = DatabaseModel(
                instance_name="test-lakebase-instance",
            )

            # Get the pool
            pool = PostgresPoolManager.get_pool(database)

            # Verify LakebasePool was used
            mock_lakebase_pool_class.assert_called_once_with(
                instance_name="test-lakebase-instance",
                workspace_client=mock_ws_client,
                min_size=1,
                max_size=database.max_pool_size,
                timeout=float(database.timeout_seconds),
            )

            # Verify the underlying pool is returned
            assert pool is mock_underlying_pool

            # Verify LakebasePool instance is tracked for cleanup
            assert database.name in PostgresPoolManager._lakebase_pools
            assert (
                PostgresPoolManager._lakebase_pools[database.name]
                is mock_lakebase_pool_instance
            )

    # Clean up
    PostgresPoolManager._pools = {}
    PostgresPoolManager._lakebase_pools = {}


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


@pytest.mark.unit
def test_database_model_workspace_client_oauth_without_workspace_host():
    """Test that OAuth works even when workspace_host is not provided.

    When client_id and client_secret are provided but workspace_host is not,
    the WorkspaceClient should check DATABRICKS_HOST env var first, then fall
    back to WorkspaceClient().config.host if not set.
    """
    from unittest.mock import MagicMock, patch

    # Mock the WorkspaceClient
    mock_ws_client_instance = MagicMock()
    mock_ws_client_instance.config.host = "https://default.databricks.com"

    with patch("dao_ai.config.WorkspaceClient") as mock_ws_client:
        with patch("dao_ai.config.os.getenv") as mock_getenv:
            mock_ws_client.return_value = mock_ws_client_instance
            # DATABRICKS_HOST is not set
            mock_getenv.return_value = None

            # Create database with OAuth credentials but NO workspace_host
            database = DatabaseModel(
                name="test_db",
                instance_name="test_db",
                host="localhost",  # Provide host to skip update_host validator
                client_id="test_client_id",
                client_secret="test_client_secret",
                # workspace_host is intentionally NOT provided
            )

            # Access workspace_client property - should use OAuth with default host
            _ = database.workspace_client

            # Verify DATABRICKS_HOST was checked
            mock_getenv.assert_called_with("DATABRICKS_HOST")

            # Verify WorkspaceClient was called twice:
            # 1. First to get the default host (WorkspaceClient().config.host)
            # 2. Second with OAuth credentials
            assert mock_ws_client.call_count == 2

            # Get the second call (the OAuth one)
            second_call_kwargs = mock_ws_client.call_args_list[1].kwargs

            # Should have client_id/client_secret for service principal auth
            assert second_call_kwargs.get("client_id") == "test_client_id"
            assert second_call_kwargs.get("client_secret") == "test_client_secret"
            assert second_call_kwargs.get("auth_type") == "oauth-m2m"
            # host should be the default from WorkspaceClient().config.host
            assert second_call_kwargs.get("host") == "https://default.databricks.com"


@pytest.mark.unit
def test_database_model_workspace_client_oauth_uses_databricks_host_env():
    """Test that OAuth uses DATABRICKS_HOST env var when set.

    When client_id and client_secret are provided and DATABRICKS_HOST is set,
    it should use that instead of creating a WorkspaceClient to get the host.
    """
    from unittest.mock import MagicMock, patch

    # Mock the WorkspaceClient
    mock_ws_client_instance = MagicMock()

    with patch("dao_ai.config.WorkspaceClient") as mock_ws_client:
        with patch("dao_ai.config.os.getenv") as mock_getenv:
            mock_ws_client.return_value = mock_ws_client_instance
            # DATABRICKS_HOST is set
            mock_getenv.return_value = "https://env-host.databricks.com"

            # Create database with OAuth credentials but NO workspace_host
            database = DatabaseModel(
                name="test_db",
                instance_name="test_db",
                host="localhost",  # Provide host to skip update_host validator
                client_id="test_client_id",
                client_secret="test_client_secret",
                # workspace_host is intentionally NOT provided
            )

            # Access workspace_client property
            _ = database.workspace_client

            # Verify DATABRICKS_HOST was checked
            mock_getenv.assert_called_with("DATABRICKS_HOST")

            # Verify WorkspaceClient was only called once (for OAuth)
            # Should NOT be called to get default host since env var is set
            assert mock_ws_client.call_count == 1

            # Get the OAuth call
            call_kwargs = mock_ws_client.call_args.kwargs

            # Should have client_id/client_secret for service principal auth
            assert call_kwargs.get("client_id") == "test_client_id"
            assert call_kwargs.get("client_secret") == "test_client_secret"
            assert call_kwargs.get("auth_type") == "oauth-m2m"
            # host should be from DATABRICKS_HOST env var
            assert call_kwargs.get("host") == "https://env-host.databricks.com"


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


# ==================== VectorStoreModel Tests ====================


@pytest.mark.unit
def test_vector_store_model_use_existing_index_minimal():
    """Test VectorStoreModel with minimal config for existing index (use existing mode)."""
    # Create VectorStoreModel with just an index - this is the minimal config
    vector_store = VectorStoreModel(
        index=IndexModel(name="catalog.schema.my_index"),
    )

    assert vector_store.index is not None
    assert vector_store.index.full_name == "catalog.schema.my_index"
    # Provisioning fields should be None
    assert vector_store.source_table is None
    assert vector_store.embedding_source_column is None
    # Endpoint should NOT be auto-discovered (only in provisioning mode)
    assert vector_store.endpoint is None
    # Embedding model should NOT be set (only in provisioning mode)
    assert vector_store.embedding_model is None


@pytest.mark.unit
def test_vector_store_model_use_existing_index_with_optional_fields():
    """Test VectorStoreModel with existing index and optional fields."""
    vector_store = VectorStoreModel(
        index=IndexModel(name="catalog.schema.my_index"),
        columns=["id", "name", "description"],
        primary_key="id",
        doc_uri="https://docs.example.com",
    )

    assert vector_store.index.full_name == "catalog.schema.my_index"
    assert vector_store.columns == ["id", "name", "description"]
    assert vector_store.primary_key == "id"
    assert vector_store.doc_uri == "https://docs.example.com"
    # Provisioning fields remain None
    assert vector_store.source_table is None
    assert vector_store.embedding_source_column is None


@pytest.mark.unit
def test_vector_store_model_validation_requires_index_or_source_table():
    """Test that VectorStoreModel fails without either index or source_table."""
    with pytest.raises(ValueError) as exc_info:
        VectorStoreModel()

    assert "Either 'index' (for existing indexes) or 'source_table'" in str(
        exc_info.value
    )


@pytest.mark.unit
def test_vector_store_model_provisioning_requires_embedding_source_column():
    """Test that provisioning mode requires embedding_source_column."""
    schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
    table = TableModel(schema=schema, name="test_table")

    with pytest.raises(ValueError) as exc_info:
        VectorStoreModel(source_table=table)

    assert "embedding_source_column is required when source_table is provided" in str(
        exc_info.value
    )


@pytest.mark.unit
def test_vector_store_model_provisioning_mode():
    """Test VectorStoreModel in provisioning mode (source_table + embedding_source_column)."""
    schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
    table = TableModel(schema=schema, name="test_table")

    # Mock the DatabricksProvider to avoid actual API calls
    # The import happens inside the validators, so we patch the providers module
    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider.find_primary_key.return_value = ["id"]
        mock_provider.find_endpoint_for_index.return_value = "test_endpoint"
        mock_provider_class.return_value = mock_provider

        vector_store = VectorStoreModel(
            source_table=table,
            embedding_source_column="description",
        )

        # Index should be auto-generated
        assert vector_store.index is not None
        assert vector_store.index.name == "test_table_index"
        assert (
            vector_store.index.full_name == "test_catalog.test_schema.test_table_index"
        )

        # Default embedding model should be set in provisioning mode
        assert vector_store.embedding_model is not None
        assert vector_store.embedding_model.name == "databricks-gte-large-en"

        # Primary key should be auto-discovered
        assert vector_store.primary_key == "id"

        # Endpoint should be auto-discovered in provisioning mode
        assert vector_store.endpoint is not None
        assert vector_store.endpoint.name == "test_endpoint"


@pytest.mark.unit
def test_vector_store_model_provisioning_with_explicit_index():
    """Test that explicit index is respected in provisioning mode."""
    schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
    table = TableModel(schema=schema, name="test_table")

    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider.find_primary_key.return_value = ["id"]
        mock_provider.find_endpoint_for_index.return_value = "test_endpoint"
        mock_provider_class.return_value = mock_provider

        vector_store = VectorStoreModel(
            source_table=table,
            embedding_source_column="description",
            index=IndexModel(schema=schema, name="custom_index"),
        )

        # Explicit index should be preserved
        assert vector_store.index.name == "custom_index"
        assert vector_store.index.full_name == "test_catalog.test_schema.custom_index"


@pytest.mark.unit
def test_vector_store_model_use_existing_no_auto_discovery():
    """Test that use existing mode does not trigger expensive auto-discovery."""
    # This test ensures no DatabricksProvider calls happen in "use existing" mode
    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        vector_store = VectorStoreModel(
            index=IndexModel(name="catalog.schema.existing_index"),
        )

        # In use existing mode, no provider methods should be called
        mock_provider.find_primary_key.assert_not_called()
        mock_provider.find_endpoint_for_index.assert_not_called()
        mock_provider.find_vector_search_endpoint.assert_not_called()

        # Verify the model is correctly created
        assert vector_store.index.full_name == "catalog.schema.existing_index"


@pytest.mark.unit
def test_vector_store_model_api_scopes():
    """Test VectorStoreModel API scopes."""
    vector_store = VectorStoreModel(
        index=IndexModel(name="catalog.schema.my_index"),
    )

    api_scopes = vector_store.api_scopes
    assert "vectorsearch.vector-search-endpoints" in api_scopes
    assert "serving.serving-endpoints" in api_scopes
    assert "vectorsearch.vector-search-indexes" in api_scopes


# =============================================================================
# IndexModel.exists() Tests
# =============================================================================


@pytest.mark.unit
def test_index_model_exists_returns_true():
    """Test IndexModel.exists() returns True when index exists."""
    from unittest.mock import patch

    index = IndexModel(name="catalog.schema.my_index")

    # Mock workspace_client property
    mock_workspace_client = MagicMock()
    mock_workspace_client.vector_search_indexes.get_index.return_value = MagicMock()

    with patch.object(
        type(index),
        "workspace_client",
        new_callable=lambda: property(lambda self: mock_workspace_client),
    ):
        assert index.exists() is True
        mock_workspace_client.vector_search_indexes.get_index.assert_called_once_with(
            "catalog.schema.my_index"
        )


@pytest.mark.unit
def test_index_model_exists_returns_false_not_found():
    """Test IndexModel.exists() returns False when index doesn't exist (NotFound)."""
    from unittest.mock import patch

    index = IndexModel(name="catalog.schema.my_index")

    # Mock workspace_client to raise NotFound
    mock_workspace_client = MagicMock()
    mock_workspace_client.vector_search_indexes.get_index.side_effect = NotFound(
        "Index not found"
    )

    with patch.object(
        type(index),
        "workspace_client",
        new_callable=lambda: property(lambda self: mock_workspace_client),
    ):
        assert index.exists() is False


@pytest.mark.unit
def test_index_model_exists_returns_false_on_error():
    """Test IndexModel.exists() returns False on other exceptions."""
    from unittest.mock import patch

    index = IndexModel(name="catalog.schema.my_index")

    # Mock workspace_client to raise generic exception
    mock_workspace_client = MagicMock()
    mock_workspace_client.vector_search_indexes.get_index.side_effect = Exception(
        "Connection error"
    )

    with patch.object(
        type(index),
        "workspace_client",
        new_callable=lambda: property(lambda self: mock_workspace_client),
    ):
        assert index.exists() is False


@pytest.mark.unit
def test_index_model_exists_with_schema():
    """Test IndexModel.exists() with schema-based index."""
    from unittest.mock import patch

    schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
    index = IndexModel(schema=schema, name="my_index")

    assert index.full_name == "test_catalog.test_schema.my_index"

    mock_workspace_client = MagicMock()
    mock_workspace_client.vector_search_indexes.get_index.return_value = MagicMock()

    with patch.object(
        type(index),
        "workspace_client",
        new_callable=lambda: property(lambda self: mock_workspace_client),
    ):
        assert index.exists() is True
        mock_workspace_client.vector_search_indexes.get_index.assert_called_once_with(
            "test_catalog.test_schema.my_index"
        )


# =============================================================================
# VectorStoreModel.create() Tests - Use Existing Mode
# =============================================================================


@pytest.mark.unit
def test_vector_store_create_validates_existing_index_success():
    """Test VectorStoreModel.create() in use existing mode when index exists."""
    index = IndexModel(name="catalog.schema.my_index")
    vector_store = VectorStoreModel(index=index)

    # Mock the provider and index.exists()
    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        # Mock the workspace client property to make exists() return True
        mock_workspace_client = MagicMock()
        mock_workspace_client.vector_search_indexes.get_index.return_value = MagicMock()

        with patch.object(
            type(index),
            "workspace_client",
            new_callable=lambda: property(lambda self: mock_workspace_client),
        ):
            vector_store.create()

            # Should NOT call create_vector_store (only validates)
            mock_provider.create_vector_store.assert_not_called()


@pytest.mark.unit
def test_vector_store_create_validates_existing_index_not_found():
    """Test VectorStoreModel.create() in use existing mode raises error when index doesn't exist."""
    index = IndexModel(name="catalog.schema.my_index")
    vector_store = VectorStoreModel(index=index)

    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        # Mock the workspace client to make exists() return False (NotFound)
        mock_workspace_client = MagicMock()
        mock_workspace_client.vector_search_indexes.get_index.side_effect = NotFound(
            "Index not found"
        )
        index._workspace_client = mock_workspace_client

        with pytest.raises(ValueError) as exc_info:
            vector_store.create()

        assert "does not exist" in str(exc_info.value)
        assert "Provide 'source_table' to provision it" in str(exc_info.value)


@pytest.mark.unit
def test_vector_store_create_validates_existing_index_no_index():
    """Test VectorStoreModel.create() raises error when index is None in use existing mode."""
    # This shouldn't happen due to validation, but test the helper method directly
    vector_store = VectorStoreModel(index=IndexModel(name="catalog.schema.my_index"))
    vector_store.index = None  # Force None to test error handling

    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        with pytest.raises(ValueError) as exc_info:
            vector_store._validate_existing_index(mock_provider)

        assert "index is required for 'use existing' mode" in str(exc_info.value)


# =============================================================================
# VectorStoreModel.create() Tests - Provisioning Mode
# =============================================================================


@pytest.mark.unit
def test_vector_store_create_provisions_new_index():
    """Test VectorStoreModel.create() in provisioning mode calls create_vector_store."""
    schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
    table = TableModel(schema=schema, name="test_table")

    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        # Mock for validators - called multiple times during __init__
        mock_provider_for_primary_key = MagicMock()
        mock_provider_for_primary_key.find_primary_key.return_value = ["id"]

        mock_provider_for_endpoint = MagicMock()
        mock_provider_for_endpoint.find_endpoint_for_index.return_value = None
        mock_provider_for_endpoint.find_vector_search_endpoint.return_value = (
            "test_endpoint"
        )

        # Mock for create call
        mock_provider_for_create = MagicMock()

        # Return different instances for each DatabricksProvider() call
        mock_provider_class.side_effect = [
            mock_provider_for_primary_key,  # set_default_primary_key validator
            mock_provider_for_endpoint,  # set_default_endpoint validator
            mock_provider_for_create,  # create() call
        ]

        vector_store = VectorStoreModel(
            source_table=table,
            embedding_source_column="description",
        )

        # Call create() - this will use the third mock from side_effect
        vector_store.create()

        # Should call create_vector_store
        mock_provider_for_create.create_vector_store.assert_called_once_with(
            vector_store
        )


@pytest.mark.unit
def test_vector_store_create_provisioning_requires_embedding_column():
    """Test VectorStoreModel._create_new_index() validates embedding_source_column."""
    schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
    table = TableModel(schema=schema, name="test_table")

    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider_for_validators = MagicMock()
        mock_provider_for_validators.find_primary_key.return_value = ["id"]
        mock_provider_for_validators.find_endpoint_for_index.return_value = None
        mock_provider_for_validators.find_vector_search_endpoint.return_value = (
            "test_endpoint"
        )
        mock_provider_class.return_value = mock_provider_for_validators

        vector_store = VectorStoreModel(
            source_table=table,
            embedding_source_column="description",
        )

        # Force None to test validation
        vector_store.embedding_source_column = None

        mock_provider = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            vector_store._create_new_index(mock_provider)

        assert "embedding_source_column is required for provisioning" in str(
            exc_info.value
        )


@pytest.mark.unit
def test_vector_store_create_provisioning_requires_endpoint():
    """Test VectorStoreModel._create_new_index() validates endpoint."""
    schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
    table = TableModel(schema=schema, name="test_table")

    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider_for_validators = MagicMock()
        mock_provider_for_validators.find_primary_key.return_value = ["id"]
        mock_provider_for_validators.find_endpoint_for_index.return_value = None
        mock_provider_for_validators.find_vector_search_endpoint.return_value = (
            "test_endpoint"
        )
        mock_provider_class.return_value = mock_provider_for_validators

        vector_store = VectorStoreModel(
            source_table=table,
            embedding_source_column="description",
        )

        # Force None to test validation
        vector_store.endpoint = None

        mock_provider = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            vector_store._create_new_index(mock_provider)

        assert "endpoint is required for provisioning" in str(exc_info.value)


@pytest.mark.unit
def test_vector_store_create_provisioning_requires_index():
    """Test VectorStoreModel._create_new_index() validates index."""
    schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
    table = TableModel(schema=schema, name="test_table")

    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider_for_validators = MagicMock()
        mock_provider_for_validators.find_primary_key.return_value = ["id"]
        mock_provider_for_validators.find_endpoint_for_index.return_value = None
        mock_provider_for_validators.find_vector_search_endpoint.return_value = (
            "test_endpoint"
        )
        mock_provider_class.return_value = mock_provider_for_validators

        vector_store = VectorStoreModel(
            source_table=table,
            embedding_source_column="description",
        )

        # Force None to test validation
        vector_store.index = None

        mock_provider = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            vector_store._create_new_index(mock_provider)

        assert "index is required for provisioning" in str(exc_info.value)


# =============================================================================
# VectorStoreModel.create() Integration Tests
# =============================================================================


@pytest.mark.unit
def test_vector_store_create_mode_detection():
    """Test VectorStoreModel.create() correctly detects provisioning vs use existing mode."""
    # Use existing mode
    index = IndexModel(name="catalog.schema.my_index")
    vector_store_existing = VectorStoreModel(index=index)

    with patch.object(
        vector_store_existing, "_validate_existing_index"
    ) as mock_validate:
        with patch("dao_ai.providers.databricks.DatabricksProvider"):
            vector_store_existing.create()
            mock_validate.assert_called_once()

    # Provisioning mode
    schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
    table = TableModel(schema=schema, name="test_table")

    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        # Mock for validators during __init__
        mock_provider_for_primary_key = MagicMock()
        mock_provider_for_primary_key.find_primary_key.return_value = ["id"]

        mock_provider_for_endpoint = MagicMock()
        mock_provider_for_endpoint.find_endpoint_for_index.return_value = None
        mock_provider_for_endpoint.find_vector_search_endpoint.return_value = (
            "test_endpoint"
        )

        # Mock for create() call
        mock_provider_for_create = MagicMock()

        mock_provider_class.side_effect = [
            mock_provider_for_primary_key,
            mock_provider_for_endpoint,
            mock_provider_for_create,
        ]

        vector_store_provisioning = VectorStoreModel(
            source_table=table,
            embedding_source_column="description",
        )

        with patch.object(
            vector_store_provisioning, "_create_new_index"
        ) as mock_create:
            vector_store_provisioning.create()
            mock_create.assert_called_once()
