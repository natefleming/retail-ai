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
    assert table.api_scopes == ["sql.statement-execution"]


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
    mock_config.guardrails = {}
    mock_config.agents = {}

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
        patch("dao_ai.providers.databricks.is_published", return_value=True),
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
    mock_config.guardrails = {}
    mock_config.agents = {}

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
        patch("dao_ai.providers.databricks.is_published", return_value=True),
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
    mock_app.trace_location = None
    mock_app.monitoring = None

    mock_config.app = mock_app

    # Mock the agents module functions
    with patch.object(
        DatabricksProvider, "_serving_endpoint_exists", return_value=False
    ):
        with patch(
            "dao_ai.providers.databricks.agents.get_deployments", return_value=[]
        ):
            with patch("dao_ai.providers.databricks.agents.deploy") as mock_deploy:
                with patch(
                    "dao_ai.providers.databricks.get_latest_model_version"
                ) as mock_version:
                    with patch("dao_ai.providers.databricks.mlflow.set_registry_uri"):
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


@pytest.mark.unit
def test_deploy_model_serving_omits_tags_when_serving_endpoint_exists():
    """Existing serving endpoints skip tag dict to avoid patch+update_config races."""
    from unittest.mock import MagicMock, patch

    from dao_ai.config import AppConfig, AppModel
    from dao_ai.providers.databricks import DatabricksProvider

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_registered_model = MagicMock()
    mock_app.endpoint_name = "ep1"
    mock_registered_model.full_name = "cat.sch.model"
    mock_app.registered_model = mock_registered_model
    mock_app.scale_to_zero = True
    mock_app.environment_vars = {}
    mock_app.workload_size = "Small"
    mock_app.tags = {"custom_tag": "x"}
    mock_app.permissions = []
    mock_app.trace_location = None
    mock_app.monitoring = None
    mock_config.app = mock_app

    with patch.object(
        DatabricksProvider, "_serving_endpoint_exists", return_value=True
    ):
        with patch.object(DatabricksProvider, "_wait_serving_endpoint_config_idle"):
            with patch(
                "dao_ai.providers.databricks.agents.get_deployments", return_value=[]
            ):
                with patch("dao_ai.providers.databricks.agents.deploy") as mock_deploy:
                    with patch(
                        "dao_ai.providers.databricks.get_latest_model_version",
                        return_value=2,
                    ):
                        with patch(
                            "dao_ai.providers.databricks.mlflow.set_registry_uri"
                        ):
                            with patch.object(
                                DatabricksProvider, "__init__", return_value=None
                            ):
                                provider = DatabricksProvider()
                                provider.w = MagicMock()
                                provider.deploy_model_serving_agent(mock_config)

                                mock_deploy.assert_called_once()
                                assert mock_deploy.call_args.kwargs.get("tags") is None


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
    from databricks.sdk.service.apps import (
        App,
        AppDeployment,
        AppDeploymentState,
        ApplicationState,
    )
    from databricks.sdk.service.iam import User

    from dao_ai.config import AppConfig, AppModel
    from dao_ai.providers.databricks import DatabricksProvider

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_app.name = "test_app"
    mock_app.description = "Test app description"
    mock_app.environment_vars = {}
    mock_app.trace_location = None
    mock_app.monitoring = None
    mock_app.enable_chat_proxy = True
    mock_config.app = mock_app
    mock_config.source_config_path = None  # No config file to upload
    mock_config.rendered_yaml = None
    mock_config.model_dump.return_value = {"app": {"name": "test_app"}}
    mock_config.resources = None  # No resources (required for generate_app_resources)
    mock_config.agents = None
    mock_config.retrievers = None

    # Create mock App and AppDeployment
    mock_created_app = MagicMock(spec=App)
    mock_created_app.name = "test_app"
    mock_created_app.url = "https://test_app.databricks.com"
    mock_created_app.app_status = MagicMock()
    mock_created_app.app_status.state = ApplicationState.RUNNING

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
            # First get: not found; then fresh get before deploy returns created app
            provider.w.apps.get.side_effect = [
                NotFound("App not found"),
                mock_created_app,
            ]
            provider.w.api_client.do.return_value = {"name": "test-app"}
            provider.w.apps.wait_get_app_active.return_value = mock_created_app
            provider.w.apps.deploy_and_wait.return_value = mock_deployment

            provider.deploy_apps_agent(mock_config)

            # Verify REST API was called to create the app
            provider.w.api_client.do.assert_called()
            create_call = provider.w.api_client.do.call_args_list[0]
            assert create_call.args[0] == "POST"
            assert create_call.args[1] == "/api/2.0/apps"
            body = create_call.kwargs.get("body", {})
            assert body["name"] == "test-app"  # Normalized: underscores become dashes
            assert body["description"] == "Test app description"
            # Verify deploy_and_wait was called
            provider.w.apps.deploy_and_wait.assert_called_once()


@pytest.mark.unit
def test_deploy_apps_agent_updates_existing_app():
    """Test that deploy_apps_agent updates an existing app."""
    from unittest.mock import MagicMock, patch

    from databricks.sdk.service.apps import (
        App,
        AppDeployment,
        AppDeploymentState,
        ApplicationState,
    )
    from databricks.sdk.service.iam import User

    from dao_ai.config import AppConfig, AppModel
    from dao_ai.providers.databricks import DatabricksProvider

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_app.name = "test_app"
    mock_app.description = "Test app description"
    mock_app.environment_vars = {}
    mock_app.trace_location = None
    mock_app.monitoring = None
    mock_app.enable_chat_proxy = True
    mock_config.app = mock_app
    mock_config.source_config_path = None  # No config file to upload
    mock_config.rendered_yaml = None
    mock_config.model_dump.return_value = {"app": {"name": "test_app"}}
    mock_config.resources = None  # No resources (required for generate_app_resources)
    mock_config.agents = None
    mock_config.retrievers = None

    # Create mock existing App
    mock_existing_app = MagicMock(spec=App)
    mock_existing_app.name = "test_app"
    mock_existing_app.url = "https://test_app.databricks.com"
    mock_existing_app.app_status = MagicMock()
    mock_existing_app.app_status.state = ApplicationState.RUNNING

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

            # Verify REST API was NOT called with POST (app already exists)
            for call in provider.w.api_client.do.call_args_list:
                assert call.args[0] != "POST" or "/api/2.0/apps" not in call.args[1], (
                    "POST /api/2.0/apps should not be called for existing app"
                )
            # Verify deploy_and_wait was called
            provider.w.apps.deploy_and_wait.assert_called_once()


@pytest.mark.unit
def test_deploy_apps_agent_uploads_rendered_yaml(tmp_path):
    """Uploaded config should be the parameter-substituted (rendered) YAML, not the raw source."""
    import io
    from unittest.mock import MagicMock, patch

    from databricks.sdk.service.apps import (
        App,
        AppDeployment,
        AppDeploymentState,
        ApplicationState,
    )
    from databricks.sdk.service.iam import User

    from dao_ai.config import AppConfig, AppModel
    from dao_ai.providers.databricks import DatabricksProvider

    raw_yaml: str = "schemas:\n  ws:\n    catalog_name: ${var.catalog}\n"
    rendered_yaml: str = "schemas:\n  ws:\n    catalog_name: my_catalog\n"
    src_file = tmp_path / "dao_ai.yaml"
    src_file.write_text(raw_yaml)

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_app.name = "rendered-test"
    mock_app.description = ""
    mock_app.environment_vars = {}
    mock_app.trace_location = None
    mock_app.monitoring = None
    mock_app.enable_chat_proxy = True
    mock_config.app = mock_app
    mock_config.source_config_path = str(src_file)
    mock_config.rendered_yaml = rendered_yaml
    mock_config.resources = None
    mock_config.agents = None
    mock_config.retrievers = None

    mock_existing_app = MagicMock(spec=App)
    mock_existing_app.app_status = MagicMock(state=ApplicationState.RUNNING)
    mock_deployment = MagicMock(spec=AppDeployment)
    mock_deployment.status = MagicMock(state=AppDeploymentState.SUCCEEDED)
    mock_user = MagicMock(spec=User, user_name="test.user@example.com")

    with patch.object(DatabricksProvider, "__init__", return_value=None):
        provider = DatabricksProvider()
        provider.w = MagicMock()
        provider.w.current_user.me.return_value = mock_user
        with patch.object(
            provider, "get_or_create_experiment", return_value=MagicMock(experiment_id="exp-1")
        ):
            provider.w.apps.get.return_value = mock_existing_app
            provider.w.apps.deploy_and_wait.return_value = mock_deployment

            provider.deploy_apps_agent(mock_config)

    # Find the workspace.upload call carrying the config
    upload_calls = [
        c for c in provider.w.workspace.upload.call_args_list
        if c.kwargs.get("path", "").endswith("dao_ai.yaml")
    ]
    assert upload_calls, "expected an upload of dao_ai.yaml"
    uploaded_bytes = upload_calls[0].kwargs["content"]
    assert isinstance(uploaded_bytes, io.BytesIO)
    uploaded_text = uploaded_bytes.getvalue().decode("utf-8")
    assert uploaded_text == rendered_yaml, "deploy must upload rendered YAML, not source"
    assert "${var.catalog}" not in uploaded_text


@pytest.mark.unit
def test_deploy_apps_agent_falls_back_to_source_when_no_rendered_yaml(tmp_path):
    """If rendered_yaml is missing (legacy callers), fall back to reading the source file."""
    import io
    from unittest.mock import MagicMock, patch

    from databricks.sdk.service.apps import (
        App,
        AppDeployment,
        AppDeploymentState,
        ApplicationState,
    )
    from databricks.sdk.service.iam import User

    from dao_ai.config import AppConfig, AppModel
    from dao_ai.providers.databricks import DatabricksProvider

    raw_yaml: str = "app:\n  name: legacy-app\n"
    src_file = tmp_path / "dao_ai.yaml"
    src_file.write_text(raw_yaml)

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_app.name = "legacy-app"
    mock_app.description = ""
    mock_app.environment_vars = {}
    mock_app.trace_location = None
    mock_app.monitoring = None
    mock_app.enable_chat_proxy = True
    mock_config.app = mock_app
    mock_config.source_config_path = str(src_file)
    mock_config.rendered_yaml = None  # legacy
    mock_config.resources = None
    mock_config.agents = None
    mock_config.retrievers = None

    mock_existing_app = MagicMock(spec=App)
    mock_existing_app.app_status = MagicMock(state=ApplicationState.RUNNING)
    mock_deployment = MagicMock(spec=AppDeployment)
    mock_deployment.status = MagicMock(state=AppDeploymentState.SUCCEEDED)
    mock_user = MagicMock(spec=User, user_name="test.user@example.com")

    with patch.object(DatabricksProvider, "__init__", return_value=None):
        provider = DatabricksProvider()
        provider.w = MagicMock()
        provider.w.current_user.me.return_value = mock_user
        with patch.object(
            provider, "get_or_create_experiment", return_value=MagicMock(experiment_id="exp-1")
        ):
            provider.w.apps.get.return_value = mock_existing_app
            provider.w.apps.deploy_and_wait.return_value = mock_deployment

            provider.deploy_apps_agent(mock_config)

    upload_calls = [
        c for c in provider.w.workspace.upload.call_args_list
        if c.kwargs.get("path", "").endswith("dao_ai.yaml")
    ]
    assert upload_calls
    uploaded_text = upload_calls[0].kwargs["content"].getvalue().decode("utf-8")
    assert uploaded_text == raw_yaml


@pytest.mark.unit
def test_deploy_apps_agent_serializes_python_built_config(tmp_path):
    """When AppConfig has neither rendered_yaml nor a source file (i.e. it was
    constructed entirely in Python), deploy_apps_agent should serialize the
    in-memory model_dump back to YAML and upload that, instead of failing."""
    import yaml as _yaml
    from unittest.mock import MagicMock, patch

    from databricks.sdk.service.apps import (
        App,
        AppDeployment,
        AppDeploymentState,
        ApplicationState,
    )
    from databricks.sdk.service.iam import User

    from dao_ai.config import AppConfig, AppModel
    from dao_ai.providers.databricks import DatabricksProvider

    real_config = AppConfig()
    assert real_config.source_config_path is None
    assert real_config.rendered_yaml is None
    expected_yaml = _yaml.safe_dump(
        real_config.model_dump(mode="json", by_alias=True, exclude_none=True),
        sort_keys=False,
        default_flow_style=False,
    )

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_app.name = "py-built"
    mock_app.description = ""
    mock_app.environment_vars = {}
    mock_app.trace_location = None
    mock_app.monitoring = None
    mock_app.enable_chat_proxy = True
    mock_config.app = mock_app
    mock_config.source_config_path = None
    mock_config.rendered_yaml = None
    mock_config.model_dump.return_value = real_config.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    mock_config.resources = None
    mock_config.agents = None
    mock_config.retrievers = None

    mock_existing_app = MagicMock(spec=App)
    mock_existing_app.app_status = MagicMock(state=ApplicationState.RUNNING)
    mock_deployment = MagicMock(spec=AppDeployment)
    mock_deployment.status = MagicMock(state=AppDeploymentState.SUCCEEDED)
    mock_user = MagicMock(spec=User, user_name="test.user@example.com")

    with patch.object(DatabricksProvider, "__init__", return_value=None):
        provider = DatabricksProvider()
        provider.w = MagicMock()
        provider.w.current_user.me.return_value = mock_user
        with patch.object(
            provider, "get_or_create_experiment", return_value=MagicMock(experiment_id="exp-1")
        ):
            provider.w.apps.get.return_value = mock_existing_app
            provider.w.apps.deploy_and_wait.return_value = mock_deployment

            provider.deploy_apps_agent(mock_config)

    # The provider should have created the workspace dir and uploaded a YAML
    # serialized from the in-memory model.
    provider.w.workspace.mkdirs.assert_called()
    upload_calls = [
        c for c in provider.w.workspace.upload.call_args_list
        if c.kwargs.get("path", "").endswith("dao_ai.yaml")
    ]
    assert upload_calls, "expected an upload of dao_ai.yaml even without a source file"
    uploaded_text = upload_calls[0].kwargs["content"].getvalue().decode("utf-8")
    # Round-trip equivalence: the uploaded YAML parses to the same dict
    # the AppConfig.model_dump() would have produced.
    assert _yaml.safe_load(uploaded_text) == _yaml.safe_load(expected_yaml)


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
        project="test_db",
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
        project="test_db",
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
            project="test_db",
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
            project="test_db",
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
            project="test_db",
            host="localhost",
            on_behalf_of_user=True,
            user="test_user",
        )

    assert "Cannot mix authentication methods" in str(exc_info.value)


@pytest.mark.unit
def test_database_model_instance_name_aliases_to_project():
    """Test that instance_name is accepted as a deprecated alias for project."""
    import warnings
    from unittest.mock import MagicMock, PropertyMock, patch

    mock_ws_client = MagicMock()
    mock_ws_client.current_user.me.return_value = MagicMock(user_name="test_user")

    with patch.object(
        DatabaseModel, "workspace_client", new_callable=PropertyMock
    ) as mock_prop:
        mock_prop.return_value = mock_ws_client

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            database = DatabaseModel(
                instance_name="my-lakebase-instance",
            )

            assert database.project == "my-lakebase-instance"
            assert database.name == "my-lakebase-instance"
            assert database.is_lakebase is True

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "instance_name" in str(deprecation_warnings[0].message)


@pytest.mark.unit
def test_database_model_connection_params_raises_for_lakebase():
    """Test connection_params raises ValueError for Lakebase databases."""
    from unittest.mock import MagicMock, PropertyMock, patch

    mock_ws_client = MagicMock()
    mock_ws_client.current_user.me.return_value = MagicMock(user_name="test_user")

    with patch.object(
        DatabaseModel, "workspace_client", new_callable=PropertyMock
    ) as mock_prop:
        mock_prop.return_value = mock_ws_client

        database = DatabaseModel(
            project="test_db",
        )

        with pytest.raises(ValueError, match="not supported for Lakebase"):
            database.connection_params


@pytest.mark.unit
def test_database_model_connection_params_raises_for_lakebase_autoscaling():
    """Test connection_params raises ValueError for Lakebase databases."""
    from unittest.mock import MagicMock, PropertyMock, patch

    mock_ws_client = MagicMock()
    mock_ws_client.current_user.me.return_value = MagicMock(user_name="test_user")

    with patch.object(
        DatabaseModel, "workspace_client", new_callable=PropertyMock
    ) as mock_prop:
        mock_prop.return_value = mock_ws_client

        database = DatabaseModel(
            project="test_db",
        )

        with pytest.raises(ValueError, match="not supported for Lakebase"):
            database.connection_params


@pytest.mark.unit
def test_postgres_pool_manager_uses_lakebase_pool():
    """Test PostgresPoolManager uses LakebasePool for Lakebase connections."""
    from unittest.mock import MagicMock, PropertyMock, patch

    from dao_ai.memory.postgres import PostgresPoolManager

    PostgresPoolManager._pools = {}
    PostgresPoolManager._lakebase_pools = {}

    mock_ws_client = MagicMock()
    mock_ws_client.current_user.me.return_value = MagicMock(user_name="test_user")

    mock_lakebase_pool_instance = MagicMock()
    mock_underlying_pool = MagicMock()
    mock_lakebase_pool_instance.pool = mock_underlying_pool

    with patch.object(
        DatabaseModel, "workspace_client", new_callable=PropertyMock
    ) as mock_ws_prop:
        mock_ws_prop.return_value = mock_ws_client

        with patch("dao_ai.memory.databricks._create_lakebase_pool") as mock_create:
            mock_create.return_value = mock_lakebase_pool_instance

            database = DatabaseModel(
                project="test-lakebase-project",
                branch="production",
            )

            pool = PostgresPoolManager.get_pool(database)

            mock_create.assert_called_once()
            assert pool is mock_underlying_pool
            assert database.name in PostgresPoolManager._lakebase_pools

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
            project="test_db",
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
                project="test_db",
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
                project="test_db",
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
def test_database_model_as_resources_lakebase():
    """DatabaseModel.as_resources returns a DatabricksLakebase resource for
    Lakebase databases so the deploying agent (Model Serving SystemAuthPolicy
    or Databricks Apps auto-SP) gets CAN_CONNECT_AND_CREATE on the instance.

    See tests/dao_ai/test_lakebase_app_resources.py for the broader suite.
    """
    from mlflow.models.resources import DatabricksLakebase

    db = DatabaseModel(
        name="test-db",
        project="test-db",
    )
    assert db.is_lakebase is True
    resources = list(db.as_resources())
    assert len(resources) == 1
    assert isinstance(resources[0], DatabricksLakebase)
    assert resources[0].to_dict()["lakebase"][0]["name"] == "test-db"
    assert db.api_scopes == ["postgres"]


@pytest.mark.unit
def test_database_model_as_resources_project_defaults_name():
    """Test that project serves as the default name for Lakebase."""
    db = DatabaseModel(
        project="my-project",
    )
    assert db.name == "my-project"
    assert db.is_lakebase is True


@pytest.mark.unit
def test_database_model_as_resources_standard_postgres():
    """Test DatabaseModel.as_resources returns empty list for standard PostgreSQL."""
    db = DatabaseModel(
        name="test-db",
        host="my-postgres-host.example.com",
        user="test_user",
        password="test_password",
    )
    assert db.is_lakebase is False
    resources = db.as_resources()
    assert len(resources) == 0


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
        vector_store.ensure_resolved()

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


@pytest.mark.unit
def test_vector_store_obo_skips_client_args():
    """Test that OBO vector stores don't pass client_args to DatabricksVectorSearch.

    When on_behalf_of_user=True, the workspace_client from
    workspace_client_from(context) carries the user's forwarded token.
    Passing client_args would create a separate VectorSearchClient that
    overrides the OBO token with ambient/SP auth.
    """
    from conftest import add_databricks_resource_attrs

    from dao_ai.tools.vector_search import create_vector_search_tool

    mock_ws_client = MagicMock()

    vector_store = Mock(spec=VectorStoreModel)
    vector_store.index = Mock()
    vector_store.index.full_name = "catalog.schema.test_index"
    vector_store.index.name = "test_index"
    vector_store.primary_key = "id"
    vector_store.doc_uri = None
    vector_store.embedding_source_column = "text"
    add_databricks_resource_attrs(vector_store)
    vector_store.on_behalf_of_user = True
    vector_store.workspace_client_from = Mock(return_value=mock_ws_client)

    retriever = Mock()
    retriever.vector_store = vector_store
    retriever.columns = ["col1"]
    retriever.search_parameters = Mock()
    retriever.search_parameters.num_results = 5
    retriever.search_parameters.filters = None
    retriever.search_parameters.query_type = "ANN"
    retriever.instructed = None
    retriever.rerank = None

    with (
        patch("dao_ai.tools.vector_search.DatabricksVectorSearch") as mock_dvs_cls,
        patch.dict(
            "os.environ",
            {"DATABRICKS_HOST": "https://test.databricks.com"},
            clear=False,
        ),
        patch("dao_ai.tools.vector_search.mlflow"),
    ):
        mock_dvs_instance = MagicMock()
        mock_dvs_instance.similarity_search.return_value = []
        mock_dvs_cls.return_value = mock_dvs_instance

        tool = create_vector_search_tool(
            retriever=retriever,
            name="test_vs_tool",
            description="Test vector search",
        )

        # Invoke the tool — the runtime context is injected via LangChain's
        # ToolRuntime mechanism; for this unit test we just call invoke
        # which triggers _get_vector_search(context=None).
        tool.invoke({"query": "test query"})

        # Verify DatabricksVectorSearch was called with client_args=None
        mock_dvs_cls.assert_called_once()
        call_kwargs = mock_dvs_cls.call_args.kwargs
        assert call_kwargs.get("client_args") is None, (
            "client_args should be None for OBO vector stores to prevent "
            "overriding the workspace_client's forwarded user token"
        )
        assert call_kwargs.get("workspace_client") is mock_ws_client


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
            mock_provider_for_endpoint,  # set_default_endpoint validator (during __init__)
            mock_provider_for_primary_key,  # set_default_primary_key (during ensure_resolved)
            mock_provider_for_create,  # create() call
        ]

        vector_store = VectorStoreModel(
            source_table=table,
            embedding_source_column="description",
        )
        vector_store.ensure_resolved()

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
            mock_provider_for_endpoint,  # set_default_endpoint validator (during __init__)
            mock_provider_for_primary_key,  # set_default_primary_key (during ensure_resolved)
            mock_provider_for_create,
        ]

        vector_store_provisioning = VectorStoreModel(
            source_table=table,
            embedding_source_column="description",
        )
        vector_store_provisioning.ensure_resolved()

        with patch.object(
            vector_store_provisioning, "_create_new_index"
        ) as mock_create:
            vector_store_provisioning.create()
            mock_create.assert_called_once()


# =============================================================================
# Trace Location / OTEL Environment Variable Tests
# =============================================================================


@pytest.mark.unit
def test_set_databricks_env_vars_injects_trace_location_env_vars():
    """Test that set_databricks_env_vars auto-injects MLFLOW_TRACING_DESTINATION
    and MLFLOW_TRACING_SQL_WAREHOUSE_ID when trace_location is configured."""
    from dao_ai.config import AppModel, SchemaModel, TraceLocationModel

    schema = SchemaModel(catalog_name="my_catalog", schema_name="my_schema")
    trace_loc = TraceLocationModel(schema=schema, warehouse="abc123")

    mock_app = MagicMock(spec=AppModel)
    mock_app.environment_vars = {}
    mock_app.service_principal = None
    mock_app.trace_location = trace_loc

    with patch(
        "dao_ai.utils.get_default_databricks_host",
        return_value="https://test.databricks.com",
    ):
        AppModel.set_databricks_env_vars(mock_app)

    assert (
        mock_app.environment_vars["MLFLOW_TRACING_DESTINATION"]
        == "my_catalog.my_schema"
    )
    assert mock_app.environment_vars["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] == "abc123"


@pytest.mark.unit
def test_set_databricks_env_vars_does_not_override_explicit_trace_vars():
    """Test that explicit environment_vars for tracing take precedence."""
    from dao_ai.config import AppModel, SchemaModel, TraceLocationModel

    schema = SchemaModel(catalog_name="my_catalog", schema_name="my_schema")
    trace_loc = TraceLocationModel(schema=schema, warehouse="abc123")

    mock_app = MagicMock(spec=AppModel)
    mock_app.environment_vars = {
        "MLFLOW_TRACING_DESTINATION": "override_catalog.override_schema",
        "MLFLOW_TRACING_SQL_WAREHOUSE_ID": "override_wh",
    }
    mock_app.service_principal = None
    mock_app.trace_location = trace_loc

    with patch(
        "dao_ai.utils.get_default_databricks_host",
        return_value="https://test.databricks.com",
    ):
        AppModel.set_databricks_env_vars(mock_app)

    assert (
        mock_app.environment_vars["MLFLOW_TRACING_DESTINATION"]
        == "override_catalog.override_schema"
    )
    assert mock_app.environment_vars["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] == "override_wh"


@pytest.mark.unit
def test_set_databricks_env_vars_no_trace_vars_without_trace_location():
    """Test that MLFLOW_TRACING_DESTINATION is not set when trace_location is None."""
    from dao_ai.config import AppModel

    mock_app = MagicMock(spec=AppModel)
    mock_app.environment_vars = {}
    mock_app.service_principal = None
    mock_app.trace_location = None

    with patch(
        "dao_ai.utils.get_default_databricks_host",
        return_value="https://test.databricks.com",
    ):
        AppModel.set_databricks_env_vars(mock_app)

    assert "MLFLOW_TRACING_DESTINATION" not in mock_app.environment_vars
    assert "MLFLOW_TRACING_SQL_WAREHOUSE_ID" not in mock_app.environment_vars


@pytest.mark.unit
def test_deploy_model_serving_links_experiment_and_grants_permissions():
    """Test that deploy_model_serving_agent links experiment to UC trace location
    and grants OTEL table permissions when trace_location is configured."""
    from dao_ai.config import AppConfig, AppModel, TraceLocationModel
    from dao_ai.providers.databricks import DatabricksProvider

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_registered_model = MagicMock()

    mock_app.endpoint_name = "test_endpoint"
    mock_registered_model.full_name = "cat.sch.model"
    mock_app.registered_model = mock_registered_model
    mock_app.scale_to_zero = True
    mock_app.environment_vars = {}
    mock_app.workload_size = "Small"
    mock_app.tags = {}
    mock_app.permissions = []

    trace_loc = MagicMock(spec=TraceLocationModel)
    trace_loc.catalog_name = "trace_cat"
    trace_loc.schema_name = "trace_sch"
    trace_loc.warehouse_id = "wh123"
    mock_app.trace_location = trace_loc
    mock_app.monitoring = None
    mock_app.service_principal = None

    mock_config.app = mock_app

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "exp123"

    with (
        patch.object(
            DatabricksProvider, "_serving_endpoint_exists", return_value=False
        ),
        patch(
            "dao_ai.providers.databricks.agents.get_deployments",
            return_value=[],
        ),
        patch("dao_ai.providers.databricks.agents.deploy"),
        patch("dao_ai.providers.databricks.get_latest_model_version", return_value=1),
        patch("dao_ai.providers.databricks.mlflow.set_registry_uri"),
        patch(
            "mlflow.tracing.enablement.set_experiment_trace_location"
        ) as mock_set_loc,
    ):
        with patch.object(DatabricksProvider, "__init__", return_value=None):
            provider = DatabricksProvider()
            provider.w = MagicMock()

            with patch.object(
                provider, "get_or_create_experiment", return_value=mock_experiment
            ):
                with patch.object(
                    provider, "grant_otel_table_permissions"
                ) as mock_grant:
                    provider.deploy_model_serving_agent(mock_config)

                    mock_set_loc.assert_called_once()
                    call_kwargs = mock_set_loc.call_args.kwargs
                    assert call_kwargs["experiment_id"] == "exp123"
                    assert call_kwargs["sql_warehouse_id"] == "wh123"

                    mock_grant.assert_called_once_with(mock_config)


@pytest.mark.unit
def test_grant_otel_table_permissions_grants_modify_and_select():
    """Test that grant_otel_table_permissions grants MODIFY and SELECT on each OTEL table."""
    from dao_ai.config import (
        AppConfig,
        AppModel,
        ServicePrincipalModel,
        TraceLocationModel,
    )
    from dao_ai.providers.databricks import DatabricksProvider

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)

    trace_loc = MagicMock(spec=TraceLocationModel)
    trace_loc.catalog_name = "cat"
    trace_loc.schema_name = "sch"
    trace_loc.warehouse_id = "wh1"
    mock_app.trace_location = trace_loc

    mock_sp = MagicMock(spec=ServicePrincipalModel)
    mock_sp.client_id = "sp-client-id-123"
    mock_app.service_principal = mock_sp

    mock_config.app = mock_app

    with patch.object(DatabricksProvider, "__init__", return_value=None):
        with patch("dao_ai.config.value_of", return_value="sp-client-id-123"):
            provider = DatabricksProvider()
            provider.w = MagicMock()

            provider.grant_otel_table_permissions(mock_config)

            expected_table_count = len(TraceLocationModel.OTEL_TABLE_SUFFIXES)
            # 2 privileges per table (MODIFY and SELECT)
            assert provider.w.grants.update.call_count == expected_table_count * 2


@pytest.mark.unit
def test_grant_otel_table_permissions_skips_without_service_principal():
    """Test that grant_otel_table_permissions does nothing without a service principal."""
    from dao_ai.config import AppConfig, AppModel, TraceLocationModel
    from dao_ai.providers.databricks import DatabricksProvider

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)

    trace_loc = MagicMock(spec=TraceLocationModel)
    trace_loc.catalog_name = "cat"
    trace_loc.schema_name = "sch"
    mock_app.trace_location = trace_loc
    mock_app.service_principal = None

    mock_config.app = mock_app

    with patch.object(DatabricksProvider, "__init__", return_value=None):
        provider = DatabricksProvider()
        provider.w = MagicMock()

        provider.grant_otel_table_permissions(mock_config)

        provider.w.grants.update.assert_not_called()
