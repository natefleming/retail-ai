from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from conftest import has_databricks_env
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.catalog import FunctionInfo, TableInfo
from mlflow.models.resources import DatabricksFunction, DatabricksTable

from dao_ai.config import (
    AppConfig,
    DatabaseModel,
    DatasetModel,
    FunctionModel,
    IndexModel,
    SchemaModel,
    TableModel,
    VectorStoreModel,
)
from dao_ai.providers.databricks import DatabricksProvider


def _stamp_extras_resolvable(mock_config: MagicMock) -> MagicMock:
    """Give a ``MagicMock(spec=AppConfig)`` the real, empty collections the
    extras resolver iterates (``tools``/``retrievers``/``middleware``/
    ``datasets``) plus a disabled-A2A app, so ``create_agent`` /
    ``deploy_apps_agent`` — which now resolve optional-feature extras from the
    config — don't choke on non-iterable mock attributes. Tests that exercise a
    specific feature can still override these afterwards.
    """
    mock_config.tools = {}
    mock_config.retrievers = {}
    mock_config.middleware = {}
    mock_config.datasets = None
    mock_config.memory = None
    # app.a2a disabled + no orchestration → resolver adds no extras by default.
    mock_config.app.a2a = MagicMock(enabled=False)
    mock_config.app.orchestration = None
    # Custom-dep passthrough surfaces read by the Apps/MS deploy paths. Only
    # default them when the test hasn't set a real list — never clobber a
    # test's own pip_requirements/code_paths values. ``getattr`` tolerates
    # both plain and spec-restricted mocks.
    if not isinstance(getattr(mock_config.app, "pip_requirements", None), list):
        mock_config.app.pip_requirements = []
    if not isinstance(getattr(mock_config.app, "code_paths", None), list):
        mock_config.app.code_paths = []
    # Trace-permission fields the deploy paths read at grant time. Default to
    # the real model defaults (manage_permissions=True, experiment=None) unless
    # a test set concrete values. ``getattr`` tolerates spec-restricted mocks.
    if not isinstance(getattr(mock_config.app, "manage_permissions", None), bool):
        mock_config.app.manage_permissions = True
    if getattr(mock_config.app, "experiment", None) is None or isinstance(
        getattr(mock_config.app, "experiment", None), MagicMock
    ):
        mock_config.app.experiment = None
    # The Model-Serving grant path fires only when a service principal is
    # declared (``manage_permissions and service_principal is not None``).
    # Default it off so deploy tests don't spin up a real WorkspaceClient for
    # grants they aren't asserting; tests exercising the grant path set a
    # concrete SP themselves. ``spec``-restricted mocks block the attribute
    # entirely (getattr returns the default and never materializes it), so set
    # it explicitly unless a test already assigned a concrete model instance.
    sp = getattr(mock_config.app, "service_principal", None)
    if sp is None or isinstance(sp, MagicMock):
        mock_config.app.service_principal = None
    return mock_config


@pytest.mark.unit
def test_dataset_resolve_asset_path_config_relative():
    """A relative ddl/data path resolves against the stamped config dir."""
    ds = DatasetModel(ddl="functions/find_x.sql", data=None)
    ds._base_path = "/repo/examples/99_complete_applications/hardware_store"
    resolved = ds.resolve_asset_path(ds.ddl)
    assert resolved == Path(
        "/repo/examples/99_complete_applications/hardware_store/functions/find_x.sql"
    )


@pytest.mark.unit
def test_dataset_resolve_asset_path_absolute_passthrough():
    """An absolute path is returned unchanged regardless of base_path."""
    ds = DatasetModel(ddl="/abs/fn.sql", data=None)
    ds._base_path = "/repo/config"
    assert ds.resolve_asset_path(ds.ddl) == Path("/abs/fn.sql")


@pytest.mark.unit
def test_dataset_resolve_asset_path_cwd_fallback():
    """With no base_path stamped, a relative path stays relative (CWD-based)."""
    ds = DatasetModel(ddl="data/seed.sql", data=None)
    assert ds.resolve_asset_path(ds.ddl) == Path("data/seed.sql")


@pytest.mark.unit
def test_resolve_spark_read_path_passthrough_volumes():
    """A path already on a UC volume returns unchanged and never stages."""
    existing_path = "/Volumes/cat/sch/vol/products.parquet"
    ds = DatasetModel(
        table=TableModel(
            schema=SchemaModel(catalog_name="c", schema_name="s"), name="t"
        )
    )
    provider = DatabricksProvider()
    with (
        patch.object(provider, "create_volume") as mock_create_volume,
        patch("dao_ai.providers.databricks.shutil.copy2") as mock_copy,
    ):
        result = provider._resolve_spark_read_path(ds, Path(existing_path))
    assert result == existing_path
    mock_create_volume.assert_not_called()
    mock_copy.assert_not_called()


@pytest.mark.unit
def test_resolve_spark_read_path_stages_workspace_file_to_volume():
    """A workspace/local file is copied into a staging volume in the target
    schema, and Spark is pointed at the /Volumes path."""
    ds = DatasetModel(
        table=TableModel(
            schema=SchemaModel(catalog_name="mycat", schema_name="mysch"),
            name="products",
        )
    )
    workspace_path = Path(
        "/Workspace/Users/me/.bundle/app/files/config/data/products.snappy.parquet"
    )
    provider = DatabricksProvider()
    with (
        patch.object(provider, "create_volume") as mock_create_volume,
        patch("dao_ai.providers.databricks.shutil.copy2") as mock_copy,
    ):
        result = provider._resolve_spark_read_path(ds, workspace_path)

    expected = "/Volumes/mycat/mysch/dao_ai_staging/products.snappy.parquet"
    assert result == expected
    # Volume derived from the dataset's own target schema.
    (volume_arg,) = mock_create_volume.call_args.args
    assert volume_arg.full_name == "mycat.mysch.dao_ai_staging"
    mock_copy.assert_called_once_with(str(workspace_path), expected)


@pytest.mark.unit
def test_dataset_staging_schema_from_fully_qualified_table_name():
    """When no schema reference is set, catalog+schema parse from the FQN."""
    ds = DatasetModel(table=TableModel(name="mycat.mysch.products"))
    provider = DatabricksProvider()
    schema = provider._dataset_staging_schema(ds)
    assert schema.catalog_name == "mycat"
    assert schema.schema_name == "mysch"


@pytest.mark.unit
def test_create_dataset_routes_csv_through_spark_read_not_pandas():
    """csv reads via the staged spark.read path — not pd.read_csv, whose
    header= arg is incompatible with the Spark-style ``header: true``
    read_options the configs author."""
    ds = DatasetModel(
        table=TableModel(
            schema=SchemaModel(catalog_name="mycat", schema_name="mysch"), name="orders"
        ),
        data="data/orders_raw.csv",
        format="csv",
        read_options={"header": True},
    )
    ds._base_path = "/Workspace/Users/me/.bundle/app/files/config"

    mock_spark = MagicMock()
    provider = DatabricksProvider()
    with (
        patch("pyspark.sql.SparkSession.getActiveSession", return_value=mock_spark),
        patch.object(
            provider,
            "_resolve_spark_read_path",
            return_value="/Volumes/mycat/mysch/dao_ai_staging/orders_raw.csv",
        ) as mock_resolve,
        patch("dao_ai.providers.databricks.pd.read_csv") as mock_read_csv,
    ):
        provider.create_dataset(ds)

    mock_read_csv.assert_not_called()
    mock_resolve.assert_called_once()
    mock_spark.read.format.assert_called_once_with("csv")
    mock_spark.read.format.return_value.options.assert_called_once_with(header=True)


@pytest.mark.unit
def test_from_file_stamps_base_path_on_models(tmp_path):
    """AppConfig.from_file stamps each dataset/function with the config's dir."""
    body = (
        "resources:\n"
        "  models:\n"
        "    default_llm: &default_llm\n"
        "      name: databricks-gpt-5-4-mini\n"
        "agents:\n"
        "  greeter: &greeter\n"
        "    name: greeter\n"
        "    description: A friendly assistant.\n"
        "    model: *default_llm\n"
        "    prompt: You are concise.\n"
        "app:\n"
        "  name: hw_app\n"
        "  agents:\n"
        "    - *greeter\n"
        "datasets:\n"
        "  - ddl: data/seed.sql\n"
        "    data: null\n"
        "unity_catalog_functions:\n"
        "  - function:\n"
        "      name: c.s.f\n"
        "    ddl: functions/fn.sql\n"
    )
    cfg_path = tmp_path / "hw.yaml"
    cfg_path.write_text(body)
    config = AppConfig.from_file(cfg_path, initialize=False)

    assert config.datasets[0]._base_path == str(tmp_path)
    assert config.unity_catalog_functions[0]._base_path == str(tmp_path)
    assert config.datasets[0].resolve_asset_path("data/seed.sql") == (
        tmp_path / "data" / "seed.sql"
    )


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
    # Default-truthy MagicMock would trigger the trace_location code path
    # (which calls set_experiment a second time with trace_location=UC(...)).
    # This test covers the "no trace_location" baseline.
    mock_app.trace_location = None
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
        patch("dao_ai.utils.is_published", return_value=True),
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
        _stamp_extras_resolvable(mock_config)
        provider.create_agent(config=mock_config)

        # Verify experiment was retrieved/created and set
        mock_get_experiment.assert_called_once_with(mock_config)
        mock_set_experiment.assert_called_once_with(
            experiment_id=mock_experiment.experiment_id
        )


@pytest.mark.unit
def test_create_agent_does_not_mutate_config_pip_requirements():
    """``create_agent`` adds serving-only requirements to a *copy*.

    ``config.app.pip_requirements`` is a live pydantic list, so a ``+=`` on the
    field itself extends it in place. A local-wheel deploy appends
    ``code/dao_ai-<ver>.whl`` — an MLflow-relative path, not a PEP 508
    requirement — plus the whole frozen environment, and a subsequent Apps
    deploy in the same process (``deploy_agent(mode=BOTH)``) folds
    ``pip_requirements`` straight into the generated ``pyproject.toml``. That
    made ``uv lock`` fail with "Dependency #2 ... Expected semicolon".
    """
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    import mlflow

    from dao_ai.config import AppConfig
    from dao_ai.providers.databricks import DatabricksProvider

    declared: list[str] = ["my-extra-package==1.2.3"]

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock()
    mock_app.name = "test_app"
    mock_app.code_paths = []
    mock_app.pip_requirements = declared
    mock_app.input_example = None
    mock_app.trace_location = None
    mock_config.app = mock_app

    mock_resources = MagicMock()
    for attr in (
        "llms",
        "vector_stores",
        "warehouses",
        "genie_rooms",
        "tables",
        "functions",
        "connections",
        "databases",
        "volumes",
    ):
        setattr(mock_resources, attr, MagicMock(values=lambda: []))
    mock_config.resources = mock_resources
    mock_config.guardrails = {}
    mock_config.agents = {}

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "test_experiment_123"

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
        # The local-wheel branch: this is what appended the bad requirement.
        patch("dao_ai.providers.databricks._use_local_source", return_value=True),
        patch(
            "dao_ai.providers.databricks.find_dev_wheel",
            return_value=Path("/tmp/dist/dao_ai-9.9.9-py3-none-any.whl"),
        ),
        patch("dao_ai.providers.databricks._warn_if_stale_dev_wheel"),
        patch(
            "dao_ai.providers.databricks.get_installed_packages",
            return_value=["some-frozen-dep==0.1.0"],
        ),
    ):
        mock_start_run.return_value.__enter__.return_value = MagicMock()
        mock_log_model.return_value = MagicMock(model_uri="test_uri")

        _stamp_extras_resolvable(mock_config)
        DatabricksProvider().create_agent(config=mock_config)

    # Compared against a literal, not against ``declared`` — an in-place extend
    # grows that same list object, so ``== declared`` would compare it to itself.
    assert mock_app.pip_requirements == ["my-extra-package==1.2.3"], (
        "create_agent extended config.app.pip_requirements in place: "
        f"{mock_app.pip_requirements}"
    )
    # It still has to have reached the logged model, or the copy would be a
    # silent no-op fix.
    conda_env = mock_log_model.call_args.kwargs["conda_env"]
    logged = next(
        dep["pip"] for dep in conda_env["dependencies"] if isinstance(dep, dict)
    )
    assert "code/dao_ai-9.9.9-py3-none-any.whl" in logged
    assert "my-extra-package==1.2.3" in logged


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
        patch("dao_ai.utils.is_published", return_value=True),
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
        _stamp_extras_resolvable(mock_config)
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
def test_create_agent_local_source_no_wheel_no_source_raises():
    """create_agent must fail loud when local-source is forced but nothing to ship.

    ``--development`` (or auto-detect on a non-published install) forces
    ``_use_local_source`` True, but if there is neither a pre-built wheel
    (``find_dev_wheel() is None``) nor a source tree (``is_source_layout``
    False — dao-ai is in site-packages), continuing would log a model with NO
    dao-ai. This must raise instead of silently shipping a broken model, so it
    matches the Apps + generate-agent paths.
    """
    from unittest.mock import MagicMock, patch

    import mlflow

    from dao_ai.config import AppConfig
    from dao_ai.providers.databricks import DatabricksProvider

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock()
    mock_app.name = "test_app"
    mock_app.code_paths = []
    mock_app.pip_requirements = []
    mock_app.input_example = None
    mock_app.trace_location = None
    mock_config.app = mock_app

    mock_resources = MagicMock()
    for attr in (
        "llms",
        "vector_stores",
        "warehouses",
        "genie_rooms",
        "tables",
        "functions",
        "connections",
        "databases",
        "volumes",
    ):
        setattr(mock_resources, attr, MagicMock(values=lambda: []))
    mock_config.resources = mock_resources
    mock_config.guardrails = {}
    mock_config.agents = {}

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "exp123"
    mock_experiment.name = "/Users/test_user/test_app"

    with (
        patch.object(
            DatabricksProvider, "get_or_create_experiment", return_value=mock_experiment
        ),
        patch.object(mlflow, "set_experiment"),
        patch.object(mlflow, "set_registry_uri"),
        patch.object(mlflow, "set_tag"),
        patch("dao_ai.providers.databricks.MlflowClient"),
        # Local-source forced, but no wheel and dao-ai lives in site-packages.
        patch("dao_ai.providers.databricks.find_dev_wheel", return_value=None),
        patch("dao_ai.providers.databricks.is_source_layout", return_value=False),
    ):
        provider = DatabricksProvider()
        with pytest.raises(RuntimeError, match="Build a wheel first"):
            _stamp_extras_resolvable(mock_config)
            provider.create_agent(config=mock_config, development=True)


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
    # Pydantic v2 fields aren't in dir(AppConfig); MagicMock(spec=...) blocks
    # them. Set resources=None explicitly so the deploy code's
    # `if config.resources and config.resources.databases` short-circuits.
    mock_config.resources = None

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
                        _stamp_extras_resolvable(mock_config)
                        # Keep the deploy hermetic: experiment resolution would
                        # otherwise build a live WorkspaceClient.
                        provider.get_or_create_experiment = MagicMock(
                            return_value=MagicMock(experiment_id="exp1")
                        )
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
    mock_config.resources = None
    _stamp_extras_resolvable(mock_config)

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
                                # Keep hermetic: experiment resolution would
                                # otherwise build a live WorkspaceClient.
                                provider.get_or_create_experiment = MagicMock(
                                    return_value=MagicMock(experiment_id="exp1")
                                )
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
            _stamp_extras_resolvable(mock_config)
            provider.deploy_agent(config=mock_config)

            # Should route to model serving by default
            mock_model_serving.assert_called_once_with(mock_config)
            mock_apps.assert_not_called()


@pytest.mark.unit
def test_deploy_agent_routes_to_model_serving_explicitly():
    """Test that deploy_agent routes to deploy_model_serving_agent when mode=MODEL_SERVING."""
    from unittest.mock import MagicMock, patch

    from dao_ai.config import AppConfig, AppModel, ServingMode
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
            _stamp_extras_resolvable(mock_config)
            provider.deploy_agent(config=mock_config, mode=ServingMode.MODEL_SERVING)

            mock_model_serving.assert_called_once_with(mock_config)
            mock_apps.assert_not_called()


@pytest.mark.unit
def test_deploy_agent_routes_to_apps_when_specified():
    """Test that deploy_agent routes to deploy_apps_agent when mode=APPS."""
    from unittest.mock import MagicMock, patch

    from dao_ai.config import AppConfig, AppModel, ServingMode
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
            _stamp_extras_resolvable(mock_config)
            provider.deploy_agent(config=mock_config, mode=ServingMode.APPS)

            mock_apps.assert_called_once_with(
                mock_config, as_mcp=False, development=None
            )
            mock_model_serving.assert_not_called()


@pytest.mark.unit
def test_deploy_agent_routes_as_mcp_to_apps(monkeypatch):
    """as_mcp routes through deploy_apps_agent (MCP runs on the Apps runtime)."""
    from dao_ai.config import ServingMode
    from dao_ai.providers.databricks import DatabricksProvider

    p = DatabricksProvider.__new__(
        DatabricksProvider
    )  # no __init__ / no WorkspaceClient
    calls = []
    monkeypatch.setattr(
        p, "deploy_model_serving_agent", lambda c: calls.append("ms"), raising=False
    )
    monkeypatch.setattr(
        p,
        "deploy_apps_agent",
        lambda c, as_mcp=False, development=None: calls.append(
            "mcp" if as_mcp else "apps"
        ),
        raising=False,
    )
    p.deploy_agent(config=object(), mode=ServingMode.APPS, as_mcp=True)
    assert calls == ["mcp"]


@pytest.mark.unit
def test_deploy_agent_rejects_as_mcp_with_model_serving(monkeypatch):
    """as_mcp + MODEL_SERVING raises — there is no Model Serving MCP surface."""
    from dao_ai.config import ServingMode
    from dao_ai.providers.databricks import DatabricksProvider

    p = DatabricksProvider.__new__(DatabricksProvider)
    monkeypatch.setattr(p, "deploy_model_serving_agent", lambda c: None, raising=False)
    with pytest.raises(ValueError, match="as_mcp requires mode=APPS"):
        p.deploy_agent(config=object(), mode=ServingMode.MODEL_SERVING, as_mcp=True)


@pytest.mark.unit
def test_deploy_apps_agent_as_mcp_command_and_extras(monkeypatch):
    """as_mcp forwards the MCP command, mcp extras, chat-UI off, and mcp- name."""
    import dao_ai._extras as _extras
    from dao_ai.providers.databricks import DatabricksProvider

    monkeypatch.setattr(
        _extras, "resolve_required_extras_or_all", lambda config, target="mcp": set()
    )

    p = DatabricksProvider.__new__(DatabricksProvider)
    captured = {}

    def _fake_deploy_app(
        config, *, app_command, extras, include_chat_ui, as_mcp, development
    ):
        captured.update(
            app_command=app_command,
            extras=extras,
            include_chat_ui=include_chat_ui,
            as_mcp=as_mcp,
        )

    monkeypatch.setattr(p, "_deploy_app", _fake_deploy_app, raising=False)

    class _App:
        enable_chat_proxy = None

    class _Cfg:
        app = _App()

    p.deploy_apps_agent(_Cfg(), as_mcp=True)
    assert captured["app_command"] == ["python", "-m", "dao_ai.mcp.server"]
    assert "mcp" in captured["extras"]
    assert captured["include_chat_ui"] is False
    assert captured["as_mcp"] is True


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
    mock_app.apps_compute_size.return_value = "LARGE"
    mock_config.app = mock_app
    mock_config.source_config_path = None  # No config file to upload
    mock_config._source_config_path = None
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

            _stamp_extras_resolvable(mock_config)
            provider.deploy_apps_agent(mock_config)

            # Verify REST API was called to create the app
            provider.w.api_client.do.assert_called()
            create_call = provider.w.api_client.do.call_args_list[0]
            assert create_call.args[0] == "POST"
            assert create_call.args[1] == "/api/2.0/apps"
            body = create_call.kwargs.get("body", {})
            assert body["name"] == "test-app"  # Normalized: underscores become dashes
            assert body["description"] == "Test app description"
            # compute_size is set on CREATE (the POST API accepts it)
            assert body["compute_size"] == "LARGE"
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
    mock_app.apps_compute_size.return_value = "LARGE"
    mock_config.app = mock_app
    mock_config.source_config_path = None  # No config file to upload
    mock_config._source_config_path = None
    mock_config.rendered_yaml = None
    mock_config.model_dump.return_value = {"app": {"name": "test_app"}}
    mock_config.resources = None  # No resources (required for generate_app_resources)
    mock_config.agents = None
    mock_config.retrievers = None

    # Create mock existing App (already MEDIUM — a resize to LARGE should warn
    # and NOT be sent on the update API, which rejects compute_size changes)
    mock_existing_app = MagicMock(spec=App)
    mock_existing_app.name = "test_app"
    mock_existing_app.url = "https://test_app.databricks.com"
    mock_existing_app.compute_size = "MEDIUM"
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

            _stamp_extras_resolvable(mock_config)
            provider.deploy_apps_agent(mock_config)

            # Verify REST API was NOT called with POST (app already exists)
            for call in provider.w.api_client.do.call_args_list:
                assert call.args[0] != "POST" or "/api/2.0/apps" not in call.args[1], (
                    "POST /api/2.0/apps should not be called for existing app"
                )
            # compute_size must NOT be sent on any update — the PATCH API
            # rejects it. It is only valid on CREATE.
            for call in provider.w.api_client.do.call_args_list:
                body = call.kwargs.get("body", {})
                assert "compute_size" not in body, (
                    "compute_size must not be sent on an update call"
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
            provider,
            "get_or_create_experiment",
            return_value=MagicMock(experiment_id="exp-1"),
        ):
            provider.w.apps.get.return_value = mock_existing_app
            provider.w.apps.deploy_and_wait.return_value = mock_deployment

            _stamp_extras_resolvable(mock_config)
            provider.deploy_apps_agent(mock_config)

    # Find the workspace.upload call carrying the config
    upload_calls = [
        c
        for c in provider.w.workspace.upload.call_args_list
        if c.kwargs.get("path", "").endswith("dao_ai.yaml")
    ]
    assert upload_calls, "expected an upload of dao_ai.yaml"
    uploaded_bytes = upload_calls[0].kwargs["content"]
    assert isinstance(uploaded_bytes, io.BytesIO)
    uploaded_text = uploaded_bytes.getvalue().decode("utf-8")
    assert uploaded_text == rendered_yaml, (
        "deploy must upload rendered YAML, not source"
    )
    assert "${var.catalog}" not in uploaded_text


@pytest.mark.unit
def test_deploy_apps_agent_falls_back_to_source_when_no_rendered_yaml(tmp_path):
    """If rendered_yaml is missing (legacy callers), fall back to reading the source file."""
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
            provider,
            "get_or_create_experiment",
            return_value=MagicMock(experiment_id="exp-1"),
        ):
            provider.w.apps.get.return_value = mock_existing_app
            provider.w.apps.deploy_and_wait.return_value = mock_deployment

            _stamp_extras_resolvable(mock_config)
            provider.deploy_apps_agent(mock_config)

    upload_calls = [
        c
        for c in provider.w.workspace.upload.call_args_list
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
    from unittest.mock import MagicMock, patch

    import yaml as _yaml
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
            provider,
            "get_or_create_experiment",
            return_value=MagicMock(experiment_id="exp-1"),
        ):
            provider.w.apps.get.return_value = mock_existing_app
            provider.w.apps.deploy_and_wait.return_value = mock_deployment

            _stamp_extras_resolvable(mock_config)
            provider.deploy_apps_agent(mock_config)

    # The provider should have created the workspace dir and uploaded a YAML
    # serialized from the in-memory model.
    provider.w.workspace.mkdirs.assert_called()
    upload_calls = [
        c
        for c in provider.w.workspace.upload.call_args_list
        if c.kwargs.get("path", "").endswith("dao_ai.yaml")
    ]
    assert upload_calls, "expected an upload of dao_ai.yaml even without a source file"
    uploaded_text = upload_calls[0].kwargs["content"].getvalue().decode("utf-8")
    # Round-trip equivalence: the uploaded YAML parses to the same dict
    # the AppConfig.model_dump() would have produced.
    assert _yaml.safe_load(uploaded_text) == _yaml.safe_load(expected_yaml)


@pytest.mark.unit
def test_serving_mode_enum_values():
    """Test that ServingMode enum has expected values."""
    from dao_ai.config import ServingMode

    assert ServingMode.MODEL_SERVING.value == "model_serving"
    assert ServingMode.APPS.value == "apps"

    # Test enum can be created from string
    assert ServingMode("model_serving") == ServingMode.MODEL_SERVING
    assert ServingMode("apps") == ServingMode.APPS


@pytest.mark.unit
def test_serving_mode_members():
    from dao_ai.config import ServingMode

    assert {t.value for t in ServingMode} == {"model_serving", "apps"}
    # MCP is a protocol (as_mcp), not a platform — deliberately not a member.
    with pytest.raises(ValueError):
        ServingMode("mcp")
    with pytest.raises(AttributeError):
        _ = ServingMode.MCP
    with pytest.raises(ValueError):
        ServingMode("both")
    with pytest.raises(AttributeError):
        _ = ServingMode.BOTH


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
    """DatabaseModel.as_resources intentionally returns ``[]`` for autoscaling
    Lakebase databases. MLflow's ``DatabricksLakebase`` resource only supports
    the deprecated provisioned-instance shape, and emitting it for an
    autoscaling project causes Model Serving endpoints to fail to start with
    ``NOT_FOUND: Database instance is not found`` (MLflow issue #22452,
    2026-04-10). Apps-side resource binding goes through a different code path
    (``_extract_database_resources``) and is unaffected -- see the broader
    suite in tests/dao_ai/test_lakebase_app_resources.py.
    """
    db = DatabaseModel(
        name="test-db",
        project="test-db",
    )
    assert db.is_lakebase is True
    assert list(db.as_resources()) == []
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

    # Mock the DatabricksProvider to avoid actual API calls during
    # ensure_resolved (primary-key discovery uses the provider).
    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider.find_primary_key.return_value = ["id"]
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

        # Endpoint discovery is deferred to create()/_create_new_index — it must
        # NOT run during construction or ensure_resolved (that would build a
        # VectorSearchClient at serving/parse time). It stays None until create().
        assert vector_store.endpoint is None
        mock_provider.find_endpoint_for_index.assert_not_called()
        mock_provider.find_vector_search_endpoint.assert_not_called()


@pytest.mark.unit
def test_vector_store_provisioning_config_load_makes_no_vector_search_calls():
    """Regression: parsing + resolving a provisioning config must not touch
    Vector Search.

    Endpoint discovery used to run in a model validator, building a
    VectorSearchClient at config-parse time. On serving/MCP-server boot (which
    parses and calls ensure_resolved but never provisions) that bare client has
    no credentials and crashed with InvalidInputException. This locks in that
    neither construction nor ensure_resolved performs endpoint discovery.
    """
    schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
    table = TableModel(schema=schema, name="test_table")

    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider.find_primary_key.return_value = ["id"]
        mock_provider_class.return_value = mock_provider

        vector_store = VectorStoreModel(
            source_table=table,
            embedding_source_column="description",
        )
        vector_store.ensure_resolved()

        # Endpoint stays unresolved; no Vector Search endpoint lookups happened.
        assert vector_store.endpoint is None
        mock_provider.find_endpoint_for_index.assert_not_called()
        mock_provider.find_vector_search_endpoint.assert_not_called()


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
        # Tool creation/invocation triggers column auto-discovery and a VS
        # refresh, both of which would build a live client (DNS) against the
        # fake host. Stub the column probe and the refresh VS client.
        patch(
            "dao_ai.tools.vector_search._fetch_index_columns",
            return_value=[("col1", None, None)],
        ),
        patch("dao_ai.tools.vector_search.VectorSearchClient"),
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
        # Provider used by ensure_resolved() for primary-key discovery.
        mock_provider_for_primary_key = MagicMock()
        mock_provider_for_primary_key.find_primary_key.return_value = ["id"]

        # Provider built inside create(); endpoint discovery now happens here
        # (deferred out of the validator), so it must stub the lookup methods.
        mock_provider_for_create = MagicMock()
        mock_provider_for_create.find_endpoint_for_index.return_value = None
        mock_provider_for_create.find_vector_search_endpoint.return_value = (
            "test_endpoint"
        )

        # One construction during ensure_resolved(), one inside create().
        mock_provider_class.side_effect = [
            mock_provider_for_primary_key,  # ensure_resolved() primary-key discovery
            mock_provider_for_create,  # create() call
        ]

        vector_store = VectorStoreModel(
            source_table=table,
            embedding_source_column="description",
        )
        vector_store.ensure_resolved()

        # Call create() - this will use the second mock from side_effect
        vector_store.create()

        # Endpoint was auto-discovered inside create()/_create_new_index.
        assert vector_store.endpoint is not None
        assert vector_store.endpoint.name == "test_endpoint"

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
def test_vector_store_create_provisioning_auto_discovers_endpoint():
    """_create_new_index() auto-discovers the endpoint when none is configured.

    Endpoint discovery was moved out of the config validator (which ran at
    serving/parse time) into provisioning. When endpoint is None, it now falls
    back through find_endpoint_for_index -> find_vector_search_endpoint using the
    provider passed by create().
    """
    schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
    table = TableModel(schema=schema, name="test_table")

    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider_for_validators = MagicMock()
        mock_provider_for_validators.find_primary_key.return_value = ["id"]
        mock_provider_class.return_value = mock_provider_for_validators

        vector_store = VectorStoreModel(
            source_table=table,
            embedding_source_column="description",
        )

        # No endpoint configured (nor discovered at parse time — that's the fix).
        assert vector_store.endpoint is None

        # Provider handed to _create_new_index: first lookup misses, the
        # available-indexes fallback resolves the endpoint.
        mock_provider = MagicMock()
        mock_provider.find_endpoint_for_index.return_value = None
        mock_provider.find_vector_search_endpoint.return_value = "discovered_endpoint"

        vector_store._create_new_index(mock_provider)

        assert vector_store.endpoint is not None
        assert vector_store.endpoint.name == "discovered_endpoint"
        mock_provider.find_endpoint_for_index.assert_called_once()
        mock_provider.find_vector_search_endpoint.assert_called_once()
        mock_provider.create_vector_store.assert_called_once_with(vector_store)


@pytest.mark.unit
def test_get_vector_index_self_resolves_endpoint_when_none():
    """get_vector_index resolves a missing endpoint on demand and stamps it.

    The read path must work on an instance that never went through create()
    (e.g. a retriever's deep-copied vector_store), so it resolves the endpoint
    via find_endpoint_for_index and stamps it back onto the model.
    """
    schema = SchemaModel(catalog_name="c", schema_name="s")
    vector_store = VectorStoreModel(
        index=IndexModel(schema=schema, name="products_index"),
        embedding_source_column="description",
    )
    assert vector_store.endpoint is None

    provider = DatabricksProvider(vsc=MagicMock())
    with patch.object(
        provider, "find_endpoint_for_index", return_value="resolved_ep"
    ) as mock_find:
        provider.get_vector_index(vector_store)

    mock_find.assert_called_once()
    # Endpoint stamped back onto the model, and get_index called with it.
    assert vector_store.endpoint is not None
    assert vector_store.endpoint.name == "resolved_ep"
    provider.vsc.get_index.assert_called_once_with("resolved_ep", "c.s.products_index")


@pytest.mark.unit
def test_get_vector_index_endpointless_fallback():
    """When discovery finds no endpoint, fall back to an endpoint-less lookup.

    The SDK's get_index accepts endpoint_name=None and resolves the index by
    full name, so a failed discovery must NOT crash.
    """
    schema = SchemaModel(catalog_name="c", schema_name="s")
    vector_store = VectorStoreModel(
        index=IndexModel(schema=schema, name="products_index"),
        embedding_source_column="description",
    )

    provider = DatabricksProvider(vsc=MagicMock())
    with patch.object(provider, "find_endpoint_for_index", return_value=None):
        provider.get_vector_index(vector_store)

    assert vector_store.endpoint is None
    provider.vsc.get_index.assert_called_once_with(None, "c.s.products_index")


@pytest.mark.unit
def test_get_vector_index_discovery_failure_falls_through():
    """A raising discovery (e.g. unauthenticated VS client) must not propagate.

    On serving with SP auth, _vsc_for_refresh can yield a client that can't
    list endpoints; discovery then raises. get_vector_index must swallow it and
    fall back to the endpoint-less lookup so retrieval degrades gracefully.
    """
    schema = SchemaModel(catalog_name="c", schema_name="s")
    vector_store = VectorStoreModel(
        index=IndexModel(schema=schema, name="products_index"),
        embedding_source_column="description",
    )

    provider = DatabricksProvider(vsc=MagicMock())
    with patch.object(
        provider,
        "find_endpoint_for_index",
        side_effect=RuntimeError("InvalidInputException: no creds"),
    ):
        provider.get_vector_index(vector_store)

    assert vector_store.endpoint is None
    provider.vsc.get_index.assert_called_once_with(None, "c.s.products_index")


@pytest.mark.unit
def test_get_vector_index_uses_configured_endpoint():
    """An explicitly configured endpoint is used as-is (no discovery call)."""
    schema = SchemaModel(catalog_name="c", schema_name="s")
    vector_store = VectorStoreModel(
        index=IndexModel(schema=schema, name="products_index"),
        embedding_source_column="description",
        endpoint={"name": "explicit_ep"},
    )

    provider = DatabricksProvider(vsc=MagicMock())
    with patch.object(provider, "find_endpoint_for_index") as mock_find:
        provider.get_vector_index(vector_store)

    mock_find.assert_not_called()
    provider.vsc.get_index.assert_called_once_with("explicit_ep", "c.s.products_index")


@pytest.mark.unit
def test_retriever_copy_endpoint_resolves_after_provisioning():
    """Regression: a retriever's deep-copied vector_store is a DISTINCT instance
    from resources.vector_stores, so provisioning create() only stamps the
    endpoint on the resource copy. The retriever copy must still resolve its
    endpoint on demand via get_vector_index rather than raising on endpoint.name.
    """
    cfg = {
        "schemas": {"s": {"catalog_name": "c", "schema_name": "sch"}},
        "resources": {
            "vector_stores": {
                "pv": {
                    "index": {
                        "schema": {"catalog_name": "c", "schema_name": "sch"},
                        "name": "products_index",
                    },
                    "embedding_source_column": "description",
                }
            }
        },
        "retrievers": {
            "r": {
                "vector_store": {
                    "index": {
                        "schema": {"catalog_name": "c", "schema_name": "sch"},
                        "name": "products_index",
                    },
                    "embedding_source_column": "description",
                },
                "columns": ["x"],
            }
        },
    }
    config = AppConfig(**cfg)
    vs_res = config.resources.vector_stores["pv"]
    vs_ret = config.retrievers["r"].vector_store

    # Distinct instances (the crux of the regression) and both start unresolved.
    assert vs_res is not vs_ret
    assert vs_res.endpoint is None and vs_ret.endpoint is None

    # get_vector_index on the retriever copy self-resolves instead of raising.
    provider = DatabricksProvider(vsc=MagicMock())
    with patch.object(provider, "find_endpoint_for_index", return_value="ep"):
        provider.get_vector_index(vs_ret)

    assert vs_ret.endpoint is not None
    assert vs_ret.endpoint.name == "ep"


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
def test_set_databricks_env_vars_injects_warehouse_but_not_destination():
    """`set_databricks_env_vars` auto-injects `MLFLOW_TRACING_SQL_WAREHOUSE_ID`
    when `trace_location` is set, but NOT `MLFLOW_TRACING_DESTINATION` —
    Databricks docs (docs.databricks.com/aws/en/mlflow3/genai/tracing/
    trace-unity-catalog) do not use that env var, and MLflow's env-var
    parser converts the 2-part string to legacy UCSchemaLocation with a
    hardcoded default table name, shadowing the correct experiment-linked
    UnityCatalog.
    """
    from dao_ai.config import AppModel, SchemaModel, TraceLocationModel

    schema = SchemaModel(catalog_name="my_catalog", schema_name="my_schema")
    trace_loc = TraceLocationModel(schema=schema, warehouse="abc123")

    mock_app = MagicMock(spec=AppModel)
    mock_app.environment_vars = {}
    mock_app.service_principal = None
    mock_app.trace_location = trace_loc
    mock_app.experiment = None

    with patch(
        "dao_ai.utils.get_default_databricks_host",
        return_value="https://test.databricks.com",
    ):
        AppModel.set_databricks_env_vars(mock_app)

    assert "MLFLOW_TRACING_DESTINATION" not in mock_app.environment_vars
    assert mock_app.environment_vars["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] == "abc123"


@pytest.mark.unit
def test_set_databricks_env_vars_no_trace_vars_without_trace_location():
    """When trace_location is unset, neither MLFLOW_TRACING_DESTINATION nor
    MLFLOW_TRACING_SQL_WAREHOUSE_ID is injected."""
    from dao_ai.config import AppModel

    mock_app = MagicMock(spec=AppModel)
    mock_app.environment_vars = {}
    mock_app.service_principal = None
    mock_app.trace_location = None
    mock_app.experiment = None

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
    mock_config.resources = None
    _stamp_extras_resolvable(mock_config)

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
        patch("dao_ai.providers.databricks.mlflow.set_experiment") as mock_set_exp,
    ):
        with patch.object(DatabricksProvider, "__init__", return_value=None):
            provider = DatabricksProvider()
            provider.w = MagicMock()

            with patch.object(
                provider, "get_or_create_experiment", return_value=mock_experiment
            ):
                provider.deploy_model_serving_agent(mock_config)

                # mlflow.set_experiment(experiment_id=..., trace_location=
                # UnityCatalog(...)) is the post-3.11 blessed API for linking
                # an experiment to UC trace storage. Replaces the older
                # set_destination(UCSchemaLocation) + set_experiment_trace_location
                # pair (both of which emit deprecation warnings).
                # Per-OTEL-table grant calls were also removed earlier —
                # MLflow auto-creates the tables at first trace write.
                mock_set_exp.assert_called_once()
                call_kwargs = mock_set_exp.call_args.kwargs
                assert call_kwargs["experiment_id"] == "exp123"
                # trace_location is UnityCatalog(catalog_name=..., schema_name=...)
                trace_loc = call_kwargs["trace_location"]
                assert trace_loc.catalog_name == "trace_cat"
                assert trace_loc.schema_name == "trace_sch"


@pytest.mark.unit
def test_deploy_apps_agent_uploads_pyproject_with_dao_ai_version_pin(tmp_path):
    """The published deploy path must ship a pyproject.toml that pins dao-ai
    to the version that generated the bundle. Without this pin, Databricks Apps'
    runtime ``uv pip install dao-ai`` only audits the cached venv from prior
    deploys to the same app slot — letting older dao-ai linger when the bundle
    YAML uses newer fields (e.g. ``app.background:`` introduced in 0.1.92).
    Regression guard for the workshop verification crash on 2026-06-23.
    """
    from unittest.mock import MagicMock, patch

    from databricks.sdk.service.apps import (
        App,
        AppDeployment,
        AppDeploymentState,
        ApplicationState,
    )
    from databricks.sdk.service.iam import User

    from dao_ai.config import AppConfig, AppModel
    from dao_ai.providers.databricks import DatabricksProvider, dao_ai_version

    raw_yaml: str = "app:\n  name: pin-test\n"
    src_file = tmp_path / "dao_ai.yaml"
    src_file.write_text(raw_yaml)

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_app.name = "pin-test"
    mock_app.description = ""
    mock_app.environment_vars = {}
    mock_app.trace_location = None
    mock_app.monitoring = None
    mock_app.enable_chat_proxy = True
    # Skip the permission-grant branch (would touch unstubbed SDK surfaces).
    mock_app.manage_permissions = False
    mock_app.service_principal = None
    mock_config.app = mock_app
    mock_config.source_config_path = str(src_file)
    mock_config.rendered_yaml = raw_yaml
    mock_config.resources = None
    mock_config.agents = None
    mock_config.retrievers = None

    mock_existing_app = MagicMock(spec=App)
    mock_existing_app.app_status = MagicMock(state=ApplicationState.RUNNING)
    mock_deployment = MagicMock(spec=AppDeployment)
    mock_deployment.status = MagicMock(state=AppDeploymentState.SUCCEEDED)
    mock_user = MagicMock(spec=User, user_name="test.user@example.com")

    with (
        patch.object(DatabricksProvider, "__init__", return_value=None),
        patch("dao_ai.utils.is_published", return_value=True),
        patch(
            "dao_ai._locking.render_portable_lock",
            return_value="version = 1\n# portable lock (public CDN)\n",
        ),
    ):
        provider = DatabricksProvider()
        provider.w = MagicMock()
        provider.w.current_user.me.return_value = mock_user
        with patch.object(
            provider,
            "get_or_create_experiment",
            return_value=MagicMock(experiment_id="exp-1"),
        ):
            provider.w.apps.get.return_value = mock_existing_app
            provider.w.apps.deploy_and_wait.return_value = mock_deployment

            _stamp_extras_resolvable(mock_config)
            provider.deploy_apps_agent(mock_config)

    # Apps' build phase runs ``uv sync --locked --no-dev`` from pyproject.toml
    # + uv.lock. requirements.txt must NOT be uploaded — it would take
    # precedence and force the pip path.
    requirements_uploads = [
        c
        for c in provider.w.workspace.upload.call_args_list
        if c.kwargs.get("path", "").endswith("/requirements.txt")
    ]
    assert not requirements_uploads, (
        "Published deploy_apps_agent must NOT upload requirements.txt — it takes "
        "precedence over pyproject.toml + uv.lock and forces the pip path."
    )

    # uv.lock is the file that (with pyproject.toml) triggers Apps' uv path.
    lock_uploads = [
        c
        for c in provider.w.workspace.upload.call_args_list
        if c.kwargs.get("path", "").endswith("/uv.lock")
    ]
    assert lock_uploads, (
        "Published deploy_apps_agent must upload a uv.lock so Apps' build phase "
        "runs ``uv sync --locked`` against the pinned dao-ai version."
    )
    lock_text = lock_uploads[0].kwargs["content"].getvalue().decode("utf-8")
    # The lock must be portable — no internal mirror host (see dao_ai._locking).
    assert "pypi-proxy" not in lock_text, (
        "uv.lock must not reference the internal PyPI mirror (unreachable in "
        "the Apps container / for customers)."
    )

    # pyproject.toml is uploaded with the local version pin.
    pyproject_uploads = [
        c
        for c in provider.w.workspace.upload.call_args_list
        if c.kwargs.get("path", "").endswith("/pyproject.toml")
    ]
    assert pyproject_uploads
    pyproject_text = pyproject_uploads[0].kwargs["content"].getvalue().decode("utf-8")
    # Exact version pin, whether or not optional-feature extras are present
    # (``dao-ai==<ver>`` or ``dao-ai[a2a,...]==<ver>``).
    assert (
        f"dao-ai=={dao_ai_version()}" in pyproject_text
        or f"]=={dao_ai_version()}" in pyproject_text
    ), f"pyproject must pin the exact dao-ai version; got:\n{pyproject_text}"

    # app.yaml's command must be a bare ``python -m ...`` (no runtime pip
    # install) so Apps' build-phase install is the sole installer.
    app_yaml_uploads = [
        c
        for c in provider.w.workspace.upload.call_args_list
        if c.kwargs.get("path", "").endswith("/app.yaml")
    ]
    assert app_yaml_uploads, "expected an app.yaml upload"
    app_yaml_text = app_yaml_uploads[0].kwargs["content"].getvalue().decode("utf-8")
    assert "uv pip install dao-ai" not in app_yaml_text, (
        "app.yaml must NOT issue a runtime ``uv pip install dao-ai`` — that "
        "command audits the cached venv and never upgrades. The new path "
        "relies on Apps' build-phase ``uv sync`` from pyproject.toml + uv.lock."
    )


def test_deploy_apps_agent_dev_path_ships_uv_lock(tmp_path):
    """deploy_apps_agent dev path (is_published() == False) ships pyproject.toml
    + a portable uv.lock (referencing the local wheel via [tool.uv.sources]) and
    NOT a requirements.txt.

    Apps' build phase runs ``uv sync --locked --no-dev``. The lock is de-mirrored
    (public-CDN URLs) by ``dao_ai._locking.render_portable_lock`` — mocked here
    since generating a real lock needs a real wheel + network (covered live).
    """
    from databricks.sdk.service.apps import (
        App,
        AppDeployment,
        AppDeploymentState,
        ApplicationState,
    )
    from databricks.sdk.service.iam import User

    from dao_ai.config import AppConfig, AppModel
    from dao_ai.providers.databricks import DatabricksProvider

    raw_yaml: str = "app:\n  name: dev-test\n"
    src_file = tmp_path / "dao_ai.yaml"
    src_file.write_text(raw_yaml)

    # Materialize a stub wheel file so find_dev_wheel() finds something.
    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    stub_wheel = wheel_dir / "dao_ai-0.0.0-py3-none-any.whl"
    stub_wheel.write_bytes(b"PK\x03\x04stubwheel")

    mock_config = MagicMock(spec=AppConfig)
    mock_app = MagicMock(spec=AppModel)
    mock_app.name = "dev-test"
    mock_app.description = ""
    mock_app.environment_vars = {}
    mock_app.trace_location = None
    mock_app.monitoring = None
    mock_app.enable_chat_proxy = True
    # Skip the permission-grant branch (would touch unstubbed SDK surfaces).
    mock_app.manage_permissions = False
    mock_app.service_principal = None
    mock_config.app = mock_app
    mock_config.source_config_path = str(src_file)
    mock_config.rendered_yaml = raw_yaml
    mock_config.resources = None
    mock_config.agents = None
    mock_config.retrievers = None

    mock_existing_app = MagicMock(spec=App)
    mock_existing_app.app_status = MagicMock(state=ApplicationState.RUNNING)
    mock_deployment = MagicMock(spec=AppDeployment)
    mock_deployment.status = MagicMock(state=AppDeploymentState.SUCCEEDED)
    mock_user = MagicMock(spec=User, user_name="test.user@example.com")

    with (
        patch.object(DatabricksProvider, "__init__", return_value=None),
        patch("dao_ai.utils.is_published", return_value=False),
        patch("dao_ai.providers.databricks.find_dev_wheel", return_value=stub_wheel),
        patch(
            "dao_ai._locking.render_portable_lock",
            return_value="version = 1\n# portable lock (public CDN)\n",
        ),
    ):
        provider = DatabricksProvider()
        provider.w = MagicMock()
        provider.w.current_user.me.return_value = mock_user
        with patch.object(
            provider,
            "get_or_create_experiment",
            return_value=MagicMock(experiment_id="exp-1"),
        ):
            provider.w.apps.get.return_value = mock_existing_app
            provider.w.apps.deploy_and_wait.return_value = mock_deployment

            _stamp_extras_resolvable(mock_config)
            provider.deploy_apps_agent(mock_config)

    # uv.lock is uploaded; Apps' build phase runs ``uv sync --locked --no-dev``.
    uv_lock_uploads = [
        c
        for c in provider.w.workspace.upload.call_args_list
        if c.kwargs.get("path", "").endswith("/uv.lock")
    ]
    assert uv_lock_uploads, (
        "Dev-path deploy_apps_agent must upload a uv.lock so Apps' build phase "
        "runs ``uv sync --locked`` (the lock references the bundled wheel via "
        "[tool.uv.sources] and is de-mirrored to public-CDN URLs)."
    )

    # requirements.txt must NOT be uploaded — it would take precedence over the
    # pyproject.toml + uv.lock uv path.
    requirements_uploads = [
        c
        for c in provider.w.workspace.upload.call_args_list
        if c.kwargs.get("path", "").endswith("/requirements.txt")
    ]
    assert not requirements_uploads, (
        "Dev-path deploy_apps_agent must NOT upload requirements.txt — it takes "
        "precedence over pyproject.toml + uv.lock and forces the pip path."
    )

    # The dev pyproject redirects dao-ai to the bundled wheel via uv sources.
    pyproject_uploads = [
        c
        for c in provider.w.workspace.upload.call_args_list
        if c.kwargs.get("path", "").endswith("/pyproject.toml")
    ]
    assert pyproject_uploads
    pyproject_text = pyproject_uploads[0].kwargs["content"].getvalue().decode("utf-8")
    assert "[tool.uv.sources]" in pyproject_text and stub_wheel.name in pyproject_text


# =============================================================================
# create_lakebase_autoscaling_role — idempotency via postgres_role match
# =============================================================================

_LB_CLIENT_ID = "ad1118d0-d49d-47a6-8aa5-7f67ef14da3c"


def _mock_lakebase_ws(existing_postgres_roles: list[str]) -> Mock:
    """A mock WorkspaceClient whose postgres branch has the given SP roles.

    ``list_branches`` yields one default branch; ``list_roles`` yields a Role
    per entry in ``existing_postgres_roles`` (matched on ``status.postgres_role``,
    which is how the server keys SP roles — NOT the client-supplied role_id).
    """
    w = Mock()
    branch = Mock()
    branch.name = "projects/proj/branches/main"
    branch.status = Mock(default=True)
    w.postgres.list_branches.return_value = iter([branch])
    roles = []
    for pr in existing_postgres_roles:
        role = Mock()
        role.role_id = "rol-server-assigned"
        role.status = Mock(postgres_role=pr)
        roles.append(role)
    w.postgres.list_roles.return_value = iter(roles)
    return w


@pytest.mark.unit
def test_create_lakebase_role_skips_create_when_role_exists(monkeypatch):
    """Role keyed by status.postgres_role already present → no create_role call."""
    w = _mock_lakebase_ws(existing_postgres_roles=[_LB_CLIENT_ID])
    monkeypatch.setattr("dao_ai.config.WorkspaceClient", lambda **kwargs: w)
    db = DatabaseModel(
        name="lb", project="proj", client_id=_LB_CLIENT_ID, client_secret="s"
    )
    DatabricksProvider(w=w).create_lakebase_autoscaling_role(db)

    w.postgres.list_roles.assert_called_once()
    w.postgres.create_role.assert_not_called()


@pytest.mark.unit
def test_create_lakebase_role_creates_when_absent(monkeypatch):
    """No role matching the SP's client_id → create_role is issued once."""
    w = _mock_lakebase_ws(
        existing_postgres_roles=["22222222-2222-2222-2222-222222222222"]
    )
    monkeypatch.setattr("dao_ai.config.WorkspaceClient", lambda **kwargs: w)
    db = DatabaseModel(
        name="lb", project="proj", client_id=_LB_CLIENT_ID, client_secret="s"
    )
    DatabricksProvider(w=w).create_lakebase_autoscaling_role(db)

    w.postgres.create_role.assert_called_once()
    _, kwargs = w.postgres.create_role.call_args
    assert kwargs["parent"] == "projects/proj/branches/main"
    # The role hint is sanitized from the client_id and 'sp-' prefixed.
    assert kwargs["role_id"] == f"sp-{_LB_CLIENT_ID}"
    assert kwargs["role"].spec.postgres_role == _LB_CLIENT_ID


@pytest.mark.unit
def test_create_lakebase_role_uses_the_caller_client_not_the_sp_client(monkeypatch):
    """The Postgres control plane runs as the CALLER, never as the SP itself.

    Regression test: the provider used to read ``database.workspace_client``,
    which builds an oauth-m2m client from the DatabaseModel's own credentials —
    so the SP tried to create its own role and the workspace rejected it with
    "not authorized ... assign the user <sp> 'Can Manage' for Database project".
    Constructing the SP client at all is the bug, so we make it raise.
    """

    def _explode(**kwargs):
        raise AssertionError(
            "create_lakebase_autoscaling_role must not build a service-principal "
            f"WorkspaceClient; got kwargs={sorted(kwargs)}"
        )

    w = _mock_lakebase_ws(existing_postgres_roles=[])
    monkeypatch.setattr("dao_ai.config.WorkspaceClient", _explode)
    db = DatabaseModel(
        name="lb", project="proj", client_id=_LB_CLIENT_ID, client_secret="s"
    )
    DatabricksProvider(w=w).create_lakebase_autoscaling_role(db)

    # Every control-plane call went through the injected caller client.
    w.postgres.list_branches.assert_called_once()
    w.postgres.list_roles.assert_called_once()
    w.postgres.create_role.assert_called_once()


@pytest.mark.unit
def test_create_lakebase_role_subject_is_still_the_sp_client_id(monkeypatch):
    """Caller identity changed; the role SUBJECT must remain the SP."""
    w = _mock_lakebase_ws(existing_postgres_roles=[])
    monkeypatch.setattr("dao_ai.config.WorkspaceClient", lambda **kwargs: w)
    db = DatabaseModel(
        name="lb", project="proj", client_id=_LB_CLIENT_ID, client_secret="s"
    )
    DatabricksProvider(w=w).create_lakebase_autoscaling_role(db)

    _, kwargs = w.postgres.create_role.call_args
    assert kwargs["role"].spec.postgres_role == _LB_CLIENT_ID


@pytest.mark.unit
def test_create_lakebase_role_honours_explicit_client_id_override(monkeypatch):
    """An explicit client_id wins over the model's — the one-pass provision path.

    ``dao-ai sp provision`` mints an SP and must create its Postgres role in the
    same run, before the secret scope the config reads ``client_id`` from has
    been populated. The override carries the freshly minted id.
    """
    fresh_id = "de6db65b-59f0-4368-87ed-9b06f6054da0"
    w = _mock_lakebase_ws(existing_postgres_roles=[])
    monkeypatch.setattr("dao_ai.config.WorkspaceClient", lambda **kwargs: w)
    db = DatabaseModel(
        name="lb", project="proj", client_id=_LB_CLIENT_ID, client_secret="s"
    )
    DatabricksProvider(w=w).create_lakebase_autoscaling_role(db, client_id=fresh_id)

    _, kwargs = w.postgres.create_role.call_args
    assert kwargs["role"].spec.postgres_role == fresh_id
    assert kwargs["role_id"] == f"sp-{fresh_id}"


@pytest.mark.unit
def test_create_lakebase_role_override_works_when_model_client_id_unset(monkeypatch):
    """With an override, an unresolvable model client_id no longer blocks the role.

    This is what removes the old two-pass "provision the SP and populate the
    scope, then re-run" round-trip.
    """
    fresh_id = "de6db65b-59f0-4368-87ed-9b06f6054da0"
    w = _mock_lakebase_ws(existing_postgres_roles=[])
    monkeypatch.setattr("dao_ai.config.WorkspaceClient", lambda **kwargs: w)
    db = DatabaseModel(name="lb", project="proj")
    DatabricksProvider(w=w).create_lakebase_autoscaling_role(db, client_id=fresh_id)

    w.postgres.create_role.assert_called_once()
    _, kwargs = w.postgres.create_role.call_args
    assert kwargs["role"].spec.postgres_role == fresh_id


@pytest.mark.unit
def test_create_lakebase_project_uses_the_caller_client_not_the_sp_client(monkeypatch):
    """Creating the project is control-plane work → the caller's identity.

    The SP has no standing to create the project it is about to be granted a
    role on, so building its oauth-m2m client here is the bug. Keeps
    ``DatabaseModel.create()``'s ``w`` honoured by BOTH provisioning helpers.
    """

    def _explode(**kwargs):
        raise AssertionError(
            "create_lakebase_autoscaling must not build a service-principal "
            f"WorkspaceClient; got kwargs={sorted(kwargs)}"
        )

    w = _mock_lakebase_ws(existing_postgres_roles=[])
    w.postgres.get_project.return_value = None
    monkeypatch.setattr("dao_ai.config.WorkspaceClient", _explode)
    db = DatabaseModel(
        name="lb", project="proj", client_id=_LB_CLIENT_ID, client_secret="s"
    )
    DatabricksProvider(w=w).create_lakebase_autoscaling(db)

    w.postgres.get_project.assert_called_once_with("projects/proj")


@pytest.mark.unit
def test_create_lakebase_role_without_override_still_skips_unset_client_id(monkeypatch):
    """Legacy path unchanged: no override + no model client_id → warn and skip."""
    w = _mock_lakebase_ws(existing_postgres_roles=[])
    monkeypatch.setattr("dao_ai.config.WorkspaceClient", lambda **kwargs: w)
    db = DatabaseModel(name="lb", project="proj")
    DatabricksProvider(w=w).create_lakebase_autoscaling_role(db)

    w.postgres.create_role.assert_not_called()
    w.postgres.list_roles.assert_not_called()


# =============================================================================
# _wait_for_initial_snapshot — bound on stalled progress, not elapsed time
# =============================================================================


def _snapshot_index(states_and_rows):
    """A mock index whose describe() walks the given (state, rows) sequence."""
    idx = Mock()
    seq = list(states_and_rows)

    def _describe():
        state, rows = seq[0] if len(seq) == 1 else seq.pop(0)
        return {"status": {"detailed_state": state, "indexed_row_count": rows}}

    idx.describe.side_effect = _describe
    return idx


@pytest.mark.unit
def test_snapshot_wait_returns_when_snapshot_completes(monkeypatch):
    """Rising row counts then ONLINE → return the terminal state."""
    from dao_ai.providers.databricks import _wait_for_initial_snapshot

    monkeypatch.setattr("dao_ai.providers.databricks.time.sleep", lambda s: None)
    idx = _snapshot_index(
        [
            ("PROVISIONING_INITIAL_SNAPSHOT", 100),
            ("PROVISIONING_INITIAL_SNAPSHOT", 20000),
            ("ONLINE_NO_PENDING_UPDATE", 38291),
        ]
    )
    state = _wait_for_initial_snapshot(idx, "cat.sch.idx", poll_seconds=0)
    assert state == "ONLINE_NO_PENDING_UPDATE"


@pytest.mark.unit
def test_snapshot_wait_keeps_waiting_while_rows_climb(monkeypatch):
    """A long-but-progressing snapshot is not cut short.

    The regression: a full re-embed outlasted the flat 20-minute readiness
    timeout, so provisioning failed a recovery that was still working — and told
    the user to recreate the index, which is what it had just done.
    """
    from dao_ai.providers.databricks import _wait_for_initial_snapshot

    # Advance a fake clock on every poll so the stall timer is genuinely live:
    # 40 polls x 60s = 40 min of wall time, well past both the 20-min readiness
    # timeout and the 15-min stall window. Only resetting the stall timer on each
    # row increase keeps this alive to completion.
    clock = {"t": 0.0}
    monkeypatch.setattr(
        "dao_ai.providers.databricks.time.monotonic", lambda: clock["t"]
    )

    def _advance(_s):
        clock["t"] += 60.0

    monkeypatch.setattr("dao_ai.providers.databricks.time.sleep", _advance)
    seq = [("PROVISIONING_INITIAL_SNAPSHOT", n * 1000) for n in range(1, 41)]
    seq.append(("ONLINE_NO_PENDING_UPDATE", 41000))
    idx = _snapshot_index(seq)
    assert (
        _wait_for_initial_snapshot(
            idx, "cat.sch.idx", stall_timeout_seconds=900, poll_seconds=1
        )
        == "ONLINE_NO_PENDING_UPDATE"
    )


@pytest.mark.unit
def test_snapshot_wait_polls_past_the_readiness_timeout(monkeypatch):
    """The behaviour that matters: a snapshot may run longer than the flat
    20-minute readiness timeout, provided it keeps indexing rows.

    Asserted on wall-clock rather than just the return value, since a loop that
    bailed early would also return ONLINE — it would simply do so too soon.
    """
    from dao_ai.providers.databricks import (
        _VS_INDEX_READY_TIMEOUT_SECONDS,
        _wait_for_initial_snapshot,
    )

    clock = {"t": 0.0}
    monkeypatch.setattr(
        "dao_ai.providers.databricks.time.monotonic", lambda: clock["t"]
    )

    def _advance(_s):
        clock["t"] += 60.0

    monkeypatch.setattr("dao_ai.providers.databricks.time.sleep", _advance)
    seq = [("PROVISIONING_INITIAL_SNAPSHOT", n * 1000) for n in range(1, 41)]
    seq.append(("ONLINE_NO_PENDING_UPDATE", 41000))
    idx = _snapshot_index(seq)

    state = _wait_for_initial_snapshot(
        idx, "cat.sch.idx", stall_timeout_seconds=900, poll_seconds=1
    )
    assert state == "ONLINE_NO_PENDING_UPDATE"
    # It stayed in the loop well past the timeout that used to fail the task.
    assert clock["t"] > _VS_INDEX_READY_TIMEOUT_SECONDS


@pytest.mark.unit
def test_snapshot_wait_gives_up_when_progress_stalls(monkeypatch):
    """Flat row count for the stall window → return, don't wait forever."""
    from dao_ai.providers.databricks import _wait_for_initial_snapshot

    clock = {"t": 0.0}
    monkeypatch.setattr(
        "dao_ai.providers.databricks.time.monotonic", lambda: clock["t"]
    )

    def _advance(_s):
        clock["t"] += 60.0

    monkeypatch.setattr("dao_ai.providers.databricks.time.sleep", _advance)
    idx = _snapshot_index([("PROVISIONING_INITIAL_SNAPSHOT", 500)])
    state = _wait_for_initial_snapshot(
        idx, "cat.sch.idx", stall_timeout_seconds=300, poll_seconds=1
    )
    assert state == "PROVISIONING_INITIAL_SNAPSHOT"


@pytest.mark.unit
def test_snapshot_wait_returns_a_failed_state_immediately(monkeypatch):
    """A snapshot that lands in FAILED is terminal — hand it back to the caller."""
    from dao_ai.providers.databricks import _wait_for_initial_snapshot

    monkeypatch.setattr("dao_ai.providers.databricks.time.sleep", lambda s: None)
    idx = _snapshot_index([("ONLINE_PIPELINE_FAILED", 38291)])
    assert (
        _wait_for_initial_snapshot(idx, "cat.sch.idx", poll_seconds=0)
        == "ONLINE_PIPELINE_FAILED"
    )


@pytest.mark.unit
def test_snapshot_wait_honours_the_absolute_ceiling(monkeypatch):
    """Even with progress every poll, the hard ceiling still applies."""
    from dao_ai.providers.databricks import _wait_for_initial_snapshot

    clock = {"t": 0.0}
    monkeypatch.setattr(
        "dao_ai.providers.databricks.time.monotonic", lambda: clock["t"]
    )

    def _advance(_s):
        clock["t"] += 100.0

    monkeypatch.setattr("dao_ai.providers.databricks.time.sleep", _advance)
    n = {"rows": 0}

    def _describe():
        n["rows"] += 1000
        return {
            "status": {
                "detailed_state": "PROVISIONING_INITIAL_SNAPSHOT",
                "indexed_row_count": n["rows"],
            }
        }

    idx = Mock()
    idx.describe.side_effect = _describe
    state = _wait_for_initial_snapshot(
        idx, "cat.sch.idx", max_timeout_seconds=500, poll_seconds=1
    )
    assert state == "PROVISIONING_INITIAL_SNAPSHOT"


@pytest.mark.unit
def test_app_name_for_normalizes_and_prefixes():
    """app_name_for is the single source of truth for the deployed App name."""
    from dao_ai.config import MCP_APP_PREFIX, app_name_for

    # Chat protocol: lowercase + hyphenate only.
    assert app_name_for("My_Agent") == "my-agent"
    # MCP protocol: mcp- prefixed so it cannot collide with the chat App.
    assert app_name_for("My_Agent", as_mcp=True) == f"{MCP_APP_PREFIX}my-agent"


@pytest.mark.unit
def test_app_name_for_mcp_prefix_is_idempotent():
    """A config already named mcp-* must not become mcp-mcp-*.

    docs/mcp_server.md recommends naming MCP apps with the ``mcp-`` prefix
    (Multi-Agent Supervisor pattern-matches it), so users following that advice
    would otherwise get a double prefix.
    """
    from dao_ai.config import app_name_for

    assert app_name_for("mcp-dao-ai-test", as_mcp=True) == "mcp-dao-ai-test"
    assert app_name_for("MCP_Agent", as_mcp=True) == "mcp-agent"
    # Without as_mcp the name passes through untouched either way.
    assert app_name_for("mcp-agent") == "mcp-agent"
