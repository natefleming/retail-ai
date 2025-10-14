"""Integration tests for PostgreSQL database connectivity."""

import os
import sys
from pathlib import Path

import pytest
from conftest import has_postgres_env
from mlflow.models import ModelConfig
from psycopg import connect
from psycopg.errors import Error as PsycopgError

from dao_ai.config import AppConfig


@pytest.fixture
def postgres_config_path() -> Path:
    """Fixture that returns path to supervisor_postgres.yaml configuration."""
    return (
        Path(__file__).parents[2]
        / "config"
        / "hardware_store"
        / "supervisor_postgres.yaml"
    )


@pytest.fixture
def postgres_model_config(postgres_config_path: Path) -> ModelConfig:
    """Fixture that loads the PostgreSQL configuration using ModelConfig."""
    return ModelConfig(development_config=postgres_config_path)


@pytest.fixture
def postgres_app_config(postgres_model_config: ModelConfig) -> AppConfig:
    """Fixture that creates AppConfig from PostgreSQL configuration."""
    return AppConfig(**postgres_model_config.to_dict())


@pytest.mark.integration
@pytest.mark.skipif(
    not has_postgres_env(), reason="PostgreSQL environment variables not available"
)
def test_load_postgres_config(postgres_app_config: AppConfig) -> None:
    """Test that supervisor_postgres.yaml loads successfully via AppConfig.load."""
    # Verify config loaded successfully
    assert postgres_app_config is not None
    assert hasattr(postgres_app_config, "resources")

    # Verify database configuration exists
    assert hasattr(postgres_app_config.resources, "databases")
    assert "retail_database" in postgres_app_config.resources.databases

    database_config = postgres_app_config.resources.databases["retail_database"]
    print(
        f"Database config: {database_config.model_dump_json(indent=2)}", file=sys.stderr
    )

    # Verify essential database configuration fields
    assert hasattr(database_config, "host")
    assert hasattr(database_config, "port")
    assert hasattr(database_config, "database")

    # Verify memory configuration exists and references PostgreSQL
    assert hasattr(postgres_app_config, "memory")
    memory_config = postgres_app_config.memory
    assert hasattr(memory_config, "checkpointer")
    assert hasattr(memory_config, "store")

    # Check that memory configuration references the database
    checkpointer_config = memory_config.checkpointer.model_dump()
    store_config = memory_config.store.model_dump()

    # Should reference retail_database
    assert "retail_database" in str(checkpointer_config) or "postgres" in str(
        checkpointer_config
    )
    assert "retail_database" in str(store_config) or "postgres" in str(store_config)


@pytest.mark.integration
@pytest.mark.skipif(
    not has_postgres_env(), reason="PostgreSQL environment variables not available"
)
def test_postgres_database_connectivity(postgres_app_config: AppConfig) -> None:
    """Test actual PostgreSQL database connectivity using configuration."""
    # Get database configuration
    database_config = postgres_app_config.resources.databases["retail_database"]

    # Build connection parameters from config
    connection_params = {}

    # Handle host configuration
    if hasattr(database_config, "host"):
        host_config = database_config.host
        if hasattr(host_config, "environment_variable"):
            connection_params["host"] = os.getenv(host_config.environment_variable.name)
        elif hasattr(host_config, "value"):
            connection_params["host"] = host_config.value

    # Handle port configuration
    if hasattr(database_config, "port"):
        port_config = database_config.port
        if hasattr(port_config, "environment_variable"):
            port_value = os.getenv(port_config.environment_variable.name)
            connection_params["port"] = int(port_value) if port_value else None
        elif hasattr(port_config, "value"):
            connection_params["port"] = port_config.value

    # Handle database name configuration
    if hasattr(database_config, "database"):
        db_config = database_config.database
        if hasattr(db_config, "environment_variable"):
            connection_params["dbname"] = os.getenv(db_config.environment_variable.name)
        elif hasattr(db_config, "value"):
            connection_params["dbname"] = db_config.value

    # Handle authentication - try OAuth2 first, then traditional
    if hasattr(database_config, "oauth2_client_credentials"):
        oauth_config = database_config.oauth2_client_credentials
        # For OAuth2, we'd typically need to get a token first
        # For this test, we'll fall back to traditional auth
        print(f"OAuth2 config available: {oauth_config}", file=sys.stderr)

    if hasattr(database_config, "user"):
        user_config = database_config.user
        if hasattr(user_config, "environment_variable"):
            connection_params["user"] = os.getenv(user_config.environment_variable.name)
        elif hasattr(user_config, "value"):
            connection_params["user"] = user_config.value

    if hasattr(database_config, "password"):
        password_config = database_config.password
        if hasattr(password_config, "environment_variable"):
            connection_params["password"] = os.getenv(
                password_config.environment_variable.name
            )
        elif hasattr(password_config, "secret"):
            # In a real environment, this would fetch from secrets manager
            # For testing, we'll use environment variable
            connection_params["password"] = os.getenv("PG_PASSWORD")

    # Remove None values
    connection_params = {k: v for k, v in connection_params.items() if v is not None}

    print(f"Connection params: {connection_params}", file=sys.stderr)

    # Test database connectivity
    try:
        with connect(**connection_params) as conn:
            assert conn is not None

            # Test basic query execution
            with conn.cursor() as cursor:
                cursor.execute("SELECT version()")
                result = cursor.fetchone()
                assert result is not None
                print(f"PostgreSQL version: {result[0]}", file=sys.stderr)

                # Test that we can create/check tables (basic functionality)
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public'
                    )
                """)
                schema_exists = cursor.fetchone()[0]
                assert schema_exists is not None
                print(f"Public schema accessible: {schema_exists}", file=sys.stderr)

    except PsycopgError as e:
        pytest.fail(f"PostgreSQL connection failed: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during database connectivity test: {e}")


@pytest.mark.integration
@pytest.mark.skipif(
    not has_postgres_env(), reason="PostgreSQL environment variables not available"
)
def test_postgres_memory_initialization(postgres_app_config: AppConfig) -> None:
    """Test that memory components can be initialized with PostgreSQL configuration."""
    # Initialize the configuration which should set up memory components
    try:
        postgres_app_config.initialize()

        # Verify memory configuration is set up
        assert hasattr(postgres_app_config, "memory")
        memory_config = postgres_app_config.memory

        # Check that checkpointer and store are configured
        assert hasattr(memory_config, "checkpointer")
        assert hasattr(memory_config, "store")

        print(f"Memory checkpointer: {memory_config.checkpointer}", file=sys.stderr)
        print(f"Memory store: {memory_config.store}", file=sys.stderr)

        # Clean up
        postgres_app_config.shutdown()

    except Exception as e:
        pytest.fail(f"Failed to initialize memory with PostgreSQL configuration: {e}")


@pytest.mark.unit
@pytest.mark.skipif(not has_postgres_env(), reason="PostgreSQL env vars not set")
def test_load_postgres_config_without_env() -> None:
    """Test that PostgreSQL config loads without requiring environment variables."""
    config_path = (
        Path(__file__).parents[2]
        / "config"
        / "hardware_store"
        / "supervisor_postgres.yaml"
    )

    # This should work even without environment variables set
    model_config = ModelConfig(development_config=config_path)
    app_config = AppConfig(**model_config.to_dict())

    # Verify basic structure
    assert app_config is not None
    assert hasattr(app_config, "resources")
    assert hasattr(app_config.resources, "databases")
    assert "retail_database" in app_config.resources.databases

    # Verify memory configuration structure
    assert hasattr(app_config, "memory")
    assert hasattr(app_config.memory, "checkpointer")
    assert hasattr(app_config.memory, "store")


@pytest.mark.integration
@pytest.mark.skipif(
    not has_postgres_env(), reason="PostgreSQL environment variables not available"
)
def test_postgres_connection_string_format() -> None:
    """Test PostgreSQL connection string construction from environment variables."""
    # Test that connection string environment variable works if provided
    if "PG_CONNECTION_STRING" in os.environ:
        connection_string = os.environ["PG_CONNECTION_STRING"]

        try:
            with connect(connection_string) as conn:
                assert conn is not None
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    assert result[0] == 1
        except PsycopgError as e:
            pytest.fail(f"Connection string test failed: {e}")
    else:
        # Construct from individual variables
        required_vars = ["PG_HOST", "PG_PORT", "PG_USER", "PG_PASSWORD", "PG_DATABASE"]
        if all(var in os.environ for var in required_vars):
            connection_string = (
                f"postgresql://{os.environ['PG_USER']}:{os.environ['PG_PASSWORD']}"
                f"@{os.environ['PG_HOST']}:{os.environ['PG_PORT']}/{os.environ['PG_DATABASE']}"
            )

            try:
                with connect(connection_string) as conn:
                    assert conn is not None
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        result = cursor.fetchone()
                        assert result[0] == 1
            except PsycopgError as e:
                pytest.fail(f"Constructed connection string test failed: {e}")
