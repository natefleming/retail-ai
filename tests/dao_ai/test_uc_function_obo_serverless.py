"""Test that OBO (On-Behalf-Of) authentication works with UC Functions on serverless.

Databricks UC function execution via DatabricksFunctionClient uses serverless Spark
compute. Integration tests using two distinct identities (user PAT + service principal)
confirmed that serverless honours the WorkspaceClient's identity — current_user()
returns different results depending on which credentials are used.

This means OBO is supported for UC functions: when _create_obo_uc_tool forwards a
user's token via workspace_client_from(context), DatabricksFunctionClient executes
the function as that user on serverless.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from conftest import has_databricks_connect
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import PermissionDenied

from dao_ai.config import (
    FunctionModel,
    FunctionType,
    SchemaModel,
    UnityCatalogFunctionModel,
)
from dao_ai.state import Context


def _has_databricks_auth() -> bool:
    """Check for minimal Databricks auth env vars (HOST + TOKEN)."""
    return bool(os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"))


# ---------------------------------------------------------------------------
# Unit tests: FunctionModel accepts OBO
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFunctionModelAcceptsOBO:
    """FunctionModel must preserve on_behalf_of_user setting.

    Serverless Spark honours the WorkspaceClient identity, so OBO is supported
    for UC functions via DatabricksFunctionClient.
    """

    def test_obo_true_is_preserved(self) -> None:
        func = FunctionModel(
            name="catalog.schema.my_function",
            on_behalf_of_user=True,
        )
        assert func.on_behalf_of_user is True

    def test_obo_true_with_schema_is_preserved(self) -> None:
        schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
        func = FunctionModel(
            schema=schema,
            name="my_function",
            on_behalf_of_user=True,
        )
        assert func.on_behalf_of_user is True

    def test_obo_false_is_accepted(self) -> None:
        func = FunctionModel(
            name="catalog.schema.my_function",
            on_behalf_of_user=False,
        )
        assert func.on_behalf_of_user is False

    def test_default_obo_is_false(self) -> None:
        func = FunctionModel(name="catalog.schema.my_function")
        assert func.on_behalf_of_user is False


# ---------------------------------------------------------------------------
# Unit tests: create_uc_tools routes to OBO path when on_behalf_of_user=True
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateUcToolsOBOPath:
    """Verify that create_uc_tools takes the _create_obo_uc_tool path when
    on_behalf_of_user=True on the FunctionModel resource."""

    def test_create_uc_tools_uses_obo_path(self) -> None:
        """When on_behalf_of_user=True, create_uc_tools should call
        _create_obo_uc_tool instead of the standard UCFunctionToolkit."""
        from dao_ai.tools.unity_catalog import create_uc_tools

        uc_func = UnityCatalogFunctionModel(
            type=FunctionType.UNITY_CATALOG,
            resource=FunctionModel(
                name="catalog.schema.test_func",
                on_behalf_of_user=True,
            ),
        )

        with patch("dao_ai.tools.unity_catalog._create_obo_uc_tool") as mock_obo_tool:
            mock_obo_tool.return_value = MagicMock()

            create_uc_tools(uc_func)

            mock_obo_tool.assert_called_once_with(uc_func)

    def test_create_uc_tools_uses_standard_path_when_obo_false(self) -> None:
        """When on_behalf_of_user=False (default), create_uc_tools should use
        the standard UCFunctionToolkit path."""
        from dao_ai.tools.unity_catalog import create_uc_tools

        uc_func = UnityCatalogFunctionModel(
            type=FunctionType.UNITY_CATALOG,
            resource=FunctionModel(
                name="catalog.schema.test_func",
                on_behalf_of_user=False,
            ),
        )

        with (
            patch("dao_ai.tools.unity_catalog.UCFunctionToolkit") as mock_toolkit_class,
            patch(
                "dao_ai.tools.unity_catalog.DatabricksFunctionClient"
            ) as mock_client_class,
        ):
            mock_toolkit = MagicMock()
            mock_toolkit.tools = []
            mock_toolkit_class.return_value = mock_toolkit
            mock_client_class.return_value = MagicMock()

            create_uc_tools(uc_func)

            mock_toolkit_class.assert_called_once()

    def test_execute_uc_function_delegates_to_client(self) -> None:
        """_execute_uc_function passes parameters through to
        DatabricksFunctionClient.execute_function."""
        from dao_ai.tools.unity_catalog import _execute_uc_function

        mock_func_client = MagicMock()
        mock_func_client.execute_function.return_value = MagicMock(
            error=None, value="result"
        )

        result = _execute_uc_function(
            client=mock_func_client,
            function_name="catalog.schema.test_func",
            query="test",
        )

        mock_func_client.execute_function.assert_called_once_with(
            function_name="catalog.schema.test_func",
            parameters={"query": "test"},
        )
        assert result == "result"


# ---------------------------------------------------------------------------
# Integration tests: OBO with UC functions on serverless (two-identity proof)
# ---------------------------------------------------------------------------

_HARDWARE_STORE_SCHEMA = SchemaModel(
    catalog_name="retail_consumer_goods", schema_name="hardware_store"
)
_FIND_PRODUCT_BY_SKU: str = "retail_consumer_goods.hardware_store.find_product_by_sku"
_CURRENT_USER_FUNCTION: str = "main.dao_ai_test.get_current_user"

# SQL to (re-)create the identity test function:
# CREATE OR REPLACE FUNCTION main.dao_ai_test.get_current_user()
# RETURNS STRING
# LANGUAGE SQL
# COMMENT 'Returns current_user() to test execution identity on serverless'
# RETURN current_user();


def _has_two_identities() -> bool:
    """Check that we have both a user PAT and a service principal."""
    has_user = bool(os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"))
    has_sp = bool(
        os.getenv("RETAIL_AI_DATABRICKS_HOST")
        and os.getenv("RETAIL_AI_DATABRICKS_CLIENT_ID")
        and os.getenv("RETAIL_AI_DATABRICKS_CLIENT_SECRET")
    )
    return has_user and has_sp


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not _has_two_identities(),
    reason=(
        "Requires two distinct identities: "
        "DATABRICKS_HOST/TOKEN (user PAT) + "
        "RETAIL_AI_DATABRICKS_HOST/CLIENT_ID/CLIENT_SECRET (service principal)"
    ),
)
@pytest.mark.skipif(
    not has_databricks_connect(),
    reason="databricks-connect not importable (UC fn serverless execution path)",
)
class TestOBOUCFunctionServerlessIntegration:
    """Verify that serverless Spark honours the WorkspaceClient identity,
    enabling OBO for UC functions via DatabricksFunctionClient.

    Uses two distinct identities — a user PAT and a service principal — to
    confirm that current_user() returns different results depending on the
    WorkspaceClient credentials.

    NOTE: Must be run in isolation due to Spark Connect gRPC session hangs.
    """

    @pytest.fixture(scope="class")
    def user_ws(self) -> WorkspaceClient:
        """WorkspaceClient authenticated with the user's PAT."""
        return WorkspaceClient(
            host=os.environ["DATABRICKS_HOST"],
            token=os.environ["DATABRICKS_TOKEN"],
        )

    @pytest.fixture(scope="class")
    def sp_ws(self) -> WorkspaceClient:
        """WorkspaceClient authenticated with the service principal.

        Uses databricks.sdk.Config directly to avoid the SDK picking up
        ambient DATABRICKS_TOKEN from the environment.
        """
        from databricks.sdk import config as sdk_config

        cfg = sdk_config.Config(
            host=os.environ["RETAIL_AI_DATABRICKS_HOST"],
            client_id=os.environ["RETAIL_AI_DATABRICKS_CLIENT_ID"],
            client_secret=os.environ["RETAIL_AI_DATABRICKS_CLIENT_SECRET"],
            auth_type="oauth-m2m",
        )
        return WorkspaceClient(config=cfg)

    @pytest.fixture(scope="class", autouse=True)
    def _require_functions(self, user_ws: WorkspaceClient) -> None:
        """Skip all tests if the required UC functions don't exist."""
        from databricks.sdk.errors.platform import NotFound

        for fn in [_FIND_PRODUCT_BY_SKU, _CURRENT_USER_FUNCTION]:
            try:
                user_ws.functions.get(name=fn)
            except NotFound:
                pytest.skip(
                    f"UC function '{fn}' not found — skipping integration tests"
                )

    @pytest.fixture(autouse=True)
    def _cleanup_spark(self) -> None:
        """Stop Spark Connect sessions after each test to prevent gRPC hangs."""
        yield
        try:
            from pyspark.sql import SparkSession

            active = SparkSession.getActiveSession()
            if active is not None:
                active.stop()
        except Exception:
            pass

    @pytest.fixture(scope="class")
    def user_identity(self, user_ws: WorkspaceClient) -> str:
        return user_ws.current_user.me().user_name

    @pytest.fixture(scope="class")
    def sp_identity(self, sp_ws: WorkspaceClient) -> str:
        return sp_ws.current_user.me().user_name

    # ------------------------------------------------------------------
    # Precondition
    # ------------------------------------------------------------------

    def test_two_identities_are_distinct(
        self, user_identity: str, sp_identity: str
    ) -> None:
        """Sanity check: the user PAT and SP are different identities."""
        assert user_identity != sp_identity

    # ------------------------------------------------------------------
    # Core: serverless honours WorkspaceClient identity
    # ------------------------------------------------------------------

    def test_serverless_identity_with_user_client(
        self, user_ws: WorkspaceClient, user_identity: str
    ) -> None:
        """current_user() on serverless returns the user's identity."""
        from databricks_langchain import DatabricksFunctionClient

        client = DatabricksFunctionClient(client=user_ws)
        result = client.execute_function(
            function_name=_CURRENT_USER_FUNCTION, parameters={}
        )
        assert result.error is None
        assert result.value.strip() == user_identity

    def test_serverless_identity_with_sp_client(
        self, sp_ws: WorkspaceClient, sp_identity: str
    ) -> None:
        """current_user() on serverless returns the SP's identity."""
        from databricks_langchain import DatabricksFunctionClient

        client = DatabricksFunctionClient(client=sp_ws)
        result = client.execute_function(
            function_name=_CURRENT_USER_FUNCTION, parameters={}
        )
        assert result.error is None
        assert result.value.strip() == sp_identity

    def test_serverless_identity_changes_with_client(
        self,
        user_ws: WorkspaceClient,
        sp_ws: WorkspaceClient,
        user_identity: str,
        sp_identity: str,
    ) -> None:
        """Definitive test: two different WorkspaceClients produce different
        current_user() results on serverless, proving OBO is viable."""
        from databricks_langchain import DatabricksFunctionClient

        user_client = DatabricksFunctionClient(client=user_ws)
        sp_client = DatabricksFunctionClient(client=sp_ws)

        user_result = user_client.execute_function(
            function_name=_CURRENT_USER_FUNCTION, parameters={}
        )
        sp_result = sp_client.execute_function(
            function_name=_CURRENT_USER_FUNCTION, parameters={}
        )

        assert user_result.error is None
        assert sp_result.error is None
        assert user_result.value.strip() == user_identity
        assert sp_result.value.strip() == sp_identity
        assert user_result.value.strip() != sp_result.value.strip()

    # ------------------------------------------------------------------
    # _create_obo_uc_tool end-to-end
    # ------------------------------------------------------------------

    def test_create_obo_uc_tool_with_hardware_store_function(
        self,
        user_identity: str,
    ) -> None:
        """_create_obo_uc_tool with find_product_by_sku executes successfully
        when OBO context provides a user's forwarded token."""
        from dao_ai.tools.unity_catalog import _create_obo_uc_tool

        uc_func = UnityCatalogFunctionModel(
            type=FunctionType.UNITY_CATALOG,
            resource=FunctionModel(
                schema=_HARDWARE_STORE_SCHEMA,
                name="find_product_by_sku",
                on_behalf_of_user=True,
            ),
        )

        tool = _create_obo_uc_tool(uc_func)
        assert tool.name == "find_product_by_sku"

        result = tool.invoke(
            {"sku": ["ABC123"]},
            config={
                "configurable": {
                    "context": Context(
                        user_id=user_identity,
                        headers={
                            "x-forwarded-access-token": os.environ["DATABRICKS_TOKEN"],
                            "x-forwarded-user": user_identity,
                        },
                    ),
                }
            },
        )

        assert result is not None
        assert isinstance(result, str)

    def test_create_obo_uc_tool_identity_via_get_current_user(
        self,
        user_identity: str,
        sp_identity: str,
    ) -> None:
        """_create_obo_uc_tool with get_current_user confirms the function
        executes as the user whose token was forwarded, not the ambient SP."""
        from dao_ai.tools.unity_catalog import _create_obo_uc_tool

        uc_func = UnityCatalogFunctionModel(
            type=FunctionType.UNITY_CATALOG,
            resource=FunctionModel(
                name=_CURRENT_USER_FUNCTION,
                on_behalf_of_user=True,
            ),
        )

        tool = _create_obo_uc_tool(uc_func)

        result = tool.invoke(
            {},
            config={
                "configurable": {
                    "context": Context(
                        user_id=user_identity,
                        headers={
                            "x-forwarded-access-token": os.environ["DATABRICKS_TOKEN"],
                            "x-forwarded-user": user_identity,
                        },
                    ),
                }
            },
        )

        assert result is not None
        obo_tool_identity = result.strip()
        assert obo_tool_identity == user_identity, (
            f"OBO tool returned identity '{obo_tool_identity}' but the forwarded "
            f"token belongs to '{user_identity}'."
        )

    # ------------------------------------------------------------------
    # SQL Warehouse comparison
    # ------------------------------------------------------------------

    def test_sql_warehouse_also_honours_caller_identity(
        self,
        user_ws: WorkspaceClient,
        sp_ws: WorkspaceClient,
        user_identity: str,
        sp_identity: str,
    ) -> None:
        """SQL Warehouse via statement_execution also honours caller identity."""
        warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
        if not warehouse_id:
            pytest.skip("DATABRICKS_WAREHOUSE_ID not set")

        def _run_current_user(ws: WorkspaceClient) -> str:
            try:
                resp = ws.statement_execution.execute_statement(
                    warehouse_id=warehouse_id,
                    statement="SELECT current_user()",
                    wait_timeout="30s",
                )
            except PermissionDenied as e:
                pytest.skip(f"Caller lacks CAN_USE on warehouse {warehouse_id}: {e}")
            assert resp.status and resp.status.state.value == "SUCCEEDED"
            assert resp.result and resp.result.data_array
            return resp.result.data_array[0][0].strip()

        assert _run_current_user(user_ws) == user_identity
        assert _run_current_user(sp_ws) == sp_identity
