"""Tests for the generic REST API tool."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from dao_ai.config import ConnectionModel
from dao_ai.tools.rest_api import create_rest_api_tool
from dao_ai.tools.tracing import ResourceInfo

# ---------------------------------------------------------------------------
# Factory validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFactoryValidation:
    """Tests for mutual exclusivity and required-parameter validation."""

    def test_raises_when_both_connection_and_base_url(self) -> None:
        connection = ConnectionModel(name="my-conn")
        with pytest.raises(ValueError, match="not both"):
            create_rest_api_tool(
                connection=connection,
                base_url="https://example.com",
            )

    def test_raises_when_neither_connection_nor_base_url(self) -> None:
        with pytest.raises(ValueError, match="Provide either"):
            create_rest_api_tool()

    def test_connection_as_dict(self) -> None:
        tool = create_rest_api_tool(
            connection={"name": "my-conn", "on_behalf_of_user": False},
        )
        assert tool is not None
        assert tool.name == "rest_api_call"

    def test_custom_name_and_description(self) -> None:
        tool = create_rest_api_tool(
            base_url="https://example.com",
            name="my_api",
            description="Call my API",
        )
        assert tool.name == "my_api"
        assert tool.description == "Call my API"

    def test_default_name_and_description(self) -> None:
        tool = create_rest_api_tool(base_url="https://example.com")
        assert tool.name == "rest_api_call"
        assert "HTTP request" in tool.description


# ---------------------------------------------------------------------------
# UC Connection mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUCConnectionMode:
    """Tests for REST API calls through a UC Connection."""

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch.object(ConnectionModel, "workspace_client_from")
    def test_calls_http_request_with_correct_params(
        self, mock_wcf: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        from databricks.sdk.service.serving import (
            ExternalFunctionRequestHttpMethod,
            HttpRequestResponse,
        )

        mock_ws = MagicMock()
        mock_response = HttpRequestResponse(contents=BytesIO(b'{"result": "ok"}'))
        mock_ws.serving_endpoints.http_request.return_value = mock_response
        mock_wcf.return_value = mock_ws

        connection = ConnectionModel(name="test-conn", on_behalf_of_user=False)
        tool = create_rest_api_tool(connection=connection)

        result = tool.invoke(
            {
                "method": "POST",
                "path": "/api/v1/data",
                "json_body": {"key": "value"},
                "query_params": {"limit": "10"},
            }
        )

        mock_ws.serving_endpoints.http_request.assert_called_once_with(
            connection_name="test-conn",
            method=ExternalFunctionRequestHttpMethod.POST,
            path="/api/v1/data",
            json=json.dumps({"key": "value"}),
            params="limit=10",
        )
        assert result == '{"result": "ok"}'

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch.object(ConnectionModel, "workspace_client_from")
    def test_get_request_no_body_no_params(
        self, mock_wcf: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        from databricks.sdk.service.serving import HttpRequestResponse

        mock_ws = MagicMock()
        mock_response = HttpRequestResponse(contents=BytesIO(b"hello"))
        mock_ws.serving_endpoints.http_request.return_value = mock_response
        mock_wcf.return_value = mock_ws

        connection = ConnectionModel(name="test-conn", on_behalf_of_user=False)
        tool = create_rest_api_tool(connection=connection)

        result = tool.invoke({"method": "GET", "path": "/health"})

        call_kwargs = mock_ws.serving_endpoints.http_request.call_args
        assert (
            call_kwargs.kwargs.get("json") is None or call_kwargs[1].get("json") is None
        )
        assert result == "hello"

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch.object(ConnectionModel, "workspace_client_from")
    def test_obo_workspace_client_from_called(
        self, mock_wcf: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        from databricks.sdk.service.serving import HttpRequestResponse

        mock_ws = MagicMock()
        mock_response = HttpRequestResponse(contents=BytesIO(b"ok"))
        mock_ws.serving_endpoints.http_request.return_value = mock_response
        mock_wcf.return_value = mock_ws

        connection = ConnectionModel(name="obo-conn", on_behalf_of_user=True)
        tool = create_rest_api_tool(connection=connection)

        tool.invoke({"method": "GET", "path": "/test"})

        mock_wcf.assert_called_once()

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch.object(ConnectionModel, "workspace_client_from")
    def test_tracing_attributes_uc_connection(
        self, mock_wcf: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        from databricks.sdk.service.serving import HttpRequestResponse

        mock_ws = MagicMock()
        mock_response = HttpRequestResponse(contents=BytesIO(b"ok"))
        mock_ws.serving_endpoints.http_request.return_value = mock_response
        mock_wcf.return_value = mock_ws

        connection = ConnectionModel(name="traced-conn", on_behalf_of_user=True)
        tool = create_rest_api_tool(connection=connection)

        tool.invoke({"method": "GET", "path": "/test"})

        mock_set_attrs.assert_called_once()
        info: ResourceInfo = mock_set_attrs.call_args[0][0]
        assert info.resource_type == "rest_api"
        assert info.on_behalf_of_user is True
        assert info.name == "traced-conn"

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch.object(ConnectionModel, "workspace_client_from", return_value=MagicMock())
    def test_unsupported_method_returns_error(
        self, mock_wcf: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        connection = ConnectionModel(name="test-conn", on_behalf_of_user=False)
        tool = create_rest_api_tool(connection=connection)

        result = tool.invoke({"method": "OPTIONS", "path": "/test"})

        assert "Unsupported HTTP method" in result

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch.object(ConnectionModel, "workspace_client_from")
    def test_error_returns_message_not_raises(
        self, mock_wcf: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        mock_ws = MagicMock()
        mock_ws.serving_endpoints.http_request.side_effect = RuntimeError("boom")
        mock_wcf.return_value = mock_ws

        connection = ConnectionModel(name="err-conn", on_behalf_of_user=False)
        tool = create_rest_api_tool(connection=connection)

        result = tool.invoke({"method": "GET", "path": "/fail"})

        assert "REST API call failed" in result
        assert "boom" in result


# ---------------------------------------------------------------------------
# Traditional HTTP mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTraditionalHTTPMode:
    """Tests for REST API calls via the requests library."""

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch("dao_ai.tools.rest_api.requests.request")
    def test_get_request(
        self, mock_request: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = '{"data": [1, 2, 3]}'
        mock_request.return_value = mock_response

        tool = create_rest_api_tool(base_url="https://api.example.com/v1")
        result = tool.invoke({"method": "GET", "path": "/items"})

        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.example.com/v1/items",
            json=None,
            params=None,
            headers=None,
            timeout=30,
        )
        assert result == '{"data": [1, 2, 3]}'

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch("dao_ai.tools.rest_api.requests.request")
    def test_post_with_body_and_params(
        self, mock_request: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = "created"
        mock_request.return_value = mock_response

        tool = create_rest_api_tool(base_url="https://api.example.com")
        result = tool.invoke(
            {
                "method": "post",
                "path": "/users",
                "json_body": {"name": "Alice"},
                "query_params": {"notify": "true"},
            }
        )

        mock_request.assert_called_once_with(
            method="POST",
            url="https://api.example.com/users",
            json={"name": "Alice"},
            params={"notify": "true"},
            headers=None,
            timeout=30,
        )
        assert result == "created"

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch("dao_ai.tools.rest_api.requests.request")
    def test_bearer_token_in_headers(
        self, mock_request: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_request.return_value = mock_response

        tool = create_rest_api_tool(
            base_url="https://api.example.com",
            auth_token="my-secret-token",
        )
        tool.invoke({"method": "GET", "path": "/secure"})

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer my-secret-token"

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch("dao_ai.tools.rest_api.requests.request")
    def test_default_headers(
        self, mock_request: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_request.return_value = mock_response

        tool = create_rest_api_tool(
            base_url="https://api.example.com",
            default_headers={"X-API-Key": "abc123", "Accept": "application/json"},
        )
        tool.invoke({"method": "GET", "path": "/data"})

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["headers"]["X-API-Key"] == "abc123"
        assert call_kwargs["headers"]["Accept"] == "application/json"

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch("dao_ai.tools.rest_api.requests.request")
    def test_bearer_and_default_headers_combined(
        self, mock_request: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_request.return_value = mock_response

        tool = create_rest_api_tool(
            base_url="https://api.example.com",
            auth_token="tok",
            default_headers={"X-Custom": "val"},
        )
        tool.invoke({"method": "GET", "path": "/test"})

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer tok"
        assert call_kwargs["headers"]["X-Custom"] == "val"

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch("dao_ai.tools.rest_api.requests.request")
    def test_tracing_attributes_http(
        self, mock_request: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_request.return_value = mock_response

        tool = create_rest_api_tool(base_url="https://api.example.com/v2")
        tool.invoke({"method": "GET", "path": "/ping"})

        mock_set_attrs.assert_called_once()
        info: ResourceInfo = mock_set_attrs.call_args[0][0]
        assert info.resource_type == "rest_api"
        assert info.on_behalf_of_user is False
        assert info.name == "https://api.example.com/v2"

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch("dao_ai.tools.rest_api.requests.request")
    def test_url_construction_trailing_slash(
        self, mock_request: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_request.return_value = mock_response

        tool = create_rest_api_tool(base_url="https://api.example.com/v1/")
        tool.invoke({"method": "GET", "path": "/items"})

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["url"] == "https://api.example.com/v1/items"

    @patch("dao_ai.tools.rest_api.set_resource_attributes")
    @patch("dao_ai.tools.rest_api.requests.request")
    def test_error_returns_message_not_raises(
        self, mock_request: MagicMock, mock_set_attrs: MagicMock
    ) -> None:
        mock_request.side_effect = ConnectionError("network error")

        tool = create_rest_api_tool(base_url="https://api.example.com")
        result = tool.invoke({"method": "GET", "path": "/fail"})

        assert "REST API call failed" in result
        assert "network error" in result
