"""Unit tests for :mod:`dao_ai.auth`."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from dao_ai.auth import OBONotAvailableError, WorkspaceBearerAuth


class TestWorkspaceBearerAuth:
    def test_injects_authorization_header_from_workspace_client(self) -> None:
        ws = MagicMock()
        ws.config.authenticate.return_value = {"Authorization": "Bearer token-x"}
        auth = WorkspaceBearerAuth(ws)

        request = httpx.Request("POST", "https://example.test/invocations")
        flow = auth.auth_flow(request)
        prepared = next(flow)
        assert prepared.headers["Authorization"] == "Bearer token-x"

    def test_calls_authenticate_on_every_request(self) -> None:
        """Token refresh: the callable is invoked per request, not cached."""
        ws = MagicMock()
        ws.config.authenticate.return_value = {"Authorization": "Bearer t1"}
        auth = WorkspaceBearerAuth(ws)

        r1 = httpx.Request("GET", "https://e.test/a")
        r2 = httpx.Request("GET", "https://e.test/b")
        next(auth.auth_flow(r1))
        next(auth.auth_flow(r2))
        assert ws.config.authenticate.call_count == 2

    def test_works_for_oauth_m2m_strategy(self) -> None:
        """The auth class doesn't care about auth_type — just reads the header."""
        ws = MagicMock()
        ws.config.authenticate.return_value = {"Authorization": "Bearer m2m-oauth"}
        ws.config.auth_type = "oauth-m2m"
        auth = WorkspaceBearerAuth(ws)
        prepared = next(auth.auth_flow(httpx.Request("POST", "https://e.test/")))
        assert prepared.headers["Authorization"] == "Bearer m2m-oauth"

    def test_works_for_pat_strategy(self) -> None:
        """The auth class also accepts PAT-auth WCs — the original databricks_openai
        BearerAuth gate is NOT applied here, by design."""
        ws = MagicMock()
        ws.config.authenticate.return_value = {"Authorization": "Bearer pat-token"}
        ws.config.auth_type = "pat"
        auth = WorkspaceBearerAuth(ws)
        prepared = next(auth.auth_flow(httpx.Request("POST", "https://e.test/")))
        assert prepared.headers["Authorization"] == "Bearer pat-token"


class TestOBONotAvailableError:
    def test_message_includes_resource_name_when_supplied(self) -> None:
        err = OBONotAvailableError(resource_name="my-app")
        assert "my-app" in str(err)
        assert err.resource_name == "my-app"

    def test_message_includes_field_when_supplied(self) -> None:
        err = OBONotAvailableError(field="DatabricksAppModel.on_behalf_of_user")
        assert "DatabricksAppModel.on_behalf_of_user" in str(err)
        assert err.field == "DatabricksAppModel.on_behalf_of_user"

    def test_message_works_without_optional_args(self) -> None:
        err = OBONotAvailableError()
        assert "on_behalf_of_user" in str(err)

    def test_is_a_value_error(self) -> None:
        """Subclasses ValueError so callers can catch broadly if they want."""
        with pytest.raises(ValueError):
            raise OBONotAvailableError(resource_name="r")
