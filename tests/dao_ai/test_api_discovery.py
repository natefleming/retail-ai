"""Unit tests for :mod:`dao_ai.tools._api_discovery`.

Covers the two discovery probes and the precedence helper without any
network or workspace dependency.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from dao_ai.tools._api_discovery import (
    discover_app_agent_api,
    discover_serving_endpoint_api,
    resolve_api,
)


def _ws_with_auth_header() -> MagicMock:
    """Return a stub WorkspaceClient that successfully mints an auth header."""
    ws = MagicMock(name="WorkspaceClient")
    ws.config.authenticate.return_value = {"Authorization": "Bearer test-token"}
    return ws


def _patch_httpx_get(
    monkeypatch: pytest.MonkeyPatch,
    handler: callable,
) -> None:
    """Replace dao_ai.tools._api_discovery.httpx.get with a handler that
    receives (url, headers, timeout) and returns an httpx.Response."""

    def _fake_get(url: str, **kwargs: Any) -> httpx.Response:
        return handler(url, kwargs.get("headers"), kwargs.get("timeout"))

    monkeypatch.setattr("dao_ai.tools._api_discovery.httpx.get", _fake_get)


# ---------------------------------------------------------------------------
# discover_app_agent_api — /agent/info probe
# ---------------------------------------------------------------------------


class TestDiscoverAppAgentApi:
    def test_200_with_agent_api_responses_returns_responses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_httpx_get(
            monkeypatch,
            lambda url, headers, timeout: httpx.Response(
                200,
                json={
                    "name": "test-app",
                    "use_case": "agent",
                    "mlflow_version": "3.14.0",
                    "agent_api": "responses",
                },
            ),
        )
        result = discover_app_agent_api(
            "https://test-app.databricksapps.com",
            _ws_with_auth_header(),
        )
        assert result == "responses"

    def test_200_without_agent_api_field_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_httpx_get(
            monkeypatch,
            lambda url, headers, timeout: httpx.Response(
                200,
                json={"name": "x", "use_case": "agent"},
            ),
        )
        assert discover_app_agent_api("https://x.test", _ws_with_auth_header()) is None

    def test_200_with_unknown_agent_api_value_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_httpx_get(
            monkeypatch,
            lambda url, headers, timeout: httpx.Response(
                200, json={"agent_api": "a2a"}
            ),
        )
        assert discover_app_agent_api("https://x.test", _ws_with_auth_header()) is None

    def test_200_with_non_json_body_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_httpx_get(
            monkeypatch,
            lambda url, headers, timeout: httpx.Response(
                200, content=b"<html>not json</html>"
            ),
        )
        assert discover_app_agent_api("https://x.test", _ws_with_auth_header()) is None

    def test_401_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_httpx_get(
            monkeypatch,
            lambda url, headers, timeout: httpx.Response(401),
        )
        assert discover_app_agent_api("https://x.test", _ws_with_auth_header()) is None

    def test_404_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_httpx_get(
            monkeypatch,
            lambda url, headers, timeout: httpx.Response(404),
        )
        assert discover_app_agent_api("https://x.test", _ws_with_auth_header()) is None

    def test_500_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_httpx_get(
            monkeypatch,
            lambda url, headers, timeout: httpx.Response(500),
        )
        assert discover_app_agent_api("https://x.test", _ws_with_auth_header()) is None

    def test_network_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(url, **kwargs):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr("dao_ai.tools._api_discovery.httpx.get", _raise)
        assert discover_app_agent_api("https://x.test", _ws_with_auth_header()) is None

    def test_auth_failure_returns_none_without_probing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If authenticate() raises, the probe should never run.
        ws = MagicMock(name="WorkspaceClient")
        ws.config.authenticate.side_effect = RuntimeError("auth broken")

        get_calls: list[str] = []

        def _spy_get(url, **kwargs):
            get_calls.append(url)
            return httpx.Response(200, json={"agent_api": "responses"})

        monkeypatch.setattr("dao_ai.tools._api_discovery.httpx.get", _spy_get)
        assert discover_app_agent_api("https://x.test", ws) is None
        assert get_calls == []  # auth failed before any HTTP

    def test_url_strips_trailing_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[str] = []

        def _spy_get(url, **kwargs):
            captured.append(url)
            return httpx.Response(200, json={"agent_api": "responses"})

        monkeypatch.setattr("dao_ai.tools._api_discovery.httpx.get", _spy_get)
        discover_app_agent_api("https://x.test/", _ws_with_auth_header())
        assert captured == ["https://x.test/agent/info"]


# ---------------------------------------------------------------------------
# discover_serving_endpoint_api — SDK task probe
# ---------------------------------------------------------------------------


class TestDiscoverServingEndpointApi:
    def _ws_with_task(self, task: Any) -> MagicMock:
        ws = MagicMock(name="WorkspaceClient")
        ep = MagicMock()
        ep.task = task
        ws.serving_endpoints.get.return_value = ep
        return ws

    def test_task_agent_v1_responses_returns_responses(self) -> None:
        ws = self._ws_with_task("agent/v1/responses")
        assert discover_serving_endpoint_api("hardware_store_dao", ws) == "responses"

    def test_task_llm_v1_chat_returns_completions(self) -> None:
        ws = self._ws_with_task("llm/v1/chat")
        assert (
            discover_serving_endpoint_api("databricks-claude-sonnet-4", ws)
            == "completions"
        )

    def test_task_embeddings_returns_none(self) -> None:
        ws = self._ws_with_task("llm/v1/embeddings")
        assert discover_serving_endpoint_api("databricks-gte-large-en", ws) is None

    def test_task_unknown_future_value_returns_none(self) -> None:
        ws = self._ws_with_task("agent/v2/somethingnew")
        assert discover_serving_endpoint_api("future-endpoint", ws) is None

    def test_task_none_returns_none(self) -> None:
        ws = self._ws_with_task(None)
        assert discover_serving_endpoint_api("untyped", ws) is None

    def test_sdk_raises_returns_none(self) -> None:
        ws = MagicMock(name="WorkspaceClient")
        ws.serving_endpoints.get.side_effect = RuntimeError("not found")
        assert discover_serving_endpoint_api("missing", ws) is None


# ---------------------------------------------------------------------------
# resolve_api — precedence helper
# ---------------------------------------------------------------------------


class TestResolveApi:
    def test_explicit_wins_over_discovery(self) -> None:
        calls: list[int] = []

        def _discover():
            calls.append(1)
            return "responses"

        result = resolve_api(
            explicit="completions", discover=_discover, default="responses"
        )
        assert result.value == "completions"
        assert result.origin == "explicit"
        # Critical invariant: probe MUST NOT run when explicit is set.
        assert calls == []

    def test_discovered_wins_over_default(self) -> None:
        result = resolve_api(
            explicit=None,
            discover=lambda: "responses",
            default="completions",
        )
        assert result.value == "responses"
        assert result.origin == "discovery"

    def test_default_when_no_explicit_or_discovered(self) -> None:
        result = resolve_api(
            explicit=None,
            discover=lambda: None,
            default="completions",
        )
        assert result.value == "completions"
        assert result.origin == "default"

    def test_explicit_responses_wins_when_discovery_would_say_completions(
        self,
    ) -> None:
        result = resolve_api(
            explicit="responses",
            discover=lambda: "completions",
            default="completions",
        )
        assert result.value == "responses"
        assert result.origin == "explicit"

    def test_discovery_origin_reported_when_value_matches_default(self) -> None:
        """Discovery actually ran and returned a value; origin should be
        'discovery' even though that value happens to equal the default."""
        result = resolve_api(
            explicit=None,
            discover=lambda: "completions",
            default="completions",
        )
        assert result.value == "completions"
        assert result.origin == "discovery"
