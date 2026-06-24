"""Unit tests for :func:`dao_ai.tools.create_app_dispatcher`.

The dispatcher does four things:
1. Lazily resolves the OpenAI API contract on first invocation.
2. Caches the resolution for subsequent invocations.
3. POSTs to ``<app.url>/invocations`` using ``httpx.AsyncClient`` with
   :class:`WorkspaceBearerAuth`, bypassing the ``databricks_openai`` OAuth
   gate that breaks PAT-auth WCs.
4. Surfaces ``OBONotAvailableError`` when the calling agent should be in
   OBO mode but didn't propagate the forwarded user token.

Tests monkeypatch ``httpx.AsyncClient`` + ``discover_app_agent_api`` +
``DatabricksAppModel.workspace_client_from`` so no network is needed.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from dao_ai.auth import OBONotAvailableError
from dao_ai.config import DatabricksAppModel
from dao_ai.state import Context
from dao_ai.tools._api_discovery import ApiContract
from dao_ai.tools.app_agent_dispatcher import create_app_dispatcher

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubResponse:
    """httpx.Response stub."""

    def __init__(self, json_payload: dict[str, Any], status_code: int = 200) -> None:
        self._json = json_payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _StubAsyncClient:
    """httpx.AsyncClient stub. Records every .post call."""

    instances: list["_StubAsyncClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs: dict[str, Any] = kwargs
        self.posts: list[dict[str, Any]] = []
        # Pre-configured envelope; tests may override via instance.envelope_for
        self.envelope_for: dict[str, dict[str, Any]] = {}
        _StubAsyncClient.instances.append(self)

    async def __aenter__(self) -> "_StubAsyncClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any], **_kw: Any) -> _StubResponse:
        self.posts.append({"url": url, "json": json})
        # If a test pre-loaded a response, use it. Otherwise echo the prompt
        # in a Responses-API-shaped envelope.
        if url in self.envelope_for:
            return _StubResponse(self.envelope_for[url])
        prompt: str = ""
        if json.get("input"):
            prompt = json["input"][0].get("content", "")
        elif json.get("messages"):
            prompt = json["messages"][0].get("content", "")
        # Echo back a Responses-shaped envelope by default.
        if "messages" in json:
            envelope = {
                "choices": [{"message": {"content": f"chat:{prompt}"}}]
            }
        else:
            envelope = {
                "output": [{"content": [{"text": f"resp:{prompt}"}]}]
            }
        return _StubResponse(envelope)


@pytest.fixture(autouse=True)
def reset_stub_instances() -> None:
    _StubAsyncClient.instances.clear()


@pytest.fixture
def stub_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dao_ai.tools.app_agent_dispatcher.httpx.AsyncClient",
        _StubAsyncClient,
    )


@pytest.fixture
def stub_workspace_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake_ws: MagicMock = MagicMock(name="WorkspaceClient")
    fake_ws.config.authenticate.return_value = {"Authorization": "Bearer stub-token"}
    monkeypatch.setattr(
        DatabricksAppModel,
        "workspace_client_from",
        lambda self, context, *, strict=False: fake_ws,
    )
    # Also stub the .url property so the dispatcher's probe-lambda
    # doesn't trigger a real `apps.get()` SDK call on first invoke.
    monkeypatch.setattr(
        DatabricksAppModel,
        "url",
        property(lambda self: f"https://{self.name}.test"),
    )
    return fake_ws


def _set_probe(monkeypatch: pytest.MonkeyPatch, returns: Optional[ApiContract]) -> None:
    """Force discover_app_agent_api to return the given value."""

    def _probe(app_url: str, ws: Any, **_kwargs: Any) -> Optional[ApiContract]:
        return returns

    monkeypatch.setattr(
        "dao_ai.tools.app_agent_dispatcher.discover_app_agent_api",
        _probe,
    )


# ---------------------------------------------------------------------------
# Factory output
# ---------------------------------------------------------------------------


class TestFactoryOutput:
    def test_factory_returns_structured_tool(self) -> None:
        tool = create_app_dispatcher(DatabricksAppModel(name="my-app"))
        assert tool.coroutine is not None
        assert tool.name == "my-app"

    def test_factory_accepts_dict_app(self) -> None:
        tool = create_app_dispatcher({"name": "dict-app"})
        assert tool.name == "dict-app"

    def test_factory_custom_name_and_description(self) -> None:
        tool = create_app_dispatcher(
            DatabricksAppModel(name="my-app"),
            name="custom",
            description="Custom desc.",
        )
        assert tool.name == "custom"
        assert "Custom desc." in tool.description


# ---------------------------------------------------------------------------
# Explicit api: precedence (no probe runs)
# ---------------------------------------------------------------------------


class TestExplicitApiSkipsProbe:
    def test_explicit_responses_posts_input_body_and_skips_probe(
        self,
        stub_httpx: None,
        stub_workspace_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        probe_calls: list[int] = []
        monkeypatch.setattr(
            "dao_ai.tools.app_agent_dispatcher.discover_app_agent_api",
            lambda *a, **kw: probe_calls.append(1) or "completions",
        )

        tool = create_app_dispatcher(
            DatabricksAppModel(name="explicit-resp"), api="responses"
        )
        result: str = asyncio.run(tool.ainvoke({"prompt": "hi"}))

        assert result == "resp:hi"
        assert probe_calls == []  # CRITICAL: no probe when explicit
        instance = _StubAsyncClient.instances[0]
        assert len(instance.posts) == 1
        assert instance.posts[0]["url"] == "https://explicit-resp.test/invocations"
        # Responses API uses {"input": [...]} body shape.
        assert "input" in instance.posts[0]["json"]
        assert "messages" not in instance.posts[0]["json"]

    def test_explicit_completions_posts_messages_body_and_skips_probe(
        self,
        stub_httpx: None,
        stub_workspace_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        probe_calls: list[int] = []
        monkeypatch.setattr(
            "dao_ai.tools.app_agent_dispatcher.discover_app_agent_api",
            lambda *a, **kw: probe_calls.append(1) or "responses",
        )

        tool = create_app_dispatcher(
            DatabricksAppModel(name="explicit-cc"), api="completions"
        )
        result: str = asyncio.run(tool.ainvoke({"prompt": "hi"}))

        assert result == "chat:hi"
        assert probe_calls == []
        instance = _StubAsyncClient.instances[0]
        # Chat Completions uses {"messages": [...]} body shape.
        assert "messages" in instance.posts[0]["json"]
        assert "input" not in instance.posts[0]["json"]


# ---------------------------------------------------------------------------
# Discovery resolves api on first invoke
# ---------------------------------------------------------------------------


class TestDiscoveryResolvesApi:
    def test_probe_returns_responses_routes_to_responses(
        self,
        stub_httpx: None,
        stub_workspace_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_probe(monkeypatch, "responses")
        tool = create_app_dispatcher(DatabricksAppModel(name="probed-app"))
        asyncio.run(tool.ainvoke({"prompt": "hi"}))
        instance = _StubAsyncClient.instances[0]
        assert "input" in instance.posts[0]["json"]

    def test_probe_returns_none_falls_back_to_default_responses(
        self,
        stub_httpx: None,
        stub_workspace_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_probe(monkeypatch, None)
        tool = create_app_dispatcher(
            DatabricksAppModel(name="unknown-app"),
            default_api="responses",
        )
        asyncio.run(tool.ainvoke({"prompt": "hi"}))
        instance = _StubAsyncClient.instances[0]
        assert "input" in instance.posts[0]["json"]

    def test_probe_returns_none_default_completions_routes_to_chat(
        self,
        stub_httpx: None,
        stub_workspace_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_probe(monkeypatch, None)
        tool = create_app_dispatcher(
            DatabricksAppModel(name="unknown-app"),
            default_api="completions",
        )
        asyncio.run(tool.ainvoke({"prompt": "hi"}))
        instance = _StubAsyncClient.instances[0]
        assert "messages" in instance.posts[0]["json"]


# ---------------------------------------------------------------------------
# Caching — probe runs once per tool
# ---------------------------------------------------------------------------


class TestProbeCaching:
    def test_probe_runs_only_once_across_invocations(
        self,
        stub_httpx: None,
        stub_workspace_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        probe_count: list[int] = []

        def _probe(*a, **kw):
            probe_count.append(1)
            return "responses"

        monkeypatch.setattr(
            "dao_ai.tools.app_agent_dispatcher.discover_app_agent_api", _probe
        )
        tool = create_app_dispatcher(DatabricksAppModel(name="cached-app"))
        asyncio.run(tool.ainvoke({"prompt": "first"}))
        asyncio.run(tool.ainvoke({"prompt": "second"}))
        asyncio.run(tool.ainvoke({"prompt": "third"}))
        assert len(probe_count) == 1  # probe cached after first call


# ---------------------------------------------------------------------------
# Strict-mode OBO surfacing (matrix cell #10 close-out)
# ---------------------------------------------------------------------------


class TestStrictModeOBO:
    def test_obo_target_no_forwarded_token_raises(
        self,
        stub_httpx: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cell #10: target.on_behalf_of_user=true + no forwarded token in context."""

        _set_probe(monkeypatch, "responses")
        # Stub the URL so the (never-reached) discovery branch wouldn't crash.
        monkeypatch.setattr(
            DatabricksAppModel,
            "url",
            property(lambda self: f"https://{self.name}.test"),
        )
        # DO NOT stub workspace_client_from — let the real implementation
        # raise OBONotAvailableError when strict=True + on_behalf_of_user=true
        # + no headers in context.
        app_model = DatabricksAppModel(name="obo-app", on_behalf_of_user=True)
        tool = create_app_dispatcher(app_model)

        # ToolRuntime.context.headers is None when the caller didn't propagate.
        # The dispatcher itself extracts context via runtime.context, so the
        # cleanest way to assert is via tool.coroutine directly with an empty
        # runtime stub.
        runtime = MagicMock()
        runtime.context = Context(headers=None)

        with pytest.raises(OBONotAvailableError) as exc_info:
            asyncio.run(tool.coroutine(prompt="hi", runtime=runtime))
        assert "obo-app" in str(exc_info.value) or exc_info.value.resource_name == "obo-app"

    def test_obo_target_with_forwarded_token_succeeds(
        self,
        stub_httpx: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cell #1 happy path: target OBO + forwarded token present → tool runs."""

        _set_probe(monkeypatch, "responses")
        monkeypatch.setattr(
            DatabricksAppModel,
            "url",
            property(lambda self: f"https://{self.name}.test"),
        )
        # Stub workspace_client_from to behave as the real implementation
        # would with valid headers: return a WC instance.
        fake_ws = MagicMock()
        fake_ws.config.authenticate.return_value = {"Authorization": "Bearer u-token"}

        def _ws_from(self, context, *, strict=False) -> Any:
            # If strict and OBO and no token, raise — same shape as real impl
            if strict and self.on_behalf_of_user:
                headers = (context.headers if context else None) or {}
                if not headers.get("x-forwarded-access-token"):
                    raise OBONotAvailableError(resource_name=self.name)
            return fake_ws

        monkeypatch.setattr(DatabricksAppModel, "workspace_client_from", _ws_from)

        app_model = DatabricksAppModel(name="obo-ok", on_behalf_of_user=True)
        tool = create_app_dispatcher(app_model)
        runtime = MagicMock()
        runtime.context = Context(
            headers={"x-forwarded-access-token": "u-token"}
        )

        result = asyncio.run(tool.coroutine(prompt="hola", runtime=runtime))
        assert result == "resp:hola"


# ---------------------------------------------------------------------------
# Auth header propagation
# ---------------------------------------------------------------------------


class TestAuthHeaderPropagation:
    def test_workspace_bearer_auth_is_used(
        self,
        stub_httpx: None,
        stub_workspace_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Confirm the httpx.AsyncClient was constructed with WorkspaceBearerAuth."""

        _set_probe(monkeypatch, "responses")
        tool = create_app_dispatcher(DatabricksAppModel(name="auth-app"))
        asyncio.run(tool.ainvoke({"prompt": "hi"}))
        instance = _StubAsyncClient.instances[0]
        auth = instance.init_kwargs.get("auth")
        assert auth is not None
        # WorkspaceBearerAuth holds the workspace_client by attribute.
        assert hasattr(auth, "_workspace_client")
