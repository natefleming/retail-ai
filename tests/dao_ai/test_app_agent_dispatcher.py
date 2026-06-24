"""Unit tests for :func:`dao_ai.tools.create_app_dispatcher`.

The dispatcher does three things:
1. Lazily resolves the OpenAI API contract on first invocation.
2. Caches the resolution for subsequent invocations.
3. Calls DatabricksOpenAI.responses.create OR
   DatabricksOpenAI.chat.completions.create based on the resolved api.

Tests monkeypatch DatabricksOpenAI + discover_app_agent_api +
DatabricksAppModel.workspace_client_from so no network is needed.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from dao_ai.config import DatabricksAppModel
from dao_ai.tools._api_discovery import ApiContract
from dao_ai.tools.app_agent_dispatcher import create_app_dispatcher


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, output_text: str = "stub-response") -> None:
        self.output_text: str = output_text


class _StubResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        return _StubResponse(f"resp:{kwargs.get('input')[0]['content']}")


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content: str = content


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message: _StubMessage = _StubMessage(content)


class _StubChatCompletion:
    def __init__(self, content: str) -> None:
        self.choices: list[_StubChoice] = [_StubChoice(content)]


class _StubChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _StubChatCompletion:
        self.calls.append(kwargs)
        return _StubChatCompletion(f"chat:{kwargs.get('messages')[0]['content']}")


class _StubChat:
    def __init__(self) -> None:
        self.completions: _StubChatCompletions = _StubChatCompletions()


class _StubDatabricksOpenAI:
    """Stub that records calls; both `.responses` and `.chat.completions`."""

    instances: list["_StubDatabricksOpenAI"] = []

    def __init__(self, **_kwargs: Any) -> None:
        self.responses: _StubResponses = _StubResponses()
        self.chat: _StubChat = _StubChat()
        _StubDatabricksOpenAI.instances.append(self)


@pytest.fixture(autouse=True)
def reset_stub_instances() -> None:
    _StubDatabricksOpenAI.instances.clear()


@pytest.fixture
def stub_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dao_ai.tools.app_agent_dispatcher.DatabricksOpenAI",
        _StubDatabricksOpenAI,
    )


@pytest.fixture
def stub_workspace_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake_ws: MagicMock = MagicMock(name="WorkspaceClient")
    monkeypatch.setattr(
        DatabricksAppModel,
        "workspace_client_from",
        lambda self, context: fake_ws,
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
    def test_explicit_responses_calls_responses_create_and_skips_probe(
        self,
        stub_openai: None,
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
        # responses.create was called; chat.completions.create was NOT
        instance = _StubDatabricksOpenAI.instances[0]
        assert len(instance.responses.calls) == 1
        assert instance.responses.calls[0]["model"] == "apps/explicit-resp"
        assert len(instance.chat.completions.calls) == 0

    def test_explicit_completions_calls_chat_completions_create_and_skips_probe(
        self,
        stub_openai: None,
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
        instance = _StubDatabricksOpenAI.instances[0]
        assert len(instance.chat.completions.calls) == 1
        assert instance.chat.completions.calls[0]["model"] == "apps/explicit-cc"
        assert len(instance.responses.calls) == 0


# ---------------------------------------------------------------------------
# Discovery resolves api on first invoke
# ---------------------------------------------------------------------------


class TestDiscoveryResolvesApi:
    def test_probe_returns_responses_routes_to_responses(
        self,
        stub_openai: None,
        stub_workspace_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_probe(monkeypatch, "responses")
        tool = create_app_dispatcher(DatabricksAppModel(name="probed-app"))
        asyncio.run(tool.ainvoke({"prompt": "hi"}))
        assert len(_StubDatabricksOpenAI.instances[0].responses.calls) == 1

    def test_probe_returns_none_falls_back_to_default_responses(
        self,
        stub_openai: None,
        stub_workspace_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_probe(monkeypatch, None)
        tool = create_app_dispatcher(
            DatabricksAppModel(name="unknown-app"),
            default_api="responses",
        )
        asyncio.run(tool.ainvoke({"prompt": "hi"}))
        assert len(_StubDatabricksOpenAI.instances[0].responses.calls) == 1

    def test_probe_returns_none_default_completions_routes_to_chat(
        self,
        stub_openai: None,
        stub_workspace_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_probe(monkeypatch, None)
        tool = create_app_dispatcher(
            DatabricksAppModel(name="unknown-app"),
            default_api="completions",
        )
        asyncio.run(tool.ainvoke({"prompt": "hi"}))
        assert len(_StubDatabricksOpenAI.instances[0].chat.completions.calls) == 1


# ---------------------------------------------------------------------------
# Caching — probe runs once per tool
# ---------------------------------------------------------------------------


class TestProbeCaching:
    def test_probe_runs_only_once_across_invocations(
        self,
        stub_openai: None,
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
