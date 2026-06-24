"""Unit tests for :func:`dao_ai.tools.create_chat_completions_agent_tool`."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from dao_ai.config import DatabricksAppModel
from dao_ai.tools.chat_completions_agent import (
    _coerce_app,
    create_chat_completions_agent_tool,
)


class TestCoerceApp:
    def test_coerce_app_passes_through_model_instance(self) -> None:
        app: DatabricksAppModel = DatabricksAppModel(name="my-app")
        assert _coerce_app(app) is app

    def test_coerce_app_validates_dict_into_model(self) -> None:
        coerced: DatabricksAppModel = _coerce_app({"name": "my-app"})
        assert isinstance(coerced, DatabricksAppModel)
        assert coerced.name == "my-app"

    def test_coerce_app_rejects_other_types(self) -> None:
        with pytest.raises(TypeError, match="DatabricksAppModel or dict"):
            _coerce_app("my-app")  # type: ignore[arg-type]


class TestFactoryOutput:
    def test_factory_returns_structured_tool(self) -> None:
        tool = create_chat_completions_agent_tool(DatabricksAppModel(name="my-app"))
        assert tool.coroutine is not None
        assert tool.name == "my-app"

    def test_factory_accepts_dict_app(self) -> None:
        tool = create_chat_completions_agent_tool({"name": "dict-app"})
        assert tool.name == "dict-app"

    def test_factory_custom_name_and_description(self) -> None:
        tool = create_chat_completions_agent_tool(
            DatabricksAppModel(name="my-app"),
            name="custom_tool",
            description="Custom description.",
        )
        assert tool.name == "custom_tool"
        assert "Custom description." in tool.description


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content: str = content


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message: _StubMessage = _StubMessage(content)


class _StubChatCompletion:
    def __init__(self, content: str) -> None:
        self.choices: list[_StubChoice] = [_StubChoice(content)]


class _StubCompletions:
    def __init__(self, content: str = "stub-reply") -> None:
        self.content: str = content
        self.create_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _StubChatCompletion:
        self.create_calls.append(kwargs)
        return _StubChatCompletion(self.content)


class _StubChat:
    def __init__(self, content: str = "stub-reply") -> None:
        self.completions: _StubCompletions = _StubCompletions(content)


class _StubDatabricksOpenAI:
    def __init__(self, content: str = "stub-reply") -> None:
        self.chat: _StubChat = _StubChat(content)
        self.init_kwargs: dict[str, Any] = {}


@pytest.fixture
def stub_databricks_openai(monkeypatch: pytest.MonkeyPatch) -> _StubDatabricksOpenAI:
    stub: _StubDatabricksOpenAI = _StubDatabricksOpenAI(content="canned-reply")

    def _factory(**kwargs: Any) -> _StubDatabricksOpenAI:
        stub.init_kwargs = kwargs
        return stub

    monkeypatch.setattr(
        "dao_ai.tools.chat_completions_agent.DatabricksOpenAI",
        _factory,
    )
    return stub


@pytest.fixture
def stub_workspace_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake_ws: MagicMock = MagicMock(name="WorkspaceClient")
    monkeypatch.setattr(
        DatabricksAppModel,
        "workspace_client_from",
        lambda self, context: fake_ws,
    )
    return fake_ws


class TestInvocation:
    def test_invoke_calls_chat_completions_create_with_apps_prefix(
        self,
        stub_databricks_openai: _StubDatabricksOpenAI,
        stub_workspace_client: MagicMock,
    ) -> None:
        tool = create_chat_completions_agent_tool(
            DatabricksAppModel(name="a2a-minimal")
        )
        result: str = asyncio.run(tool.ainvoke({"prompt": "Hello"}))

        assert result == "canned-reply"
        calls: list[dict[str, Any]] = (
            stub_databricks_openai.chat.completions.create_calls
        )
        assert len(calls) == 1
        assert calls[0]["model"] == "apps/a2a-minimal"
        assert calls[0]["messages"] == [{"role": "user", "content": "Hello"}]

    def test_invoke_uses_workspace_client_from_context(
        self,
        stub_databricks_openai: _StubDatabricksOpenAI,
        stub_workspace_client: MagicMock,
    ) -> None:
        tool = create_chat_completions_agent_tool(DatabricksAppModel(name="my-app"))
        asyncio.run(tool.ainvoke({"prompt": "Hi"}))
        assert (
            stub_databricks_openai.init_kwargs.get("workspace_client")
            is stub_workspace_client
        )

    def test_invoke_returns_message_content(
        self,
        stub_databricks_openai: _StubDatabricksOpenAI,
        stub_workspace_client: MagicMock,
    ) -> None:
        stub_databricks_openai.chat.completions.content = "PONG"
        tool = create_chat_completions_agent_tool(DatabricksAppModel(name="my-app"))
        result: str = asyncio.run(tool.ainvoke({"prompt": "Ping?"}))
        assert result == "PONG"

    def test_invoke_handles_none_content_gracefully(
        self,
        stub_databricks_openai: _StubDatabricksOpenAI,
        stub_workspace_client: MagicMock,
    ) -> None:
        """OpenAI ChatCompletion.choices[0].message.content can be None
        (e.g. tool-only responses). The factory normalizes that to ''."""
        # Mutate the stub so message.content is None this call.
        stub_databricks_openai.chat.completions.content = None  # type: ignore[assignment]
        tool = create_chat_completions_agent_tool(DatabricksAppModel(name="my-app"))
        result: str = asyncio.run(tool.ainvoke({"prompt": "Hi"}))
        assert result == ""
