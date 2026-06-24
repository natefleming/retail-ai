"""Unit tests for :func:`dao_ai.tools.create_responses_agent_tool`.

These tests monkeypatch :class:`DatabricksOpenAI` and the
:meth:`DatabricksAppModel.workspace_client_from` method — no network or
real workspace credentials required.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from dao_ai.config import DatabricksAppModel
from dao_ai.tools.responses_agent import (
    _coerce_app,
    create_responses_agent_tool,
)


# ---------------------------------------------------------------------------
# _coerce_app — Pydantic validation of YAML-delivered dicts
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Factory output — StructuredTool wrapping
# ---------------------------------------------------------------------------


class TestFactoryOutput:
    def test_factory_returns_structured_tool(self) -> None:
        tool = create_responses_agent_tool(DatabricksAppModel(name="my-app"))
        assert tool.coroutine is not None
        # Default tool name is the app name.
        assert tool.name == "my-app"

    def test_factory_accepts_dict_app(self) -> None:
        tool = create_responses_agent_tool({"name": "dict-app"})
        assert tool.name == "dict-app"

    def test_factory_custom_name_and_description(self) -> None:
        tool = create_responses_agent_tool(
            DatabricksAppModel(name="my-app"),
            name="custom_tool",
            description="Custom description.",
        )
        assert tool.name == "custom_tool"
        assert "Custom description." in tool.description


# ---------------------------------------------------------------------------
# Invocation — DatabricksOpenAI is called with the right args
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text: str = output_text


class _StubResponses:
    def __init__(self, output_text: str = "stub-reply") -> None:
        self.output_text: str = output_text
        self.create_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _StubResponse:
        self.create_calls.append(kwargs)
        return _StubResponse(self.output_text)


class _StubDatabricksOpenAI:
    def __init__(self, output_text: str = "stub-reply") -> None:
        self.responses: _StubResponses = _StubResponses(output_text)
        self.init_kwargs: dict[str, Any] = {}


@pytest.fixture
def stub_databricks_openai(monkeypatch: pytest.MonkeyPatch) -> _StubDatabricksOpenAI:
    """Replace DatabricksOpenAI in the module under test with a stub instance."""
    stub: _StubDatabricksOpenAI = _StubDatabricksOpenAI(output_text="canned-reply")

    def _factory(**kwargs: Any) -> _StubDatabricksOpenAI:
        stub.init_kwargs = kwargs
        return stub

    monkeypatch.setattr(
        "dao_ai.tools.responses_agent.DatabricksOpenAI",
        _factory,
    )
    return stub


@pytest.fixture
def stub_workspace_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace DatabricksAppModel.workspace_client_from with a stub."""
    fake_ws: MagicMock = MagicMock(name="WorkspaceClient")
    monkeypatch.setattr(
        DatabricksAppModel,
        "workspace_client_from",
        lambda self, context: fake_ws,
    )
    return fake_ws


class TestInvocation:
    def test_invoke_calls_responses_create_with_apps_prefix(
        self,
        stub_databricks_openai: _StubDatabricksOpenAI,
        stub_workspace_client: MagicMock,
    ) -> None:
        tool = create_responses_agent_tool(DatabricksAppModel(name="a2a-minimal"))
        result: str = asyncio.run(tool.ainvoke({"prompt": "Hello"}))

        assert result == "canned-reply"
        assert len(stub_databricks_openai.responses.create_calls) == 1
        call_kwargs: dict[str, Any] = stub_databricks_openai.responses.create_calls[0]
        assert call_kwargs["model"] == "apps/a2a-minimal"
        assert call_kwargs["input"] == [{"role": "user", "content": "Hello"}]

    def test_invoke_uses_workspace_client_from_context(
        self,
        stub_databricks_openai: _StubDatabricksOpenAI,
        stub_workspace_client: MagicMock,
    ) -> None:
        tool = create_responses_agent_tool(DatabricksAppModel(name="my-app"))
        asyncio.run(tool.ainvoke({"prompt": "Hi"}))

        # DatabricksOpenAI must have been constructed with the stub workspace_client.
        assert (
            stub_databricks_openai.init_kwargs.get("workspace_client")
            is stub_workspace_client
        )

    def test_invoke_returns_output_text(
        self,
        stub_databricks_openai: _StubDatabricksOpenAI,
        stub_workspace_client: MagicMock,
    ) -> None:
        stub_databricks_openai.responses.output_text = "PONG"
        tool = create_responses_agent_tool(DatabricksAppModel(name="my-app"))
        result: str = asyncio.run(tool.ainvoke({"prompt": "Ping?"}))
        assert result == "PONG"

    def test_invoke_obo_app_uses_workspace_client_from(
        self,
        stub_databricks_openai: _StubDatabricksOpenAI,
        stub_workspace_client: MagicMock,
    ) -> None:
        """`on_behalf_of_user=True` apps still go through workspace_client_from
        — the OBO branching happens inside that method, not in this factory."""
        tool = create_responses_agent_tool(
            DatabricksAppModel(name="my-app", on_behalf_of_user=True)
        )
        asyncio.run(tool.ainvoke({"prompt": "Hi"}))
        assert (
            stub_databricks_openai.init_kwargs.get("workspace_client")
            is stub_workspace_client
        )
