"""Tests for first-class tool model types: Genie, VectorSearch, Search, Agent, A2A.

These models are thin Pydantic wrappers around the existing factory functions
(``dao_ai.tools.create_genie_tool``, ``create_vector_search_tool``,
``create_search_tool``, ``create_agent_endpoint_tool``,
``create_a2a_agent_tool``). The tests verify:

- Pydantic discriminator dispatch: ``type: genie`` deserializes to
  ``GenieToolModel``, etc.
- ``.as_tools()`` delegates to the existing factory and returns the same
  tool count / names as the equivalent ``type: factory`` configuration.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool

from dao_ai.config import (
    A2AToolModel,
    AgentToolModel,
    FunctionType,
    GenieToolModel,
    SearchToolModel,
    ToolModel,
    VectorSearchToolModel,
)
from dao_ai.tools.genie import GenieToolkit


@pytest.fixture
def genie_room_dict() -> dict:
    return {
        "space_id": "test-space-id",
        "name": "Test Room",
        "on_behalf_of_user": False,
    }


@pytest.fixture
def vector_store_dict() -> dict:
    return {
        "embedding_model": {"name": "dummy-embedding"},
        "endpoint": {"name": "test-endpoint", "type": "STANDARD"},
        "index": {"name": "test_idx"},
        "source_table": {"name": "test_table"},
        "primary_key": "id",
        "embedding_source_column": "text",
        "columns": ["id", "text"],
    }


# ---------------------------------------------------------------------------
# Discriminator dispatch — confirm the right subclass is constructed
# ---------------------------------------------------------------------------


class TestDiscriminatorDispatch:
    def test_type_genie_dispatches_to_genie_tool_model(
        self, genie_room_dict: dict
    ) -> None:
        m = ToolModel.model_validate(
            {
                "name": "sales_genie",
                "function": {"type": "genie", "genie_room": genie_room_dict},
            }
        )
        assert isinstance(m.function, GenieToolModel)
        assert m.function.type == FunctionType.GENIE.value

    def test_type_vector_search_dispatches_to_vector_search_tool_model(
        self, vector_store_dict: dict
    ) -> None:
        m = ToolModel.model_validate(
            {
                "name": "product_search",
                "function": {
                    "type": "vector_search",
                    "vector_store": vector_store_dict,
                },
            }
        )
        assert isinstance(m.function, VectorSearchToolModel)
        assert m.function.type == FunctionType.VECTOR_SEARCH.value

    def test_type_search_dispatches_to_search_tool_model(self) -> None:
        m = ToolModel.model_validate(
            {"name": "web_search", "function": {"type": "search"}}
        )
        assert isinstance(m.function, SearchToolModel)
        assert m.function.type == FunctionType.SEARCH.value


# ---------------------------------------------------------------------------
# Validation rules on the new models
# ---------------------------------------------------------------------------


class TestValidation:
    def test_genie_requires_genie_room(self) -> None:
        with pytest.raises(Exception):
            ToolModel.model_validate(
                {"name": "x", "function": {"type": "genie"}}
            )

    def test_vector_search_rejects_neither_retriever_nor_store(self) -> None:
        with pytest.raises(Exception, match="retriever.*vector_store"):
            ToolModel.model_validate(
                {"name": "x", "function": {"type": "vector_search"}}
            )

    def test_vector_search_rejects_both_retriever_and_store(
        self, vector_store_dict: dict
    ) -> None:
        with pytest.raises(Exception, match="retriever.*vector_store"):
            ToolModel.model_validate(
                {
                    "name": "x",
                    "function": {
                        "type": "vector_search",
                        "vector_store": vector_store_dict,
                        "retriever": {"vector_store": vector_store_dict},
                    },
                }
            )


# ---------------------------------------------------------------------------
# .as_tools() parity vs equivalent FactoryFunctionModel
# ---------------------------------------------------------------------------


class TestParityWithFactory:
    def test_genie_no_cache_parity(self, genie_room_dict: dict) -> None:
        """type: genie with no caches should produce identical tools to factory shape."""
        new = ToolModel.model_validate(
            {
                "name": "g",
                "function": {
                    "type": "genie",
                    "genie_room": genie_room_dict,
                    "name": "my_genie",
                },
            }
        )
        old = ToolModel.model_validate(
            {
                "name": "g",
                "function": {
                    "type": "factory",
                    "name": "dao_ai.tools.create_genie_tool",
                    "args": {
                        "genie_room": genie_room_dict,
                        "name": "my_genie",
                    },
                },
            }
        )
        new_tools = new.function.as_tools()
        old_tools = old.function.as_tools()
        assert len(new_tools) == len(old_tools) == 1
        assert [t.name for t in new_tools] == [t.name for t in old_tools]

    def test_genie_with_lru_cache_parity(self, genie_room_dict: dict) -> None:
        """type: genie with LRU cache should produce GenieToolkit (query + feedback)."""
        from dao_ai.config import GenieLRUCacheParametersModel, WarehouseModel

        lru = GenieLRUCacheParametersModel(
            warehouse=WarehouseModel(warehouse_id="test-warehouse"),
            capacity=10,
            time_to_live_seconds=60,
        )
        m = ToolModel.model_validate(
            {
                "name": "g",
                "function": {
                    "type": "genie",
                    "genie_room": genie_room_dict,
                    "name": "cached_genie",
                    "lru_cache": lru.model_dump(),
                },
            }
        )
        tools = m.function.as_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"cached_genie", "cached_genie_feedback"}

    def test_genie_enable_feedback_promotes_to_toolkit(
        self, genie_room_dict: dict
    ) -> None:
        m = ToolModel.model_validate(
            {
                "name": "g",
                "function": {
                    "type": "genie",
                    "genie_room": genie_room_dict,
                    "name": "feedback_genie",
                    "enable_feedback": True,
                },
            }
        )
        tools = m.function.as_tools()
        assert len(tools) == 2
        assert {t.name for t in tools} == {
            "feedback_genie",
            "feedback_genie_feedback",
        }


# ---------------------------------------------------------------------------
# All new tool models inherit BaseFunctionModel behavior (human_in_the_loop, etc.)
# ---------------------------------------------------------------------------


class TestInheritedBehavior:
    def test_genie_accepts_human_in_the_loop(self, genie_room_dict: dict) -> None:
        m = ToolModel.model_validate(
            {
                "name": "g",
                "function": {
                    "type": "genie",
                    "genie_room": genie_room_dict,
                    "human_in_the_loop": {
                        "review_prompt": "Approve this query?",
                    },
                },
            }
        )
        assert m.function.human_in_the_loop is not None
        assert m.function.human_in_the_loop.review_prompt == "Approve this query?"

    def test_vector_search_accepts_human_in_the_loop(
        self, vector_store_dict: dict
    ) -> None:
        m = ToolModel.model_validate(
            {
                "name": "vs",
                "function": {
                    "type": "vector_search",
                    "vector_store": vector_store_dict,
                    "human_in_the_loop": {"review_prompt": "Approve search?"},
                },
            }
        )
        assert m.function.human_in_the_loop is not None


# ---------------------------------------------------------------------------
# AgentToolModel — Supervisor API knowledge_assistant / serving_endpoint
# ---------------------------------------------------------------------------


class TestAgentToolModel:
    def test_type_agent_dispatches_to_agent_tool_model(self) -> None:
        m = ToolModel.model_validate(
            {
                "name": "ka_reviews",
                "function": {"type": "agent", "endpoint": "ka-customer-reviews"},
            }
        )
        assert isinstance(m.function, AgentToolModel)
        assert m.function.type == FunctionType.AGENT.value
        assert m.function.endpoint == "ka-customer-reviews"

    def test_agent_requires_endpoint_or_app(self) -> None:
        """At least one of endpoint / app must be set."""
        with pytest.raises(Exception):
            ToolModel.model_validate({"name": "x", "function": {"type": "agent"}})

    def test_agent_rejects_both_endpoint_and_app(self) -> None:
        """Endpoint and app are mutually exclusive."""
        # Direct validation gives the clean error; ToolModel dispatch produces
        # a union error (existing dao-ai behavior — see GenieToolModel tests).
        from dao_ai.config import AgentToolModel as _ATM

        with pytest.raises(Exception):
            _ATM(
                type=FunctionType.AGENT,
                endpoint="my-ep",
                app={"name": "my-app"},
            )

    def test_agent_app_dispatches_to_responses_agent_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """type: agent with app: dispatches to create_responses_agent_tool."""
        from dao_ai.config import DatabricksAppModel
        from langchain_core.tools import StructuredTool

        captured: dict[str, object] = {}

        def _stub(
            app, *, name=None, description=None
        ):  # type: ignore[no-untyped-def]
            captured["app"] = app
            captured["name"] = name
            captured["description"] = description
            return StructuredTool.from_function(
                func=lambda prompt: "stub",
                name=name or "stub_tool",
                description=description or "stub description",
            )

        monkeypatch.setattr("dao_ai.tools.create_responses_agent_tool", _stub)

        m = ToolModel.model_validate(
            {
                "name": "delegate",
                "function": {
                    "type": "agent",
                    "app": {"name": "dao-ai-some-app", "on_behalf_of_user": True},
                    "name": "delegate_tool",
                    "description": "Delegate to the deployed dao-ai app.",
                },
            }
        )
        assert isinstance(m.function, AgentToolModel)
        assert isinstance(m.function.app, DatabricksAppModel)
        assert m.function.app.name == "dao-ai-some-app"
        assert m.function.endpoint is None

        tools = m.function.as_tools()
        assert len(tools) == 1
        assert tools[0].name == "delegate_tool"
        # The Responses-API factory was called, NOT the A2A factory.
        assert isinstance(captured["app"], DatabricksAppModel)
        assert captured["app"].name == "dao-ai-some-app"
        assert captured["name"] == "delegate_tool"
        assert captured["description"] == "Delegate to the deployed dao-ai app."

    def test_agent_app_rejects_mcp_prefix_at_factory_time(self) -> None:
        """mcp- prefix apps should error with a clear pointer to type: mcp."""
        m = AgentToolModel(
            type=FunctionType.AGENT,
            app={"name": "mcp-hardware-store"},
        )
        with pytest.raises(ValueError, match="type: mcp"):
            m.as_tools()

    def test_agent_api_defaults_to_none_and_app_resolves_to_responses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`api:` defaults to None (auto); the app: branch resolves to responses."""
        from langchain_core.tools import StructuredTool

        called: dict[str, int] = {"responses": 0, "completions": 0}

        def _responses_stub(app, *, name=None, description=None):  # type: ignore[no-untyped-def]
            called["responses"] += 1
            return StructuredTool.from_function(
                func=lambda prompt: "stub", name=name or "stub", description="stub"
            )

        def _completions_stub(*args, **kwargs):  # type: ignore[no-untyped-def]
            called["completions"] += 1
            raise AssertionError("completions factory must not be called for api=None on app:")

        monkeypatch.setattr("dao_ai.tools.create_responses_agent_tool", _responses_stub)
        monkeypatch.setattr(
            "dao_ai.tools.create_chat_completions_agent_tool", _completions_stub
        )

        m = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "agent",
                    "app": {"name": "my-app", "on_behalf_of_user": True},
                },
            }
        )
        assert isinstance(m.function, AgentToolModel)
        assert m.function.api is None
        m.function.as_tools()
        assert called["responses"] == 1
        assert called["completions"] == 0

    def test_agent_api_completions_dispatches_to_chat_completions_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`api: completions` routes through create_chat_completions_agent_tool."""
        from dao_ai.config import DatabricksAppModel
        from langchain_core.tools import StructuredTool

        captured: dict[str, object] = {}

        def _completions_stub(
            app, *, name=None, description=None
        ):  # type: ignore[no-untyped-def]
            captured["app"] = app
            captured["name"] = name
            captured["description"] = description
            return StructuredTool.from_function(
                func=lambda prompt: "stub",
                name=name or "stub",
                description=description or "stub",
            )

        def _responses_stub(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError(
                "create_responses_agent_tool must not be called when api=completions"
            )

        monkeypatch.setattr(
            "dao_ai.tools.create_chat_completions_agent_tool", _completions_stub
        )
        monkeypatch.setattr(
            "dao_ai.tools.create_responses_agent_tool", _responses_stub
        )

        m = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "agent",
                    "app": {"name": "legacy-app", "on_behalf_of_user": True},
                    "api": "completions",
                    "name": "legacy_tool",
                },
            }
        )
        tools = m.function.as_tools()
        assert len(tools) == 1
        assert tools[0].name == "legacy_tool"
        assert isinstance(captured["app"], DatabricksAppModel)
        assert captured["app"].name == "legacy-app"

    def test_agent_api_responses_explicit_dispatches_to_responses_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`api: responses` (explicit) routes through create_responses_agent_tool."""
        from langchain_core.tools import StructuredTool

        call_count: dict[str, int] = {"responses": 0, "completions": 0}

        def _responses_stub(
            app, *, name=None, description=None
        ):  # type: ignore[no-untyped-def]
            call_count["responses"] += 1
            return StructuredTool.from_function(
                func=lambda prompt: "stub",
                name=name or "stub",
                description=description or "stub",
            )

        def _completions_stub(*args, **kwargs):  # type: ignore[no-untyped-def]
            call_count["completions"] += 1
            raise AssertionError("completions factory called for api=responses")

        monkeypatch.setattr(
            "dao_ai.tools.create_responses_agent_tool", _responses_stub
        )
        monkeypatch.setattr(
            "dao_ai.tools.create_chat_completions_agent_tool", _completions_stub
        )

        m = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "agent",
                    "app": {"name": "modern-app", "on_behalf_of_user": True},
                    "api": "responses",
                },
            }
        )
        m.function.as_tools()
        assert call_count["responses"] == 1
        assert call_count["completions"] == 0

    def test_agent_api_rejects_invalid_value(self) -> None:
        """Unknown api: values are rejected by Pydantic at validate-time."""
        with pytest.raises(Exception):
            ToolModel.model_validate(
                {
                    "name": "a",
                    "function": {
                        "type": "agent",
                        "app": {"name": "x", "on_behalf_of_user": True},
                        "api": "rest",  # not allowed
                    },
                }
            )

    def test_agent_endpoint_default_passes_auto_detect_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`endpoint:` with api=None → auto_detect_responses_api=True."""
        from langchain_core.tools import StructuredTool

        captured: dict[str, object] = {}

        def _stub(llm, *, name=None, description=None, auto_detect_responses_api=False):  # type: ignore[no-untyped-def]
            captured["llm"] = llm
            captured["auto_detect"] = auto_detect_responses_api
            return StructuredTool.from_function(
                func=lambda prompt: "stub", name=name or "stub", description="stub"
            )

        monkeypatch.setattr("dao_ai.tools.create_agent_endpoint_tool", _stub)
        m = ToolModel.model_validate(
            {
                "name": "a",
                "function": {"type": "agent", "endpoint": "my-endpoint"},
            }
        )
        m.function.as_tools()
        assert captured["auto_detect"] is True
        # llm.use_responses_api unchanged from default (False) when api=None.
        assert captured["llm"].use_responses_api is False  # type: ignore[union-attr]

    def test_agent_endpoint_api_responses_forces_use_responses_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`endpoint:` + api=responses → llm.use_responses_api=True, no auto-detect."""
        from langchain_core.tools import StructuredTool

        captured: dict[str, object] = {}

        def _stub(llm, *, name=None, description=None, auto_detect_responses_api=False):  # type: ignore[no-untyped-def]
            captured["llm"] = llm
            captured["auto_detect"] = auto_detect_responses_api
            return StructuredTool.from_function(
                func=lambda prompt: "stub", name=name or "stub", description="stub"
            )

        monkeypatch.setattr("dao_ai.tools.create_agent_endpoint_tool", _stub)
        m = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "agent",
                    "endpoint": "hardware_store_dao",
                    "api": "responses",
                },
            }
        )
        m.function.as_tools()
        assert captured["auto_detect"] is False
        assert captured["llm"].use_responses_api is True  # type: ignore[union-attr]

    def test_agent_endpoint_api_completions_forces_legacy_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`endpoint:` + api=completions → llm.use_responses_api=False, no auto-detect."""
        from langchain_core.tools import StructuredTool

        captured: dict[str, object] = {}

        def _stub(llm, *, name=None, description=None, auto_detect_responses_api=False):  # type: ignore[no-untyped-def]
            captured["llm"] = llm
            captured["auto_detect"] = auto_detect_responses_api
            return StructuredTool.from_function(
                func=lambda prompt: "stub", name=name or "stub", description="stub"
            )

        monkeypatch.setattr("dao_ai.tools.create_agent_endpoint_tool", _stub)
        m = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "agent",
                    "endpoint": "databricks-claude-sonnet-4",
                    "api": "completions",
                },
            }
        )
        m.function.as_tools()
        assert captured["auto_detect"] is False
        assert captured["llm"].use_responses_api is False  # type: ignore[union-attr]

    def test_agent_app_does_not_dispatch_to_a2a(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: type: agent + app: MUST NOT call create_a2a_agent_tool.

        Earlier PR #126 routed through A2A; this assertion locks in the
        new behavior (Responses API). type: a2a remains the explicit A2A
        path.
        """
        call_count: dict[str, int] = {"a2a": 0, "responses": 0}

        def _a2a_stub(*args, **kwargs):  # type: ignore[no-untyped-def]
            call_count["a2a"] += 1
            raise AssertionError(
                "create_a2a_agent_tool must not be called from type: agent"
            )

        from langchain_core.tools import StructuredTool

        def _responses_stub(
            app, *, name=None, description=None
        ):  # type: ignore[no-untyped-def]
            call_count["responses"] += 1
            return StructuredTool.from_function(
                func=lambda prompt: "stub",
                name=name or "stub",
                description=description or "stub",
            )

        monkeypatch.setattr("dao_ai.tools.create_a2a_agent_tool", _a2a_stub)
        monkeypatch.setattr(
            "dao_ai.tools.create_responses_agent_tool", _responses_stub
        )

        m = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "agent",
                    "app": {"name": "dao-ai-supplier-app", "on_behalf_of_user": True},
                },
            }
        )
        m.function.as_tools()

        assert call_count["a2a"] == 0
        assert call_count["responses"] == 1

    def test_agent_accepts_obo_toggle(self) -> None:
        m = ToolModel.model_validate(
            {
                "name": "user_agent",
                "function": {
                    "type": "agent",
                    "endpoint": "my-agent",
                    "on_behalf_of_user": True,
                },
            }
        )
        assert m.function.on_behalf_of_user is True

    def test_agent_accepts_human_in_the_loop(self) -> None:
        m = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "agent",
                    "endpoint": "my-agent",
                    "human_in_the_loop": {"review_prompt": "Approve call?"},
                },
            }
        )
        assert m.function.human_in_the_loop is not None

    def test_agent_as_tools_returns_structured_tool(self) -> None:
        m = AgentToolModel(
            type=FunctionType.AGENT,
            endpoint="my-agent",
            name="my_agent_tool",
            description="Answers questions",
        )
        tools = m.as_tools()
        assert len(tools) == 1
        assert tools[0].name == "my_agent_tool"

    def test_agent_endpoint_accepts_full_inference_endpoint_model(self) -> None:
        """endpoint: can be a full InferenceEndpointModel (with temp / max_tokens / ai_gateway)."""
        from dao_ai.config import InferenceEndpointModel

        m = ToolModel.model_validate(
            {
                "name": "ka",
                "function": {
                    "type": "agent",
                    "endpoint": {
                        "name": "ka-customer-reviews",
                        "temperature": 0.7,
                        "max_tokens": 1000,
                    },
                    "name": "ka_tool",
                },
            }
        )
        assert isinstance(m.function, AgentToolModel)
        assert isinstance(m.function.endpoint, InferenceEndpointModel)
        assert m.function.endpoint.name == "ka-customer-reviews"
        assert m.function.endpoint.temperature == 0.7
        assert m.function.endpoint.max_tokens == 1000

        llm = m.function._resolved_llm()
        assert isinstance(llm, InferenceEndpointModel)
        assert llm.temperature == 0.7
        assert llm.max_tokens == 1000

        tools = m.function.as_tools()
        assert len(tools) == 1
        assert tools[0].name == "ka_tool"

    def test_agent_endpoint_string_promoted_to_inference_endpoint_model(self) -> None:
        """String endpoint: gets promoted to InferenceEndpointModel(name=...) internally."""
        from dao_ai.config import InferenceEndpointModel

        m = AgentToolModel(
            type=FunctionType.AGENT,
            endpoint="my-endpoint",
            on_behalf_of_user=True,
        )
        llm = m._resolved_llm()
        assert isinstance(llm, InferenceEndpointModel)
        assert llm.name == "my-endpoint"
        assert llm.on_behalf_of_user is True

    def test_agent_parity_with_factory(self) -> None:
        new = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "agent",
                    "endpoint": "my-agent",
                    "name": "my_agent_tool",
                },
            }
        )
        old = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "factory",
                    "name": "dao_ai.tools.create_agent_endpoint_tool",
                    "args": {
                        "llm": {"name": "my-agent"},
                        "name": "my_agent_tool",
                    },
                },
            }
        )
        new_tools = new.function.as_tools()
        old_tools = old.function.as_tools()
        assert len(new_tools) == len(old_tools) == 1
        assert [t.name for t in new_tools] == [t.name for t in old_tools]

    def test_agent_parity_with_factory_full_model(self) -> None:
        """type: agent with full InferenceEndpointModel matches the original factory shape."""
        llm = {
            "name": "agent-bricks-customer-support-endpoint",
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        new = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "agent",
                    "endpoint": llm,
                    "name": "customer_support_specialist",
                },
            }
        )
        old = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "factory",
                    "name": "dao_ai.tools.create_agent_endpoint_tool",
                    "args": {
                        "llm": llm,
                        "name": "customer_support_specialist",
                    },
                },
            }
        )
        new_tools = new.function.as_tools()
        old_tools = old.function.as_tools()
        assert len(new_tools) == len(old_tools) == 1
        assert [t.name for t in new_tools] == [t.name for t in old_tools]


# ---------------------------------------------------------------------------
# A2AToolModel — Google A2A v0.3 protocol
# ---------------------------------------------------------------------------


class TestA2AToolModel:
    def test_type_a2a_dispatches_to_a2a_tool_model(self) -> None:
        m = ToolModel.model_validate(
            {
                "name": "remote_agent",
                "function": {
                    "type": "a2a",
                    "endpoint": "https://agent.example.com",
                    "auth_type": "none",
                },
            }
        )
        assert isinstance(m.function, A2AToolModel)
        assert m.function.type == FunctionType.A2A.value
        assert m.function.endpoint == "https://agent.example.com"
        assert m.function.auth_type == "none"

    def test_a2a_requires_endpoint_or_app(self) -> None:
        with pytest.raises(Exception, match="endpoint.*app"):
            ToolModel.model_validate(
                {"name": "x", "function": {"type": "a2a"}}
            )

    def test_a2a_rejects_unknown_auth_type(self) -> None:
        with pytest.raises(Exception):
            ToolModel.model_validate(
                {
                    "name": "x",
                    "function": {
                        "type": "a2a",
                        "endpoint": "https://example.com",
                        "auth_type": "magic_token",
                    },
                }
            )

    def test_a2a_streaming_default_true(self) -> None:
        m = A2AToolModel(
            type=FunctionType.A2A,
            endpoint="https://agent.example.com",
            auth_type="none",
        )
        assert m.streaming is True
        assert m.timeout_seconds == 300

    def test_a2a_as_tools_mode_1_none_auth(self) -> None:
        m = A2AToolModel(
            type=FunctionType.A2A,
            endpoint="https://agent.example.com",
            auth_type="none",
            name="external_a2a",
        )
        tools = m.as_tools()
        assert len(tools) == 1
        assert tools[0].name == "external_a2a"

    def test_a2a_parity_with_factory(self) -> None:
        new = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "a2a",
                    "endpoint": "https://agent.example.com",
                    "auth_type": "none",
                    "name": "external_a2a",
                },
            }
        )
        old = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "factory",
                    "name": "dao_ai.tools.create_a2a_agent_tool",
                    "args": {
                        "endpoint": "https://agent.example.com",
                        "auth_type": "none",
                        "name": "external_a2a",
                    },
                },
            }
        )
        new_tools = new.function.as_tools()
        old_tools = old.function.as_tools()
        assert len(new_tools) == len(old_tools) == 1
        assert [t.name for t in new_tools] == [t.name for t in old_tools]

    def test_a2a_accepts_human_in_the_loop(self) -> None:
        m = ToolModel.model_validate(
            {
                "name": "a",
                "function": {
                    "type": "a2a",
                    "endpoint": "https://agent.example.com",
                    "auth_type": "none",
                    "human_in_the_loop": {"review_prompt": "Approve?"},
                },
            }
        )
        assert m.function.human_in_the_loop is not None
