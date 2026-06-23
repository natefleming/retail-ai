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

    def test_agent_requires_endpoint(self) -> None:
        with pytest.raises(Exception):
            ToolModel.model_validate({"name": "x", "function": {"type": "agent"}})

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
