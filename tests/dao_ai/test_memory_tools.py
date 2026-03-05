"""Tests for Databricks-compatible memory tool wrappers and memory schemas."""

import asyncio
import json
import uuid

import pytest

from dao_ai.memory.schemas import (
    SCHEMA_REGISTRY,
    EpisodeMemory,
    PreferenceMemory,
    UserProfile,
    resolve_schemas,
)
from dao_ai.tools.memory import create_manage_memory_tool, create_search_memory_tool

NAMESPACE: tuple[str, ...] = ("memory", "test")


class TestCreateSearchMemoryTool:
    """Tests for the search_memory tool wrapper."""

    @pytest.mark.unit
    def test_tool_name(self) -> None:
        tool = create_search_memory_tool(namespace=NAMESPACE)
        assert tool.name == "search_memory"

    @pytest.mark.unit
    def test_schema_has_no_additional_properties(self) -> None:
        tool = create_search_memory_tool(namespace=NAMESPACE)
        schema = tool.args_schema.model_json_schema()
        raw = json.dumps(schema)
        assert "additionalProperties" not in raw

    @pytest.mark.unit
    def test_schema_has_no_any_of(self) -> None:
        tool = create_search_memory_tool(namespace=NAMESPACE)
        schema = tool.args_schema.model_json_schema()
        raw = json.dumps(schema)
        assert "anyOf" not in raw

    @pytest.mark.unit
    def test_schema_fields(self) -> None:
        tool = create_search_memory_tool(namespace=NAMESPACE)
        schema = tool.args_schema.model_json_schema()
        props = schema["properties"]

        assert "query" in props
        assert props["query"]["type"] == "string"

        assert "limit" in props
        assert props["limit"]["type"] == "integer"

        assert "offset" in props
        assert props["offset"]["type"] == "integer"

    @pytest.mark.unit
    def test_schema_omits_filter_field(self) -> None:
        tool = create_search_memory_tool(namespace=NAMESPACE)
        schema = tool.args_schema.model_json_schema()
        assert "filter" not in schema["properties"]


class TestCreateManageMemoryTool:
    """Tests for the manage_memory tool wrapper."""

    @pytest.mark.unit
    def test_tool_name(self) -> None:
        tool = create_manage_memory_tool(namespace=NAMESPACE)
        assert tool.name == "manage_memory"

    @pytest.mark.unit
    def test_schema_has_no_any_of(self) -> None:
        tool = create_manage_memory_tool(namespace=NAMESPACE)
        schema = tool.args_schema.model_json_schema()
        raw = json.dumps(schema)
        assert "anyOf" not in raw, f"Schema still contains anyOf: {raw}"

    @pytest.mark.unit
    def test_schema_has_no_uuid_format(self) -> None:
        tool = create_manage_memory_tool(namespace=NAMESPACE)
        schema = tool.args_schema.model_json_schema()
        raw = json.dumps(schema)
        assert '"format": "uuid"' not in raw

    @pytest.mark.unit
    def test_schema_content_field(self) -> None:
        tool = create_manage_memory_tool(namespace=NAMESPACE)
        schema = tool.args_schema.model_json_schema()
        props = schema["properties"]

        assert "content" in props
        assert props["content"]["type"] == "string"
        assert "content" in schema.get("required", [])

    @pytest.mark.unit
    def test_schema_action_field(self) -> None:
        tool = create_manage_memory_tool(namespace=NAMESPACE)
        schema = tool.args_schema.model_json_schema()
        props = schema["properties"]

        assert "action" in props
        assert props["action"]["type"] == "string"
        assert set(props["action"]["enum"]) == {"create", "update", "delete"}
        assert props["action"]["default"] == "create"

    @pytest.mark.unit
    def test_schema_id_field_is_simple_string(self) -> None:
        tool = create_manage_memory_tool(namespace=NAMESPACE)
        schema = tool.args_schema.model_json_schema()
        props = schema["properties"]

        assert "id" in props
        assert props["id"]["type"] == "string"
        assert props["id"]["default"] == ""

    @pytest.mark.unit
    def test_handle_tool_error_enabled(self) -> None:
        tool = create_manage_memory_tool(namespace=NAMESPACE)
        assert tool.handle_tool_error is True

    @pytest.mark.unit
    def test_invalid_uuid_returns_error_string(self) -> None:
        tool = create_manage_memory_tool(namespace=NAMESPACE)
        result = asyncio.run(
            tool.coroutine(
                content="test memory",
                action="update",
                id="previous conversation ID",
            )
        )
        assert isinstance(result, str)
        assert "not a valid memory ID" in result
        assert "previous conversation ID" in result

    @pytest.mark.unit
    def test_valid_uuid_accepted_by_schema(self) -> None:
        tool = create_manage_memory_tool(namespace=NAMESPACE)
        valid_id = str(uuid.uuid4())
        validated = tool.args_schema.model_validate(
            {"content": "test", "action": "update", "id": valid_id}
        )
        assert validated.id == valid_id


class TestMemorySchemas:
    """Tests for structured memory schemas."""

    @pytest.mark.unit
    def test_user_profile_defaults(self) -> None:
        profile = UserProfile()
        assert profile.name == ""
        assert profile.preferred_name == ""
        assert profile.expertise == []
        assert profile.preferences == []
        assert profile.goals == []

    @pytest.mark.unit
    def test_user_profile_populated(self) -> None:
        profile = UserProfile(
            name="Alice",
            preferred_name="Ali",
            role="Data Engineer",
            organization="Acme Corp",
            communication_style="casual",
            expertise=["Python", "SQL"],
            preferences=["dark mode"],
            goals=["migrate to Databricks"],
            context="Working on a data pipeline project",
        )
        assert profile.name == "Alice"
        assert profile.expertise == ["Python", "SQL"]

    @pytest.mark.unit
    def test_preference_memory_required_fields(self) -> None:
        pref = PreferenceMemory(
            category="response_style",
            preference="concise and technical",
        )
        assert pref.category == "response_style"
        assert pref.context == ""

    @pytest.mark.unit
    def test_episode_memory_required_fields(self) -> None:
        episode = EpisodeMemory(
            situation="User asked about query optimization",
            approach="Used EXPLAIN ANALYZE to identify bottleneck",
            outcome="User applied the suggestion and saw 10x improvement",
        )
        assert episode.lesson == ""

    @pytest.mark.unit
    def test_schema_registry_keys(self) -> None:
        assert "user_profile" in SCHEMA_REGISTRY
        assert "preference" in SCHEMA_REGISTRY
        assert "episode" in SCHEMA_REGISTRY
        assert SCHEMA_REGISTRY["user_profile"] is UserProfile
        assert SCHEMA_REGISTRY["preference"] is PreferenceMemory
        assert SCHEMA_REGISTRY["episode"] is EpisodeMemory

    @pytest.mark.unit
    def test_resolve_schemas_all(self) -> None:
        schemas = resolve_schemas(["user_profile", "preference", "episode"])
        assert schemas == [UserProfile, PreferenceMemory, EpisodeMemory]

    @pytest.mark.unit
    def test_resolve_schemas_partial(self) -> None:
        schemas = resolve_schemas(["user_profile"])
        assert schemas == [UserProfile]

    @pytest.mark.unit
    def test_resolve_schemas_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown memory schema 'nonexistent'"):
            resolve_schemas(["nonexistent"])
