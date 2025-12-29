"""Tests for AgentModel.response_format validation and conversion."""

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from dao_ai.config import AgentModel, LLMModel, ResponseFormatModel


class SamplePydanticModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


@dataclass
class SampleDataclass:
    """Sample dataclass for testing."""

    name: str
    value: int


def create_minimal_agent(**kwargs) -> AgentModel:
    """Helper to create a minimal AgentModel with only required fields."""
    defaults = {
        "name": "test_agent",
        "model": LLMModel(name="databricks-meta-llama-3-1-70b-instruct"),
    }
    defaults.update(kwargs)
    return AgentModel(**defaults)


# Test 1: None response_format (optional field)
def test_response_format_none():
    """Test that response_format can be None (optional)."""
    agent = create_minimal_agent(response_format=None)
    assert agent.response_format is None


# Test 2: Direct ResponseFormatModel
def test_response_format_as_response_format_model():
    """Test providing a ResponseFormatModel directly."""
    response_format = ResponseFormatModel(response_schema=SamplePydanticModel)
    agent = create_minimal_agent(response_format=response_format)

    assert isinstance(agent.response_format, ResponseFormatModel)
    assert agent.response_format.response_schema == SamplePydanticModel
    assert agent.response_format.is_type_schema is True


# Test 3: Direct type (Pydantic model)
def test_response_format_as_direct_type_pydantic():
    """Test providing a Pydantic model type directly."""
    agent = create_minimal_agent(response_format=SamplePydanticModel)

    assert isinstance(agent.response_format, ResponseFormatModel)
    assert agent.response_format.response_schema == SamplePydanticModel
    assert agent.response_format.is_type_schema is True


# Test 4: Direct type (dataclass)
def test_response_format_as_direct_type_dataclass():
    """Test providing a dataclass type directly."""
    agent = create_minimal_agent(response_format=SampleDataclass)

    assert isinstance(agent.response_format, ResponseFormatModel)
    assert agent.response_format.response_schema == SampleDataclass
    assert agent.response_format.is_type_schema is True


# Test 5: Direct type (built-in)
def test_response_format_as_direct_type_builtin():
    """Test providing a built-in type directly."""
    agent = create_minimal_agent(response_format=dict)

    assert isinstance(agent.response_format, ResponseFormatModel)
    assert agent.response_format.response_schema is dict
    assert agent.response_format.is_type_schema is True


# Test 6: String as FQN (fully qualified name) - Pydantic
def test_response_format_as_string_fqn_pydantic():
    """Test providing a string FQN that resolves to a Pydantic model."""
    agent = create_minimal_agent(response_format="pydantic.BaseModel")

    assert isinstance(agent.response_format, ResponseFormatModel)
    assert agent.response_format.response_schema == BaseModel
    assert agent.response_format.is_type_schema is True


# Test 7: String as FQN - DAO module
def test_response_format_as_string_fqn_dao_module():
    """Test providing a string FQN from dao_ai module."""
    agent = create_minimal_agent(response_format="dao_ai.config.ResponseFormatModel")

    assert isinstance(agent.response_format, ResponseFormatModel)
    assert agent.response_format.response_schema == ResponseFormatModel
    assert agent.response_format.is_type_schema is True


# Test 8: String as FQN - Built-in type
def test_response_format_as_string_fqn_builtin():
    """Test providing a string FQN for built-in types."""
    agent = create_minimal_agent(response_format="builtins.dict")

    assert isinstance(agent.response_format, ResponseFormatModel)
    assert agent.response_format.response_schema is dict
    assert agent.response_format.is_type_schema is True


# Test 9: String fallback to json_schema (invalid FQN format)
def test_response_format_string_fallback_invalid_fqn():
    """Test that invalid FQN strings fall back to JSON schema."""
    agent = create_minimal_agent(response_format="not_a_valid_fqn")

    assert isinstance(agent.response_format, ResponseFormatModel)
    assert agent.response_format.response_schema == "not_a_valid_fqn"
    assert agent.response_format.is_json_schema is True


# Test 10: String fallback to json_schema (module not found)
def test_response_format_string_fallback_module_not_found():
    """Test that non-existent modules fall back to JSON schema."""
    agent = create_minimal_agent(response_format="nonexistent.module.SomeClass")

    assert isinstance(agent.response_format, ResponseFormatModel)
    assert agent.response_format.response_schema == "nonexistent.module.SomeClass"
    assert agent.response_format.is_json_schema is True


# Test 11: String fallback to json_schema (class not found)
def test_response_format_string_fallback_class_not_found():
    """Test that non-existent classes fall back to JSON schema."""
    agent = create_minimal_agent(response_format="pydantic.NonexistentClass")

    assert isinstance(agent.response_format, ResponseFormatModel)
    assert agent.response_format.response_schema == "pydantic.NonexistentClass"
    assert agent.response_format.is_json_schema is True


# Test 12: String as json_schema (intentional)
def test_response_format_string_as_json_schema():
    """Test providing an intentional JSON schema string."""
    json_schema_str = '{"type": "object", "properties": {"name": {"type": "string"}}}'
    agent = create_minimal_agent(response_format=json_schema_str)

    assert isinstance(agent.response_format, ResponseFormatModel)
    assert agent.response_format.response_schema == json_schema_str
    assert agent.response_format.is_json_schema is True


# Test 13: Invalid type (not None, ResponseFormatModel, type, or str)
def test_response_format_invalid_type():
    """Test that invalid types raise ValidationError (Pydantic validates before our validator)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        create_minimal_agent(response_format=123)  # int is not valid


def test_response_format_invalid_type_list():
    """Test that list types raise ValidationError (Pydantic validates before our validator)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        create_minimal_agent(response_format=[SamplePydanticModel])


def test_response_format_invalid_type_dict():
    """Test that dict instances (with wrong keys) raise ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        create_minimal_agent(response_format={"key": "value"})


# Test 14: Complex scenario - ensure no mutation
def test_response_format_no_mutation_of_original():
    """Test that the original ResponseFormatModel is not mutated."""
    original = ResponseFormatModel(response_schema=SamplePydanticModel)
    agent1 = create_minimal_agent(response_format=original)
    agent2 = create_minimal_agent(response_format=original)

    # Both agents should reference the same ResponseFormatModel (no mutation)
    assert agent1.response_format is original
    assert agent2.response_format is original
    assert agent1.response_format is agent2.response_format


# Test 15: Multiple agents with different formats
def test_multiple_agents_different_formats():
    """Test creating multiple agents with different response formats."""
    agent1 = create_minimal_agent(name="agent1", response_format=SamplePydanticModel)
    agent2 = create_minimal_agent(name="agent2", response_format="pydantic.BaseModel")
    agent3 = create_minimal_agent(name="agent3", response_format="json_schema_string")
    agent4 = create_minimal_agent(name="agent4", response_format=None)

    assert agent1.response_format.response_schema == SamplePydanticModel
    assert agent1.response_format.is_type_schema is True

    assert agent2.response_format.response_schema == BaseModel
    assert agent2.response_format.is_type_schema is True

    assert agent3.response_format.response_schema == "json_schema_string"
    assert agent3.response_format.is_json_schema is True

    assert agent4.response_format is None


# Test 16: Edge case - type that is actually a module
def test_response_format_string_is_module_not_type():
    """Test that strings pointing to modules (not types) fall back to JSON schema."""
    agent = create_minimal_agent(response_format="os.path")

    # os.path is a module, not a type, so it should fall back to JSON schema
    assert isinstance(agent.response_format, ResponseFormatModel)
    assert agent.response_format.response_schema == "os.path"
    assert agent.response_format.is_json_schema is True


# Test 17: Verify proper fallback behavior
def test_response_format_fallback_behavior():
    """Test the fallback behavior when string cannot be resolved as type."""
    # Should fall back without raising exception
    agent = create_minimal_agent(response_format="invalid_format")

    assert agent.response_format.response_schema == "invalid_format"
    assert agent.response_format.is_json_schema is True


# Test 18: Test ResponseFormatModel with use_tool flag
def test_response_format_with_use_tool():
    """Test that ResponseFormatModel's other fields work correctly."""
    response_format = ResponseFormatModel(
        response_schema=SamplePydanticModel, use_tool=True
    )
    agent = create_minimal_agent(response_format=response_format)

    assert agent.response_format.response_schema == SamplePydanticModel
    assert agent.response_format.use_tool is True
    assert agent.response_format.is_type_schema is True
