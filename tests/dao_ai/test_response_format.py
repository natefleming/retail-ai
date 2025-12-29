"""Tests for ResponseFormatModel response_schema validation."""

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from dao_ai.config import ResponseFormatModel


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


@dataclass
class SampleDataclass:
    """Sample dataclass for testing."""

    name: str
    value: int


# Test 1: Direct type (Pydantic model)
def test_direct_type_pydantic():
    """Test response_schema with direct Pydantic model reference."""
    response_format = ResponseFormatModel(response_schema=SampleModel)
    assert response_format.response_schema == SampleModel
    assert isinstance(response_format.response_schema, type)
    assert response_format.is_type_schema is True
    assert response_format.is_json_schema is False
    assert response_format.use_tool is None  # Default: auto-detect


# Test 2: Direct type (built-in)
def test_direct_type_builtin():
    """Test response_schema with built-in type."""
    response_format = ResponseFormatModel(response_schema=dict)
    assert response_format.response_schema is dict
    assert isinstance(response_format.response_schema, type)
    assert response_format.is_type_schema is True
    assert response_format.is_json_schema is False


# Test 3: Direct type (dataclass)
def test_direct_type_dataclass():
    """Test response_schema with dataclass."""
    response_format = ResponseFormatModel(response_schema=SampleDataclass)
    assert response_format.response_schema == SampleDataclass
    assert isinstance(response_format.response_schema, type)
    assert response_format.is_type_schema is True
    assert response_format.is_json_schema is False


# Test 4: String as FQN (Pydantic)
def test_string_fqn_pydantic():
    """Test response_schema with string FQN that resolves to Pydantic model."""
    response_format = ResponseFormatModel(response_schema="pydantic.BaseModel")
    assert response_format.response_schema == BaseModel
    assert isinstance(response_format.response_schema, type)
    assert response_format.is_type_schema is True
    assert response_format.is_json_schema is False


# Test 5: String as FQN (DAO module)
def test_string_fqn_dao_module():
    """Test response_schema with string FQN from dao_ai module."""
    response_format = ResponseFormatModel(
        response_schema="dao_ai.config.ResponseFormatModel"
    )
    assert response_format.response_schema == ResponseFormatModel
    assert isinstance(response_format.response_schema, type)
    assert response_format.is_type_schema is True
    assert response_format.is_json_schema is False


# Test 6: String as FQN (built-in)
def test_string_fqn_builtin():
    """Test response_schema with string FQN for built-in types."""
    response_format = ResponseFormatModel(response_schema="builtins.dict")
    assert response_format.response_schema is dict
    assert isinstance(response_format.response_schema, type)
    assert response_format.is_type_schema is True
    assert response_format.is_json_schema is False


# Test 7: String fallback to JSON schema (invalid FQN format)
def test_string_fallback_invalid_fqn():
    """Test that invalid FQN strings fall back to JSON schema."""
    response_format = ResponseFormatModel(response_schema="not_a_valid_fqn")
    assert response_format.response_schema == "not_a_valid_fqn"
    assert isinstance(response_format.response_schema, str)
    assert response_format.is_type_schema is False
    assert response_format.is_json_schema is True


# Test 8: String fallback to JSON schema (module not found)
def test_string_fallback_module_not_found():
    """Test that non-existent modules fall back to JSON schema."""
    response_format = ResponseFormatModel(
        response_schema="nonexistent.module.SomeClass"
    )
    assert response_format.response_schema == "nonexistent.module.SomeClass"
    assert isinstance(response_format.response_schema, str)
    assert response_format.is_type_schema is False
    assert response_format.is_json_schema is True


# Test 9: String fallback to JSON schema (class not found)
def test_string_fallback_class_not_found():
    """Test that non-existent classes fall back to JSON schema."""
    response_format = ResponseFormatModel(response_schema="pydantic.NonexistentClass")
    assert response_format.response_schema == "pydantic.NonexistentClass"
    assert isinstance(response_format.response_schema, str)
    assert response_format.is_type_schema is False
    assert response_format.is_json_schema is True


# Test 10: String fallback to JSON schema (not a type - module)
def test_string_fallback_not_a_type():
    """Test that non-type objects fall back to JSON schema."""
    # os.path is a module, not a type
    response_format = ResponseFormatModel(response_schema="os.path")
    assert response_format.response_schema == "os.path"
    assert isinstance(response_format.response_schema, str)
    assert response_format.is_type_schema is False
    assert response_format.is_json_schema is True


# Test 11: String as JSON schema (intentional)
def test_string_as_json_schema_intentional():
    """Test providing an intentional JSON schema string."""
    json_schema_str = '{"type": "object", "properties": {"name": {"type": "string"}}}'
    response_format = ResponseFormatModel(response_schema=json_schema_str)
    assert response_format.response_schema == json_schema_str
    assert isinstance(response_format.response_schema, str)
    assert response_format.is_type_schema is False
    assert response_format.is_json_schema is True


# Test 12: None value (optional)
def test_none_value():
    """Test response_schema with None value (optional field)."""
    response_format = ResponseFormatModel(response_schema=None)
    assert response_format.response_schema is None
    assert response_format.is_type_schema is False
    assert response_format.is_json_schema is False


# Test 13: Empty ResponseFormatModel (all optional)
def test_empty_response_format():
    """Test ResponseFormatModel with no fields set."""
    response_format = ResponseFormatModel()
    assert response_format.response_schema is None
    assert response_format.use_tool is None  # Default: auto-detect
    assert response_format.is_type_schema is False
    assert response_format.is_json_schema is False


# Test 14: response_schema with other fields
def test_response_schema_with_other_fields():
    """Test response_schema works alongside other ResponseFormatModel fields."""
    response_format = ResponseFormatModel(response_schema=SampleModel, use_tool=True)
    assert response_format.response_schema == SampleModel
    assert response_format.use_tool is True
    assert response_format.is_type_schema is True


# Test 15: Invalid type
def test_invalid_type():
    """Test that invalid types raise ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ResponseFormatModel(response_schema=123)


def test_invalid_type_list():
    """Test that list types raise ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ResponseFormatModel(response_schema=[SampleModel])


def test_invalid_type_dict():
    """Test that dict instances raise ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ResponseFormatModel(response_schema={"key": "value"})


# Test use_tool attribute
def test_use_tool_auto_detect():
    """Test use_tool=None (auto-detect, default)."""
    response_format = ResponseFormatModel(response_schema=SampleModel)
    assert response_format.use_tool is None


def test_use_tool_force_provider():
    """Test use_tool=False (force ProviderStrategy)."""
    response_format = ResponseFormatModel(response_schema=SampleModel, use_tool=False)
    assert response_format.use_tool is False


def test_use_tool_force_tool():
    """Test use_tool=True (force ToolStrategy)."""
    response_format = ResponseFormatModel(response_schema=SampleModel, use_tool=True)
    assert response_format.use_tool is True
