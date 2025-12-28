"""Tests for type_from_fqn utility function."""

import pytest
from pydantic import BaseModel

from dao_ai.utils import type_from_fqn


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


def test_type_from_fqn_pydantic():
    """Test loading Pydantic BaseModel type."""
    result = type_from_fqn("pydantic.BaseModel")
    assert result == BaseModel
    assert isinstance(result, type)


def test_type_from_fqn_builtin():
    """Test loading built-in type."""
    result = type_from_fqn("builtins.dict")
    assert result is dict
    assert isinstance(result, type)


def test_type_from_fqn_invalid_format():
    """Test error when format is invalid (no dot)."""
    with pytest.raises(ValueError, match="Expected format"):
        type_from_fqn("InvalidFormat")


def test_type_from_fqn_module_not_found():
    """Test error when module doesn't exist."""
    with pytest.raises(ImportError, match="Could not import module"):
        type_from_fqn("nonexistent_module.SomeClass")


def test_type_from_fqn_class_not_found():
    """Test error when class doesn't exist in module."""
    with pytest.raises(AttributeError, match="does not have attribute"):
        type_from_fqn("pydantic.NonexistentClass")


def test_type_from_fqn_not_a_type():
    """Test error when resolved object is not a type."""
    with pytest.raises(TypeError, match="is not a type"):
        # os.path is a module, not a type
        type_from_fqn("os.path")


def test_type_from_fqn_dao_config():
    """Test loading DAO config type."""
    from dao_ai.config import ResponseFormatModel

    result = type_from_fqn("dao_ai.config.ResponseFormatModel")
    assert result == ResponseFormatModel
    assert isinstance(result, type)


def test_type_from_fqn_instantiation():
    """Test that loaded type can be instantiated."""
    result = type_from_fqn("pydantic.BaseModel")

    # Create a simple subclass and instantiate it
    class TestModel(result):
        value: str

    instance = TestModel(value="test")
    assert instance.value == "test"
