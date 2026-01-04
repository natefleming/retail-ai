"""
Tests for message validation middleware.

This module tests the validation middleware with the input/output structure:
- thread_id and conversation_id are interchangeable in configurable
- session is for accumulated state only
"""

import json
from unittest.mock import MagicMock

import pytest

from dao_ai.middleware.message_validation import (
    CustomFieldValidationMiddleware,
    RequiredField,
    ThreadIdValidationMiddleware,
    UserIdValidationMiddleware,
    create_custom_field_validation_middleware,
)
from dao_ai.state import AgentState, Context


def create_mock_runtime(context: Context) -> MagicMock:
    """Create a mock runtime with the given context."""
    runtime = MagicMock()
    runtime.context = context
    return runtime


# =============================================================================
# UserIdValidationMiddleware Tests
# =============================================================================


@pytest.mark.unit
class TestUserIdValidationMiddleware:
    """Tests for UserIdValidationMiddleware."""

    def test_valid_user_id(self) -> None:
        """Test that a valid user_id passes validation."""
        middleware = UserIdValidationMiddleware()
        state: AgentState = {"messages": []}
        context = Context(user_id="test_user", thread_id="conv_1")
        runtime = create_mock_runtime(context)

        result = middleware.validate(state, runtime)

        assert result is None  # No error

    def test_missing_user_id_error_message(self) -> None:
        """Test error message format when user_id is missing."""
        middleware = UserIdValidationMiddleware()
        state: AgentState = {"messages": []}
        context = Context(user_id=None, thread_id="conv_1")
        runtime = create_mock_runtime(context)

        with pytest.raises(ValueError) as exc_info:
            middleware.validate(state, runtime)

        error = str(exc_info.value)

        # Should have thread_id in configurable and conversation_id in session
        assert "thread_id" in error
        assert "conversation_id" in error

        # Parse the JSON in the error message
        json_start = error.find("```json") + 7
        json_end = error.find("```", json_start)
        config_json = error[json_start:json_end].strip()
        config = json.loads(config_json)

        assert "configurable" in config
        assert "session" in config
        # thread_id in configurable, conversation_id in session
        assert "thread_id" in config["configurable"]
        assert "conversation_id" not in config["configurable"]
        # Session has conversation_id (alias of thread_id)
        assert config["session"]["conversation_id"] == "conv_1"

    def test_user_id_with_dots_error_message(self) -> None:
        """Test error message uses conversation_id when user_id has dots."""
        middleware = UserIdValidationMiddleware()
        state: AgentState = {"messages": []}
        context = Context(user_id="user.with.dots", thread_id="conv_1")
        runtime = create_mock_runtime(context)

        with pytest.raises(ValueError) as exc_info:
            middleware.validate(state, runtime)

        error = str(exc_info.value)

        # Should suggest corrected user_id
        assert "user_with_dots" in error
        assert "conversation_id" in error


# =============================================================================
# ThreadIdValidationMiddleware Tests
# =============================================================================


@pytest.mark.unit
class TestThreadIdValidationMiddleware:
    """Tests for ThreadIdValidationMiddleware."""

    def test_valid_thread_id(self) -> None:
        """Test that a valid thread_id passes validation."""
        middleware = ThreadIdValidationMiddleware()
        state: AgentState = {"messages": []}
        context = Context(user_id="test_user", thread_id="conv_1")
        runtime = create_mock_runtime(context)

        result = middleware.validate(state, runtime)

        assert result is None  # No error

    def test_missing_thread_id_error_message(self) -> None:
        """Test error message asks for thread_id or conversation_id."""
        middleware = ThreadIdValidationMiddleware()
        state: AgentState = {"messages": []}
        context = Context(user_id="test_user", thread_id=None)
        runtime = create_mock_runtime(context)

        with pytest.raises(ValueError) as exc_info:
            middleware.validate(state, runtime)

        error = str(exc_info.value)

        # Should mention both are acceptable (interchangeable)
        assert "thread_id" in error
        assert "conversation_id" in error


# =============================================================================
# CustomFieldValidationMiddleware Tests
# =============================================================================


@pytest.mark.unit
class TestCustomFieldValidationMiddleware:
    """Tests for CustomFieldValidationMiddleware."""

    def test_all_required_fields_present(self) -> None:
        """Test validation passes when all required fields are present."""
        middleware = CustomFieldValidationMiddleware(
            fields=[
                RequiredField(name="store_num", description="Store number"),
            ]
        )
        state: AgentState = {"messages": []}
        context = Context(
            user_id="test_user",
            thread_id="conv_1",
            store_num="87887",
        )
        runtime = create_mock_runtime(context)

        result = middleware.validate(state, runtime)

        assert result is None  # No error

    def test_missing_required_field_error_message(self) -> None:
        """Test error message format when required field is missing."""
        middleware = CustomFieldValidationMiddleware(
            fields=[
                RequiredField(
                    name="store_num",
                    description="Your store number",
                    example_value="12345",
                ),
            ]
        )
        state: AgentState = {"messages": []}
        context = Context(
            user_id="test_user",
            thread_id="conv_1",
            # Missing store_num
        )
        runtime = create_mock_runtime(context)

        with pytest.raises(ValueError) as exc_info:
            middleware.validate(state, runtime)

        error = str(exc_info.value)

        # Should include store_num as missing
        assert "store_num" in error

        # Parse the JSON in the error message
        json_start = error.find("```json") + 7
        json_end = error.find("```", json_start)
        config_json = error[json_start:json_end].strip()
        config = json.loads(config_json)

        # Check structure: thread_id in configurable, conversation_id in session
        assert "configurable" in config
        assert "session" in config

        cfg = config["configurable"]
        assert cfg["thread_id"] == "conv_1"
        assert "conversation_id" not in cfg  # conversation_id is in session
        assert cfg["user_id"] == "test_user"
        assert cfg["store_num"] == "12345"  # Example value

        # Session has conversation_id (alias of thread_id)
        assert config["session"]["conversation_id"] == "conv_1"

    def test_preserves_provided_values(self) -> None:
        """Test error message preserves values user already provided."""
        middleware = CustomFieldValidationMiddleware(
            fields=[
                RequiredField(name="field_a", example_value="a"),
                RequiredField(name="field_b", example_value="b"),
            ]
        )
        state: AgentState = {"messages": []}
        context = Context(
            user_id="test_user",
            thread_id="conv_1",
            field_a="my_value_a",  # Provided field_a, missing field_b
        )
        runtime = create_mock_runtime(context)

        with pytest.raises(ValueError) as exc_info:
            middleware.validate(state, runtime)

        error = str(exc_info.value)

        # Parse the JSON
        json_start = error.find("```json") + 7
        json_end = error.find("```", json_start)
        config_json = error[json_start:json_end].strip()
        config = json.loads(config_json)

        cfg = config["configurable"]

        # field_a should keep user's value
        assert cfg["field_a"] == "my_value_a"
        # field_b should show example
        assert cfg["field_b"] == "b"

    def test_optional_fields_not_validated(self) -> None:
        """Test that optional fields are not required."""
        middleware = CustomFieldValidationMiddleware(
            fields=[
                RequiredField(name="required_field", required=True),
                RequiredField(name="optional_field", required=False),
            ]
        )
        state: AgentState = {"messages": []}
        context = Context(
            user_id="test_user",
            thread_id="conv_1",
            required_field="value",  # Only required field provided
        )
        runtime = create_mock_runtime(context)

        result = middleware.validate(state, runtime)

        assert result is None  # Should pass

    def test_factory_function(self) -> None:
        """Test create_custom_field_validation_middleware factory."""
        middleware = create_custom_field_validation_middleware(
            fields=[
                {"name": "store_num", "description": "Store", "example_value": "123"},
            ]
        )

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, CustomFieldValidationMiddleware)
        assert len(middleware.fields) == 1
        assert middleware.fields[0].name == "store_num"
