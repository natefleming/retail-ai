import pytest

from dao_ai.hooks import null_hook


@pytest.mark.unit
def test_null_hook_returns_empty_dict() -> None:
    """Test that null_hook returns an empty dictionary."""
    state = {"messages": ["test"], "route": "search"}
    config = {"user_id": "123", "store_num": "456"}

    result = null_hook(state, config)

    assert result == {}
    assert isinstance(result, dict)


@pytest.mark.unit
def test_null_hook_with_empty_inputs() -> None:
    """Test null_hook with empty state and config."""
    result = null_hook({}, {})

    assert result == {}
    assert isinstance(result, dict)


@pytest.mark.unit
def test_null_hook_with_none_values() -> None:
    """Test null_hook with None values in inputs."""
    state = {"key": None, "other": "value"}
    config = {"setting": None}

    result = null_hook(state, config)

    assert result == {}
    assert isinstance(result, dict)


@pytest.mark.unit
def test_null_hook_does_not_modify_inputs() -> None:
    """Test that null_hook doesn't modify the input dictionaries."""
    original_state = {"messages": ["test"], "route": "search"}
    original_config = {"user_id": "123"}

    state = original_state.copy()
    config = original_config.copy()

    result = null_hook(state, config)

    # Inputs should remain unchanged
    assert state == original_state
    assert config == original_config
    assert result == {}


@pytest.mark.unit
def test_null_hook_with_complex_data_types() -> None:
    """Test null_hook with complex data types in state and config."""
    state = {
        "messages": [{"content": "hello", "type": "human"}],
        "context": [{"doc": "test", "metadata": {"id": 1}}],
        "nested": {"deep": {"value": 42}},
    }
    config = {
        "settings": {"timeout": 30, "retries": 3},
        "features": ["feature1", "feature2"],
    }

    result = null_hook(state, config)

    assert result == {}
    assert isinstance(result, dict)
