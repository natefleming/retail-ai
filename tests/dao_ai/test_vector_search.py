from unittest.mock import Mock, patch

import pytest

from dao_ai.vector_search import endpoint_exists, index_exists


@pytest.mark.unit
def test_endpoint_exists_with_matching_endpoint() -> None:
    """Test endpoint_exists when the endpoint exists in the list."""
    mock_vsc = Mock()
    mock_vsc.list_endpoints.return_value = {
        "endpoints": [
            {"name": "endpoint1"},
            {"name": "target_endpoint"},
            {"name": "endpoint3"},
        ]
    }

    result = endpoint_exists(mock_vsc, "target_endpoint")

    assert result is True
    mock_vsc.list_endpoints.assert_called_once()


@pytest.mark.unit
def test_endpoint_exists_with_no_matching_endpoint() -> None:
    """Test endpoint_exists when the endpoint doesn't exist in the list."""
    mock_vsc = Mock()
    mock_vsc.list_endpoints.return_value = {
        "endpoints": [
            {"name": "endpoint1"},
            {"name": "endpoint2"},
            {"name": "endpoint3"},
        ]
    }

    result = endpoint_exists(mock_vsc, "missing_endpoint")

    assert result is False
    mock_vsc.list_endpoints.assert_called_once()


@pytest.mark.unit
def test_endpoint_exists_with_empty_endpoints() -> None:
    """Test endpoint_exists when no endpoints are returned."""
    mock_vsc = Mock()
    mock_vsc.list_endpoints.return_value = {"endpoints": []}

    result = endpoint_exists(mock_vsc, "any_endpoint")

    assert result is False
    mock_vsc.list_endpoints.assert_called_once()


@pytest.mark.unit
def test_endpoint_exists_with_rate_limit_error() -> None:
    """Test endpoint_exists handles rate limit errors gracefully."""
    mock_vsc = Mock()
    mock_vsc.list_endpoints.side_effect = Exception(
        "REQUEST_LIMIT_EXCEEDED: Too many requests"
    )

    with patch("builtins.print") as mock_print:
        result = endpoint_exists(mock_vsc, "any_endpoint")

    assert result is True  # Should assume endpoint exists during rate limit
    mock_print.assert_called_once_with(
        "WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error."
    )


@pytest.mark.unit
def test_endpoint_exists_with_other_exception() -> None:
    """Test endpoint_exists re-raises non-rate-limit exceptions."""
    mock_vsc = Mock()
    mock_vsc.list_endpoints.side_effect = Exception("Some other error")

    with pytest.raises(Exception, match="Some other error"):
        endpoint_exists(mock_vsc, "any_endpoint")


@pytest.mark.unit
def test_endpoint_exists_with_missing_endpoints_key() -> None:
    """Test endpoint_exists when the response doesn't have endpoints key."""
    mock_vsc = Mock()
    mock_vsc.list_endpoints.return_value = {}

    result = endpoint_exists(mock_vsc, "any_endpoint")

    assert result is False


@pytest.mark.unit
def test_index_exists_when_index_exists() -> None:
    """Test index_exists when the index exists and describe succeeds."""
    mock_vsc = Mock()
    mock_index = Mock()
    mock_index.describe.return_value = {"status": "READY"}
    mock_vsc.get_index.return_value = mock_index

    result = index_exists(mock_vsc, "test_endpoint", "catalog.schema.table")

    assert result is True
    mock_vsc.get_index.assert_called_once_with("test_endpoint", "catalog.schema.table")
    mock_index.describe.assert_called_once()


@pytest.mark.unit
def test_index_exists_when_index_does_not_exist() -> None:
    """Test index_exists when the index doesn't exist."""
    mock_vsc = Mock()
    mock_index = Mock()
    mock_index.describe.side_effect = Exception(
        "RESOURCE_DOES_NOT_EXIST: Index not found"
    )
    mock_vsc.get_index.return_value = mock_index

    result = index_exists(mock_vsc, "test_endpoint", "catalog.schema.missing_table")

    assert result is False
    mock_vsc.get_index.assert_called_once_with(
        "test_endpoint", "catalog.schema.missing_table"
    )
    mock_index.describe.assert_called_once()


@pytest.mark.unit
def test_index_exists_with_permission_error() -> None:
    """Test index_exists with permission errors."""
    mock_vsc = Mock()
    mock_index = Mock()
    mock_index.describe.side_effect = Exception("PERMISSION_DENIED: Access denied")
    mock_vsc.get_index.return_value = mock_index

    with patch("builtins.print") as mock_print:
        with pytest.raises(Exception, match="PERMISSION_DENIED: Access denied"):
            index_exists(mock_vsc, "test_endpoint", "catalog.schema.table")

    mock_print.assert_called_once_with(
        "Unexpected error describing the index. This could be a permission issue."
    )


@pytest.mark.unit
def test_index_exists_with_other_unexpected_error() -> None:
    """Test index_exists with other unexpected errors."""
    mock_vsc = Mock()
    mock_index = Mock()
    mock_index.describe.side_effect = Exception("UNKNOWN_ERROR: Something went wrong")
    mock_vsc.get_index.return_value = mock_index

    with patch("builtins.print") as mock_print:
        with pytest.raises(Exception, match="UNKNOWN_ERROR: Something went wrong"):
            index_exists(mock_vsc, "test_endpoint", "catalog.schema.table")

    mock_print.assert_called_once_with(
        "Unexpected error describing the index. This could be a permission issue."
    )


@pytest.mark.unit
def test_index_exists_get_index_failure() -> None:
    """Test index_exists when get_index itself fails."""
    mock_vsc = Mock()
    mock_vsc.get_index.side_effect = Exception(
        "ENDPOINT_NOT_FOUND: Endpoint doesn't exist"
    )

    with pytest.raises(Exception, match="ENDPOINT_NOT_FOUND: Endpoint doesn't exist"):
        index_exists(mock_vsc, "missing_endpoint", "catalog.schema.table")
