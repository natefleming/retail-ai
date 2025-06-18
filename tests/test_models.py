from unittest.mock import MagicMock, patch

import pytest

from dao_ai.models import get_latest_model_version


@pytest.mark.unit
def test_get_latest_model_version_single_version() -> None:
    """Test getting latest version when only one version exists."""
    with patch("dao_ai.models.MlflowClient") as mock_client:
        # Mock model version
        mock_version = MagicMock()
        mock_version.version = "1"

        mock_instance = mock_client.return_value
        mock_instance.search_model_versions.return_value = [mock_version]

        result = get_latest_model_version("test_model")

        assert result == 1
        mock_instance.search_model_versions.assert_called_once_with("name='test_model'")


@pytest.mark.unit
def test_get_latest_model_version_multiple_versions() -> None:
    """Test getting latest version when multiple versions exist."""
    with patch("dao_ai.models.MlflowClient") as mock_client:
        # Mock multiple model versions
        mock_versions = []
        for version in ["1", "3", "2", "5"]:
            mock_version = MagicMock()
            mock_version.version = version
            mock_versions.append(mock_version)

        mock_instance = mock_client.return_value
        mock_instance.search_model_versions.return_value = mock_versions

        result = get_latest_model_version("test_model")

        assert result == 5
        mock_instance.search_model_versions.assert_called_once_with("name='test_model'")


@pytest.mark.unit
def test_get_latest_model_version_no_versions() -> None:
    """Test getting latest version when no versions exist."""
    with patch("dao_ai.models.MlflowClient") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.search_model_versions.return_value = []

        result = get_latest_model_version("nonexistent_model")

        # Should return 1 as default
        assert result == 1
        mock_instance.search_model_versions.assert_called_once_with(
            "name='nonexistent_model'"
        )


@pytest.mark.unit
def test_get_latest_model_version_string_versions() -> None:
    """Test getting latest version with version numbers as strings."""
    with patch("dao_ai.models.MlflowClient") as mock_client:
        # Mock model versions with string version numbers
        mock_versions = []
        for version in ["10", "2", "21", "1"]:
            mock_version = MagicMock()
            mock_version.version = version
            mock_versions.append(mock_version)

        mock_instance = mock_client.return_value
        mock_instance.search_model_versions.return_value = mock_versions

        result = get_latest_model_version("test_model")

        # Should correctly identify 21 as the highest
        assert result == 21
