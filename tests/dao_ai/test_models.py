from unittest.mock import MagicMock, patch

import pytest

from dao_ai.models import (
    _extract_text_content,
    get_latest_model_version,
)


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


class TestExtractTextContent:
    """Tests for _extract_text_content which normalizes Claude content blocks."""

    @pytest.mark.unit
    def test_string_passthrough(self) -> None:
        assert _extract_text_content("hello world") == "hello world"

    @pytest.mark.unit
    def test_single_text_block(self) -> None:
        content = [{"type": "text", "text": "hello"}]
        assert _extract_text_content(content) == "hello"

    @pytest.mark.unit
    def test_reasoning_block_formatted_as_markdown(self) -> None:
        content = [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking..."}],
            },
            {"type": "text", "text": "Got it!"},
        ]
        result = _extract_text_content(content)
        assert "Got it!" in result
        assert "> *thinking...*" in result

    @pytest.mark.unit
    def test_multiple_text_blocks_concatenated(self) -> None:
        content = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world!"},
        ]
        assert _extract_text_content(content) == "Hello world!"

    @pytest.mark.unit
    def test_empty_list(self) -> None:
        assert _extract_text_content([]) == ""

    @pytest.mark.unit
    def test_mixed_string_and_dict_blocks(self) -> None:
        content = ["raw string", {"type": "text", "text": " and dict"}]
        assert _extract_text_content(content) == "raw string and dict"

    @pytest.mark.unit
    def test_non_list_non_string_fallback(self) -> None:
        assert _extract_text_content(42) == "42"
