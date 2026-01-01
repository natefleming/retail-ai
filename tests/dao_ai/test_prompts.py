"""Tests for MLflow Prompt Registry integration."""

from unittest.mock import Mock, patch

import pytest
from conftest import has_databricks_env

from dao_ai.config import PromptModel, SchemaModel
from dao_ai.providers.databricks import DatabricksProvider


class TestPromptRegistryUnit:
    """Unit tests for prompt registry functionality (mocked)."""

    @pytest.mark.unit
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    def test_get_prompt_with_alias(self):
        """Test loading a prompt using an alias."""
        prompt_model = PromptModel(
            name="test_prompt", alias="production", default_template="Default template"
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        # Mock the load_prompt function in config module where it's imported
        with patch("dao_ai.config.load_prompt") as mock_load:
            mock_prompt = Mock()
            mock_prompt.to_single_brace_format.return_value = (
                "Registry template content"
            )
            mock_load.return_value = mock_prompt

            result = provider.get_prompt(prompt_model)

            # Verify correct URI was used
            mock_load.assert_called_once_with("prompts:/test_prompt@production")
            assert result == mock_prompt
            assert result.to_single_brace_format() == "Registry template content"

    @pytest.mark.unit
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    def test_get_prompt_with_version(self):
        """Test loading a prompt using a specific version."""
        prompt_model = PromptModel(
            name="test_prompt", version=2, default_template="Default template"
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with patch("dao_ai.config.load_prompt") as mock_load:
            mock_prompt = Mock()
            mock_prompt.to_single_brace_format.return_value = "Version 2 content"
            mock_load.return_value = mock_prompt

            result = provider.get_prompt(prompt_model)

            # Verify correct URI was used
            mock_load.assert_called_once_with("prompts:/test_prompt/2")
            assert result == mock_prompt
            assert result.to_single_brace_format() == "Version 2 content"

    @pytest.mark.unit
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    def test_get_prompt_defaults_to_latest(self):
        """Test that prompt loading falls back to @latest alias when @champion doesn't exist."""
        prompt_model = PromptModel(
            name="test_prompt", default_template="Default template", auto_register=True
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with patch("dao_ai.providers.databricks.load_prompt") as mock_load:
            # Mock successful load on @latest alias
            mock_latest = Mock()
            mock_latest.version = "3"

            def load_side_effect(uri: str):
                if "@champion" in uri:
                    raise Exception("Champion alias not found")
                elif "@default" in uri:
                    raise Exception("Default alias not found")
                elif "@latest" in uri:
                    return mock_latest
                else:
                    raise Exception("Unexpected URI")

            mock_load.side_effect = load_side_effect

            result = provider.get_prompt(prompt_model)

            # Should try: default (for sync check), champion, default (for load), latest
            assert mock_load.call_count == 4
            mock_load.assert_any_call("prompts:/test_prompt@champion")
            mock_load.assert_any_call("prompts:/test_prompt@default")
            mock_load.assert_any_call("prompts:/test_prompt@latest")
            assert result == mock_latest

    @pytest.mark.unit
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    def test_get_prompt_prefers_champion_alias(self):
        """Test that prompt loading prefers @champion when it exists."""
        prompt_model = PromptModel(
            name="test_prompt", default_template="Default template", auto_register=True
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with patch("dao_ai.providers.databricks.load_prompt") as mock_load:
            mock_default_prompt = Mock()
            mock_default_prompt.template = "Default template"
            mock_champion_prompt = Mock()
            mock_champion_prompt.to_single_brace_format.return_value = (
                "Champion content"
            )

            def load_side_effect(uri: str):
                if "@default" in uri:
                    return mock_default_prompt
                elif "@champion" in uri:
                    return mock_champion_prompt
                else:
                    raise Exception("Unexpected URI")

            mock_load.side_effect = load_side_effect

            result = provider.get_prompt(prompt_model)

            # Should try default (for sync check), then champion
            assert mock_load.call_count >= 2
            mock_load.assert_any_call("prompts:/test_prompt@default")
            mock_load.assert_any_call("prompts:/test_prompt@champion")
            assert result == mock_champion_prompt
            assert result.to_single_brace_format() == "Champion content"

    @pytest.mark.unit
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    def test_get_prompt_fallback_to_default_template(self):
        """Test fallback to default_template when registry load fails."""
        prompt_model = PromptModel(
            name="test_prompt", default_template="Fallback template content"
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with patch("dao_ai.providers.databricks.load_prompt") as mock_load:
            # Simulate registry failure for all aliases
            mock_load.side_effect = Exception("Registry not found")

            result = provider.get_prompt(prompt_model)

            # Should fall back to using default_template directly
            from mlflow.entities.model_registry import PromptVersion

            assert isinstance(result, PromptVersion)
            assert result.name == "test_prompt"
            assert result.template == "Fallback template content"
            # version should be 1 (mock version)
            assert result.version == 1

    @pytest.mark.unit
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    def test_get_prompt_no_registry_no_default_raises_error(self):
        """Test that an error is raised when registry fails and no default_template."""
        prompt_model = PromptModel(
            name="test_prompt"
            # No default_template provided
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with patch("dao_ai.config.load_prompt") as mock_load:
            mock_load.side_effect = Exception("Registry not found")

            with pytest.raises(ValueError) as exc_info:
                provider.get_prompt(prompt_model)

            assert "Prompt 'test_prompt' not found in registry" in str(exc_info.value)
            assert "no default_template provided" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    def test_get_prompt_does_not_sync_when_registry_succeeds(self):
        """Test that default_template is NOT synced when champion alias exists."""
        template_content = "Same content in registry and config"
        prompt_model = PromptModel(
            name="test_prompt", default_template=template_content, auto_register=True
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with (
            patch("dao_ai.providers.databricks.load_prompt") as mock_load,
            patch.object(provider, "_register_default_template") as mock_register,
        ):
            mock_default_prompt = Mock()
            mock_default_prompt.template = template_content
            mock_champion_prompt = Mock()
            mock_champion_prompt.to_single_brace_format.return_value = (
                "Champion content"
            )

            def load_side_effect(uri: str):
                if "@default" in uri:
                    return mock_default_prompt
                elif "@champion" in uri:
                    return mock_champion_prompt
                else:
                    raise Exception("Unexpected URI")

            mock_load.side_effect = load_side_effect

            result = provider.get_prompt(prompt_model)

            # Should use registry content from champion and return PromptVersion
            assert result == mock_champion_prompt
            assert result.to_single_brace_format() == "Champion content"

            # Should call default (for sync check) and champion
            assert mock_load.call_count >= 2
            mock_load.assert_any_call("prompts:/test_prompt@default")
            mock_load.assert_any_call("prompts:/test_prompt@champion")

            # Should NOT register default_template when champion exists and template matches
            mock_register.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    def test_register_default_template_creates_new_version(self):
        """Test that _register_default_template creates a new prompt version."""
        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with (
            patch(
                "dao_ai.providers.databricks.mlflow.genai.register_prompt"
            ) as mock_register,
            patch(
                "dao_ai.providers.databricks.mlflow.genai.set_prompt_alias"
            ) as mock_set_alias,
        ):
            # Mock the registered prompt version
            mock_prompt_version = Mock()
            mock_prompt_version.version = 1
            mock_register.return_value = mock_prompt_version

            result = provider._register_default_template(
                "test_prompt", "New template content", None
            )

            # Should register the prompt with default commit message and dao_ai tag
            from dao_ai.utils import dao_ai_version

            mock_register.assert_called_once_with(
                name="test_prompt",
                template="New template content",
                commit_message="Auto-synced from default_template",
                tags={"dao_ai": dao_ai_version()},
            )

            # Should set default and champion aliases
            assert mock_set_alias.call_count == 2
            mock_set_alias.assert_any_call(
                name="test_prompt",
                alias="default",
                version=1,
            )
            mock_set_alias.assert_any_call(
                name="test_prompt",
                alias="champion",
                version=1,
            )

            # Should return the prompt version
            assert result == mock_prompt_version

    @pytest.mark.unit
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    def test_register_default_template_uses_description_as_commit_message(self):
        """Test that _register_default_template uses description as commit message."""
        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with (
            patch(
                "dao_ai.providers.databricks.mlflow.genai.register_prompt"
            ) as mock_register,
            patch(
                "dao_ai.providers.databricks.mlflow.genai.set_prompt_alias"
            ) as mock_set_alias,
        ):
            # Mock the registered prompt version
            mock_prompt_version = Mock()
            mock_prompt_version.version = 1
            mock_register.return_value = mock_prompt_version

            result = provider._register_default_template(
                "test_prompt",
                "New template content",
                "Custom description for commit message",
            )

            # Should register the prompt with custom description as commit message and dao_ai tag
            from dao_ai.utils import dao_ai_version

            mock_register.assert_called_once_with(
                name="test_prompt",
                template="New template content",
                commit_message="Custom description for commit message",
                tags={"dao_ai": dao_ai_version()},
            )

            # Should set default and champion aliases
            assert mock_set_alias.call_count == 2
            mock_set_alias.assert_any_call(
                name="test_prompt",
                alias="default",
                version=1,
            )
            mock_set_alias.assert_any_call(
                name="test_prompt",
                alias="champion",
                version=1,
            )

            # Should return the prompt version
            assert result == mock_prompt_version

    @pytest.mark.unit
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    def test_get_prompt_fallback_passes_description_to_sync(self):
        """Test that get_prompt passes description to sync when using fallback."""
        prompt_model = PromptModel(
            name="test_prompt",
            default_template="Fallback template content",
            description="Test prompt for hardware store",
            auto_register=True,
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with (
            patch("dao_ai.providers.databricks.load_prompt") as mock_load,
            patch.object(provider, "_register_default_template") as mock_register,
        ):
            # Simulate registry failure for all aliases
            mock_load.side_effect = Exception("Registry not found")

            result = provider.get_prompt(prompt_model)

            # Should fall back to using default_template directly
            from mlflow.entities.model_registry import PromptVersion

            assert isinstance(result, PromptVersion)
            assert result.name == "test_prompt"
            assert result.template == "Fallback template content"

            # Should have attempted to register with description
            mock_register.assert_called_once_with(
                "test_prompt",
                "Fallback template content",
                "Test prompt for hardware store",
                set_champion=True,
            )

    @pytest.mark.unit
    def test_prompt_model_validation_alias_xor_version(self):
        """Test that PromptModel validates mutually exclusive alias and version."""
        # Should fail with both alias and version
        with pytest.raises(ValueError, match="Cannot specify both alias and version"):
            PromptModel(name="test_prompt", alias="production", version=2)

        # Should succeed with only alias
        prompt = PromptModel(name="test_prompt", alias="production")
        assert prompt.alias == "production"
        assert prompt.version is None

        # Should succeed with only version
        prompt = PromptModel(name="test_prompt", version=2)
        assert prompt.version == 2
        assert prompt.alias is None

    @pytest.mark.unit
    def test_prompt_model_full_name_with_schema(self):
        """Test PromptModel full_name property with schema."""
        schema = SchemaModel(catalog_name="main", schema_name="prompts")
        prompt = PromptModel(
            name="agent_prompt", schema=schema, default_template="Template"
        )

        assert prompt.full_name == "main.prompts.agent_prompt"

    @pytest.mark.unit
    def test_prompt_model_full_name_without_schema(self):
        """Test PromptModel full_name property without schema."""
        prompt = PromptModel(name="simple_prompt", default_template="Template")

        assert prompt.full_name == "simple_prompt"


class TestPromptModelConfiguration:
    """Tests for PromptModel configuration and properties."""

    @pytest.mark.unit
    def test_prompt_template_property_calls_provider(self):
        """Test that PromptModel.template property calls DatabricksProvider.get_prompt."""
        mock_provider_instance = Mock()
        # Mock get_prompt to return a PromptVersion-like object
        mock_prompt_version = Mock()
        mock_prompt_version.to_single_brace_format.return_value = "Mocked template"
        mock_provider_instance.get_prompt.return_value = mock_prompt_version

        # Patch where DatabricksProvider is imported - inside the template property
        with patch(
            "dao_ai.providers.databricks.DatabricksProvider",
            return_value=mock_provider_instance,
        ):
            prompt_model = PromptModel(name="test_prompt", default_template="Default")

            # Access the template property
            result = prompt_model.template

            # Should call provider with self
            mock_provider_instance.get_prompt.assert_called_once_with(prompt_model)
            assert result == "Mocked template"

    @pytest.mark.unit
    def test_prompt_model_tags(self):
        """Test that PromptModel supports tags."""
        prompt = PromptModel(
            name="tagged_prompt",
            default_template="Template",
            tags={"environment": "production", "version": "v1"},
        )

        assert prompt.tags == {"environment": "production", "version": "v1"}

    @pytest.mark.unit
    def test_prompt_model_empty_tags(self):
        """Test that PromptModel has empty tags by default."""
        prompt = PromptModel(name="untagged_prompt", default_template="Template")

        assert prompt.tags == {}
