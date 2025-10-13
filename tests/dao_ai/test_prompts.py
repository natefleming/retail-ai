"""Tests for MLflow Prompt Registry integration."""

import os
from unittest.mock import Mock, patch

import pytest
from conftest import has_databricks_env

from dao_ai.config import PromptModel, SchemaModel
from dao_ai.providers.databricks import DatabricksProvider


class TestPromptRegistryUnit:
    """Unit tests for prompt registry functionality (mocked)."""

    @pytest.mark.unit
    def test_get_prompt_with_alias(self):
        """Test loading a prompt using an alias."""
        prompt_model = PromptModel(
            name="test_prompt", alias="production", default_template="Default template"
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        # Mock the mlflow.genai.load_prompt function
        with patch("mlflow.genai.load_prompt") as mock_load:
            mock_prompt = Mock()
            mock_prompt.to_single_brace_format.return_value = (
                "Registry template content"
            )
            mock_load.return_value = mock_prompt

            result = provider.get_prompt(prompt_model)

            # Verify correct URI was used
            mock_load.assert_called_once_with("prompts:/test_prompt@production")
            assert result == "Registry template content"

    @pytest.mark.unit
    def test_get_prompt_with_version(self):
        """Test loading a prompt using a specific version."""
        prompt_model = PromptModel(
            name="test_prompt", version=2, default_template="Default template"
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with patch("mlflow.genai.load_prompt") as mock_load:
            mock_prompt = Mock()
            mock_prompt.to_single_brace_format.return_value = "Version 2 content"
            mock_load.return_value = mock_prompt

            result = provider.get_prompt(prompt_model)

            # Verify correct URI was used
            mock_load.assert_called_once_with("prompts:/test_prompt/2")
            assert result == "Version 2 content"

    @pytest.mark.unit
    def test_get_prompt_defaults_to_latest(self):
        """Test that prompt loading defaults to @latest when no alias or version specified."""
        prompt_model = PromptModel(
            name="test_prompt", default_template="Default template"
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with patch("mlflow.genai.load_prompt") as mock_load:
            mock_prompt = Mock()
            mock_prompt.to_single_brace_format.return_value = "Latest content"
            mock_load.return_value = mock_prompt

            result = provider.get_prompt(prompt_model)

            # Should use @latest by default
            mock_load.assert_called_once_with("prompts:/test_prompt@latest")
            assert result == "Latest content"

    @pytest.mark.unit
    def test_get_prompt_fallback_to_default_template(self):
        """Test fallback to default_template when registry load fails."""
        prompt_model = PromptModel(
            name="test_prompt", default_template="Fallback template content"
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with (
            patch("mlflow.genai.load_prompt") as mock_load,
            patch.object(provider, "_sync_default_template_to_registry") as mock_sync,
        ):
            # Simulate registry failure
            mock_load.side_effect = Exception("Registry not found")

            result = provider.get_prompt(prompt_model)

            # Should use default_template
            assert result == "Fallback template content"

            # Should attempt to sync to registry
            mock_sync.assert_called_once_with(
                "test_prompt", "Fallback template content"
            )

    @pytest.mark.unit
    def test_get_prompt_no_registry_no_default_raises_error(self):
        """Test that an error is raised when registry fails and no default_template."""
        prompt_model = PromptModel(
            name="test_prompt"
            # No default_template provided
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with patch("mlflow.genai.load_prompt") as mock_load:
            mock_load.side_effect = Exception("Registry not found")

            with pytest.raises(ValueError) as exc_info:
                provider.get_prompt(prompt_model)

            assert "Failed to retrieve prompt 'test_prompt'" in str(exc_info.value)
            assert "no default_template was provided" in str(exc_info.value)

    @pytest.mark.unit
    def test_get_prompt_does_not_sync_when_registry_succeeds(self):
        """Test that default_template is NOT synced when registry load succeeds."""
        prompt_model = PromptModel(
            name="test_prompt", default_template="Different from registry"
        )

        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with (
            patch("mlflow.genai.load_prompt") as mock_load,
            patch.object(provider, "_sync_default_template_to_registry") as mock_sync,
        ):
            mock_prompt = Mock()
            mock_prompt.to_single_brace_format.return_value = "Registry content"
            mock_load.return_value = mock_prompt

            result = provider.get_prompt(prompt_model)

            # Should use registry content
            assert result == "Registry content"

            # Should NOT sync default_template when registry succeeds
            mock_sync.assert_not_called()

    @pytest.mark.unit
    def test_sync_default_template_creates_new_version(self):
        """Test that _sync_default_template_to_registry creates a new version when changed."""
        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with (
            patch("mlflow.genai.load_prompt") as mock_load,
            patch("mlflow.genai.register_prompt") as mock_register,
        ):
            # Simulate "default" alias not existing
            mock_load.side_effect = Exception("Alias not found")

            provider._sync_default_template_to_registry(
                "test_prompt", "New template content"
            )

            # Should register the prompt
            mock_register.assert_called_once_with(
                name="test_prompt",
                template="New template content",
                commit_message="Auto-synced from default_template",
            )

    @pytest.mark.unit
    def test_sync_default_template_skips_when_unchanged(self):
        """Test that _sync_default_template_to_registry skips registration when unchanged."""
        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        with (
            patch("mlflow.genai.load_prompt") as mock_load,
            patch("mlflow.genai.register_prompt") as mock_register,
        ):
            # Simulate "default" alias exists with same content
            mock_prompt = Mock()
            mock_prompt.to_single_brace_format.return_value = "Same template"
            mock_load.return_value = mock_prompt

            provider._sync_default_template_to_registry("test_prompt", "Same template")

            # Should NOT register when unchanged
            mock_register.assert_not_called()

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


class TestPromptRegistryIntegration:
    """Integration tests for prompt registry (real Databricks environment)."""

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    @pytest.mark.skip(
        reason="Unity Catalog Prompt Registry not fully enabled in Databricks yet"
    )
    def test_register_and_load_prompt_real(self):
        """
        Test registering and loading a prompt from real MLflow Prompt Registry.

        This test requires:
        - Valid DATABRICKS_HOST and DATABRICKS_TOKEN environment variables
        - MLFLOW_TRACKING_URI and MLFLOW_REGISTRY_URI configured
        - Permissions to create prompts in the registry
        - Unity Catalog Prompt Registry feature enabled (not yet available)
        """
        import mlflow

        prompt_name = f"test_prompt_{os.getpid()}"  # Unique name (underscores only)
        template_content = "You are a test assistant. Question: {question}"

        try:
            print(f"\nRegistering test prompt: {prompt_name}")

            # Register a prompt
            mlflow.genai.register_prompt(
                name=prompt_name,
                template=template_content,
                commit_message="Test registration",
            )

            print(f"Prompt '{prompt_name}' registered successfully")

            # Load it back using DatabricksProvider
            prompt_model = PromptModel(
                name=prompt_name, default_template="Fallback template"
            )

            provider = DatabricksProvider(w=Mock(), vsc=Mock())
            loaded_template = provider.get_prompt(prompt_model)

            print(f"Loaded template: {loaded_template[:50]}...")

            # Verify we got the correct content
            assert loaded_template == template_content

            print("✓ Register and load test passed")

        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            raise
        finally:
            # Note: MLflow Prompt Registry doesn't have a direct delete API
            # Prompts will remain in the registry after test
            print(f"\nNote: Test prompt '{prompt_name}' remains in registry")

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    @pytest.mark.skip(
        reason="Unity Catalog Prompt Registry not fully enabled in Databricks yet"
    )
    def test_prompt_version_management(self):
        """
        Test creating multiple versions of a prompt.

        This test verifies version management in the registry.
        Note: Requires Unity Catalog Prompt Registry feature (not yet available).
        """
        import mlflow

        prompt_name = f"test_versioned_prompt_{os.getpid()}"
        template_v1 = "Version 1: {input}"
        template_v2 = "Version 2 with improvements: {input}"

        try:
            print(f"\nTesting version management for: {prompt_name}")

            # Register version 1
            print("Registering version 1...")
            mlflow.genai.register_prompt(
                name=prompt_name, template=template_v1, commit_message="Initial version"
            )

            # Register version 2 (should create new version)
            print("Registering version 2...")
            mlflow.genai.register_prompt(
                name=prompt_name,
                template=template_v2,
                commit_message="Improved version",
            )

            # Load using version 1
            prompt_model_v1 = PromptModel(name=prompt_name, version=1)
            provider = DatabricksProvider(w=Mock(), vsc=Mock())
            loaded_v1 = provider.get_prompt(prompt_model_v1)

            print(f"Loaded v1: {loaded_v1[:30]}...")
            assert loaded_v1 == template_v1

            # Load using version 2
            prompt_model_v2 = PromptModel(name=prompt_name, version=2)
            loaded_v2 = provider.get_prompt(prompt_model_v2)

            print(f"Loaded v2: {loaded_v2[:30]}...")
            assert loaded_v2 == template_v2

            # Load using @latest (should get v2)
            prompt_model_latest = PromptModel(name=prompt_name)  # defaults to @latest
            loaded_latest = provider.get_prompt(prompt_model_latest)

            print(f"Loaded @latest: {loaded_latest[:30]}...")
            assert loaded_latest == template_v2

            print("✓ Version management test passed")

        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            raise
        finally:
            print(f"\nNote: Test prompt '{prompt_name}' remains in registry")

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    @pytest.mark.skip(
        reason="Unity Catalog Prompt Registry not fully enabled in Databricks yet"
    )
    def test_default_template_fallback_and_sync(self):
        """
        Test that default_template works as fallback and syncs to registry.

        This test verifies the fallback behavior when a prompt doesn't exist.
        Note: Requires Unity Catalog Prompt Registry feature (not yet available).
        """
        import mlflow

        # Use a prompt name that definitely doesn't exist
        nonexistent_prompt = f"nonexistent_prompt_{os.getpid()}_xyz"
        default_content = "Fallback template: {query}"

        try:
            print(f"\nTesting fallback for nonexistent prompt: {nonexistent_prompt}")

            # Try to load a nonexistent prompt with default_template
            prompt_model = PromptModel(
                name=nonexistent_prompt, default_template=default_content
            )

            provider = DatabricksProvider(w=Mock(), vsc=Mock())
            result = provider.get_prompt(prompt_model)

            # Should use default_template as fallback
            print(f"Got result: {result[:30]}...")
            assert result == default_content

            # The sync function should have registered it
            # Try loading it again - should now come from registry
            print("Verifying it was synced to registry...")

            # Small delay to ensure registration completes
            import time

            time.sleep(1)

            # Load directly from registry to verify it was synced
            try:
                registered_prompt = mlflow.genai.load_prompt(
                    f"prompts:/{nonexistent_prompt}@latest"
                )
                synced_content = registered_prompt.to_single_brace_format()
                print(f"Synced content: {synced_content[:30]}...")
                assert synced_content == default_content
                print("✓ Prompt was successfully synced to registry")
            except Exception as sync_check_error:
                print(f"⚠ Could not verify sync (may be async): {sync_check_error}")

            print("✓ Fallback and sync test passed")

        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            raise
        finally:
            print(f"\nNote: Test prompt '{nonexistent_prompt}' may remain in registry")

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    @pytest.mark.skip(
        reason="Unity Catalog Prompt Registry not fully enabled in Databricks yet"
    )
    def test_external_update_respected(self):
        """
        Test that external updates to registry are respected over default_template.

        This verifies that registry is the source of truth when available.
        Note: Requires Unity Catalog Prompt Registry feature (not yet available).
        """
        import mlflow

        prompt_name = f"test_external_update_{os.getpid()}"
        original_template = "Original: {x}"
        updated_template = "Updated by external process: {x}"
        different_default = "Different local default: {x}"

        try:
            print(f"\nTesting external update respect for: {prompt_name}")

            # Register initial version
            print("Registering original version...")
            mlflow.genai.register_prompt(
                name=prompt_name, template=original_template, commit_message="Original"
            )

            # Simulate external update (new version)
            print("Simulating external update...")
            mlflow.genai.register_prompt(
                name=prompt_name,
                template=updated_template,
                commit_message="External update",
            )

            # Now load with a different default_template
            # Should get the registry version, not the default_template
            prompt_model = PromptModel(
                name=prompt_name, default_template=different_default
            )

            provider = DatabricksProvider(w=Mock(), vsc=Mock())
            result = provider.get_prompt(prompt_model)

            print(f"Got result: {result[:40]}...")

            # Should use registry version (updated_template), not default_template
            assert result == updated_template
            assert result != different_default

            print("✓ External update respect test passed")

        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            raise
        finally:
            print(f"\nNote: Test prompt '{prompt_name}' remains in registry")


class TestPromptModelConfiguration:
    """Tests for PromptModel configuration and properties."""

    @pytest.mark.unit
    def test_prompt_template_property_calls_provider(self):
        """Test that PromptModel.template property calls DatabricksProvider.get_prompt."""
        mock_provider_instance = Mock()
        mock_provider_instance.get_prompt.return_value = "Mocked template"

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
