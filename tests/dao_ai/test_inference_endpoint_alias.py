"""Backward-compatibility tests for the InferenceEndpointModel / resources.models rename.

dao-ai 0.1.75 renamed the Pydantic class ``LLMModel`` to
``InferenceEndpointModel`` and renamed the ``ResourcesModel.llms`` field to
``ResourcesModel.models``. Both renames carry aliases for backward
compatibility:

- Class rename: ``LLMModel = InferenceEndpointModel`` at module scope, so
  ``from dao_ai.config import LLMModel`` keeps working and
  ``isinstance(x, LLMModel)`` is still true.
- Field rename: ``ResourcesModel.models`` declares ``alias="llms"`` and the
  parent's ``model_config`` sets ``populate_by_name=True`` so both keys
  parse under ``extra="forbid"``.

This test guards against accidental regression on either alias.
"""

from __future__ import annotations

import warnings

import pytest

from dao_ai.config import (
    AppConfig,
    AppModel,
    InferenceEndpointModel,
    LLMModel,
    ResourcesModel,
)


class TestClassAlias:
    """LLMModel must be the same class object as InferenceEndpointModel."""

    def test_llmmodel_is_inference_endpoint_model(self) -> None:
        assert LLMModel is InferenceEndpointModel

    def test_isinstance_check_works_with_legacy_name(self) -> None:
        endpoint = InferenceEndpointModel(name="databricks-claude-sonnet-4-5")
        assert isinstance(endpoint, LLMModel)

    def test_legacy_constructor_path(self) -> None:
        # Customer code using `LLMModel(name=...)` should produce an
        # InferenceEndpointModel instance.
        endpoint = LLMModel(name="databricks-gte-large-en")
        assert isinstance(endpoint, InferenceEndpointModel)
        assert endpoint.name == "databricks-gte-large-en"


class TestResourcesModelFieldAlias:
    """ResourcesModel must parse both `models:` and `llms:` keys identically."""

    SHARED_ENDPOINT = {
        "name": "databricks-claude-sonnet-4-5",
        "temperature": 0.2,
        "max_tokens": 4096,
    }

    def test_models_key_parses(self) -> None:
        resources = ResourcesModel(models={"primary": self.SHARED_ENDPOINT})
        assert "primary" in resources.models
        assert resources.models["primary"].name == "databricks-claude-sonnet-4-5"
        assert isinstance(resources.models["primary"], InferenceEndpointModel)

    def test_llms_key_still_parses_via_alias(self) -> None:
        # Pydantic should accept the legacy key via field alias.
        resources = ResourcesModel.model_validate({"llms": {"primary": self.SHARED_ENDPOINT}})
        assert "primary" in resources.models
        assert resources.models["primary"].name == "databricks-claude-sonnet-4-5"
        assert isinstance(resources.models["primary"], InferenceEndpointModel)

    def test_both_keys_produce_identical_objects(self) -> None:
        models_based = ResourcesModel(models={"primary": self.SHARED_ENDPOINT})
        llms_based = ResourcesModel.model_validate({"llms": {"primary": self.SHARED_ENDPOINT}})
        assert models_based.models == llms_based.models

    def test_models_attribute_access(self) -> None:
        resources = ResourcesModel(models={"primary": self.SHARED_ENDPOINT})
        # Canonical attribute access (new code path).
        assert resources.models["primary"].name == "databricks-claude-sonnet-4-5"

    def test_legacy_llms_attribute_emits_deprecation_warning(self) -> None:
        # Customer Python code reading `.llms` should still work but emit
        # a DeprecationWarning telling them to migrate to `.models`.
        resources = ResourcesModel(models={"primary": self.SHARED_ENDPOINT})
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            value = resources.llms
            assert any(issubclass(w.category, DeprecationWarning) for w in captured), (
                "Reading ResourcesModel.llms should emit a DeprecationWarning."
            )
        assert value == resources.models


class TestAppConfigEndToEnd:
    """A minimal AppConfig should accept either key at the YAML / dict level."""

    @pytest.fixture
    def base_config_dict(self) -> dict:
        return {
            "version": "0.1.75",
            "resources": {
                # placeholder; tests overwrite this key
            },
            "agents": {
                "tester": {
                    "name": "tester",
                    "description": "Smoke-test agent for the alias test.",
                    "model": {"name": "databricks-claude-sonnet-4-5"},
                },
            },
            "app": {
                "name": "alias-test",
                "deployment_target": "apps",
                "agents": [
                    {
                        "name": "tester",
                        "description": "Smoke-test agent for the alias test.",
                        "model": {"name": "databricks-claude-sonnet-4-5"},
                    },
                ],
                "orchestration": {
                    "swarm": {"default_agent": "tester"},
                },
            },
        }

    def test_appconfig_accepts_models_key(self, base_config_dict: dict) -> None:
        base_config_dict["resources"] = {
            "models": {"primary": {"name": "databricks-claude-sonnet-4-5"}},
        }
        cfg = AppConfig(**base_config_dict)
        assert "primary" in cfg.resources.models

    def test_appconfig_accepts_llms_key_via_alias(self, base_config_dict: dict) -> None:
        base_config_dict["resources"] = {
            "llms": {"primary": {"name": "databricks-claude-sonnet-4-5"}},
        }
        cfg = AppConfig(**base_config_dict)
        assert "primary" in cfg.resources.models

    def test_models_and_llms_produce_equivalent_configs(self, base_config_dict: dict) -> None:
        cfg_models = AppConfig(
            **{
                **base_config_dict,
                "resources": {"models": {"primary": {"name": "databricks-claude-sonnet-4-5"}}},
            }
        )
        cfg_llms = AppConfig(
            **{
                **base_config_dict,
                "resources": {"llms": {"primary": {"name": "databricks-claude-sonnet-4-5"}}},
            }
        )
        assert cfg_models.resources.models == cfg_llms.resources.models
