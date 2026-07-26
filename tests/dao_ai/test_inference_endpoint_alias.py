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
        resources = ResourcesModel.model_validate(
            {"llms": {"primary": self.SHARED_ENDPOINT}}
        )
        assert "primary" in resources.models
        assert resources.models["primary"].name == "databricks-claude-sonnet-4-5"
        assert isinstance(resources.models["primary"], InferenceEndpointModel)

    def test_both_keys_produce_identical_objects(self) -> None:
        models_based = ResourcesModel(models={"primary": self.SHARED_ENDPOINT})
        llms_based = ResourcesModel.model_validate(
            {"llms": {"primary": self.SHARED_ENDPOINT}}
        )
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

    def test_models_and_llms_produce_equivalent_configs(
        self, base_config_dict: dict
    ) -> None:
        cfg_models = AppConfig(
            **{
                **base_config_dict,
                "resources": {
                    "models": {"primary": {"name": "databricks-claude-sonnet-4-5"}}
                },
            }
        )
        cfg_llms = AppConfig(
            **{
                **base_config_dict,
                "resources": {
                    "llms": {"primary": {"name": "databricks-claude-sonnet-4-5"}}
                },
            }
        )
        assert cfg_models.resources.models == cfg_llms.resources.models


@pytest.mark.unit
class TestNullableSamplingDefaults:
    """0.2.x: ``temperature`` / ``max_tokens`` default to ``None`` and, when
    unset, are omitted from the outbound request so the serving endpoint uses
    its own default (unblocks reasoning-mode endpoints that reject any
    ``temperature``). Set values pass through unchanged."""

    def test_defaults_are_none(self) -> None:
        m = InferenceEndpointModel(name="databricks-claude-sonnet-4-5")
        assert m.temperature is None
        assert m.max_tokens is None

    def test_unset_fields_omitted_from_request_payload(self) -> None:
        from langchain_core.messages import HumanMessage

        chat = InferenceEndpointModel(
            name="databricks-claude-sonnet-4-5"
        ).as_chat_model()
        # as_chat_model may wrap in fallbacks/best_of_n; unwrap to the base client.
        base = getattr(chat, "runnable", chat)
        inputs = base._prepare_inputs([HumanMessage("hi")])
        assert "temperature" not in inputs, (
            f"unset temperature must be omitted from the payload; got {inputs.keys()}"
        )
        assert "max_tokens" not in inputs, (
            f"unset max_tokens must be omitted from the payload; got {inputs.keys()}"
        )

    def test_set_fields_present_in_request_payload(self) -> None:
        from langchain_core.messages import HumanMessage

        chat = InferenceEndpointModel(
            name="databricks-claude-sonnet-4-5", temperature=0.7, max_tokens=128
        ).as_chat_model()
        base = getattr(chat, "runnable", chat)
        inputs = base._prepare_inputs([HumanMessage("hi")])
        assert inputs.get("temperature") == 0.7
        assert inputs.get("max_tokens") == 128


@pytest.mark.unit
class TestPromptModelInlineTemplate:
    """0.2.x removed the MLflow prompt registry: ``PromptModel`` carries its
    template inline via ``template`` (required) and rejects the old
    registry-era ``default_template`` field (``extra="forbid"``)."""

    def test_template_is_required_and_inline(self) -> None:
        from dao_ai.config import PromptModel

        p = PromptModel(name="greeting", template="You are helpful.")
        assert p.template == "You are helpful."

    def test_legacy_default_template_field_rejected(self) -> None:
        from pydantic import ValidationError

        from dao_ai.config import PromptModel

        with pytest.raises(ValidationError):
            PromptModel(name="greeting", default_template="You are helpful.")
