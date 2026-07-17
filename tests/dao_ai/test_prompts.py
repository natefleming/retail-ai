"""Tests for reusable inline prompts (PromptModel) and make_prompt."""

from unittest.mock import MagicMock

import pytest

from dao_ai.config import PromptModel, SchemaModel


class TestPromptModelConfiguration:
    """Tests for PromptModel configuration and properties."""

    @pytest.mark.unit
    def test_template_returns_inline_text(self):
        """PromptModel.template returns the inline template verbatim."""
        prompt = PromptModel(name="test_prompt", template="Hello {user_id}")

        assert prompt.template == "Hello {user_id}"

    @pytest.mark.unit
    def test_jinja_template_converts_judge_vars(self):
        """jinja_template converts single-brace judge vars to double-brace."""
        prompt = PromptModel(
            name="judge_prompt",
            template="Query: {inputs}\nResponse: {outputs}\nOther: {user_id}",
        )

        rendered = prompt.jinja_template

        assert "{{ inputs }}" in rendered
        assert "{{ outputs }}" in rendered
        # Non-judge variables are left untouched.
        assert "{user_id}" in rendered

    @pytest.mark.unit
    def test_jinja_template_leaves_existing_double_brace(self):
        """Already double-brace judge vars are not double-converted."""
        prompt = PromptModel(
            name="judge_prompt",
            template="Query: {{ inputs }}\nResponse: {{ outputs }}",
        )

        rendered = prompt.jinja_template

        assert rendered.count("{{ inputs }}") == 1
        assert rendered.count("{{ outputs }}") == 1

    @pytest.mark.unit
    def test_prompt_model_full_name_with_schema(self):
        """full_name prepends the schema's full name when a schema is set."""
        schema = SchemaModel(catalog_name="main", schema_name="prompts")
        prompt = PromptModel(name="agent_prompt", schema=schema, template="Template")

        assert prompt.full_name == "main.prompts.agent_prompt"

    @pytest.mark.unit
    def test_prompt_model_full_name_without_schema(self):
        """full_name is just the name when no schema is set."""
        prompt = PromptModel(name="simple_prompt", template="Template")

        assert prompt.full_name == "simple_prompt"

    @pytest.mark.unit
    def test_prompt_model_tags(self):
        """PromptModel supports tags."""
        prompt = PromptModel(
            name="tagged_prompt",
            template="Template",
            tags={"environment": "production", "team": "retail"},
        )

        assert prompt.tags == {"environment": "production", "team": "retail"}

    @pytest.mark.unit
    def test_prompt_model_empty_tags(self):
        """PromptModel has empty tags by default."""
        prompt = PromptModel(name="untagged_prompt", template="Template")

        assert prompt.tags == {}

    @pytest.mark.unit
    def test_template_is_required(self):
        """template is a required field."""
        with pytest.raises(ValueError):
            PromptModel(name="no_template")


class TestMakePrompt:
    """Tests for make_prompt middleware creation and rendering."""

    @staticmethod
    def _invoke_middleware(middleware, context_dict: dict) -> str:
        """Invoke the dynamic prompt middleware and return the system prompt string.

        The ``@dynamic_prompt`` decorator produces an ``AgentMiddleware``
        subclass whose ``wrap_model_call`` internally calls the original
        closure and then forwards the request to a handler.  We supply a
        handler that captures the system message so we can assert on it.
        """
        mock_context = MagicMock()
        mock_context.model_dump.return_value = context_dict
        mock_request = MagicMock()
        mock_request.runtime.context = mock_context

        captured: dict[str, str] = {}

        def handler(req):
            captured["prompt"] = (
                req.system_message.content if hasattr(req, "system_message") else ""
            )
            return MagicMock()

        def override_fn(**kwargs):
            mock_overridden = MagicMock()
            mock_overridden.system_message = kwargs.get("system_message")
            return mock_overridden

        mock_request.override = override_fn

        middleware.wrap_model_call(mock_request, handler)
        return captured.get("prompt", "")

    @pytest.mark.unit
    def test_make_prompt_from_string(self):
        """make_prompt renders a static string prompt."""
        from dao_ai.prompts import make_prompt

        middleware = make_prompt("Static prompt")
        assert middleware is not None

        result = self._invoke_middleware(middleware, {})
        assert result == "Static prompt"

    @pytest.mark.unit
    def test_make_prompt_from_prompt_model(self):
        """make_prompt renders a PromptModel's inline template with context."""
        from dao_ai.prompts import make_prompt

        prompt_model = PromptModel(name="test_prompt", template="Hello {user_id}")
        middleware = make_prompt(prompt_model)
        assert middleware is not None

        result = self._invoke_middleware(middleware, {"user_id": "alice"})
        assert "alice" in result

    @pytest.mark.unit
    def test_make_prompt_returns_none_when_no_prompt(self):
        """None or empty input returns None."""
        from dao_ai.prompts import make_prompt

        assert make_prompt(None) is None
        assert make_prompt("") is None
