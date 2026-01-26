"""Unit tests for query router."""

from unittest.mock import MagicMock, patch

import pytest

from dao_ai.config import RouterModel
from dao_ai.tools.router import RouterDecision, route_query


@pytest.mark.unit
class TestRouterModel:
    """Unit tests for RouterModel configuration."""

    def test_default_values(self) -> None:
        """Test that RouterModel has sensible defaults."""
        model = RouterModel()
        assert model.model is None
        assert model.default_mode == "standard"
        assert model.auto_bypass is True

    def test_custom_values(self) -> None:
        """Test RouterModel with custom configuration."""
        model = RouterModel(
            default_mode="instructed",
            auto_bypass=False,
        )
        assert model.default_mode == "instructed"
        assert model.auto_bypass is False

    def test_mode_literal_validation(self) -> None:
        """Test that only valid modes are accepted."""
        with pytest.raises(ValueError):
            RouterModel(default_mode="invalid")


@pytest.mark.unit
class TestRouterDecision:
    """Unit tests for RouterDecision structured output."""

    def test_standard_mode(self) -> None:
        """Test creating decision for standard mode."""
        decision = RouterDecision(mode="standard")
        assert decision.mode == "standard"

    def test_instructed_mode(self) -> None:
        """Test creating decision for instructed mode."""
        decision = RouterDecision(mode="instructed")
        assert decision.mode == "instructed"

    def test_invalid_mode_rejected(self) -> None:
        """Test that invalid modes are rejected."""
        with pytest.raises(ValueError):
            RouterDecision(mode="invalid")


@pytest.mark.unit
class TestRouteQuery:
    """Unit tests for route_query function."""

    def _create_mock_llm(self, mode: str) -> MagicMock:
        """Helper to create mock LLM with with_structured_output behavior."""
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        # with_structured_output returns Pydantic instance directly
        mock_structured_llm.invoke.return_value = RouterDecision(mode=mode)
        return mock_llm

    @patch("dao_ai.tools.router._load_prompt_template")
    @patch("dao_ai.tools.router.mlflow")
    def test_routes_simple_query_to_standard(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that simple queries route to standard mode."""
        mock_load_prompt.return_value = {
            "template": "Test prompt: {schema_description} {query}"
        }
        mock_llm = self._create_mock_llm("standard")

        result = route_query(
            llm=mock_llm,
            query="drill bits",
            schema_description="products table with price, brand columns",
        )

        assert result == "standard"
        mock_mlflow.set_tag.assert_called_with("router.mode", "standard")

    @patch("dao_ai.tools.router._load_prompt_template")
    @patch("dao_ai.tools.router.mlflow")
    def test_routes_constrained_query_to_instructed(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that queries with constraints route to instructed mode."""
        mock_load_prompt.return_value = {
            "template": "Test prompt: {schema_description} {query}"
        }
        mock_llm = self._create_mock_llm("instructed")

        result = route_query(
            llm=mock_llm,
            query="Milwaukee drills under $200",
            schema_description="products table with price, brand columns",
        )

        assert result == "instructed"
        mock_mlflow.set_tag.assert_called_with("router.mode", "instructed")

    @patch("dao_ai.tools.router._load_prompt_template")
    @patch("dao_ai.tools.router.mlflow")
    def test_uses_with_structured_output(
        self,
        mock_mlflow: MagicMock,
        mock_load_prompt: MagicMock,
    ) -> None:
        """Test that route_query uses with_structured_output for parsing."""
        mock_load_prompt.return_value = {"template": "{schema_description} {query}"}
        mock_llm = self._create_mock_llm("standard")

        route_query(
            llm=mock_llm,
            query="test query",
            schema_description="test schema",
        )

        # Verify with_structured_output is called with RouterDecision schema
        mock_llm.with_structured_output.assert_called_once_with(RouterDecision)

    @patch("dao_ai.tools.router._load_prompt_template")
    @patch("dao_ai.tools.router.mlflow")
    def test_formats_prompt_correctly(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that the prompt is formatted with schema and query."""
        mock_load_prompt.return_value = {
            "template": "Schema: {schema_description}\nQuery: {query}"
        }
        mock_llm = self._create_mock_llm("standard")

        route_query(
            llm=mock_llm,
            query="my test query",
            schema_description="my schema desc",
        )

        # Get the structured LLM and check invoke args
        structured_llm = mock_llm.with_structured_output.return_value
        call_args = structured_llm.invoke.call_args[0][0]
        assert "my schema desc" in call_args
        assert "my test query" in call_args
