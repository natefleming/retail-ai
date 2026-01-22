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
        assert model.enabled is False
        assert model.model is None
        assert model.default_mode == "standard"
        assert model.auto_bypass is True

    def test_custom_values(self) -> None:
        """Test RouterModel with custom configuration."""
        model = RouterModel(
            enabled=True,
            default_mode="instructed",
            auto_bypass=False,
        )
        assert model.enabled is True
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

    def _create_mock_llm(self, response_json: str) -> MagicMock:
        """Helper to create mock LLM with bind() behavior."""
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_json
        mock_llm.bind.return_value = mock_bound_llm
        mock_bound_llm.invoke.return_value = mock_response
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
        mock_llm = self._create_mock_llm('{"mode": "standard"}')

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
        mock_llm = self._create_mock_llm('{"mode": "instructed"}')

        result = route_query(
            llm=mock_llm,
            query="Milwaukee drills under $200",
            schema_description="products table with price, brand columns",
        )

        assert result == "instructed"
        mock_mlflow.set_tag.assert_called_with("router.mode", "instructed")

    @patch("dao_ai.tools.router._load_prompt_template")
    @patch("dao_ai.tools.router._get_router_schema")
    @patch("dao_ai.tools.router.mlflow")
    def test_uses_response_format_for_databricks(
        self,
        mock_mlflow: MagicMock,
        mock_get_schema: MagicMock,
        mock_load_prompt: MagicMock,
    ) -> None:
        """Test that route_query uses response_format for Databricks compatibility."""
        mock_load_prompt.return_value = {"template": "{schema_description} {query}"}
        test_schema = {"type": "json_schema", "json_schema": {"name": "test"}}
        mock_get_schema.return_value = test_schema
        mock_llm = self._create_mock_llm('{"mode": "standard"}')

        route_query(
            llm=mock_llm,
            query="test query",
            schema_description="test schema",
        )

        # Verify bind() is called with response_format
        mock_llm.bind.assert_called_once_with(response_format=test_schema)

    @patch("dao_ai.tools.router._load_prompt_template")
    @patch("dao_ai.tools.router.mlflow")
    def test_formats_prompt_correctly(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that the prompt is formatted with schema and query."""
        mock_load_prompt.return_value = {
            "template": "Schema: {schema_description}\nQuery: {query}"
        }
        mock_llm = self._create_mock_llm('{"mode": "standard"}')

        route_query(
            llm=mock_llm,
            query="my test query",
            schema_description="my schema desc",
        )

        # Get the bound LLM and check invoke args
        bound_llm = mock_llm.bind.return_value
        call_args = bound_llm.invoke.call_args[0][0]
        assert "my schema desc" in call_args
        assert "my test query" in call_args
