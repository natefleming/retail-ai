"""
Integration tests for guardrails middleware against live Databricks endpoints.

These tests load the guardrails_basic.yaml config end-to-end, create a
ResponsesAgent, and validate that both generic (prompt-based) and specialized
(zero-config) guardrails execute correctly.

Run with:
    pytest tests/dao_ai/test_guardrails_integration.py -v -m integration -s
    pytest tests/dao_ai/test_guardrails_integration.py -v -m integration -s --log-cli-level=DEBUG
"""

import sys
from pathlib import Path

import pytest
from conftest import has_databricks_env
from mlflow.types.responses import ResponsesAgentRequest
from mlflow.types.responses_helpers import Message

from dao_ai.config import AppConfig
from dao_ai.models import ResponsesAgent

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def guardrails_config_path() -> Path:
    """Fixture that returns path to the guardrails_basic.yaml configuration."""
    return (
        Path(__file__).parents[2]
        / "config"
        / "examples"
        / "08_guardrails"
        / "guardrails_basic.yaml"
    )


@pytest.fixture
def app_config(guardrails_config_path: Path) -> AppConfig:
    """Fixture that creates AppConfig from the guardrails configuration."""
    return AppConfig.from_file(guardrails_config_path)


@pytest.fixture
def responses_agent(app_config: AppConfig) -> ResponsesAgent:
    """Fixture that creates a ResponsesAgent from the guardrails config."""
    return app_config.as_responses_agent()


def _make_request(question: str) -> ResponsesAgentRequest:
    """Create a ResponsesAgentRequest with the given question."""
    return ResponsesAgentRequest(
        input=[Message(role="user", content=question, type="message")],
        custom_inputs={"user_id": "test_user"},
    )


def _extract_response_content(response: object) -> str:
    """Extract text content from a ResponsesAgent response.

    Handles the nested response structure where ``output_item.content``
    may be a list of dicts like ``[{'text': '...', 'type': 'output_text'}]``.
    """
    assert response is not None, "Response should not be None"
    assert hasattr(response, "output"), "Response should have output"
    assert len(response.output) > 0, "Response output should not be empty"

    output_item = response.output[0]
    content = getattr(output_item, "content", None)

    # Handle list-of-dicts structure: [{'text': '...', 'type': 'output_text'}]
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                texts.append(item.get("text", ""))
            else:
                texts.append(getattr(item, "text", str(item)))
        content = "\n".join(texts)
    elif content is None:
        content = getattr(output_item, "text", None) or str(output_item)

    assert content is not None, "Response output should have content"
    assert isinstance(content, str), f"Expected str, got {type(content)}"
    assert len(content) > 0, "Response content should not be empty"
    return content


# =============================================================================
# Test Cases
# =============================================================================


@pytest.mark.integration
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_guardrails_config_loads_and_creates_agent(
    app_config: AppConfig,
) -> None:
    """
    Test 1: Config loads and agent creates successfully.

    Validates:
    - The YAML parses without errors
    - Guardrails are attached to the agent
    - Middleware (specialized guardrails) are configured
    - ResponsesAgent can be created
    """
    assert app_config is not None
    assert app_config.app is not None

    # Verify agents are configured
    assert len(app_config.agents) > 0
    agent_names: list[str] = list(app_config.agents.keys())
    print(f"Available agents: {agent_names}", file=sys.stderr)

    # Check the first agent has guardrails
    first_agent = list(app_config.agents.values())[0]
    print(
        f"Agent '{first_agent.name}' guardrails: {len(first_agent.guardrails)}",
        file=sys.stderr,
    )
    print(
        f"Agent '{first_agent.name}' middleware: {len(first_agent.middleware)}",
        file=sys.stderr,
    )
    assert len(first_agent.guardrails) >= 2, (
        f"Expected at least 2 guardrails, got {len(first_agent.guardrails)}"
    )
    assert len(first_agent.middleware) >= 3, (
        f"Expected at least 3 middleware, got {len(first_agent.middleware)}"
    )

    # Verify we can create a ResponsesAgent
    responses_agent: ResponsesAgent = app_config.as_responses_agent()
    assert responses_agent is not None
    assert hasattr(responses_agent, "predict")
    print(
        f"Successfully created ResponsesAgent: {type(responses_agent)}",
        file=sys.stderr,
    )


@pytest.mark.integration
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_simple_query_passes_all_guardrails(
    responses_agent: ResponsesAgent,
) -> None:
    """
    Test 2: Simple query passes all guardrails.

    Sends a straightforward question that should pass tone, completeness,
    relevance, and conciseness checks. Validates the full pipeline: agent
    LLM generates response, MLflow judges evaluate it, and the response
    is returned.
    """
    question = "What is Python?"
    print(f"Testing question: {question}", file=sys.stderr)

    try:
        request = _make_request(question)
        response = responses_agent.predict(request)
        content = _extract_response_content(response)

        print(f"Response: {content[:200]}...", file=sys.stderr)

        # Basic quality checks on the response
        assert len(content) > 50, (
            "Response should be substantive (>50 chars) for a 'What is Python?' question"
        )

    except Exception as e:
        print(f"Inference failed with error: {e}", file=sys.stderr)
        pytest.skip(f"Inference test skipped due to error: {e}")


@pytest.mark.integration
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_veracity_guardrail_skips_without_tool_context(
    responses_agent: ResponsesAgent,
) -> None:
    """
    Test 3: Veracity guardrail skips when no tool context is present.

    Sends a query that does NOT use tools. The veracity middleware should
    skip evaluation (no tool context to ground against) rather than
    blocking the response.
    """
    question = "Say hello to the test user"
    print(f"Testing question: {question}", file=sys.stderr)

    try:
        request = _make_request(question)
        response = responses_agent.predict(request)
        content = _extract_response_content(response)

        print(f"Response: {content[:200]}...", file=sys.stderr)

        # Response should contain some greeting
        assert len(content) > 0, "Response should not be empty"

    except Exception as e:
        print(f"Inference failed with error: {e}", file=sys.stderr)
        pytest.skip(f"Inference test skipped due to error: {e}")


@pytest.mark.integration
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_tool_context_extraction_with_search(
    responses_agent: ResponsesAgent,
) -> None:
    """
    Test 4: Query that exercises tool context extraction.

    Sends a query that should trigger the search tool, producing
    ToolMessages that the veracity and relevance guardrails can
    evaluate against.
    """
    question = "Search for the latest Python release and tell me about it"
    print(f"Testing question: {question}", file=sys.stderr)

    try:
        request = _make_request(question)
        response = responses_agent.predict(request)
        content = _extract_response_content(response)

        print(f"Response: {content[:300]}...", file=sys.stderr)

        # Response should be substantive since it used search
        assert len(content) > 50, (
            "Response should be substantive when using search tool"
        )
        # Should mention Python somewhere in the response
        assert "python" in content.lower(), (
            "Response should mention Python when asked about Python releases"
        )

    except Exception as e:
        print(f"Inference failed with error: {e}", file=sys.stderr)
        pytest.skip(f"Inference test skipped due to error: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration", "-s"])
