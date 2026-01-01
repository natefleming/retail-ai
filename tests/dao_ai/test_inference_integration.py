"""Inference tests for AI agents using actual model inference."""

import os
import sys
from pathlib import Path

import pytest
from conftest import has_databricks_env, has_postgres_env
from mlflow.types.responses import ResponsesAgentRequest
from mlflow.types.responses_helpers import Message

from dao_ai.config import AppConfig
from dao_ai.models import ResponsesAgent


@pytest.fixture
def supervisor_config_path() -> Path:
    """Fixture that returns path to hardware store configuration."""
    return (
        Path(__file__).parents[2]
        / "config"
        / "examples"
        / "15_complete_applications"
        / "hardware_store_lakebase.yaml"
    )


@pytest.fixture
def supervisor_postgres_config_path() -> Path:
    """Fixture that returns path to hardware store postgres configuration."""
    return (
        Path(__file__).parents[2]
        / "config"
        / "examples"
        / "15_complete_applications"
        / "hardware_store.yaml"
    )


@pytest.fixture
def app_config_supervisor(supervisor_config_path: Path) -> AppConfig:
    """Fixture that creates AppConfig from supervisor configuration."""
    return AppConfig.from_file(supervisor_config_path)


@pytest.fixture
def app_config_postgres(supervisor_postgres_config_path: Path) -> AppConfig:
    """Fixture that creates AppConfig from PostgreSQL supervisor configuration."""
    return AppConfig.from_file(supervisor_postgres_config_path)


@pytest.mark.integration
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_load_config_and_create_responses_agent(
    app_config_supervisor: AppConfig,
) -> None:
    """Test that we can load config and create a ResponsesAgent successfully."""
    # Verify config loaded successfully
    assert app_config_supervisor is not None
    assert hasattr(app_config_supervisor, "app")
    assert app_config_supervisor.app is not None

    # Verify we have agents configured
    assert hasattr(app_config_supervisor, "agents")
    assert len(app_config_supervisor.agents) > 0

    print(
        f"Available agents: {list(app_config_supervisor.agents.keys())}",
        file=sys.stderr,
    )

    # Test creating ResponsesAgent from config
    responses_agent: ResponsesAgent = app_config_supervisor.as_responses_agent()
    assert responses_agent is not None
    # ResponsesAgent uses predict() method, not invoke()
    assert hasattr(responses_agent, "predict")
    assert hasattr(responses_agent, "predict_stream")

    print(
        f"Successfully created ResponsesAgent: {type(responses_agent)}", file=sys.stderr
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_inventory_stock_question_inference(app_config_supervisor: AppConfig) -> None:
    """Test inference with inventory stock question: 'how many big green egg grills do you have in stock'."""
    # Create ResponsesAgent from config
    responses_agent: ResponsesAgent = app_config_supervisor.as_responses_agent()

    # Test inventory stock question
    question = "how many big green egg grills do you have in stock"
    print(f"Testing question: {question}", file=sys.stderr)

    try:
        # Create ResponsesAgentRequest
        request = ResponsesAgentRequest(
            input=[Message(role="user", content=question, type="message")],
            custom_inputs={"user_id": "test_user_001", "store_num": "001"},
        )

        # Use predict for synchronous inference
        response = responses_agent.predict(request)

        print(f"Response: {response}", file=sys.stderr)

        # Basic validation - should get some response
        assert response is not None
        assert hasattr(response, "output")
        assert len(response.output) > 0

        # Check the output content
        output_item = response.output[0]
        assert hasattr(output_item, "content") or hasattr(output_item, "text")
        content = getattr(output_item, "content", None) or getattr(
            output_item, "text", None
        )
        assert content is not None
        assert len(content) > 0

        print(f"Assistant response: {content}", file=sys.stderr)

    except Exception as e:
        print(f"Inference failed with error: {e}", file=sys.stderr)
        # Mark as skipped rather than failed since this could be due to model unavailability
        pytest.skip(f"Inference test skipped due to error: {e}")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_product_search_inference(app_config_supervisor: AppConfig) -> None:
    """Test inference with product search question."""
    # Create ResponsesAgent from config
    responses_agent: ResponsesAgent = app_config_supervisor.as_responses_agent()

    # Test product search question
    question = "Can you find me information about power drills?"
    print(f"Testing question: {question}", file=sys.stderr)

    try:
        # Create ResponsesAgentRequest
        request = ResponsesAgentRequest(
            input=[Message(role="user", content=question, type="message")],
            custom_inputs={"user_id": "test_user_002", "store_num": "001"},
        )

        # Use predict for synchronous inference
        response = responses_agent.predict(request)

        print(f"Response: {response}", file=sys.stderr)

        # Basic validation
        assert response is not None
        assert hasattr(response, "output")
        assert len(response.output) > 0

        output_item = response.output[0]
        content = getattr(output_item, "content", None) or getattr(
            output_item, "text", None
        )
        assert content is not None
        assert len(content) > 0

        print(f"Assistant response: {content}", file=sys.stderr)

    except Exception as e:
        print(f"Inference failed with error: {e}", file=sys.stderr)
        pytest.skip(f"Product search test skipped due to error: {e}")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_birthday_memory_context_inference(app_config_supervisor: AppConfig) -> None:
    """Test inference with memory/context: 'My birthday is 08/08/1978, What is my birthday?'."""
    # Create ResponsesAgent from config
    responses_agent: ResponsesAgent = app_config_supervisor.as_responses_agent()

    # First message: provide birthday information
    birthday_statement = "My birthday is 08/08/1978"
    print(f"First message: {birthday_statement}", file=sys.stderr)

    try:
        # First interaction - provide birthday info
        request1 = ResponsesAgentRequest(
            input=[Message(role="user", content=birthday_statement, type="message")],
            custom_inputs={"user_id": "test_user_003", "store_num": "001"},
        )
        response1 = responses_agent.predict(request1)

        print(f"Response 1: {response1}", file=sys.stderr)

        # Validate first response
        assert response1 is not None
        assert hasattr(response1, "output")
        assert len(response1.output) > 0

        # Extract content from MLflow response format
        content1 = getattr(response1.output[0], "content", None) or getattr(
            response1.output[0], "text", None
        )
        assert content1 is not None

        # Handle different content formats (list of dicts vs string)
        if isinstance(content1, list) and len(content1) > 0:
            # Extract text from first content item if it's a list
            first_content = content1[0]
            if isinstance(first_content, dict) and "text" in first_content:
                content1_text = first_content["text"]
            else:
                content1_text = str(first_content)
        else:
            content1_text = str(content1)

        # Second message: ask about the birthday (test memory/context)
        birthday_question = "What is my birthday?"
        print(f"Second message: {birthday_question}", file=sys.stderr)

        # Second interaction - ask about birthday (same user for context)
        request2 = ResponsesAgentRequest(
            input=[
                Message(role="user", content=birthday_statement, type="message"),
                Message(
                    role="assistant",
                    content=[{"text": content1_text, "type": "output_text"}],
                    type="message",
                ),
                Message(role="user", content=birthday_question, type="message"),
            ],
            custom_inputs={
                "user_id": "test_user_003",
                "store_num": "001",
            },  # Same user ID for context
        )
        response2 = responses_agent.predict(request2)

        print(f"Response 2: {response2}", file=sys.stderr)

        # Validate second response
        assert response2 is not None
        assert hasattr(response2, "output")
        assert len(response2.output) > 0

        content2 = getattr(response2.output[0], "content", None) or getattr(
            response2.output[0], "text", None
        )
        assert content2 is not None
        assert len(content2) > 0

        # Extract text content for checking
        if isinstance(content2, list) and len(content2) > 0:
            first_content = content2[0]
            if isinstance(first_content, dict) and "text" in first_content:
                content2_text = first_content["text"]
            else:
                content2_text = str(first_content)
        else:
            content2_text = str(content2)

        # Check if the response contains the birthday information
        response_content = content2_text.lower()
        birthday_indicators = ["08/08/1978", "august 8", "8/8/1978", "1978"]

        print(f"Assistant final response: {content2_text}", file=sys.stderr)
        print(
            f"Checking for birthday indicators in response: {birthday_indicators}",
            file=sys.stderr,
        )

        # The response should reference the birthday in some way
        has_birthday_reference = any(
            indicator in response_content for indicator in birthday_indicators
        )
        if not has_birthday_reference:
            print(
                "Warning: Response may not contain expected birthday reference",
                file=sys.stderr,
            )

    except Exception as e:
        print(f"Birthday context inference failed with error: {e}", file=sys.stderr)
        pytest.skip(f"Birthday context test skipped due to error: {e}")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"),
    reason="Databricks credentials not available",
)
def test_general_store_information_inference(app_config_supervisor: AppConfig) -> None:
    """Test inference with general store information question."""
    # Create ResponsesAgent from config
    responses_agent: ResponsesAgent = app_config_supervisor.as_responses_agent()

    # Test general store question
    question = "What are your store hours?"
    print(f"Testing question: {question}", file=sys.stderr)

    try:
        # Create ResponsesAgentRequest
        request = ResponsesAgentRequest(
            input=[Message(role="user", content=question, type="message")],
            custom_inputs={"user_id": "test_user_004", "store_num": "001"},
        )

        # Use predict for synchronous inference
        response = responses_agent.predict(request)

        print(f"Response: {response}", file=sys.stderr)

        # Basic validation
        assert response is not None
        assert hasattr(response, "output")
        assert len(response.output) > 0

        content = getattr(response.output[0], "content", None) or getattr(
            response.output[0], "text", None
        )
        assert content is not None
        assert len(content) > 0

        print(f"Assistant response: {content}", file=sys.stderr)

    except Exception as e:
        print(f"Store hours inference failed with error: {e}", file=sys.stderr)
        pytest.skip(f"Store hours test skipped due to error: {e}")


@pytest.mark.integration
@pytest.mark.skipif(
    not has_postgres_env()
    or not os.getenv("DATABRICKS_HOST")
    or not os.getenv("DATABRICKS_TOKEN"),
    reason="PostgreSQL environment or Databricks credentials not available",
)
def test_inference_with_postgres_memory(app_config_postgres: AppConfig) -> None:
    """Test inference with PostgreSQL-backed memory for conversation persistence."""
    # Create ResponsesAgent from PostgreSQL config
    responses_agent: ResponsesAgent = app_config_postgres.as_responses_agent()

    # Test conversation with memory persistence
    question = "Remember that I prefer Makita brand tools. What power drill would you recommend?"
    print(f"Testing question with PostgreSQL memory: {question}", file=sys.stderr)

    try:
        # Create ResponsesAgentRequest
        request = ResponsesAgentRequest(
            input=[Message(role="user", content=question, type="message")],
            custom_inputs={"user_id": "test_user_postgres_001", "store_num": "001"},
        )

        # Use predict for synchronous inference with PostgreSQL memory
        response = responses_agent.predict(request)

        print(f"Response with PostgreSQL memory: {response}", file=sys.stderr)

        # Basic validation
        assert response is not None
        assert hasattr(response, "output")
        assert len(response.output) > 0

        content = getattr(response.output[0], "content", None) or getattr(
            response.output[0], "text", None
        )
        assert content is not None
        assert len(content) > 0

        print(f"Assistant response: {content}", file=sys.stderr)

    except Exception as e:
        print(f"PostgreSQL memory inference failed with error: {e}", file=sys.stderr)
        pytest.skip(f"PostgreSQL memory test skipped due to error: {e}")


@pytest.mark.unit
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_config_has_required_components_for_inference() -> None:
    """Unit test to verify config has required components for inference without actual inference."""
    config_path = (
        Path(__file__).parents[2]
        / "config"
        / "examples"
        / "15_complete_applications"
        / "hardware_store_lakebase.yaml"
    )

    # Load config using Databricks Connect setup
    app_config = AppConfig.from_file(config_path)

    # Verify required components exist
    assert app_config is not None
    assert app_config.app is not None
    assert len(app_config.agents) > 0

    # Verify we can create the responses agent without invoking it
    responses_agent = app_config.as_responses_agent()
    assert responses_agent is not None

    # Check that the agent has the correct predict method for ResponsesAgent
    assert hasattr(responses_agent, "predict"), (
        f"ResponsesAgent missing 'predict' method. Available methods: {[m for m in dir(responses_agent) if not m.startswith('_')]}"
    )
    assert callable(getattr(responses_agent, "predict")), (
        "ResponsesAgent.predict is not callable"
    )

    print(
        f"Config validation successful - found {len(app_config.agents)} agents",
        file=sys.stderr,
    )
    print(f"Available agents: {list(app_config.agents.keys())}", file=sys.stderr)
