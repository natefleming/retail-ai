"""
Tests for prompt optimization functionality.

This test module provides comprehensive coverage for the prompt optimization feature in dao-ai,
including:

Unit Tests:
-----------
- PromptOptimizationModel: Creation, validation, defaults, custom parameters, optimize method
- OptimizationsModel: Creation, empty dictionary handling, optimize method
- AppConfig Integration: Configuration with and without optimizations
- DatabricksProvider: Error handling for unsupported features (agent string references)

Integration Tests:
------------------
- End-to-end workflow testing with mocked dependencies

System Tests (Skipped by default):
-----------------------------------
- Real Databricks connection tests for prompt optimization
- Config loading and execution tests
- These tests require valid Databricks credentials and datasets

Test Patterns:
--------------
- Uses pytest marks: @pytest.mark.unit, @pytest.mark.system, @pytest.mark.slow
- Uses skipif decorators with conftest.has_databricks_env() for environment-dependent tests
- Mocks external dependencies for unit tests
- Follows existing dao-ai test patterns and conventions

Note:
-----
The implementation uses MLflow 3.5+ API as documented at:
https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/

API Components:
- `from mlflow.genai.optimize import GepaPromptOptimizer, optimize_prompts`
- `from mlflow.genai.datasets import get_dataset`
- `from mlflow.genai.scorers import Correctness`

GepaPromptOptimizer signature:
- reflection_model: str (e.g., "databricks:/model-name")
- max_metric_calls: int (default: 100)
- display_progress_bar: bool (default: False)

Some tests that directly invoke DatabricksProvider.optimize_prompt are skipped
because they would require mocking complex MLflow internals. The integration and
system tests provide coverage for the end-to-end workflow.
"""

from unittest.mock import Mock, patch

import pytest
from conftest import has_databricks_env
from loguru import logger

from dao_ai.config import (
    AgentModel,
    AppConfig,
    ChatPayload,
    EvaluationDatasetEntryModel,
    EvaluationDatasetExpectationsModel,
    EvaluationDatasetModel,
    LLMModel,
    Message,
    MessageRole,
    OptimizationsModel,
    PromptModel,
    PromptOptimizationModel,
)
from dao_ai.providers.databricks import DatabricksProvider


def _create_test_dataset() -> EvaluationDatasetModel:
    """Create a test dataset for optimization tests."""
    return EvaluationDatasetModel(
        name="test_dataset",
        data=[
            EvaluationDatasetEntryModel(
                inputs=ChatPayload(
                    messages=[Message(role=MessageRole.USER, content="Test question")]
                ),
                expectations=EvaluationDatasetExpectationsModel(
                    expected_response="Test response"
                ),
            )
        ],
    )


class TestPromptOptimizationModelUnit:
    """Unit tests for PromptOptimizationModel (mocked)."""

    @pytest.mark.unit
    def test_prompt_optimization_model_creation(self):
        """Test that PromptOptimizationModel can be created with required fields."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)
        dataset = _create_test_dataset()

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset=dataset,
        )

        assert opt.name == "test_optimization"
        assert opt.prompt.name == "test_prompt"
        assert opt.agent.name == "test_agent"
        assert opt.dataset.name == "test_dataset"
        assert isinstance(opt.dataset, EvaluationDatasetModel)

    @pytest.mark.unit
    def test_prompt_optimization_model_defaults(self):
        """Test that PromptOptimizationModel has correct default values."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)
        dataset = _create_test_dataset()

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset=dataset,
        )

        assert opt.num_candidates == 50  # Default is 50
        # reflection_model defaults to None (will use agent.model at runtime)
        assert opt.reflection_model is None

    @pytest.mark.unit
    def test_prompt_optimization_model_custom_params(self):
        """Test that PromptOptimizationModel accepts custom optimizer parameters."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)
        reflection_llm = LLMModel(name="gpt-4o")
        dataset = _create_test_dataset()

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            reflection_model=reflection_llm,
            num_candidates=10,
        )

        assert opt.num_candidates == 10
        assert opt.reflection_model.name == "gpt-4o"

    @pytest.mark.unit
    def test_prompt_optimization_model_reflection_model_as_string(self):
        """Test that PromptOptimizationModel accepts reflection_model as string reference."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)
        dataset = _create_test_dataset()

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            reflection_model="gpt-4o",  # String reference
        )

        assert opt.reflection_model == "gpt-4o"
        assert isinstance(opt.reflection_model, str)

    @pytest.mark.unit
    @patch("dao_ai.optimization.optimize_prompt")
    def test_prompt_optimization_model_optimize_method(self, mock_optimize):
        """Test that PromptOptimizationModel.optimize() delegates to optimize_prompt."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)
        dataset = _create_test_dataset()

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset=dataset,
        )

        # Create mock result
        from dao_ai.optimization import OptimizationResult

        mock_result = OptimizationResult(
            optimized_prompt=PromptModel(
                name="test_prompt", version=2, default_template="Optimized {{text}}"
            ),
            optimized_template="Optimized {{text}}",
            original_score=0.5,
            optimized_score=0.8,
            improvement=0.6,
            num_evaluations=50,
        )
        mock_optimize.return_value = mock_result

        result = opt.optimize()

        # Verify the method was called
        mock_optimize.assert_called_once()
        assert result.name == "test_prompt"

    @pytest.mark.unit
    @patch("dao_ai.optimization.optimize_prompt")
    def test_optimized_prompt_has_optimizer_tag(self, mock_optimize):
        """Test that optimized prompt includes optimizer tag."""
        from dao_ai.optimization import OptimizationResult

        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm, prompt=prompt)
        dataset = _create_test_dataset()

        opt = PromptOptimizationModel(
            name="test_optimization",
            agent=agent,
            dataset=dataset,
        )

        # Create mock result with tags
        mock_result = OptimizationResult(
            optimized_prompt=PromptModel(
                name="test_prompt",
                default_template="Optimized {{text}}",
                tags={"optimizer": "gepa"},
            ),
            optimized_template="Optimized {{text}}",
            original_score=0.5,
            optimized_score=0.8,
            improvement=0.6,
            num_evaluations=50,
        )
        mock_optimize.return_value = mock_result

        result = opt.optimize()

        # Verify the optimized prompt has the optimizer tag
        assert result.tags is not None
        assert "optimizer" in result.tags
        assert result.tags["optimizer"] == "gepa"

    @pytest.mark.unit
    @patch("dao_ai.optimization.optimize_prompt")
    def test_optimized_prompt_has_latest_alias(self, mock_optimize):
        """Test that optimized prompt is tagged with 'latest' alias."""
        from dao_ai.optimization import OptimizationResult

        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm, prompt=prompt)
        dataset = _create_test_dataset()

        opt = PromptOptimizationModel(
            name="test_optimization",
            agent=agent,
            dataset=dataset,
        )

        # Create mock result with latest alias
        mock_result = OptimizationResult(
            optimized_prompt=PromptModel(
                name="test_prompt",
                default_template="Optimized {{text}}",
                alias="latest",
            ),
            optimized_template="Optimized {{text}}",
            original_score=0.5,
            optimized_score=0.5,  # No improvement
            improvement=0.0,
            num_evaluations=50,
        )
        mock_optimize.return_value = mock_result

        result = opt.optimize()

        # Verify the optimized prompt has the 'latest' alias
        assert result.alias == "latest"


class TestTrainingDatasetModelUnit:
    """Unit tests for EvaluationDatasetModel."""

    @pytest.mark.unit
    def test_training_dataset_model_creation(self):
        """Test that EvaluationDatasetModel can be created with just a name."""
        dataset = EvaluationDatasetModel(name="test_dataset")

        assert dataset.name == "test_dataset"
        assert dataset.data == []
        assert dataset.full_name == "test_dataset"

    @pytest.mark.unit
    def test_training_dataset_model_with_data(self):
        """Test that EvaluationDatasetModel can be created with data entries."""
        entries = [
            EvaluationDatasetEntryModel(
                inputs=ChatPayload(
                    messages=[Message(role=MessageRole.USER, content="Hello world")]
                ),
                expectations=EvaluationDatasetExpectationsModel(
                    expected_response="Hello! How can I help you?"
                ),
            ),
            EvaluationDatasetEntryModel(
                inputs=ChatPayload(
                    messages=[Message(role=MessageRole.USER, content="Goodbye world")]
                ),
                expectations=EvaluationDatasetExpectationsModel(
                    expected_response="Goodbye! Have a great day!"
                ),
            ),
        ]

        dataset = EvaluationDatasetModel(name="test_dataset", data=entries)

        assert dataset.name == "test_dataset"
        assert len(dataset.data) == 2
        assert dataset.data[0].inputs.messages[0].content == "Hello world"
        assert (
            dataset.data[0].expectations.expected_response
            == "Hello! How can I help you?"
        )
        assert dataset.full_name == "test_dataset"

    @pytest.mark.unit
    def test_training_dataset_with_expected_facts(self):
        """Test that EvaluationDatasetModel can be created with expected_facts."""
        entry = EvaluationDatasetEntryModel(
            inputs=ChatPayload(
                messages=[
                    Message(
                        role=MessageRole.USER, content="What is the capital of France?"
                    )
                ]
            ),
            expectations=EvaluationDatasetExpectationsModel(
                expected_facts=["Paris", "Capital city of France"]
            ),
        )

        dataset = EvaluationDatasetModel(name="test_dataset", data=[entry])

        assert dataset.name == "test_dataset"
        assert len(dataset.data) == 1
        assert (
            dataset.data[0].inputs.messages[0].content
            == "What is the capital of France?"
        )
        assert dataset.data[0].expectations.expected_facts == [
            "Paris",
            "Capital city of France",
        ]
        assert dataset.data[0].expectations.expected_response is None

    @pytest.mark.unit
    def test_training_dataset_with_expected_response_only(self):
        """Test that expected_response works without expected_facts."""
        entry = EvaluationDatasetEntryModel(
            inputs=ChatPayload(
                messages=[Message(role=MessageRole.USER, content="Hello")]
            ),
            expectations=EvaluationDatasetExpectationsModel(
                expected_response="Hi there!"
            ),
        )

        assert entry.inputs.messages[0].content == "Hello"
        assert entry.expectations.expected_response == "Hi there!"
        assert entry.expectations.expected_facts is None

    @pytest.mark.unit
    def test_training_dataset_with_custom_inputs(self):
        """Test that ChatPayload accepts custom_inputs for additional context."""
        # Test with messages and custom inputs
        entry1 = EvaluationDatasetEntryModel(
            inputs=ChatPayload(
                messages=[Message(role=MessageRole.USER, content="Some text")],
                custom_inputs={"context": "Additional context"},
            ),
            expectations=EvaluationDatasetExpectationsModel(
                expected_response="Response"
            ),
        )
        assert entry1.inputs.messages[0].content == "Some text"
        assert entry1.inputs.custom_inputs["context"] == "Additional context"

        # Test with system and user messages
        entry2 = EvaluationDatasetEntryModel(
            inputs=ChatPayload(
                messages=[
                    Message(
                        role=MessageRole.SYSTEM, content="You are a helpful assistant"
                    ),
                    Message(role=MessageRole.USER, content="Some question"),
                ]
            ),
            expectations=EvaluationDatasetExpectationsModel(expected_response="Answer"),
        )
        assert entry2.inputs.messages[0].role == MessageRole.SYSTEM
        assert entry2.inputs.messages[1].content == "Some question"

    @pytest.mark.unit
    def test_training_dataset_mutual_exclusion_validator(self):
        """Test that expected_response and expected_facts are mutually exclusive."""
        import pytest

        with pytest.raises(ValueError, match="Cannot specify both"):
            EvaluationDatasetExpectationsModel(
                expected_response="Paris is the capital",
                expected_facts=["Paris", "Capital city of France"],
            )

    @pytest.mark.unit
    def test_training_dataset_with_schema(self):
        """Test that EvaluationDatasetModel full_name includes catalog and schema."""
        from dao_ai.config import SchemaModel

        schema = SchemaModel(catalog_name="my_catalog", schema_name="my_schema")
        dataset = EvaluationDatasetModel(name="test_dataset", schema=schema)

        assert dataset.name == "test_dataset"
        assert dataset.full_name == "my_catalog.my_schema.test_dataset"
        assert dataset.schema_model == schema

    @pytest.mark.unit
    def test_training_dataset_in_optimizations_model(self):
        """Test that EvaluationDatasetModel works within OptimizationsModel."""
        dataset = EvaluationDatasetModel(
            name="test_dataset",
            data=[
                EvaluationDatasetEntryModel(
                    inputs=ChatPayload(
                        messages=[Message(role=MessageRole.USER, content="Hello")]
                    ),
                    expectations=EvaluationDatasetExpectationsModel(
                        expected_response="Hi there!"
                    ),
                )
            ],
        )

        optimizations_model = OptimizationsModel(
            training_datasets={"test_dataset": dataset}
        )

        assert len(optimizations_model.training_datasets) == 1
        assert "test_dataset" in optimizations_model.training_datasets
        assert (
            optimizations_model.training_datasets["test_dataset"].name == "test_dataset"
        )

    @pytest.mark.unit
    def test_chat_payload_to_mlflow_messages_conversion(self):
        """Test that ChatPayload messages can be converted to MLflow Message format."""
        from mlflow.types.responses_helpers import Message as MLflowMessage

        # Test with single user message
        payload = ChatPayload(
            messages=[Message(role=MessageRole.USER, content="Hello world")]
        )

        # Convert to MLflow messages (as done in predict_fn)
        mlflow_messages = [
            MLflowMessage(role=msg.role, content=msg.content)
            for msg in payload.messages
        ]

        assert len(mlflow_messages) == 1
        assert mlflow_messages[0].role == "user"
        assert mlflow_messages[0].content == "Hello world"

        # Test with multiple messages including system
        payload2 = ChatPayload(
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are helpful"),
                Message(role=MessageRole.USER, content="What is AI?"),
            ]
        )

        mlflow_messages2 = [
            MLflowMessage(role=msg.role, content=msg.content)
            for msg in payload2.messages
        ]

        assert len(mlflow_messages2) == 2
        assert mlflow_messages2[0].role == "system"
        assert mlflow_messages2[0].content == "You are helpful"
        assert mlflow_messages2[1].role == "user"
        assert mlflow_messages2[1].content == "What is AI?"

    @pytest.mark.unit
    def test_chat_payload_with_custom_inputs(self):
        """Test that ChatPayload handles custom_inputs correctly."""
        payload = ChatPayload(
            messages=[Message(role=MessageRole.USER, content="Test")],
            custom_inputs={"configurable": {"thread_id": "123", "user_id": "user1"}},
        )

        assert payload.custom_inputs is not None
        assert "configurable" in payload.custom_inputs
        assert payload.custom_inputs["configurable"]["thread_id"] == "123"
        assert payload.custom_inputs["configurable"]["user_id"] == "user1"

    @pytest.mark.unit
    def test_evaluation_dataset_entry_model_dump(self):
        """Test that EvaluationDatasetEntryModel.model_dump() serializes ChatPayload correctly."""
        entry = EvaluationDatasetEntryModel(
            inputs=ChatPayload(
                messages=[Message(role=MessageRole.USER, content="Test input")],
                custom_inputs={"key": "value"},
            ),
            expectations=EvaluationDatasetExpectationsModel(
                expected_response="Test response"
            ),
        )

        dumped = entry.model_dump()

        assert "inputs" in dumped
        assert "messages" in dumped["inputs"]
        assert len(dumped["inputs"]["messages"]) == 1
        assert dumped["inputs"]["messages"][0]["role"] == "user"
        assert dumped["inputs"]["messages"][0]["content"] == "Test input"
        assert dumped["inputs"]["custom_inputs"]["key"] == "value"

    @pytest.mark.unit
    def test_evaluation_dataset_entry_to_mlflow_format(self):
        """Test that EvaluationDatasetEntryModel.to_mlflow_format() flattens expectations correctly."""
        # Test with expected_facts
        entry_with_facts = EvaluationDatasetEntryModel(
            inputs=ChatPayload(
                messages=[Message(role=MessageRole.USER, content="Test input")],
                custom_inputs={"key": "value"},
            ),
            expectations=EvaluationDatasetExpectationsModel(
                expected_facts=["Fact 1", "Fact 2"]
            ),
        )

        mlflow_format = entry_with_facts.to_mlflow_format()

        # Verify structure is flattened
        assert "inputs" in mlflow_format
        assert "expected_facts" in mlflow_format  # Flattened to top level
        assert (
            "expectations" not in mlflow_format
        )  # Should not have nested expectations
        assert mlflow_format["expected_facts"] == ["Fact 1", "Fact 2"]
        assert (
            "expected_response" not in mlflow_format
        )  # Should not include None values

        # Test with expected_response
        entry_with_response = EvaluationDatasetEntryModel(
            inputs=ChatPayload(
                messages=[Message(role=MessageRole.USER, content="Test input")],
            ),
            expectations=EvaluationDatasetExpectationsModel(
                expected_response="Expected response"
            ),
        )

        mlflow_format_response = entry_with_response.to_mlflow_format()

        assert "inputs" in mlflow_format_response
        assert "expected_response" in mlflow_format_response  # Flattened to top level
        assert "expectations" not in mlflow_format_response
        assert mlflow_format_response["expected_response"] == "Expected response"
        assert "expected_facts" not in mlflow_format_response

    @pytest.mark.unit
    def test_chat_payload_input_alias(self):
        """Test that ChatPayload correctly handles 'input' as alias for 'messages'."""
        # Test with 'input' field
        payload1 = ChatPayload(input=[Message(role=MessageRole.USER, content="Test")])
        assert payload1.messages is not None
        assert len(payload1.messages) == 1
        assert payload1.input == payload1.messages

        # Test with 'messages' field
        payload2 = ChatPayload(
            messages=[Message(role=MessageRole.USER, content="Test")]
        )
        assert payload2.input is not None
        assert len(payload2.input) == 1
        assert payload2.input == payload2.messages


class TestOptimizationsModelUnit:
    """Unit tests for OptimizationsModel (mocked)."""

    @pytest.mark.unit
    def test_optimizations_model_creation(self):
        """Test that OptimizationsModel can be created with prompt_optimizations dict."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)
        dataset = _create_test_dataset()

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset=dataset,
        )

        optimizations_model = OptimizationsModel(prompt_optimizations={"test_opt": opt})

        assert len(optimizations_model.prompt_optimizations) == 1
        assert "test_opt" in optimizations_model.prompt_optimizations
        assert (
            optimizations_model.prompt_optimizations["test_opt"].name
            == "test_optimization"
        )

    @pytest.mark.unit
    def test_optimizations_model_empty_dict(self):
        """Test that OptimizationsModel can be created with empty dict."""
        optimizations_model = OptimizationsModel(prompt_optimizations={})

        assert len(optimizations_model.prompt_optimizations) == 0
        assert isinstance(optimizations_model.prompt_optimizations, dict)

    @pytest.mark.unit
    @patch("dao_ai.config.PromptOptimizationModel.optimize")
    def test_optimizations_model_optimize_method(self, mock_optimize):
        """Test that OptimizationsModel.optimize() calls optimize on all optimizations."""
        prompt1 = PromptModel(name="prompt1", default_template="Test {{text}}")
        prompt2 = PromptModel(name="prompt2", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)

        dataset1 = _create_test_dataset()
        dataset2 = _create_test_dataset()

        opt1 = PromptOptimizationModel(
            name="opt1", prompt=prompt1, agent=agent, dataset=dataset1
        )
        opt2 = PromptOptimizationModel(
            name="opt2", prompt=prompt2, agent=agent, dataset=dataset2
        )

        optimizations_model = OptimizationsModel(
            prompt_optimizations={"opt1": opt1, "opt2": opt2},
        )

        # Mock the optimize method to return PromptModels
        mock_result1 = PromptModel(name="prompt1", version=2, default_template="Opt1")
        mock_result2 = PromptModel(name="prompt2", version=2, default_template="Opt2")
        mock_optimize.side_effect = [mock_result1, mock_result2]

        results = optimizations_model.optimize()

        # Verify optimize was called on each optimization
        assert mock_optimize.call_count == 2
        assert len(results) == 2
        assert results["opt1"] == mock_result1
        assert results["opt2"] == mock_result2

    @pytest.mark.unit
    def test_optimizations_model_optimize_empty_dict(self):
        """Test that OptimizationsModel.optimize() handles empty dict."""
        optimizations_model = OptimizationsModel(prompt_optimizations={})

        results = optimizations_model.optimize()

        assert len(results) == 0
        assert isinstance(results, dict)


class TestAppConfigWithOptimizations:
    """Tests for AppConfig integration with OptimizationsModel."""

    @pytest.mark.unit
    def test_app_config_with_optimizations(self):
        """Test that AppConfig can include OptimizationsModel."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)
        dataset = _create_test_dataset()

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset=dataset,
        )

        optimizations_model = OptimizationsModel(
            prompt_optimizations={"test_opt": opt},
        )

        config_dict = {
            "prompts": {"test": prompt},
            "agents": {"test": agent},
            "optimizations": optimizations_model,
            "app": {
                "name": "test",
                "registered_model": {"name": "test_model"},
                "agents": [agent],
            },
        }

        config = AppConfig(**config_dict)

        assert config.optimizations is not None
        assert isinstance(config.optimizations, OptimizationsModel)
        assert len(config.optimizations.prompt_optimizations) == 1

    @pytest.mark.unit
    def test_app_config_without_optimizations(self):
        """Test that AppConfig works without OptimizationsModel (optional field)."""
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)

        config_dict = {
            "agents": {"test": agent},
            "app": {
                "name": "test",
                "registered_model": {"name": "test_model"},
                "agents": [agent],
            },
        }

        config = AppConfig(**config_dict)

        assert config.optimizations is None


class TestResponsesAgentPredictFn:
    """Unit tests for ResponsesAgent predict_fn integration."""

    @pytest.mark.unit
    def test_predict_fn_chat_payload_to_responses_agent_request(self):
        """Test that predict_fn correctly converts ChatPayload to ResponsesAgentRequest."""

        # Create mock ResponsesAgent
        mock_agent = Mock()
        # Mock response with output items
        mock_output_item = Mock()
        mock_output_item.content = "Test response"
        mock_response = Mock()
        mock_response.output = [mock_output_item]
        mock_agent.predict.return_value = mock_response

        # Simulate the predict_fn that would be created in optimize_prompt
        def predict_fn(**inputs: dict) -> str:
            from mlflow.types.responses import ResponsesAgentRequest
            from mlflow.types.responses_helpers import Message

            from dao_ai.config import ChatPayload

            chat_payload: ChatPayload = ChatPayload(**inputs)
            mlflow_messages: list[Message] = [
                Message(role=msg.role, content=msg.content)
                for msg in chat_payload.messages
            ]
            request: ResponsesAgentRequest = ResponsesAgentRequest(
                input=mlflow_messages,
                custom_inputs=chat_payload.custom_inputs,
            )
            response = mock_agent.predict(request)
            if response.output and len(response.output) > 0:
                return response.output[0].content
            else:
                return ""

        # Test the predict_fn with proper configurable structure
        result = predict_fn(
            messages=[{"role": "user", "content": "Hello"}],
            custom_inputs={
                "configurable": {"thread_id": "123", "user_id": "test_user"}
            },
        )

        assert result == "Test response"
        mock_agent.predict.assert_called_once()

        # Verify the request was constructed correctly
        call_args = mock_agent.predict.call_args
        request = call_args[0][0]
        assert len(request.input) == 1
        assert request.input[0].role == "user"
        assert request.input[0].content == "Hello"
        # Check that thread_id is preserved in configurable
        assert "configurable" in request.custom_inputs
        assert request.custom_inputs["configurable"]["thread_id"] == "123"

    @pytest.mark.unit
    def test_predict_fn_extracts_response_correctly(self):
        """Test that predict_fn correctly extracts content from ResponsesAgentResponse."""

        mock_agent = Mock()

        # Test with valid response
        mock_output_item = Mock()
        mock_output_item.content = "Extracted content"
        mock_response = Mock()
        mock_response.output = [mock_output_item]
        mock_agent.predict.return_value = mock_response

        def predict_fn(**inputs: dict) -> str:
            from mlflow.types.responses import ResponsesAgentRequest
            from mlflow.types.responses_helpers import Message

            from dao_ai.config import ChatPayload

            chat_payload: ChatPayload = ChatPayload(**inputs)
            mlflow_messages: list[Message] = [
                Message(role=msg.role, content=msg.content)
                for msg in chat_payload.messages
            ]
            request: ResponsesAgentRequest = ResponsesAgentRequest(
                input=mlflow_messages,
                custom_inputs=chat_payload.custom_inputs,
            )
            response = mock_agent.predict(request)
            if response.output and len(response.output) > 0:
                return response.output[0].content
            else:
                return ""

        result = predict_fn(messages=[{"role": "user", "content": "Test"}])
        assert result == "Extracted content"

        # Test with empty response
        mock_empty_response = Mock()
        mock_empty_response.output = []
        mock_agent.predict.return_value = mock_empty_response
        result_empty = predict_fn(messages=[{"role": "user", "content": "Test"}])
        assert result_empty == ""

    @pytest.mark.unit
    def test_predict_fn_handles_multiple_messages(self):
        """Test that predict_fn handles multiple messages in ChatPayload."""

        mock_agent = Mock()
        mock_output_item = Mock()
        mock_output_item.content = "Multi-turn response"
        mock_response = Mock()
        mock_response.output = [mock_output_item]
        mock_agent.predict.return_value = mock_response

        def predict_fn(**inputs: dict) -> str:
            from mlflow.types.responses import ResponsesAgentRequest
            from mlflow.types.responses_helpers import Message

            from dao_ai.config import ChatPayload

            chat_payload: ChatPayload = ChatPayload(**inputs)
            mlflow_messages: list[Message] = [
                Message(role=msg.role, content=msg.content)
                for msg in chat_payload.messages
            ]
            request: ResponsesAgentRequest = ResponsesAgentRequest(
                input=mlflow_messages,
                custom_inputs=chat_payload.custom_inputs,
            )
            response = mock_agent.predict(request)
            if response.output and len(response.output) > 0:
                return response.output[0].content
            else:
                return ""

        # Test with system + user messages
        result = predict_fn(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is AI?"},
            ]
        )

        assert result == "Multi-turn response"

        # Verify all messages were passed
        call_args = mock_agent.predict.call_args
        request = call_args[0][0]
        assert len(request.input) == 2
        assert request.input[0].role == "system"
        assert request.input[0].content == "You are helpful"
        assert request.input[1].role == "user"
        assert request.input[1].content == "What is AI?"


class TestPromptOptimizationIntegration:
    """Integration tests for prompt optimization workflow."""

    @pytest.mark.unit
    @patch("dao_ai.config.PromptOptimizationModel.optimize")
    def test_end_to_end_optimization_workflow(self, mock_optimize):
        """Test complete optimization workflow from config to result."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)
        dataset1 = _create_test_dataset()
        dataset2 = _create_test_dataset()

        opt1 = PromptOptimizationModel(
            name="opt1", prompt=prompt, agent=agent, dataset=dataset1
        )
        opt2 = PromptOptimizationModel(
            name="opt2", prompt=prompt, agent=agent, dataset=dataset2
        )

        optimizations_model = OptimizationsModel(
            prompt_optimizations={"opt1": opt1, "opt2": opt2},
        )

        # Mock optimize to return new versions
        mock_result1 = PromptModel(
            name="test_prompt", version=2, default_template="Optimized1"
        )
        mock_result2 = PromptModel(
            name="test_prompt", version=3, default_template="Optimized2"
        )
        mock_optimize.side_effect = [mock_result1, mock_result2]

        # Run optimization
        results = optimizations_model.optimize()

        # Verify results
        assert len(results) == 2
        assert results["opt1"].version == 2
        assert results["opt2"].version == 3
        assert mock_optimize.call_count == 2


class TestPromptOptimizationSystem:
    """System tests for prompt optimization (requires Databricks connection)."""

    @pytest.mark.system
    @pytest.mark.slow
    @pytest.mark.skipif(
        not has_databricks_env(), reason="Missing Databricks environment variables"
    )
    @pytest.mark.skip("Skipping Databricks prompt optimization with config file")
    def test_optimize_prompt_with_config_file(self):
        """
        Test prompt optimization using the example config file.

        This test:
        - Loads config/examples/11_prompt_engineering/prompt_optimization.yaml
        - Creates/updates the training dataset
        - Runs prompt optimization
        - Verifies the optimized prompt is created

        This test requires:
        - Databricks environment variables
        - Valid MLflow tracking URI
        - Access to prompt registry
        - Access to workspace API
        """
        from dao_ai.config import AppConfig

        # Load the example config
        config_path = "config/examples/11_prompt_engineering/prompt_optimization.yaml"
        config = AppConfig.from_file(config_path)

        assert config.optimizations is not None
        assert len(config.optimizations.prompt_optimizations) > 0
        assert len(config.optimizations.training_datasets) > 0

        # Get the first optimization
        opt_name = list(config.optimizations.prompt_optimizations.keys())[0]
        optimization = config.optimizations.prompt_optimizations[opt_name]

        # Verify optimization has required fields
        assert optimization.name is not None
        assert optimization.prompt is not None
        assert optimization.agent is not None
        assert optimization.dataset is not None

        # Create/update the training dataset
        dataset_name = list(config.optimizations.training_datasets.keys())[0]
        dataset = config.optimizations.training_datasets[dataset_name]

        # This will create or update the dataset
        mlflow_dataset = dataset.as_dataset()
        assert mlflow_dataset is not None

        # Run the optimization
        optimized_prompt = optimization.optimize()

        # Verify the result
        assert optimized_prompt is not None
        assert isinstance(optimized_prompt, PromptModel)
        assert optimized_prompt.name == optimization.prompt.name
        # The optimized version should have a version number
        assert optimized_prompt.version is not None

    @pytest.mark.system
    @pytest.mark.slow
    @pytest.mark.skipif(
        not has_databricks_env(), reason="Missing Databricks environment variables"
    )
    # @pytest.mark.skip("Skipping Databricks prompt optimization system test")
    def test_optimize_prompt_end_to_end(self):
        """
        End-to-end test of prompt optimization with real Databricks connection.

        This test requires:
        - Valid Databricks credentials
        - Access to Databricks foundation models

        Note: This test is skipped by default to avoid unnecessary API calls.
        Remove the @pytest.mark.skip decorator to run this test.
        """
        from dao_ai.optimization import optimize_prompt

        # Create a simple prompt and agent
        prompt = PromptModel(
            name="main.default.system_test_prompt",
            default_template="Summarize the following text: {{text}}",
        )

        llm = LLMModel(
            name="databricks-meta-llama-3-3-70b-instruct",
        )

        agent = AgentModel(name="summarization_agent", model=llm, prompt=prompt)

        # Create a simple evaluation dataset
        dataset = EvaluationDatasetModel(
            name="main.default.test_optimization_dataset",
            overwrite=True,
            data=[
                EvaluationDatasetEntryModel(
                    inputs=ChatPayload(
                        messages=[
                            Message(
                                role=MessageRole.USER,
                                content="Write a summary of machine learning.",
                            )
                        ]
                    ),
                    expectations=EvaluationDatasetExpectationsModel(
                        expected_facts=[
                            "Machine learning is a type of artificial intelligence",
                            "It uses algorithms to learn from data",
                        ]
                    ),
                ),
                EvaluationDatasetEntryModel(
                    inputs=ChatPayload(
                        messages=[
                            Message(
                                role=MessageRole.USER,
                                content="Explain quantum computing briefly.",
                            )
                        ]
                    ),
                    expectations=EvaluationDatasetExpectationsModel(
                        expected_facts=[
                            "Quantum computing uses quantum mechanics",
                            "It can solve certain problems faster than classical computers",
                        ]
                    ),
                ),
            ],
        )

        # Run optimization using GEPA
        result = optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            num_candidates=2,  # Keep small for test speed
            register_if_improved=False,  # Don't register in system test
        )

        # Verify result
        assert result.optimized_prompt is not None
        assert result.optimized_prompt.name == "main.default.system_test_prompt"
        assert result.optimized_template is not None
        assert len(result.optimized_template) > 0

    @pytest.mark.system
    @pytest.mark.slow
    @pytest.mark.skipif(
        not has_databricks_env(), reason="Missing Databricks environment variables"
    )
    @pytest.mark.skip("Skipping Databricks optimization model config system test")
    def test_optimizations_model_from_config(self, development_config):
        """
        Test loading OptimizationsModel from YAML config and running optimizations.

        This test requires a config file with optimizations defined.

        Note: This test is skipped by default to avoid unnecessary API calls.
        Remove the @pytest.mark.skip decorator to run this test.
        """
        from mlflow.models import ModelConfig

        model_config = ModelConfig(development_config=development_config)
        config = AppConfig(**model_config.to_dict())

        # Verify optimizations were loaded
        if config.optimizations is not None:
            assert isinstance(config.optimizations, OptimizationsModel)
            assert len(config.optimizations.prompt_optimizations) > 0

            # Run optimizations (if any are defined)
            results = config.optimizations.optimize()

            # Verify results
            assert isinstance(results, dict)
            for key, result in results.items():
                assert isinstance(result, PromptModel)
                assert result.name is not None


@pytest.mark.integration
@pytest.mark.skipif(
    not has_databricks_env(), reason="Databricks environment variables not set"
)
class TestPromptOptimizationWithDatabricks:
    """Integration tests for prompt optimization with real Databricks services."""

    def test_optimize_prompt_end_to_end_with_databricks(self):
        """
        End-to-end integration test for prompt optimization using Databricks.

        This test:
        1. Loads configuration from prompt_optimization.yaml
        2. Creates/loads the prompt from default_template if needed
        3. Runs the optimization using real Databricks services
        4. Verifies the optimized prompt is registered correctly
        5. Checks that aliases are set properly

        Note: This test requires:
        - Valid Databricks workspace credentials
        - Access to Unity Catalog
        - MLflow Prompt Registry access
        - Model serving endpoint access
        """
        pytest.skip(
            "Skipping due to MLflow judge JSON parsing bug with multi-line rationales. "
            "MLflow 3.5's _invoke_databricks_judge_model fails to parse judge responses "
            "that contain unescaped newlines in JSON strings. This is an MLflow internal "
            "issue, not a dao-ai bug. See: json.decoder.JSONDecodeError in "
            "mlflow/genai/judges/utils.py:882"
        )

        from pathlib import Path

        from dao_ai.config import AppConfig

        # Load the prompt optimization configuration
        config_path = Path(
            "config/examples/11_prompt_engineering/prompt_optimization.yaml"
        )
        assert config_path.exists(), f"Config file not found: {config_path}"

        config = AppConfig.from_file(str(config_path))

        # Verify optimizations are configured
        assert config.optimizations is not None
        assert config.optimizations.prompt_optimizations is not None
        assert len(config.optimizations.prompt_optimizations) > 0

        # Get the optimization configuration
        optimization = config.optimizations.prompt_optimizations["optimize_diy_prompt"]
        assert optimization.name == "optimize_diy_prompt"
        assert optimization.prompt is not None
        assert optimization.agent is not None
        assert optimization.dataset is not None

        # Verify prompt configuration
        prompt = optimization.prompt
        assert prompt.name == "diy_prompt"
        assert prompt.default_template is not None
        assert len(prompt.default_template) > 0

        # Initialize Databricks provider
        provider = DatabricksProvider()

        # Test 1: Ensure prompt exists in registry (create from default_template if needed)
        logger.info(f"Loading/creating prompt: {prompt.full_name}")
        try:
            prompt_version = provider.get_prompt(prompt)
            assert prompt_version is not None
            assert prompt_version.version is not None
            logger.info(f"Prompt version loaded: {prompt_version.version}")
        except Exception as e:
            pytest.skip(
                f"Unable to create/load prompt '{prompt.full_name}': {e}. "
                "This test requires Unity Catalog access and prompt registry permissions."
            )

        # Verify the prompt has required aliases
        import mlflow

        mlflow.set_registry_uri("databricks-uc")

        # Check that default and latest aliases exist
        try:
            default_prompt = mlflow.genai.load_prompt(
                f"prompts:/{prompt.full_name}@default"
            )
            assert default_prompt is not None
            logger.info(f"Default alias verified for {prompt.full_name}")
        except Exception as e:
            pytest.fail(f"Default alias not found for {prompt.full_name}: {e}")

        try:
            latest_prompt = mlflow.genai.load_prompt(
                f"prompts:/{prompt.full_name}@latest"
            )
            assert latest_prompt is not None
            logger.info(f"Latest alias verified for {prompt.full_name}")
        except Exception as e:
            pytest.fail(f"Latest alias not found for {prompt.full_name}: {e}")

        # Test 2: Verify dataset is configured correctly
        logger.info("Verifying dataset configuration")
        dataset_model = optimization.dataset
        assert dataset_model is not None

        if not isinstance(dataset_model, str):
            # If it's an EvaluationDatasetModel, verify it has data
            assert hasattr(dataset_model, "data")
            assert len(dataset_model.data) > 0

            # Verify the dataset entries have proper structure
            first_entry = dataset_model.data[0]
            assert hasattr(first_entry, "inputs")
            assert hasattr(first_entry, "expectations")

            # Create the dataset in MLflow
            logger.info("Creating evaluation dataset")
            dataset = dataset_model.as_dataset()
            assert dataset is not None

        # Test 3: Run the optimization
        logger.info("Running prompt optimization (this may take several minutes)")
        logger.info(f"Generating {optimization.num_candidates} candidate prompts...")

        original_version = prompt_version.version
        logger.info(f"Original prompt version: {original_version}")

        # Run optimization
        optimized_prompt = provider.optimize_prompt(optimization)

        # Test 4: Verify optimization results
        assert optimized_prompt is not None
        assert optimized_prompt.name == prompt.name

        # The optimized prompt should either:
        # 1. Return the same prompt if no improvement was found
        # 2. Return a new prompt with updated alias

        # Load the latest version after optimization
        latest_after_optimization = mlflow.genai.load_prompt(
            f"prompts:/{prompt.full_name}@latest"
        )

        logger.info(
            f"Prompt version after optimization: {latest_after_optimization.version}"
        )

        # Test 5: Verify prompt history and aliases
        # Note: We can't directly list prompt versions via MlflowClient for UC prompts,
        # but we can verify that the latest alias points to a valid version
        assert latest_after_optimization.version >= original_version

        logger.info("Integration test completed successfully!")
        logger.info(f"Original version: {original_version}")
        logger.info(f"Latest version: {latest_after_optimization.version}")

        if latest_after_optimization.version > original_version:
            logger.info("New optimized version was registered!")
        else:
            logger.info(
                "No new version registered (optimization determined existing prompt was optimal)"
            )

    def test_optimize_prompt_only_registers_improvements(self):
        """
        Test that prompt optimization only registers new versions when there's actual improvement.

        This test verifies:
        1. Templates that are identical to the original are not registered
        2. Templates that don't improve scores are not registered
        3. Only genuinely better prompts result in new versions
        """
        pytest.skip(
            "Skipping due to MLflow judge JSON parsing bug with multi-line rationales. "
            "MLflow 3.5's _invoke_databricks_judge_model fails to parse judge responses "
            "that contain unescaped newlines in JSON strings. This is an MLflow internal "
            "issue, not a dao-ai bug. See: json.decoder.JSONDecodeError in "
            "mlflow/genai/judges/utils.py:882"
        )

        from pathlib import Path

        from dao_ai.config import AppConfig

        config_path = Path(
            "config/examples/11_prompt_engineering/prompt_optimization.yaml"
        )
        config = AppConfig.from_file(str(config_path))

        optimization = config.optimizations.prompt_optimizations["optimize_diy_prompt"]
        provider = DatabricksProvider()

        # Get the current version count
        prompt = optimization.prompt

        import mlflow

        mlflow.set_registry_uri("databricks-uc")

        # Ensure the prompt exists before running optimization
        try:
            provider.get_prompt(prompt)
        except Exception as e:
            pytest.skip(
                f"Unable to create/load prompt '{prompt.full_name}': {e}. "
                "This test requires Unity Catalog access and prompt registry permissions."
            )

        # Load the latest version before optimization
        try:
            latest_before = mlflow.genai.load_prompt(
                f"prompts:/{prompt.full_name}@latest"
            )
            version_before = latest_before.version
            logger.info(f"Version before optimization: {version_before}")
        except Exception as e:
            pytest.skip(
                f"Unable to load latest version of '{prompt.full_name}': {e}. "
                "The prompt may not have been created with the latest alias."
            )

        # Run optimization with a very small number of candidates to speed up the test
        # This makes it more likely that no improvement will be found
        optimization.num_candidates = 3

        result = provider.optimize_prompt(optimization)

        # Load the latest version after optimization
        latest_after = mlflow.genai.load_prompt(f"prompts:/{prompt.full_name}@latest")
        version_after = latest_after.version

        logger.info(f"Version after optimization: {version_after}")

        # Verify that the version only increased if there was actual improvement
        if version_after > version_before:
            logger.info("New version was registered - optimization found improvement")
            # If a new version was registered, verify the template is different
            template_before = latest_before.to_single_brace_format().strip()
            template_after = latest_after.to_single_brace_format().strip()

            # Normalize whitespace for comparison
            import re

            template_before_normalized = re.sub(r"\s+", " ", template_before)
            template_after_normalized = re.sub(r"\s+", " ", template_after)

            assert template_before_normalized != template_after_normalized, (
                "New version was registered but templates are identical"
            )
        else:
            logger.info("No new version registered - no improvement found")

        assert result is not None
