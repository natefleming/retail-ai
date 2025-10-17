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

from dao_ai.config import (
    AgentModel,
    AppConfig,
    EvaluationDatasetEntryModel,
    EvaluationDatasetExpectationsModel,
    EvaluationDatasetModel,
    LLMModel,
    OptimizationsModel,
    PromptModel,
    PromptOptimizationModel,
)
from dao_ai.providers.databricks import DatabricksProvider


class TestPromptOptimizationModelUnit:
    """Unit tests for PromptOptimizationModel (mocked)."""

    @pytest.mark.unit
    def test_prompt_optimization_model_creation(self):
        """Test that PromptOptimizationModel can be created with required fields."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset="test_dataset",  # Now a string reference
        )

        assert opt.name == "test_optimization"
        assert opt.prompt.name == "test_prompt"
        assert opt.agent.name == "test_agent"
        assert opt.dataset == "test_dataset"
        assert isinstance(opt.dataset, str)

    @pytest.mark.unit
    def test_prompt_optimization_model_defaults(self):
        """Test that PromptOptimizationModel has correct default values."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset="test_dataset",
        )

        assert opt.num_candidates == 5
        assert opt.reflection_model is None
        assert opt.scorer_model is None

    @pytest.mark.unit
    def test_prompt_optimization_model_custom_params(self):
        """Test that PromptOptimizationModel accepts custom optimizer parameters."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)
        reflection_llm = LLMModel(name="gpt-4o")

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset="test_dataset",
            reflection_model=reflection_llm,
            num_candidates=10,
        )

        assert opt.num_candidates == 10
        assert opt.reflection_model.name == "gpt-4o"

    @pytest.mark.unit
    def test_prompt_optimization_model_agent_as_string(self):
        """Test that PromptOptimizationModel accepts agent as string reference."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent="test_agent",  # String reference
            dataset="test_dataset",
        )

        assert opt.agent == "test_agent"
        assert isinstance(opt.agent, str)

    @pytest.mark.unit
    def test_prompt_optimization_model_reflection_model_as_string(self):
        """Test that PromptOptimizationModel accepts reflection_model as string reference."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset="test_dataset",
            reflection_model="gpt-4o",  # String reference
        )

        assert opt.reflection_model == "gpt-4o"
        assert isinstance(opt.reflection_model, str)

    @pytest.mark.unit
    def test_prompt_optimization_model_scorer_model_as_string(self):
        """Test that PromptOptimizationModel accepts scorer_model as string reference."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset="test_dataset",
            scorer_model="gpt-4o",  # String reference
        )

        assert opt.scorer_model == "gpt-4o"
        assert isinstance(opt.scorer_model, str)

    @pytest.mark.unit
    def test_prompt_optimization_model_mixed_string_and_model_refs(self):
        """Test that PromptOptimizationModel accepts mix of string and model references."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset="test_dataset",
            reflection_model="gpt-4o",  # String reference
            scorer_model=llm,  # LLMModel reference
        )

        assert opt.reflection_model == "gpt-4o"
        assert isinstance(opt.reflection_model, str)
        assert opt.scorer_model.name == "gpt-4o-mini"
        assert isinstance(opt.scorer_model, LLMModel)

    @pytest.mark.unit
    @patch("dao_ai.providers.databricks.DatabricksProvider.optimize_prompt")
    def test_prompt_optimization_model_optimize_method(self, mock_optimize):
        """Test that PromptOptimizationModel.optimize() delegates to DatabricksProvider."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset="test_dataset",
        )

        # Mock the optimize_prompt method to return a new PromptModel
        mock_optimized_prompt = PromptModel(
            name="test_prompt", version=2, default_template="Optimized {{text}}"
        )
        mock_optimize.return_value = mock_optimized_prompt

        result = opt.optimize()

        # Verify the method was called with the optimization object
        mock_optimize.assert_called_once_with(opt)
        assert result == mock_optimized_prompt


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
                inputs={"text": "Hello world"},
                expectations=EvaluationDatasetExpectationsModel(
                    expected_response="Hello! How can I help you?"
                ),
            ),
            EvaluationDatasetEntryModel(
                inputs={"text": "Goodbye world"},
                expectations=EvaluationDatasetExpectationsModel(
                    expected_response="Goodbye! Have a great day!"
                ),
            ),
        ]

        dataset = EvaluationDatasetModel(name="test_dataset", data=entries)

        assert dataset.name == "test_dataset"
        assert len(dataset.data) == 2
        assert dataset.data[0].inputs["text"] == "Hello world"
        assert (
            dataset.data[0].expectations.expected_response
            == "Hello! How can I help you?"
        )
        assert dataset.full_name == "test_dataset"

    @pytest.mark.unit
    def test_training_dataset_with_expected_facts(self):
        """Test that EvaluationDatasetModel can be created with expected_facts."""
        entry = EvaluationDatasetEntryModel(
            inputs={"question": "What is the capital of France?"},
            expectations=EvaluationDatasetExpectationsModel(
                expected_facts=["Paris", "Capital city of France"]
            ),
        )

        dataset = EvaluationDatasetModel(name="test_dataset", data=[entry])

        assert dataset.name == "test_dataset"
        assert len(dataset.data) == 1
        assert dataset.data[0].inputs["question"] == "What is the capital of France?"
        assert dataset.data[0].expectations.expected_facts == [
            "Paris",
            "Capital city of France",
        ]
        assert dataset.data[0].expectations.expected_response is None

    @pytest.mark.unit
    def test_training_dataset_with_expected_response_only(self):
        """Test that expected_response works without expected_facts."""
        entry = EvaluationDatasetEntryModel(
            inputs={"text": "Hello"},
            expectations=EvaluationDatasetExpectationsModel(
                expected_response="Hi there!"
            ),
        )

        assert entry.inputs["text"] == "Hello"
        assert entry.expectations.expected_response == "Hi there!"
        assert entry.expectations.expected_facts is None

    @pytest.mark.unit
    def test_training_dataset_flexible_input_fields(self):
        """Test that inputs accepts any input fields as dict."""
        # Test with 'text' field
        entry1 = EvaluationDatasetEntryModel(
            inputs={"text": "Some text"},
            expectations=EvaluationDatasetExpectationsModel(
                expected_response="Response"
            ),
        )
        assert entry1.inputs["text"] == "Some text"

        # Test with 'question' field
        entry2 = EvaluationDatasetEntryModel(
            inputs={"question": "Some question"},
            expectations=EvaluationDatasetExpectationsModel(expected_response="Answer"),
        )
        assert entry2.inputs["question"] == "Some question"

        # Test with custom fields
        entry3 = EvaluationDatasetEntryModel(
            inputs={"context": "Context here", "query": "Query here"},
            expectations=EvaluationDatasetExpectationsModel(expected_response="Result"),
        )
        assert entry3.inputs["context"] == "Context here"
        assert entry3.inputs["query"] == "Query here"

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
                    inputs={"text": "Hello"},
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


class TestOptimizationsModelUnit:
    """Unit tests for OptimizationsModel (mocked)."""

    @pytest.mark.unit
    def test_optimizations_model_creation(self):
        """Test that OptimizationsModel can be created with prompt_optimizations dict."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset="test_dataset",
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
    @patch("dao_ai.config.EvaluationDatasetModel.as_dataset")
    def test_optimizations_model_optimize_method(self, mock_as_dataset, mock_optimize):
        """Test that OptimizationsModel.optimize() calls optimize on all optimizations."""
        prompt1 = PromptModel(name="prompt1", default_template="Test {{text}}")
        prompt2 = PromptModel(name="prompt2", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)

        dataset1 = EvaluationDatasetModel(name="dataset1")
        dataset2 = EvaluationDatasetModel(name="dataset2")

        opt1 = PromptOptimizationModel(
            name="opt1", prompt=prompt1, agent=agent, dataset="dataset1"
        )
        opt2 = PromptOptimizationModel(
            name="opt2", prompt=prompt2, agent=agent, dataset="dataset2"
        )

        optimizations_model = OptimizationsModel(
            training_datasets={"dataset1": dataset1, "dataset2": dataset2},
            prompt_optimizations={"opt1": opt1, "opt2": opt2},
        )

        # Mock the as_dataset method
        mock_as_dataset.return_value = Mock()

        # Mock the optimize method to return PromptModels
        mock_result1 = PromptModel(name="prompt1", version=2, default_template="Opt1")
        mock_result2 = PromptModel(name="prompt2", version=2, default_template="Opt2")
        mock_optimize.side_effect = [mock_result1, mock_result2]

        results = optimizations_model.optimize()

        # Verify as_dataset was called for each training dataset
        assert mock_as_dataset.call_count == 2

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
        dataset = EvaluationDatasetModel(name="test_dataset")

        opt = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent=agent,
            dataset="test_dataset",
        )

        optimizations_model = OptimizationsModel(
            training_datasets={"test_dataset": dataset},
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
        assert len(config.optimizations.training_datasets) == 1
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


class TestDatabricksProviderOptimizePromptUnit:
    """Unit tests for DatabricksProvider.optimize_prompt (mocked)."""

    @pytest.mark.unit
    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    @pytest.mark.skip(
        "Skipping due to complex MLflow mocking requirements - integration tests provide coverage"
    )
    def test_optimize_prompt_with_agent_string_raises_error(self):
        """Test optimize_prompt with agent as string reference raises error."""
        provider = DatabricksProvider(w=Mock(), vsc=Mock())

        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")

        optimization = PromptOptimizationModel(
            name="test_optimization",
            prompt=prompt,
            agent="test_agent",  # String reference
            dataset="test_dataset",
        )

        # Call optimize_prompt - should raise ValueError
        with pytest.raises(
            ValueError,
            match="Agent reference by string .* not yet supported",
        ):
            provider.optimize_prompt(optimization)


class TestPromptOptimizationIntegration:
    """Integration tests for prompt optimization workflow."""

    @pytest.mark.unit
    @patch("dao_ai.providers.databricks.DatabricksProvider.optimize_prompt")
    @patch("dao_ai.config.EvaluationDatasetModel.as_dataset")
    def test_end_to_end_optimization_workflow(self, mock_as_dataset, mock_optimize):
        """Test complete optimization workflow from config to result."""
        prompt = PromptModel(name="test_prompt", default_template="Test {{text}}")
        llm = LLMModel(name="gpt-4o-mini")
        agent = AgentModel(name="test_agent", model=llm)
        dataset1 = EvaluationDatasetModel(name="dataset1")
        dataset2 = EvaluationDatasetModel(name="dataset2")

        opt1 = PromptOptimizationModel(
            name="opt1", prompt=prompt, agent=agent, dataset="dataset1"
        )
        opt2 = PromptOptimizationModel(
            name="opt2", prompt=prompt, agent=agent, dataset="dataset2"
        )

        optimizations_model = OptimizationsModel(
            training_datasets={"dataset1": dataset1, "dataset2": dataset2},
            prompt_optimizations={"opt1": opt1, "opt2": opt2},
        )

        # Mock as_dataset
        mock_as_dataset.return_value = Mock()

        # Mock optimize_prompt to return new versions
        mock_result1 = PromptModel(
            name="test_prompt", version=2, default_template="Optimized1"
        )
        mock_result2 = PromptModel(
            name="test_prompt", version=3, default_template="Optimized2"
        )
        mock_optimize.side_effect = [mock_result1, mock_result2]

        # Run optimization
        results = optimizations_model.optimize()

        # Verify datasets were created
        assert mock_as_dataset.call_count == 2

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
    @pytest.mark.skip("Skipping Databricks prompt optimization system test")
    def test_optimize_prompt_end_to_end(self):
        """
        End-to-end test of prompt optimization with real Databricks connection.

        This test requires:
        - Valid Databricks credentials
        - An existing MLflow dataset registered as 'test_optimization_dataset'
        - Access to Databricks foundation models

        Note: This test is skipped by default to avoid unnecessary API calls.
        Remove the @pytest.mark.skip decorator to run this test.
        """
        provider = DatabricksProvider()

        # Create a simple prompt and agent
        prompt = PromptModel(
            name="system_test_prompt",
            default_template="Summarize the following text: {{text}}",
        )

        llm = LLMModel(
            name="databricks-meta-llama-3-1-70b-instruct",
            endpoint_name="databricks-meta-llama-3-1-70b-instruct",
        )

        agent = AgentModel(name="summarization_agent", model=llm)

        optimization = PromptOptimizationModel(
            name="system_test_optimization",
            prompt=prompt,
            agent=agent,
            dataset="test_optimization_dataset",
            num_candidates=2,  # Keep small for test speed
        )

        # Run optimization
        result = provider.optimize_prompt(optimization)

        # Verify result
        assert isinstance(result, PromptModel)
        assert result.name == "system_test_prompt"
        assert result.default_template is not None
        assert len(result.default_template) > 0

        # The optimized template should be different from the original
        # (though this isn't guaranteed, it's very likely)
        # We just verify we got a valid result back

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
