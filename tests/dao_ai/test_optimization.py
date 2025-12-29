"""
Tests for the GEPA-based prompt optimization module.

This test module provides coverage for:
- OptimizationResult dataclass
- _TrainingExample and _Trajectory dataclasses
- DAOAgentAdapter class
- _convert_dataset function
- _register_optimized_prompt function
- optimize_prompt main function

Test Categories:
- Unit tests: Test individual components in isolation
- Integration tests: Test components working together with mocks
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from dao_ai.config import (
    AgentModel,
    ChatPayload,
    EvaluationDatasetEntryModel,
    EvaluationDatasetExpectationsModel,
    EvaluationDatasetModel,
    LLMModel,
    Message,
    MessageRole,
    PromptModel,
)
from dao_ai.optimization import (
    OptimizationResult,
    optimize_prompt,
)

# --- Test Fixtures ---


def _create_test_prompt() -> PromptModel:
    """Create a test prompt model."""
    return PromptModel(
        name="test_prompt",
        default_template="Answer the following question: {question}",
        description="Test prompt for optimization",
        tags={"env": "test"},
    )


def _create_test_agent() -> AgentModel:
    """Create a test agent model."""
    return AgentModel(
        name="test_agent",
        model=LLMModel(name="gpt-4o-mini"),
        prompt=_create_test_prompt(),
    )


def _create_test_dataset() -> EvaluationDatasetModel:
    """Create a test evaluation dataset."""
    return EvaluationDatasetModel(
        name="test_dataset",
        data=[
            EvaluationDatasetEntryModel(
                inputs=ChatPayload(
                    messages=[
                        Message(
                            role=MessageRole.USER,
                            content="What is machine learning?",
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
                            content="Explain neural networks.",
                        )
                    ]
                ),
                expectations=EvaluationDatasetExpectationsModel(
                    expected_facts=[
                        "Neural networks are inspired by the brain",
                        "They consist of layers of nodes",
                    ]
                ),
            ),
            EvaluationDatasetEntryModel(
                inputs=ChatPayload(
                    messages=[
                        Message(
                            role=MessageRole.USER,
                            content="What is deep learning?",
                        )
                    ]
                ),
                expectations=EvaluationDatasetExpectationsModel(
                    expected_response="Deep learning uses multiple layers of neural networks.",
                ),
            ),
        ],
    )


# --- Unit Tests for OptimizationResult ---


class TestOptimizationResult:
    """Unit tests for OptimizationResult dataclass."""

    @pytest.mark.unit
    def test_optimization_result_creation(self) -> None:
        """Test that OptimizationResult can be created with required fields."""
        prompt = _create_test_prompt()

        result = OptimizationResult(
            optimized_prompt=prompt,
            optimized_template="New template: {question}",
            original_score=0.5,
            optimized_score=0.8,
            improvement=0.6,
            num_evaluations=50,
        )

        assert result.optimized_prompt == prompt
        assert result.optimized_template == "New template: {question}"
        assert result.original_score == 0.5
        assert result.optimized_score == 0.8
        assert result.improvement == 0.6
        assert result.num_evaluations == 50
        assert result.registered_version is None
        assert result.metadata == {}

    @pytest.mark.unit
    def test_optimization_result_improved_property_true(self) -> None:
        """Test that improved property returns True when score increased."""
        prompt = _create_test_prompt()

        result = OptimizationResult(
            optimized_prompt=prompt,
            optimized_template="New template",
            original_score=0.5,
            optimized_score=0.8,
            improvement=0.6,
            num_evaluations=50,
        )

        assert result.improved is True

    @pytest.mark.unit
    def test_optimization_result_improved_property_false(self) -> None:
        """Test that improved property returns False when score not increased."""
        prompt = _create_test_prompt()

        result = OptimizationResult(
            optimized_prompt=prompt,
            optimized_template="New template",
            original_score=0.8,
            optimized_score=0.5,
            improvement=-0.375,
            num_evaluations=50,
        )

        assert result.improved is False

    @pytest.mark.unit
    def test_optimization_result_improved_property_equal(self) -> None:
        """Test that improved property returns False when scores are equal."""
        prompt = _create_test_prompt()

        result = OptimizationResult(
            optimized_prompt=prompt,
            optimized_template="Same template",
            original_score=0.7,
            optimized_score=0.7,
            improvement=0.0,
            num_evaluations=50,
        )

        assert result.improved is False

    @pytest.mark.unit
    def test_optimization_result_with_metadata(self) -> None:
        """Test OptimizationResult with metadata."""
        prompt = _create_test_prompt()

        result = OptimizationResult(
            optimized_prompt=prompt,
            optimized_template="New template",
            original_score=0.5,
            optimized_score=0.8,
            improvement=0.6,
            num_evaluations=50,
            metadata={
                "optimizer": "gepa",
                "reflection_model": "gpt-4o",
                "train_size": 40,
                "val_size": 10,
            },
        )

        assert result.metadata["optimizer"] == "gepa"
        assert result.metadata["reflection_model"] == "gpt-4o"
        assert result.metadata["train_size"] == 40
        assert result.metadata["val_size"] == 10


# --- Unit Tests for Internal Types ---


class TestInternalTypes:
    """Unit tests for internal dataclasses."""

    @pytest.mark.unit
    def test_training_example_creation(self) -> None:
        """Test _TrainingExample dataclass creation."""
        from dao_ai.optimization import _TrainingExample

        example = _TrainingExample(
            question="What is AI?",
            expected_facts=["AI is artificial intelligence", "AI can learn"],
            expected_response=None,
            custom_inputs={"user_id": "test_user"},
        )

        assert example.question == "What is AI?"
        assert example.expected_facts == [
            "AI is artificial intelligence",
            "AI can learn",
        ]
        assert example.expected_response is None
        assert example.custom_inputs == {"user_id": "test_user"}

    @pytest.mark.unit
    def test_training_example_defaults(self) -> None:
        """Test _TrainingExample default values."""
        from dao_ai.optimization import _TrainingExample

        example = _TrainingExample(question="What is AI?")

        assert example.question == "What is AI?"
        assert example.expected_facts is None
        assert example.expected_response is None
        assert example.custom_inputs is None

    @pytest.mark.unit
    def test_trajectory_creation(self) -> None:
        """Test _Trajectory dataclass creation."""
        from dao_ai.optimization import _Trajectory

        trajectory = _Trajectory(
            question="What is AI?",
            response="AI stands for artificial intelligence.",
            expected=["AI is artificial intelligence"],
            score=0.8,
            error=None,
        )

        assert trajectory.question == "What is AI?"
        assert trajectory.response == "AI stands for artificial intelligence."
        assert trajectory.expected == ["AI is artificial intelligence"]
        assert trajectory.score == 0.8
        assert trajectory.error is None

    @pytest.mark.unit
    def test_trajectory_with_error(self) -> None:
        """Test _Trajectory with error field populated."""
        from dao_ai.optimization import _Trajectory

        trajectory = _Trajectory(
            question="What is AI?",
            response="",
            expected=["AI is artificial intelligence"],
            score=0.0,
            error="Connection timeout",
        )

        assert trajectory.error == "Connection timeout"
        assert trajectory.score == 0.0


# --- Unit Tests for Dataset Conversion ---


class TestConvertDataset:
    """Unit tests for _convert_dataset function."""

    @pytest.mark.unit
    def test_convert_dataset_from_model(self) -> None:
        """Test converting EvaluationDatasetModel to training examples."""
        from dao_ai.optimization import _convert_dataset

        dataset = _create_test_dataset()
        examples = _convert_dataset(dataset)

        assert len(examples) == 3
        assert examples[0].question == "What is machine learning?"
        assert examples[0].expected_facts is not None
        assert len(examples[0].expected_facts) == 2

    @pytest.mark.unit
    def test_convert_dataset_from_entries(self) -> None:
        """Test converting list of entries to training examples."""
        from dao_ai.optimization import _convert_dataset

        entries = [
            EvaluationDatasetEntryModel(
                inputs=ChatPayload(
                    messages=[Message(role=MessageRole.USER, content="Question 1")]
                ),
                expectations=EvaluationDatasetExpectationsModel(
                    expected_response="Answer 1"
                ),
            ),
            EvaluationDatasetEntryModel(
                inputs=ChatPayload(
                    messages=[Message(role=MessageRole.USER, content="Question 2")]
                ),
                expectations=EvaluationDatasetExpectationsModel(
                    expected_facts=["Fact 1", "Fact 2"]
                ),
            ),
        ]

        examples = _convert_dataset(entries)

        assert len(examples) == 2
        assert examples[0].question == "Question 1"
        assert examples[0].expected_response == "Answer 1"
        assert examples[1].question == "Question 2"
        assert examples[1].expected_facts == ["Fact 1", "Fact 2"]

    @pytest.mark.unit
    def test_convert_dataset_preserves_custom_inputs(self) -> None:
        """Test that custom_inputs are preserved during conversion."""
        from dao_ai.optimization import _convert_dataset

        entries = [
            EvaluationDatasetEntryModel(
                inputs=ChatPayload(
                    messages=[Message(role=MessageRole.USER, content="Question")],
                    custom_inputs={"store_num": "12345", "user_id": "test_user"},
                ),
                expectations=EvaluationDatasetExpectationsModel(
                    expected_response="Answer"
                ),
            ),
        ]

        examples = _convert_dataset(entries)

        assert examples[0].custom_inputs is not None
        assert examples[0].custom_inputs["store_num"] == "12345"
        assert examples[0].custom_inputs["user_id"] == "test_user"

    @pytest.mark.unit
    def test_convert_dataset_empty(self) -> None:
        """Test converting empty dataset."""
        from dao_ai.optimization import _convert_dataset

        dataset = EvaluationDatasetModel(name="empty", data=[])
        examples = _convert_dataset(dataset)

        assert len(examples) == 0


# --- Unit Tests for DAOAgentAdapter ---


class TestDAOAgentAdapter:
    """Unit tests for DAOAgentAdapter class."""

    @pytest.mark.unit
    def test_adapter_initialization(self) -> None:
        """Test DAOAgentAdapter initialization."""
        from dao_ai.optimization import DAOAgentAdapter

        agent = _create_test_agent()
        adapter = DAOAgentAdapter(agent_model=agent)

        assert adapter.agent_model == agent
        assert adapter.metric_fn is not None  # Default metric
        assert adapter._agent is None

    @pytest.mark.unit
    def test_adapter_with_custom_metric(self) -> None:
        """Test DAOAgentAdapter with custom metric function."""
        from dao_ai.optimization import DAOAgentAdapter, _TrainingExample

        def custom_metric(response: str, example: _TrainingExample) -> float:
            return 1.0 if "test" in response.lower() else 0.0

        agent = _create_test_agent()
        adapter = DAOAgentAdapter(agent_model=agent, metric_fn=custom_metric)

        assert adapter.metric_fn == custom_metric

    @pytest.mark.unit
    def test_default_metric_with_expected_facts(self) -> None:
        """Test default metric calculation with expected_facts."""
        from dao_ai.optimization import DAOAgentAdapter, _TrainingExample

        agent = _create_test_agent()
        adapter = DAOAgentAdapter(agent_model=agent)

        example = _TrainingExample(
            question="What is AI?",
            expected_facts=["artificial intelligence", "machine learning"],
        )

        # Response contains one fact
        score = adapter._default_metric(
            "AI stands for artificial intelligence", example
        )
        assert score == 0.5  # 1 out of 2 facts

        # Response contains both facts
        score = adapter._default_metric(
            "AI is artificial intelligence and uses machine learning", example
        )
        assert score == 1.0  # 2 out of 2 facts

        # Response contains no facts
        score = adapter._default_metric("I don't know", example)
        assert score == 0.0  # 0 out of 2 facts

    @pytest.mark.unit
    def test_default_metric_with_expected_response(self) -> None:
        """Test default metric calculation with expected_response."""
        from dao_ai.optimization import DAOAgentAdapter, _TrainingExample

        agent = _create_test_agent()
        adapter = DAOAgentAdapter(agent_model=agent)

        example = _TrainingExample(
            question="What is the capital of France?",
            expected_response="Paris is the capital of France",
        )

        # Response with high word overlap
        score = adapter._default_metric("Paris is the capital city of France", example)
        assert score > 0.5

        # Response with no overlap
        score = adapter._default_metric("I don't know", example)
        assert score < 0.5

    @pytest.mark.unit
    def test_make_reflective_dataset(self) -> None:
        """Test make_reflective_dataset method."""
        from dao_ai.optimization import DAOAgentAdapter, _TrainingExample, _Trajectory

        agent = _create_test_agent()
        adapter = DAOAgentAdapter(agent_model=agent)

        batch = [
            _TrainingExample(
                question="What is AI?",
                expected_facts=["artificial intelligence"],
            )
        ]
        trajectories = [
            _Trajectory(
                question="What is AI?",
                response="AI is artificial intelligence",
                expected=["artificial intelligence"],
                score=1.0,
            )
        ]

        reflective_data = adapter.make_reflective_dataset(batch, trajectories, "prompt")

        assert len(reflective_data) == 1
        assert "input" in reflective_data[0]
        assert "output" in reflective_data[0]
        assert "feedback" in reflective_data[0]
        assert "What is AI?" in reflective_data[0]["input"]


# --- Integration Tests ---


class TestOptimizePromptIntegration:
    """Integration tests for optimize_prompt function."""

    @pytest.mark.unit
    @patch("dao_ai.optimization.optimize")
    def test_optimize_prompt_returns_result(
        self, mock_gepa_optimize: MagicMock
    ) -> None:
        """Test that optimize_prompt returns OptimizationResult."""
        from gepa import GEPAResult

        # Create mock GEPAResult
        mock_result = Mock(spec=GEPAResult)
        mock_result.best_candidate = {"prompt": "Optimized: {question}"}
        mock_result.val_aggregate_scores = [0.5, 0.6, 0.7, 0.8]
        mock_result.best_idx = 3
        mock_result.total_metric_calls = 50
        mock_gepa_optimize.return_value = mock_result

        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = _create_test_dataset()

        result = optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            num_candidates=50,
            register_if_improved=False,
        )

        assert isinstance(result, OptimizationResult)
        assert result.optimized_template == "Optimized: {question}"
        assert result.original_score == 0.5
        assert result.optimized_score == 0.8
        assert result.num_evaluations == 50
        mock_gepa_optimize.assert_called_once()

    @pytest.mark.unit
    @patch("dao_ai.optimization.optimize")
    def test_optimize_prompt_calculates_improvement(
        self, mock_gepa_optimize: MagicMock
    ) -> None:
        """Test that improvement is calculated correctly."""
        from gepa import GEPAResult

        mock_result = Mock(spec=GEPAResult)
        mock_result.best_candidate = {"prompt": "Better prompt"}
        mock_result.val_aggregate_scores = [0.5, 0.75]  # 50% improvement
        mock_result.best_idx = 1
        mock_result.total_metric_calls = 10
        mock_gepa_optimize.return_value = mock_result

        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = _create_test_dataset()

        result = optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            num_candidates=10,
            register_if_improved=False,
        )

        # Improvement = (0.75 - 0.5) / 0.5 = 0.5 (50%)
        assert result.improvement == pytest.approx(0.5, abs=0.01)
        assert result.improved is True

    @pytest.mark.unit
    @patch("dao_ai.optimization.optimize")
    def test_optimize_prompt_no_improvement(
        self, mock_gepa_optimize: MagicMock
    ) -> None:
        """Test result when there's no improvement."""
        from gepa import GEPAResult

        mock_result = Mock(spec=GEPAResult)
        mock_result.best_candidate = {"prompt": "Same prompt"}
        mock_result.val_aggregate_scores = [0.8, 0.7, 0.6]  # Score decreased
        mock_result.best_idx = 0  # Original is best
        mock_result.total_metric_calls = 10
        mock_gepa_optimize.return_value = mock_result

        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = _create_test_dataset()

        result = optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            num_candidates=10,
            register_if_improved=False,
        )

        assert result.improvement == 0.0
        assert result.improved is False
        assert result.original_score == result.optimized_score

    @pytest.mark.unit
    @patch("dao_ai.optimization.optimize")
    def test_optimize_prompt_uses_custom_reflection_model(
        self, mock_gepa_optimize: MagicMock
    ) -> None:
        """Test that custom reflection model is passed to GEPA."""
        from gepa import GEPAResult

        mock_result = Mock(spec=GEPAResult)
        mock_result.best_candidate = {"prompt": "Optimized"}
        mock_result.val_aggregate_scores = [0.5, 0.6]
        mock_result.best_idx = 1
        mock_result.total_metric_calls = 10
        mock_gepa_optimize.return_value = mock_result

        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = _create_test_dataset()

        optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            reflection_model="databricks-meta-llama-3-3-70b-instruct",
            num_candidates=10,
            register_if_improved=False,
        )

        # Verify reflection_lm was passed correctly
        call_kwargs = mock_gepa_optimize.call_args.kwargs
        assert call_kwargs["reflection_lm"] == "databricks-meta-llama-3-3-70b-instruct"

    @pytest.mark.unit
    @patch("dao_ai.optimization.optimize")
    def test_optimize_prompt_defaults_to_agent_model(
        self, mock_gepa_optimize: MagicMock
    ) -> None:
        """Test that reflection model defaults to agent's model."""
        from gepa import GEPAResult

        mock_result = Mock(spec=GEPAResult)
        mock_result.best_candidate = {"prompt": "Optimized"}
        mock_result.val_aggregate_scores = [0.5]
        mock_result.best_idx = 0
        mock_result.total_metric_calls = 10
        mock_gepa_optimize.return_value = mock_result

        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = _create_test_dataset()

        optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            num_candidates=10,
            register_if_improved=False,
        )

        # Verify it uses agent's model URI
        call_kwargs = mock_gepa_optimize.call_args.kwargs
        assert call_kwargs["reflection_lm"] == agent.model.uri

    @pytest.mark.unit
    @patch("dao_ai.optimization.optimize")
    def test_optimize_prompt_handles_error(self, mock_gepa_optimize: MagicMock) -> None:
        """Test error handling when GEPA fails."""
        mock_gepa_optimize.side_effect = RuntimeError("GEPA failed")

        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = _create_test_dataset()

        result = optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            num_candidates=10,
            register_if_improved=False,
        )

        assert result.original_score == 0.0
        assert result.optimized_score == 0.0
        assert result.improvement == 0.0
        assert "error" in result.metadata
        assert "GEPA failed" in result.metadata["error"]

    @pytest.mark.unit
    def test_optimize_prompt_empty_dataset_error(self) -> None:
        """Test that empty dataset raises ValueError."""
        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = EvaluationDatasetModel(name="empty", data=[])

        with pytest.raises(ValueError, match="Dataset is empty"):
            optimize_prompt(
                prompt=prompt,
                agent=agent,
                dataset=dataset,
                num_candidates=10,
            )

    @pytest.mark.unit
    def test_optimize_prompt_no_template_error(self) -> None:
        """Test that prompt without template raises ValueError."""
        prompt = PromptModel(name="no_template")  # No default_template
        agent = _create_test_agent()
        agent.prompt = prompt
        dataset = _create_test_dataset()

        # Either "has no template" or "not found in registry" error is acceptable
        with pytest.raises(
            ValueError,
            match="(has no template|not found in registry|no default_template)",
        ):
            optimize_prompt(
                prompt=prompt,
                agent=agent,
                dataset=dataset,
                num_candidates=10,
            )


# --- Tests for Prompt Registry Integration ---


class TestPromptRegistration:
    """Tests for prompt registration in MLflow."""

    @pytest.mark.unit
    @patch("dao_ai.optimization.mlflow")
    @patch("dao_ai.optimization.optimize")
    def test_registers_prompt_when_improved(
        self,
        mock_gepa_optimize: MagicMock,
        mock_mlflow: MagicMock,
    ) -> None:
        """Test that optimized prompt is registered when improved."""
        from gepa import GEPAResult
        from mlflow.entities.model_registry import PromptVersion

        # Setup GEPA mock
        mock_result = Mock(spec=GEPAResult)
        mock_result.best_candidate = {"prompt": "Improved template"}
        mock_result.val_aggregate_scores = [0.5, 0.9]
        mock_result.best_idx = 1
        mock_result.total_metric_calls = 50
        mock_gepa_optimize.return_value = mock_result

        # Setup MLflow mock
        mock_version = Mock(spec=PromptVersion)
        mock_version.version = 2
        mock_mlflow.genai.register_prompt.return_value = mock_version

        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = _create_test_dataset()

        optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            num_candidates=50,
            register_if_improved=True,
        )

        # Verify registration was called
        mock_mlflow.genai.register_prompt.assert_called_once()
        call_kwargs = mock_mlflow.genai.register_prompt.call_args.kwargs

        # Verify tags are comprehensive
        tags = call_kwargs["tags"]
        assert "dao_ai_version" in tags
        assert "optimizer" in tags
        assert tags["optimizer"] == "gepa"
        assert "target_model" in tags
        assert "target_agent" in tags
        assert "original_score" in tags
        assert "optimized_score" in tags
        assert "improvement" in tags
        assert "improvement_percent" in tags
        assert "num_evaluations" in tags
        assert "train_size" in tags
        assert "val_size" in tags
        assert "optimization_timestamp" in tags

        # Verify aliases are set
        assert mock_mlflow.genai.set_prompt_alias.call_count == 2  # latest and champion

    @pytest.mark.unit
    @patch("dao_ai.optimization.mlflow")
    @patch("dao_ai.optimization.optimize")
    def test_skips_registration_when_no_improvement(
        self,
        mock_gepa_optimize: MagicMock,
        mock_mlflow: MagicMock,
    ) -> None:
        """Test that prompt is not registered when there's no improvement."""
        from gepa import GEPAResult

        mock_result = Mock(spec=GEPAResult)
        mock_result.best_candidate = {"prompt": "Same template"}
        mock_result.val_aggregate_scores = [0.8, 0.7]  # No improvement
        mock_result.best_idx = 0
        mock_result.total_metric_calls = 10
        mock_gepa_optimize.return_value = mock_result

        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = _create_test_dataset()

        optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            num_candidates=10,
            register_if_improved=True,
        )

        # Verify registration was NOT called
        mock_mlflow.genai.register_prompt.assert_not_called()

    @pytest.mark.unit
    @patch("dao_ai.optimization.mlflow")
    @patch("dao_ai.optimization.optimize")
    def test_skips_registration_when_disabled(
        self,
        mock_gepa_optimize: MagicMock,
        mock_mlflow: MagicMock,
    ) -> None:
        """Test that registration is skipped when register_if_improved=False."""
        from gepa import GEPAResult

        mock_result = Mock(spec=GEPAResult)
        mock_result.best_candidate = {"prompt": "Improved template"}
        mock_result.val_aggregate_scores = [0.5, 0.9]
        mock_result.best_idx = 1
        mock_result.total_metric_calls = 50
        mock_gepa_optimize.return_value = mock_result

        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = _create_test_dataset()

        optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            num_candidates=50,
            register_if_improved=False,
        )

        # Verify registration was NOT called
        mock_mlflow.genai.register_prompt.assert_not_called()

    @pytest.mark.unit
    @patch("dao_ai.optimization.mlflow")
    @patch("dao_ai.optimization.optimize")
    def test_respects_min_improvement_threshold(
        self,
        mock_gepa_optimize: MagicMock,
        mock_mlflow: MagicMock,
    ) -> None:
        """Test that min_improvement threshold is respected."""
        from gepa import GEPAResult

        mock_result = Mock(spec=GEPAResult)
        mock_result.best_candidate = {"prompt": "Slightly better"}
        mock_result.val_aggregate_scores = [0.5, 0.52]  # 4% improvement
        mock_result.best_idx = 1
        mock_result.total_metric_calls = 10
        mock_gepa_optimize.return_value = mock_result

        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = _create_test_dataset()

        # Require at least 10% improvement
        optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            num_candidates=10,
            register_if_improved=True,
            min_improvement=0.1,  # 10%
        )

        # Verify registration was NOT called (4% < 10%)
        mock_mlflow.genai.register_prompt.assert_not_called()


# --- Tests for Metadata in Optimized Prompt ---


class TestOptimizedPromptMetadata:
    """Tests for metadata in the optimized prompt."""

    @pytest.mark.unit
    @patch("dao_ai.optimization.optimize")
    def test_optimized_prompt_has_tags(self, mock_gepa_optimize: MagicMock) -> None:
        """Test that optimized prompt includes comprehensive tags."""
        from gepa import GEPAResult

        mock_result = Mock(spec=GEPAResult)
        mock_result.best_candidate = {"prompt": "Optimized"}
        mock_result.val_aggregate_scores = [0.5, 0.8]
        mock_result.best_idx = 1
        mock_result.total_metric_calls = 50
        mock_gepa_optimize.return_value = mock_result

        prompt = PromptModel(
            name="test_prompt",
            default_template="Original: {question}",
            tags={"env": "prod", "version": "1.0"},
        )
        agent = _create_test_agent()
        agent.prompt = prompt
        dataset = _create_test_dataset()

        result = optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            num_candidates=50,
            register_if_improved=False,
        )

        tags = result.optimized_prompt.tags
        assert tags is not None

        # Original tags preserved
        assert tags.get("env") == "prod"
        assert tags.get("version") == "1.0"

        # Optimization tags added
        assert "dao_ai_version" in tags
        assert tags["optimizer"] == "gepa"
        assert "target_model" in tags
        assert tags["target_agent"] == agent.name
        assert "original_score" in tags
        assert "optimized_score" in tags
        assert "improvement" in tags
        assert "num_evaluations" in tags

    @pytest.mark.unit
    @patch("dao_ai.optimization.optimize")
    def test_optimized_prompt_has_correct_alias(
        self, mock_gepa_optimize: MagicMock
    ) -> None:
        """Test that optimized prompt has correct alias based on improvement."""
        from gepa import GEPAResult

        # Test with improvement
        mock_result = Mock(spec=GEPAResult)
        mock_result.best_candidate = {"prompt": "Better"}
        mock_result.val_aggregate_scores = [0.5, 0.8]
        mock_result.best_idx = 1
        mock_result.total_metric_calls = 10
        mock_gepa_optimize.return_value = mock_result

        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = _create_test_dataset()

        result = optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            register_if_improved=False,
        )

        assert result.optimized_prompt.alias == "champion"

        # Test without improvement
        mock_result.val_aggregate_scores = [0.8, 0.6]
        mock_result.best_idx = 0
        mock_gepa_optimize.return_value = mock_result

        result = optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            register_if_improved=False,
        )

        assert result.optimized_prompt.alias == "latest"

    @pytest.mark.unit
    @patch("dao_ai.optimization.optimize")
    def test_optimized_prompt_has_description(
        self, mock_gepa_optimize: MagicMock
    ) -> None:
        """Test that optimized prompt has descriptive description."""
        from gepa import GEPAResult

        mock_result = Mock(spec=GEPAResult)
        mock_result.best_candidate = {"prompt": "Optimized"}
        mock_result.val_aggregate_scores = [0.5, 0.8]
        mock_result.best_idx = 1
        mock_result.total_metric_calls = 10
        mock_gepa_optimize.return_value = mock_result

        prompt = _create_test_prompt()
        agent = _create_test_agent()
        dataset = _create_test_dataset()

        result = optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=dataset,
            register_if_improved=False,
        )

        assert "GEPA" in result.optimized_prompt.description
        assert agent.name in result.optimized_prompt.description
        assert "improvement" in result.optimized_prompt.description.lower()
