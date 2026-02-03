"""Unit tests for context-aware cache threshold optimization."""

import math
from unittest.mock import Mock, patch

import pytest

from dao_ai.genie.cache.optimization import (
    ContextAwareCacheEvalDataset,
    ContextAwareCacheEvalEntry,
    ThresholdOptimizationResult,
    _compute_l2_similarity,
    _evaluate_thresholds,
    clear_judge_cache,
    optimize_context_aware_cache_thresholds,
    semantic_match_judge,
)


class TestL2Similarity:
    """Test L2 distance to similarity conversion."""

    def test_identical_vectors_return_one(self) -> None:
        """Identical vectors should have similarity of 1.0."""
        embedding = [1.0, 0.0, 0.0]
        similarity = _compute_l2_similarity(embedding, embedding)
        assert similarity == 1.0

    def test_orthogonal_vectors_have_low_similarity(self) -> None:
        """Orthogonal unit vectors should have low similarity."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        similarity = _compute_l2_similarity(embedding1, embedding2)
        # L2 distance = sqrt(2), similarity = 1 / (1 + sqrt(2)) ≈ 0.414
        expected = 1.0 / (1.0 + math.sqrt(2))
        assert abs(similarity - expected) < 0.001

    def test_opposite_vectors_have_low_similarity(self) -> None:
        """Opposite unit vectors should have even lower similarity."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [-1.0, 0.0, 0.0]
        similarity = _compute_l2_similarity(embedding1, embedding2)
        # L2 distance = 2, similarity = 1 / (1 + 2) = 0.333
        expected = 1.0 / 3.0
        assert abs(similarity - expected) < 0.001

    def test_similar_vectors_have_high_similarity(self) -> None:
        """Similar vectors should have high similarity."""
        embedding1 = [0.9, 0.9, 0.9]
        embedding2 = [1.0, 1.0, 1.0]
        similarity = _compute_l2_similarity(embedding1, embedding2)
        # L2 distance = sqrt(0.01 + 0.01 + 0.01) = sqrt(0.03) ≈ 0.173
        # similarity = 1 / (1 + 0.173) ≈ 0.852
        assert similarity > 0.8

    def test_mismatched_dimensions_raises_error(self) -> None:
        """Vectors with different dimensions should raise ValueError."""
        embedding1 = [1.0, 0.0]
        embedding2 = [1.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="Embedding dimensions must match"):
            _compute_l2_similarity(embedding1, embedding2)

    def test_similarity_is_symmetric(self) -> None:
        """Similarity should be the same regardless of argument order."""
        embedding1 = [0.5, 0.3, 0.8]
        embedding2 = [0.7, 0.2, 0.6]
        sim1 = _compute_l2_similarity(embedding1, embedding2)
        sim2 = _compute_l2_similarity(embedding2, embedding1)
        assert sim1 == sim2


class TestSemanticMatchJudge:
    """Test LLM-as-Judge for semantic matching."""

    def test_judge_caches_results(self) -> None:
        """Judge should cache results to avoid redundant LLM calls."""
        clear_judge_cache()

        mock_chat = Mock()
        mock_response = Mock()
        mock_response.content = "MATCH"
        mock_chat.invoke.return_value = mock_response

        mock_model = Mock()
        mock_model.as_chat_model.return_value = mock_chat

        with patch("dao_ai.genie.cache.optimization.LLMModel", return_value=mock_model):
            # First call
            result1 = semantic_match_judge(
                "What are sales?",
                "Previous context",
                "Show me sales",
                "Previous context",
                mock_model,
                use_cache=True,
            )
            # Second call with same inputs
            result2 = semantic_match_judge(
                "What are sales?",
                "Previous context",
                "Show me sales",
                "Previous context",
                mock_model,
                use_cache=True,
            )

            # Should only call LLM once (cached)
            assert mock_chat.invoke.call_count == 1
            assert result1 is True
            assert result2 is True

        clear_judge_cache()

    def test_judge_returns_match(self) -> None:
        """Judge should return True for MATCH response."""
        clear_judge_cache()

        mock_chat = Mock()
        mock_response = Mock()
        mock_response.content = "MATCH"
        mock_chat.invoke.return_value = mock_response

        mock_model = Mock()
        mock_model.as_chat_model.return_value = mock_chat

        with patch("dao_ai.genie.cache.optimization.LLMModel", return_value=mock_model):
            result = semantic_match_judge(
                "Question 1", "Context 1", "Question 2", "Context 2", mock_model
            )
            assert result is True

        clear_judge_cache()

    def test_judge_returns_no_match(self) -> None:
        """Judge should return False for NO_MATCH response."""
        clear_judge_cache()

        mock_chat = Mock()
        mock_response = Mock()
        mock_response.content = "NO_MATCH"
        mock_chat.invoke.return_value = mock_response

        mock_model = Mock()
        mock_model.as_chat_model.return_value = mock_chat

        with patch("dao_ai.genie.cache.optimization.LLMModel", return_value=mock_model):
            result = semantic_match_judge(
                "Question 1", "Context 1", "Question 2", "Context 2", mock_model
            )
            assert result is False

        clear_judge_cache()

    def test_judge_handles_error_gracefully(self) -> None:
        """Judge should return False on error (conservative)."""
        clear_judge_cache()

        mock_chat = Mock()
        mock_chat.invoke.side_effect = Exception("API Error")

        mock_model = Mock()
        mock_model.as_chat_model.return_value = mock_chat

        with patch("dao_ai.genie.cache.optimization.LLMModel", return_value=mock_model):
            result = semantic_match_judge(
                "Question 1", "Context 1", "Question 2", "Context 2", mock_model
            )
            assert result is False

        clear_judge_cache()


class TestEvaluateThresholds:
    """Test threshold evaluation logic."""

    @pytest.fixture
    def sample_dataset(self) -> ContextAwareCacheEvalDataset:
        """Create a sample evaluation dataset with known similarities."""
        # Create entries with known embedding similarities
        # Entry 1: Very similar (should match with low thresholds)
        similar_entry = ContextAwareCacheEvalEntry(
            question="What are sales?",
            question_embedding=[1.0, 0.0, 0.0],
            context="Previous context",
            context_embedding=[0.0, 1.0, 0.0],
            cached_question="Show me sales",
            cached_question_embedding=[0.99, 0.05, 0.0],  # Very similar
            cached_context="Previous context",
            cached_context_embedding=[0.0, 0.99, 0.05],  # Very similar
            expected_match=True,
        )

        # Entry 2: Different (should not match)
        different_entry = ContextAwareCacheEvalEntry(
            question="What is revenue?",
            question_embedding=[1.0, 0.0, 0.0],
            context="Revenue context",
            context_embedding=[0.0, 1.0, 0.0],
            cached_question="Show me inventory",
            cached_question_embedding=[0.0, 0.0, 1.0],  # Very different
            cached_context="Inventory context",
            cached_context_embedding=[0.0, 0.0, 1.0],  # Very different
            expected_match=False,
        )

        return ContextAwareCacheEvalDataset(
            name="test_dataset",
            entries=[similar_entry, different_entry],
            description="Test dataset",
        )

    def test_high_threshold_rejects_similar_entries(
        self, sample_dataset: ContextAwareCacheEvalDataset
    ) -> None:
        """High thresholds should reject even similar entries."""
        precision, recall, f1, confusion = _evaluate_thresholds(
            dataset=sample_dataset,
            similarity_threshold=0.99,  # Very high
            context_similarity_threshold=0.99,
            question_weight=0.5,
            judge_model=None,
        )

        # With very high thresholds, nothing should match
        assert confusion["true_positives"] == 0
        assert confusion["false_negatives"] == 1  # Should have matched but didn't

    def test_low_threshold_accepts_similar_entries(
        self, sample_dataset: ContextAwareCacheEvalDataset
    ) -> None:
        """Lower thresholds should accept similar entries."""
        precision, recall, f1, confusion = _evaluate_thresholds(
            dataset=sample_dataset,
            similarity_threshold=0.5,  # Low
            context_similarity_threshold=0.5,
            question_weight=0.5,
            judge_model=None,
        )

        # With low thresholds, similar entry should match
        assert confusion["true_positives"] >= 1

    def test_metrics_calculation(
        self, sample_dataset: ContextAwareCacheEvalDataset
    ) -> None:
        """Test that metrics are calculated correctly."""
        precision, recall, f1, confusion = _evaluate_thresholds(
            dataset=sample_dataset,
            similarity_threshold=0.8,
            context_similarity_threshold=0.8,
            question_weight=0.6,
            judge_model=None,
        )

        # Verify total entries
        assert confusion["total"] == 2

        # Verify metrics are in valid range
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0


class TestOptimizeContextAwareCacheThresholds:
    """Test the main optimization function."""

    @pytest.fixture
    def simple_dataset(self) -> ContextAwareCacheEvalDataset:
        """Create a simple labeled dataset for testing."""
        entries = [
            # Positive pair (semantically similar)
            ContextAwareCacheEvalEntry(
                question="What are total sales?",
                question_embedding=[1.0, 0.0, 0.0],
                context="",
                context_embedding=[0.0, 0.0, 0.0],
                cached_question="Show me total sales",
                cached_question_embedding=[0.98, 0.1, 0.0],
                cached_context="",
                cached_context_embedding=[0.0, 0.0, 0.0],
                expected_match=True,
            ),
            # Negative pair (semantically different)
            ContextAwareCacheEvalEntry(
                question="What is inventory count?",
                question_embedding=[0.0, 1.0, 0.0],
                context="",
                context_embedding=[0.0, 0.0, 0.0],
                cached_question="Show me revenue",
                cached_question_embedding=[0.0, 0.0, 1.0],
                cached_context="",
                cached_context_embedding=[0.0, 0.0, 0.0],
                expected_match=False,
            ),
        ]
        return ContextAwareCacheEvalDataset(
            name="simple_test",
            entries=entries,
            description="Simple test dataset",
        )

    def test_optimization_runs_successfully(
        self, simple_dataset: ContextAwareCacheEvalDataset
    ) -> None:
        """Test that optimization completes without error."""
        result = optimize_context_aware_cache_thresholds(
            dataset=simple_dataset,
            judge_model=None,  # Use labels only
            n_trials=5,  # Small number for fast test
            metric="f1",
            register_if_improved=False,  # Don't log to MLflow
            show_progress_bar=False,
        )

        assert isinstance(result, ThresholdOptimizationResult)
        assert "similarity_threshold" in result.optimized_thresholds
        assert "context_similarity_threshold" in result.optimized_thresholds
        assert "question_weight" in result.optimized_thresholds
        assert result.n_trials == 5

    def test_optimization_respects_original_thresholds(
        self, simple_dataset: ContextAwareCacheEvalDataset
    ) -> None:
        """Test that original thresholds are correctly stored."""
        original = {
            "similarity_threshold": 0.9,
            "context_similarity_threshold": 0.85,
            "question_weight": 0.7,
        }

        result = optimize_context_aware_cache_thresholds(
            dataset=simple_dataset,
            original_thresholds=original,
            judge_model=None,
            n_trials=3,
            register_if_improved=False,
            show_progress_bar=False,
        )

        assert result.original_thresholds == original

    def test_thresholds_are_in_valid_range(
        self, simple_dataset: ContextAwareCacheEvalDataset
    ) -> None:
        """Test that optimized thresholds are within valid bounds."""
        result = optimize_context_aware_cache_thresholds(
            dataset=simple_dataset,
            judge_model=None,
            n_trials=5,
            register_if_improved=False,
            show_progress_bar=False,
        )

        thresholds = result.optimized_thresholds
        assert 0.5 <= thresholds["similarity_threshold"] <= 0.99
        assert 0.5 <= thresholds["context_similarity_threshold"] <= 0.99
        assert 0.1 <= thresholds["question_weight"] <= 0.9


class TestThresholdOptimizationResult:
    """Test the ThresholdOptimizationResult dataclass."""

    def test_improved_property_true(self) -> None:
        """Test improved property returns True when score increased."""
        result = ThresholdOptimizationResult(
            optimized_thresholds={"similarity_threshold": 0.8},
            original_thresholds={"similarity_threshold": 0.85},
            original_score=0.7,
            optimized_score=0.85,
            improvement=0.21,
            n_trials=50,
            best_trial_number=25,
            study_name="test_study",
        )
        assert result.improved is True

    def test_improved_property_false(self) -> None:
        """Test improved property returns False when score decreased."""
        result = ThresholdOptimizationResult(
            optimized_thresholds={"similarity_threshold": 0.8},
            original_thresholds={"similarity_threshold": 0.85},
            original_score=0.8,
            optimized_score=0.75,
            improvement=-0.0625,
            n_trials=50,
            best_trial_number=1,
            study_name="test_study",
        )
        assert result.improved is False


class TestContextAwareCacheEvalDataset:
    """Test the ContextAwareCacheEvalDataset dataclass."""

    def test_len_returns_entry_count(self) -> None:
        """Test __len__ returns correct entry count."""
        entries = [
            ContextAwareCacheEvalEntry(
                question="Q1",
                question_embedding=[1.0],
                context="",
                context_embedding=[],
                cached_question="CQ1",
                cached_question_embedding=[1.0],
                cached_context="",
                cached_context_embedding=[],
            ),
            ContextAwareCacheEvalEntry(
                question="Q2",
                question_embedding=[1.0],
                context="",
                context_embedding=[],
                cached_question="CQ2",
                cached_question_embedding=[1.0],
                cached_context="",
                cached_context_embedding=[],
            ),
        ]
        dataset = ContextAwareCacheEvalDataset(name="test", entries=entries)
        assert len(dataset) == 2

    def test_iteration(self) -> None:
        """Test that dataset is iterable."""
        entries = [
            ContextAwareCacheEvalEntry(
                question="Q1",
                question_embedding=[1.0],
                context="",
                context_embedding=[],
                cached_question="CQ1",
                cached_question_embedding=[1.0],
                cached_context="",
                cached_context_embedding=[],
            ),
        ]
        dataset = ContextAwareCacheEvalDataset(name="test", entries=entries)
        collected = list(dataset)
        assert len(collected) == 1
        assert collected[0].question == "Q1"
