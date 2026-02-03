"""
Semantic cache threshold optimization using Optuna Bayesian optimization.

This module provides optimization for Genie semantic cache thresholds using
Optuna's Tree-structured Parzen Estimator (TPE) algorithm with LLM-as-Judge
evaluation for semantic match validation.

The optimizer tunes these thresholds:
- similarity_threshold: Minimum similarity for question matching
- context_similarity_threshold: Minimum similarity for context matching
- question_weight: Weight for question similarity in combined score

Usage:
    from dao_ai.genie.cache.optimization import optimize_context_aware_cache_thresholds

    result = optimize_context_aware_cache_thresholds(
        dataset=my_eval_dataset,
        judge_model="databricks-meta-llama-3-3-70b-instruct",
        n_trials=50,
        metric="f1",
    )

    if result.improved:
        print(f"Improved by {result.improvement:.1%}")
        print(f"Best thresholds: {result.optimized_thresholds}")
"""

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Sequence

import mlflow
import optuna
from loguru import logger
from optuna.samplers import TPESampler

# Optional MLflow integration - requires optuna-integration[mlflow]
try:
    from optuna.integration import MLflowCallback

    MLFLOW_CALLBACK_AVAILABLE = True
except ModuleNotFoundError:
    MLFLOW_CALLBACK_AVAILABLE = False
    MLflowCallback = None  # type: ignore

from dao_ai.config import GenieContextAwareCacheParametersModel, LLMModel
from dao_ai.utils import dao_ai_version

__all__ = [
    "ContextAwareCacheEvalEntry",
    "ContextAwareCacheEvalDataset",
    "ThresholdOptimizationResult",
    "optimize_context_aware_cache_thresholds",
    "generate_eval_dataset_from_cache",
    "semantic_match_judge",
]


@dataclass
class ContextAwareCacheEvalEntry:
    """Single evaluation entry for threshold optimization.

    Represents a pair of question/context combinations to evaluate
    whether the cache should return a hit or miss.

    Attributes:
        question: Current question being asked
        question_embedding: Pre-computed embedding of the question
        context: Conversation context string
        context_embedding: Pre-computed embedding of the context
        cached_question: Question from the cache entry
        cached_question_embedding: Embedding of the cached question
        cached_context: Context from the cache entry
        cached_context_embedding: Embedding of the cached context
        expected_match: Whether this pair should be a cache hit (True),
            miss (False), or use LLM to determine (None)
    """

    question: str
    question_embedding: list[float]
    context: str
    context_embedding: list[float]
    cached_question: str
    cached_question_embedding: list[float]
    cached_context: str
    cached_context_embedding: list[float]
    expected_match: bool | None = None


@dataclass
class ContextAwareCacheEvalDataset:
    """Dataset for semantic cache threshold optimization.

    Attributes:
        name: Name of the dataset for tracking
        entries: List of evaluation entries
        description: Optional description of the dataset
    """

    name: str
    entries: list[ContextAwareCacheEvalEntry]
    description: str = ""

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


@dataclass
class ThresholdOptimizationResult:
    """Result of semantic cache threshold optimization.

    Attributes:
        optimized_thresholds: Dictionary of optimized threshold values
        original_thresholds: Dictionary of original threshold values
        original_score: Score with original thresholds
        optimized_score: Score with optimized thresholds
        improvement: Percentage improvement (0.0-1.0)
        n_trials: Number of optimization trials run
        best_trial_number: Trial number that produced best result
        study_name: Name of the Optuna study
        metadata: Additional optimization metadata
    """

    optimized_thresholds: dict[str, float]
    original_thresholds: dict[str, float]
    original_score: float
    optimized_score: float
    improvement: float
    n_trials: int
    best_trial_number: int
    study_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        """Whether the optimization improved the thresholds."""
        return self.optimized_score > self.original_score


# Cache for LLM judge results to avoid redundant calls
_judge_cache: dict[str, bool] = {}


def _compute_cache_key(
    question1: str, context1: str, question2: str, context2: str
) -> str:
    """Compute a cache key for judge results."""
    content = f"{question1}|{context1}|{question2}|{context2}"
    return hashlib.sha256(content.encode()).hexdigest()


def semantic_match_judge(
    question1: str,
    context1: str,
    question2: str,
    context2: str,
    model: LLMModel | str,
    use_cache: bool = True,
) -> bool:
    """
    Use LLM to determine if two question/context pairs are semantically equivalent.

    This function acts as a judge to determine whether two questions with their
    respective conversation contexts are asking for the same information and
    would expect the same SQL query response.

    Args:
        question1: First question
        context1: Conversation context for first question
        question2: Second question
        context2: Conversation context for second question
        model: LLM model to use for judging
        use_cache: Whether to cache results (default True)

    Returns:
        True if the pairs are semantically equivalent, False otherwise
    """
    global _judge_cache

    # Check cache first
    if use_cache:
        cache_key = _compute_cache_key(question1, context1, question2, context2)
        if cache_key in _judge_cache:
            return _judge_cache[cache_key]

    # Convert model to LLMModel if string
    llm_model: LLMModel = LLMModel(name=model) if isinstance(model, str) else model

    # Create the chat model
    chat = llm_model.as_chat_model()

    # Construct the prompt for semantic equivalence judgment
    prompt = f"""You are an expert at determining semantic equivalence between database queries.

Given two question-context pairs, determine if they are semantically equivalent - meaning they are asking for the same information and would expect the same SQL query result.

Consider:
1. Are both questions asking for the same data/information?
2. Do the conversation contexts provide similar filtering or constraints?
3. Would answering both require the same SQL query?

IMPORTANT: Be strict. Only return "MATCH" if the questions are truly asking for the same thing in the same context. Similar but different questions should return "NO_MATCH".

Question 1: {question1}
Context 1: {context1 if context1 else "(no context)"}

Question 2: {question2}
Context 2: {context2 if context2 else "(no context)"}

Respond with ONLY one word: "MATCH" or "NO_MATCH"
"""

    try:
        response = chat.invoke(prompt)
        result_text = response.content.strip().upper()
        is_match = "MATCH" in result_text and "NO_MATCH" not in result_text

        # Cache the result
        if use_cache:
            _judge_cache[cache_key] = is_match

        logger.trace(
            "LLM judge result",
            question1=question1[:50],
            question2=question2[:50],
            is_match=is_match,
        )

        return is_match

    except Exception as e:
        logger.warning("LLM judge failed", error=str(e))
        # Default to not matching on error (conservative)
        return False


def _compute_l2_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """
    Compute similarity from L2 (Euclidean) distance.

    Uses the same formula as the semantic cache:
    similarity = 1.0 / (1.0 + L2_distance)

    This gives a value in range [0, 1] where 1 means identical.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Similarity score in range [0, 1]
    """
    if len(embedding1) != len(embedding2):
        raise ValueError(
            f"Embedding dimensions must match: {len(embedding1)} vs {len(embedding2)}"
        )

    # Compute L2 distance
    squared_diff_sum = sum(
        (a - b) ** 2 for a, b in zip(embedding1, embedding2, strict=True)
    )
    l2_distance = math.sqrt(squared_diff_sum)

    # Convert to similarity
    similarity = 1.0 / (1.0 + l2_distance)
    return similarity


def _evaluate_thresholds(
    dataset: ContextAwareCacheEvalDataset,
    similarity_threshold: float,
    context_similarity_threshold: float,
    question_weight: float,
    judge_model: LLMModel | str | None = None,
) -> tuple[float, float, float, dict[str, int]]:
    """
    Evaluate a set of thresholds against the dataset.

    Args:
        dataset: Evaluation dataset
        similarity_threshold: Threshold for question similarity
        context_similarity_threshold: Threshold for context similarity
        question_weight: Weight for question in combined score
        judge_model: Optional LLM model for judging unlabeled entries

    Returns:
        Tuple of (precision, recall, f1, confusion_matrix_dict)
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Note: context_weight = 1.0 - question_weight could be used for weighted scoring
    # but we currently use independent thresholds for question and context similarity

    for entry in dataset.entries:
        # Compute similarities
        question_sim = _compute_l2_similarity(
            entry.question_embedding, entry.cached_question_embedding
        )
        context_sim = _compute_l2_similarity(
            entry.context_embedding, entry.cached_context_embedding
        )

        # Apply threshold logic (same as production cache)
        predicted_match = (
            question_sim >= similarity_threshold
            and context_sim >= context_similarity_threshold
        )

        # Get expected match
        expected_match = entry.expected_match
        if expected_match is None:
            if judge_model is None:
                # Skip entries without labels if no judge provided
                continue
            # Use LLM judge to determine expected match
            expected_match = semantic_match_judge(
                entry.question,
                entry.context,
                entry.cached_question,
                entry.cached_context,
                judge_model,
            )

        # Update confusion matrix
        if predicted_match and expected_match:
            true_positives += 1
        elif predicted_match and not expected_match:
            false_positives += 1
        elif not predicted_match and expected_match:
            false_negatives += 1
        else:
            true_negatives += 1

    # Calculate metrics
    total = true_positives + false_positives + true_negatives + false_negatives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    confusion = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "total": total,
    }

    return precision, recall, f1, confusion


def _create_objective(
    dataset: ContextAwareCacheEvalDataset,
    judge_model: LLMModel | str | None,
    metric: Literal["f1", "precision", "recall", "fbeta"],
    beta: float = 1.0,
) -> Callable[[optuna.Trial], float]:
    """Create the Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        # Sample parameters
        similarity_threshold = trial.suggest_float(
            "similarity_threshold", 0.5, 0.99, log=False
        )
        context_similarity_threshold = trial.suggest_float(
            "context_similarity_threshold", 0.5, 0.99, log=False
        )
        question_weight = trial.suggest_float("question_weight", 0.1, 0.9, log=False)

        # Evaluate
        precision, recall, f1, confusion = _evaluate_thresholds(
            dataset=dataset,
            similarity_threshold=similarity_threshold,
            context_similarity_threshold=context_similarity_threshold,
            question_weight=question_weight,
            judge_model=judge_model,
        )

        # Log intermediate results
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)
        trial.set_user_attr("f1", f1)
        trial.set_user_attr("confusion", confusion)

        # Return selected metric
        if metric == "f1":
            return f1
        elif metric == "precision":
            return precision
        elif metric == "recall":
            return recall
        elif metric == "fbeta":
            # F-beta score: (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
            if precision + recall == 0:
                return 0.0
            fbeta = (
                (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
            )
            return fbeta
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return objective


def optimize_context_aware_cache_thresholds(
    dataset: ContextAwareCacheEvalDataset,
    original_thresholds: dict[str, float]
    | GenieContextAwareCacheParametersModel
    | None = None,
    judge_model: LLMModel | str = "databricks-meta-llama-3-3-70b-instruct",
    n_trials: int = 50,
    metric: Literal["f1", "precision", "recall", "fbeta"] = "f1",
    beta: float = 1.0,
    register_if_improved: bool = True,
    study_name: str | None = None,
    seed: int | None = None,
    show_progress_bar: bool = True,
) -> ThresholdOptimizationResult:
    """
    Optimize semantic cache thresholds using Bayesian optimization.

    Uses Optuna's Tree-structured Parzen Estimator (TPE) to efficiently
    search the parameter space and find optimal threshold values.

    Args:
        dataset: Evaluation dataset with question/context pairs
        original_thresholds: Original thresholds to compare against.
            Can be a dict or GenieContextAwareCacheParametersModel.
            If None, uses default values.
        judge_model: LLM model for semantic match judging (for unlabeled entries)
        n_trials: Number of optimization trials to run
        metric: Optimization metric ("f1", "precision", "recall", "fbeta")
        beta: Beta value for fbeta metric (higher = favor recall)
        register_if_improved: Log results to MLflow if improved
        study_name: Optional name for the Optuna study
        seed: Random seed for reproducibility
        show_progress_bar: Whether to show progress bar during optimization

    Returns:
        ThresholdOptimizationResult with optimized thresholds and metrics

    Example:
        from dao_ai.genie.cache.optimization import (
            optimize_context_aware_cache_thresholds,
            ContextAwareCacheEvalDataset,
        )

        result = optimize_context_aware_cache_thresholds(
            dataset=my_dataset,
            judge_model="databricks-meta-llama-3-3-70b-instruct",
            n_trials=50,
            metric="f1",
        )

        if result.improved:
            print(f"New thresholds: {result.optimized_thresholds}")
    """
    logger.info(
        "Starting semantic cache threshold optimization",
        dataset_name=dataset.name,
        dataset_size=len(dataset),
        n_trials=n_trials,
        metric=metric,
    )

    # Parse original thresholds
    if original_thresholds is None:
        orig_thresholds = {
            "similarity_threshold": 0.85,
            "context_similarity_threshold": 0.80,
            "question_weight": 0.6,
        }
    elif isinstance(original_thresholds, GenieContextAwareCacheParametersModel):
        orig_thresholds = {
            "similarity_threshold": original_thresholds.similarity_threshold,
            "context_similarity_threshold": original_thresholds.context_similarity_threshold,
            "question_weight": original_thresholds.question_weight or 0.6,
        }
    else:
        orig_thresholds = original_thresholds.copy()

    # Evaluate original thresholds
    orig_precision, orig_recall, orig_f1, orig_confusion = _evaluate_thresholds(
        dataset=dataset,
        similarity_threshold=orig_thresholds["similarity_threshold"],
        context_similarity_threshold=orig_thresholds["context_similarity_threshold"],
        question_weight=orig_thresholds["question_weight"],
        judge_model=judge_model,
    )

    # Determine original score based on metric
    if metric == "f1":
        original_score = orig_f1
    elif metric == "precision":
        original_score = orig_precision
    elif metric == "recall":
        original_score = orig_recall
    elif metric == "fbeta":
        if orig_precision + orig_recall == 0:
            original_score = 0.0
        else:
            original_score = (
                (1 + beta**2)
                * (orig_precision * orig_recall)
                / (beta**2 * orig_precision + orig_recall)
            )
    else:
        raise ValueError(f"Unknown metric: {metric}")

    logger.info(
        "Evaluated original thresholds",
        precision=f"{orig_precision:.4f}",
        recall=f"{orig_recall:.4f}",
        f1=f"{orig_f1:.4f}",
        original_score=f"{original_score:.4f}",
    )

    # Create study name if not provided
    if study_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        study_name = f"context_aware_cache_threshold_optimization_{timestamp}"

    # Create Optuna study
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    # Add original thresholds as first trial for comparison
    study.enqueue_trial(orig_thresholds)

    # Create objective function
    objective = _create_objective(
        dataset=dataset,
        judge_model=judge_model,
        metric=metric,
        beta=beta,
    )

    # Set up MLflow callback if available
    callbacks = []
    if MLFLOW_CALLBACK_AVAILABLE and MLflowCallback is not None:
        try:
            mlflow_callback = MLflowCallback(
                tracking_uri=mlflow.get_tracking_uri(),
                metric_name=metric,
                create_experiment=False,
            )
            callbacks.append(mlflow_callback)
        except Exception as e:
            logger.debug("MLflow callback not available", error=str(e))

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
        callbacks=callbacks if callbacks else None,
    )

    # Get best trial
    best_trial = study.best_trial
    best_thresholds = {
        "similarity_threshold": best_trial.params["similarity_threshold"],
        "context_similarity_threshold": best_trial.params[
            "context_similarity_threshold"
        ],
        "question_weight": best_trial.params["question_weight"],
    }

    # Get metrics from best trial
    best_precision = best_trial.user_attrs.get("precision", 0.0)
    best_recall = best_trial.user_attrs.get("recall", 0.0)
    best_f1 = best_trial.user_attrs.get("f1", 0.0)
    best_confusion = best_trial.user_attrs.get("confusion", {})
    optimized_score = best_trial.value

    # Calculate improvement
    improvement = (
        (optimized_score - original_score) / original_score
        if original_score > 0
        else 0.0
    )

    logger.success(
        "Optimization complete",
        best_trial_number=best_trial.number,
        original_score=f"{original_score:.4f}",
        optimized_score=f"{optimized_score:.4f}",
        improvement=f"{improvement:.1%}",
        best_thresholds=best_thresholds,
    )

    # Log to MLflow if improved
    if register_if_improved and improvement > 0:
        try:
            _log_optimization_to_mlflow(
                study_name=study_name,
                dataset_name=dataset.name,
                dataset_size=len(dataset),
                original_thresholds=orig_thresholds,
                optimized_thresholds=best_thresholds,
                original_score=original_score,
                optimized_score=optimized_score,
                improvement=improvement,
                metric=metric,
                n_trials=n_trials,
                best_precision=best_precision,
                best_recall=best_recall,
                best_f1=best_f1,
                best_confusion=best_confusion,
                judge_model=judge_model,
            )
        except Exception as e:
            logger.warning("Failed to log optimization to MLflow", error=str(e))

    # Build result
    result = ThresholdOptimizationResult(
        optimized_thresholds=best_thresholds,
        original_thresholds=orig_thresholds,
        original_score=original_score,
        optimized_score=optimized_score,
        improvement=improvement,
        n_trials=n_trials,
        best_trial_number=best_trial.number,
        study_name=study_name,
        metadata={
            "metric": metric,
            "beta": beta if metric == "fbeta" else None,
            "judge_model": str(judge_model),
            "dataset_name": dataset.name,
            "dataset_size": len(dataset),
            "original_precision": orig_precision,
            "original_recall": orig_recall,
            "original_f1": orig_f1,
            "optimized_precision": best_precision,
            "optimized_recall": best_recall,
            "optimized_f1": best_f1,
            "confusion_matrix": best_confusion,
        },
    )

    return result


def _log_optimization_to_mlflow(
    study_name: str,
    dataset_name: str,
    dataset_size: int,
    original_thresholds: dict[str, float],
    optimized_thresholds: dict[str, float],
    original_score: float,
    optimized_score: float,
    improvement: float,
    metric: str,
    n_trials: int,
    best_precision: float,
    best_recall: float,
    best_f1: float,
    best_confusion: dict[str, int],
    judge_model: LLMModel | str,
) -> None:
    """Log optimization results to MLflow."""
    with mlflow.start_run(run_name=study_name):
        # Log parameters
        mlflow.log_params(
            {
                "optimizer": "optuna_tpe",
                "metric": metric,
                "n_trials": n_trials,
                "dataset_name": dataset_name,
                "dataset_size": dataset_size,
                "judge_model": str(judge_model),
                "dao_ai_version": dao_ai_version(),
            }
        )

        # Log original thresholds
        for key, value in original_thresholds.items():
            mlflow.log_param(f"original_{key}", value)

        # Log optimized thresholds
        for key, value in optimized_thresholds.items():
            mlflow.log_param(f"optimized_{key}", value)

        # Log metrics
        mlflow.log_metrics(
            {
                "original_score": original_score,
                "optimized_score": optimized_score,
                "improvement": improvement,
                "precision": best_precision,
                "recall": best_recall,
                "f1": best_f1,
                **{f"confusion_{k}": v for k, v in best_confusion.items()},
            }
        )

        # Log thresholds as artifact
        thresholds_artifact = {
            "study_name": study_name,
            "original": original_thresholds,
            "optimized": optimized_thresholds,
            "improvement": improvement,
            "metric": metric,
        }
        mlflow.log_dict(thresholds_artifact, "optimized_thresholds.json")

        logger.info(
            "Logged optimization results to MLflow",
            study_name=study_name,
            improvement=f"{improvement:.1%}",
        )


def generate_eval_dataset_from_cache(
    cache_entries: Sequence[dict[str, Any]],
    embedding_model: LLMModel | str = "databricks-gte-large-en",
    num_positive_pairs: int = 50,
    num_negative_pairs: int = 50,
    paraphrase_model: LLMModel | str | None = None,
    dataset_name: str = "generated_eval_dataset",
) -> ContextAwareCacheEvalDataset:
    """
    Generate an evaluation dataset from existing cache entries.

    Creates positive pairs (semantically equivalent questions) using LLM paraphrasing
    and negative pairs (different questions) from random cache entry pairs.

    Args:
        cache_entries: List of cache entries with 'question', 'conversation_context',
            'question_embedding', and 'context_embedding' keys
        embedding_model: Model for generating embeddings for paraphrased questions
        num_positive_pairs: Number of positive (matching) pairs to generate
        num_negative_pairs: Number of negative (non-matching) pairs to generate
        paraphrase_model: LLM for generating paraphrases (defaults to embedding_model)
        dataset_name: Name for the generated dataset

    Returns:
        ContextAwareCacheEvalDataset with generated entries
    """
    import random

    if len(cache_entries) < 2:
        raise ValueError("Need at least 2 cache entries to generate dataset")

    # Convert embedding model
    emb_model: LLMModel = (
        LLMModel(name=embedding_model)
        if isinstance(embedding_model, str)
        else embedding_model
    )
    embeddings = emb_model.as_embeddings_model()

    # Use paraphrase model or default to a capable LLM
    para_model: LLMModel = (
        LLMModel(name=paraphrase_model)
        if isinstance(paraphrase_model, str)
        else (
            paraphrase_model
            if paraphrase_model
            else LLMModel(name="databricks-meta-llama-3-3-70b-instruct")
        )
    )
    chat = para_model.as_chat_model()

    entries: list[ContextAwareCacheEvalEntry] = []

    # Generate positive pairs (paraphrases)
    logger.info(
        "Generating positive pairs using paraphrasing", count=num_positive_pairs
    )

    for i in range(min(num_positive_pairs, len(cache_entries))):
        entry = cache_entries[i % len(cache_entries)]
        original_question = entry.get("question", "")
        original_context = entry.get("conversation_context", "")
        original_q_emb = entry.get("question_embedding", [])
        original_c_emb = entry.get("context_embedding", [])

        if not original_question or not original_q_emb:
            continue

        # Generate paraphrase
        try:
            paraphrase_prompt = f"""Rephrase the following question to ask the same thing but using different words.
Keep the same meaning and intent. Only output the rephrased question, nothing else.

Original question: {original_question}

Rephrased question:"""
            response = chat.invoke(paraphrase_prompt)
            paraphrased_question = response.content.strip()

            # Generate embedding for paraphrase
            para_q_emb = embeddings.embed_query(paraphrased_question)
            para_c_emb = (
                embeddings.embed_query(original_context)
                if original_context
                else original_c_emb
            )

            entries.append(
                ContextAwareCacheEvalEntry(
                    question=paraphrased_question,
                    question_embedding=para_q_emb,
                    context=original_context,
                    context_embedding=para_c_emb,
                    cached_question=original_question,
                    cached_question_embedding=original_q_emb,
                    cached_context=original_context,
                    cached_context_embedding=original_c_emb,
                    expected_match=True,
                )
            )
        except Exception as e:
            logger.warning("Failed to generate paraphrase", error=str(e))

    # Generate negative pairs (random different questions)
    logger.info(
        "Generating negative pairs from different cache entries",
        count=num_negative_pairs,
    )

    for _ in range(num_negative_pairs):
        # Pick two different random entries
        if len(cache_entries) < 2:
            break
        idx1, idx2 = random.sample(range(len(cache_entries)), 2)
        entry1 = cache_entries[idx1]
        entry2 = cache_entries[idx2]

        # Use entry1 as the "question" and entry2 as the "cached" entry
        entries.append(
            ContextAwareCacheEvalEntry(
                question=entry1.get("question", ""),
                question_embedding=entry1.get("question_embedding", []),
                context=entry1.get("conversation_context", ""),
                context_embedding=entry1.get("context_embedding", []),
                cached_question=entry2.get("question", ""),
                cached_question_embedding=entry2.get("question_embedding", []),
                cached_context=entry2.get("conversation_context", ""),
                cached_context_embedding=entry2.get("context_embedding", []),
                expected_match=False,
            )
        )

    logger.info(
        "Generated evaluation dataset",
        name=dataset_name,
        total_entries=len(entries),
        positive_pairs=sum(1 for e in entries if e.expected_match is True),
        negative_pairs=sum(1 for e in entries if e.expected_match is False),
    )

    return ContextAwareCacheEvalDataset(
        name=dataset_name,
        entries=entries,
        description=f"Generated from {len(cache_entries)} cache entries",
    )


def clear_judge_cache() -> None:
    """Clear the LLM judge result cache."""
    global _judge_cache
    _judge_cache.clear()
    logger.debug("Cleared judge cache")
