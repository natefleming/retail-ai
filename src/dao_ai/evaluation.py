"""
DAO AI Evaluation Module

Provides reusable utilities for MLflow GenAI evaluation using built-in
MLflow 3.10+ judges and scorers, and production monitoring registration
via the MLflow 3 scorer lifecycle API.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import create_dataset, delete_dataset, get_dataset
from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset
from mlflow.genai.scorers import (
    Completeness,
    Guidelines,
    RelevanceToQuery,
    Safety,
    Scorer,
    ScorerSamplingConfig,
    ToolCallEfficiency,
    delete_scorer,
    list_scorers,
)
from mlflow.models.evaluation.base import EvaluationResult

if TYPE_CHECKING:
    from dao_ai.config import EvaluationModel, GuidelineModel


def normalize_eval_inputs(raw_inputs: Any) -> dict[str, Any]:
    """
    Normalize various evaluation input formats to the standard
    ``{"messages": [{"role": "user", "content": "..."}]}`` structure.

    Handles:
    - Dicts with a ``messages`` key (passthrough)
    - Lists of message dicts (wrapping)
    - Dicts with a ``request`` key (inference table format)
    - Raw strings or other types (fallback)

    Args:
        raw_inputs: Raw input data in any supported format.

    Returns:
        Normalized dict with a ``messages`` key.
    """
    if isinstance(raw_inputs, dict) and "messages" in raw_inputs:
        messages_val = raw_inputs.get("messages")
        if isinstance(messages_val, list):
            return {"messages": messages_val}
        return {"messages": [{"role": "user", "content": str(messages_val)}]}

    if isinstance(raw_inputs, list) and (
        not raw_inputs or isinstance(raw_inputs[0], dict)
    ):
        return {"messages": raw_inputs}

    if isinstance(raw_inputs, dict) and "request" in raw_inputs:
        return {"messages": [{"role": "user", "content": str(raw_inputs["request"])}]}

    return {"messages": [{"role": "user", "content": str(raw_inputs)}]}


def prepare_eval_dataframe(
    spark_df: Any,
    num_evals: int | None = None,
) -> pd.DataFrame:
    """
    Convert a Spark DataFrame with STRUCT columns into a Pandas DataFrame
    with plain JSON-serializable Python dicts.

    PySpark's ``toPandas()`` can produce Row objects or other non-serializable
    types for nested STRUCT/ARRAY columns.  This function sidesteps that by
    converting those columns to JSON strings **in Spark** (where serialization
    is reliable), calling ``toPandas()``, and parsing them back to plain Python
    dicts via ``json.loads``.

    The ``inputs`` column is automatically normalized via
    :func:`normalize_eval_inputs` after conversion.

    Args:
        spark_df: A PySpark DataFrame with ``inputs`` (and optionally
            ``expectations``) columns.
        num_evals: If provided, limits the result to the first *n* rows.

    Returns:
        A Pandas DataFrame ready for ``merge_records`` and
        ``mlflow.genai.evaluate``.
    """
    import pyspark.sql.functions as F
    from pyspark.sql.types import ArrayType, StructType

    struct_columns: list[str] = [
        field.name
        for field in spark_df.schema
        if isinstance(field.dataType, (StructType, ArrayType))
    ]

    for col_name in struct_columns:
        spark_df = spark_df.withColumn(col_name, F.to_json(F.col(col_name)))

    eval_df: pd.DataFrame = spark_df.toPandas()

    if num_evals is not None:
        eval_df = eval_df.head(num_evals)

    for col_name in struct_columns:
        if col_name in eval_df.columns:
            eval_df[col_name] = eval_df[col_name].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

    if "inputs" in eval_df.columns:
        eval_df["inputs"] = eval_df["inputs"].apply(normalize_eval_inputs)

    return eval_df


def create_guidelines_scorers(
    guidelines_config: list[GuidelineModel],
    judge_model: str | None = None,
) -> list[Guidelines]:
    """
    Create Guidelines scorers from configuration.

    Uses the default managed Databricks judge when no judge_model is specified.

    Args:
        guidelines_config: List of guideline configurations with ``name`` and
            ``guidelines`` attributes.
        judge_model: Optional model endpoint override.

    Returns:
        List of configured Guidelines scorers.
    """
    scorers: list[Guidelines] = []
    for guideline in guidelines_config:
        kwargs: dict[str, Any] = {
            "name": guideline.name,
            "guidelines": guideline.guidelines,
        }
        if judge_model:
            kwargs["model"] = judge_model
        scorers.append(Guidelines(**kwargs))
        logger.debug(f"Created Guidelines scorer: {guideline.name}")
    return scorers


def build_scorers(evaluation_config: EvaluationModel) -> list[Scorer]:
    """
    Build the complete scorer list from evaluation configuration.

    Assembles built-in MLflow judges (Safety, Completeness, RelevanceToQuery,
    ToolCallEfficiency) and any Guidelines scorers defined in the config.

    Args:
        evaluation_config: EvaluationModel configuration with optional
            ``guidelines`` attribute.

    Returns:
        List of scorer instances ready for ``mlflow.genai.evaluate()``.
    """
    scorers: list[Scorer] = [
        Safety(),
        Completeness(),
        RelevanceToQuery(),
        ToolCallEfficiency(),
    ]

    if evaluation_config.guidelines:
        guideline_scorers: list[Guidelines] = create_guidelines_scorers(
            evaluation_config.guidelines
        )
        scorers.extend(guideline_scorers)
        logger.info(f"Added {len(guideline_scorers)} Guidelines scorers")

    return scorers


def create_or_get_eval_dataset(
    name: str,
    experiment_id: str,
    source_df: pd.DataFrame,
    tags: dict[str, str] | None = None,
    replace: bool = False,
) -> EvaluationDataset:
    """
    Create or load a versioned MLflow evaluation dataset and populate it.

    On the first call for a given ``name``, a new dataset is created in the
    experiment.  Subsequent calls load the existing dataset.  In both cases
    the rows in ``source_df`` are merged into the dataset -- MLflow
    deduplicates records by input hash automatically.

    When ``replace`` is ``True``, any existing dataset with the same name
    is deleted before a fresh one is created.

    Args:
        name: Human-readable dataset name.  A good convention is to derive
            it from the evaluation table (e.g.
            ``"catalog_schema_evaluation"``).
        experiment_id: MLflow experiment ID to associate the dataset with.
        source_df: DataFrame with ``inputs`` (and optionally
            ``expectations``) columns that will be merged into the dataset.
        tags: Optional key-value tags for organising and filtering datasets.
        replace: If ``True``, delete the existing dataset and create a new
            one.  Defaults to ``False``.

    Returns:
        An ``EvaluationDataset`` that can be passed directly
        to ``mlflow.genai.evaluate(data=...)``.
    """
    dataset: EvaluationDataset

    if replace:
        try:
            existing = get_dataset(name=name)
            delete_dataset(name=name)
            logger.info(
                f"Deleted existing evaluation dataset: {name} "
                f"(dataset_id={existing.dataset_id})"
            )
        except MlflowException:
            logger.debug(f"No existing dataset to replace: {name}")

    if not replace:
        try:
            dataset = get_dataset(name=name)
            logger.info(
                f"Loaded existing evaluation dataset: {name} "
                f"(dataset_id={dataset.dataset_id})"
            )
        except MlflowException:
            replace = True

    if replace:
        kwargs: dict[str, Any] = {
            "name": name,
            "experiment_id": [experiment_id],
        }
        if tags:
            kwargs["tags"] = tags
        dataset = create_dataset(**kwargs)
        logger.info(
            f"Created new evaluation dataset: {name} "
            f"(dataset_id={dataset.dataset_id}, experiment_id={experiment_id})"
        )

    logger.debug(
        f"Dataset details: name={dataset.name}, "
        f"dataset_id={dataset.dataset_id}, "
        f"source_type={dataset.source_type}"
    )

    dataset = dataset.merge_records(source_df)
    logger.info(f"Merged {len(source_df)} records into dataset {name}")
    return dataset


def prepare_eval_results_for_display(
    eval_results: EvaluationResult,
) -> pd.DataFrame:
    """
    Prepare evaluation results DataFrame for display in Databricks.

    Converts complex columns (e.g. ``assessments``) to string representation
    to avoid Arrow serialization issues.

    Args:
        eval_results: EvaluationResult from ``mlflow.genai.evaluate()``.

    Returns:
        DataFrame copy with complex columns converted to strings.
    """
    results_df: pd.DataFrame = eval_results.tables["eval_results"].copy()

    if "assessments" in results_df.columns:
        results_df["assessments"] = results_df["assessments"].astype(str)

    for col in results_df.columns:
        if results_df[col].dtype == "object":
            try:
                results_df[col].to_list()
            except Exception:
                results_df[col] = results_df[col].astype(str)

    return results_df


_BUILTIN_SCORER_CLASSES: list[type[Scorer]] = [
    Safety,
    Completeness,
    RelevanceToQuery,
    ToolCallEfficiency,
]


def register_monitoring_scorers(
    evaluation_config: EvaluationModel,
    experiment_id: str,
) -> list[Scorer]:
    """
    Register and start evaluation scorers for production monitoring.

    Builds the same scorer set used for offline evaluation (built-in judges
    plus any configured Guidelines scorers) and registers each one against
    the given MLflow experiment so they continuously evaluate production
    traces.

    Scorers that are already registered (by name) are skipped to avoid
    duplicate registration errors.

    Args:
        evaluation_config: ``EvaluationModel`` configuration.  Must have a
            ``monitoring`` attribute with ``sample_rate`` and
            ``guidelines_sample_rate`` fields.
        experiment_id: MLflow experiment ID where production traces are
            logged.

    Returns:
        List of registered (and started) scorer instances.
    """
    import mlflow

    monitoring = evaluation_config.monitoring
    if monitoring is None:
        logger.warning(
            "No monitoring configuration found; skipping scorer registration"
        )
        return []

    mlflow.set_experiment(experiment_id=experiment_id)

    existing_names: set[str] = {s.name for s in list_scorers()}

    registered: list[Scorer] = []

    for scorer_cls in _BUILTIN_SCORER_CLASSES:
        name: str = scorer_cls.__name__
        if name in existing_names:
            logger.info(f"Scorer already registered, skipping: {name}")
            continue

        scorer: Scorer = scorer_cls().register(name=name)
        scorer = scorer.start(
            sampling_config=ScorerSamplingConfig(
                sample_rate=monitoring.sample_rate,
            ),
        )
        registered.append(scorer)
        logger.info(
            f"Registered and started scorer: {name} "
            f"(sample_rate={monitoring.sample_rate})"
        )

    if evaluation_config.guidelines:
        guideline_scorers: list[Guidelines] = create_guidelines_scorers(
            evaluation_config.guidelines
        )
        for gs in guideline_scorers:
            name = gs.name
            if name in existing_names:
                logger.info(f"Guidelines scorer already registered, skipping: {name}")
                continue

            registered_gs: Scorer = gs.register(name=name)
            registered_gs = registered_gs.start(
                sampling_config=ScorerSamplingConfig(
                    sample_rate=monitoring.guidelines_sample_rate,
                ),
            )
            registered.append(registered_gs)
            logger.info(
                f"Registered and started Guidelines scorer: {name} "
                f"(sample_rate={monitoring.guidelines_sample_rate})"
            )

    logger.info(f"Production monitoring: {len(registered)} scorers registered")
    return registered


def get_monitoring_scorers() -> list[Scorer]:
    """
    List all scorers registered for production monitoring in the active
    experiment.

    Returns:
        List of registered scorer instances.
    """
    return list_scorers()


def stop_monitoring_scorers() -> list[Scorer]:
    """
    Stop all active monitoring scorers in the current experiment.

    Each scorer is stopped (sample_rate set to 0) but remains registered
    so it can be restarted later.

    Returns:
        List of stopped scorer instances.
    """
    stopped: list[Scorer] = []
    for scorer in list_scorers():
        if scorer.sample_rate and scorer.sample_rate > 0:
            stopped_scorer: Scorer = scorer.stop()
            stopped.append(stopped_scorer)
            logger.info(f"Stopped scorer: {stopped_scorer.name}")
    logger.info(f"Stopped {len(stopped)} monitoring scorers")
    return stopped


def delete_monitoring_scorers() -> list[str]:
    """
    Delete all registered monitoring scorers from the current experiment.

    Unlike :func:`stop_monitoring_scorers`, this permanently removes the
    scorers.

    Returns:
        List of deleted scorer names.
    """
    deleted: list[str] = []
    for scorer in list_scorers():
        name: str = scorer.name
        delete_scorer(name=name)
        deleted.append(name)
        logger.info(f"Deleted scorer: {name}")
    logger.info(f"Deleted {len(deleted)} monitoring scorers")
    return deleted
