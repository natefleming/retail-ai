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
    from dao_ai.config import (
        EvaluationModel,
        GuidelineModel,
        MonitoringModel,
    )


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
        except Exception:
            logger.debug(f"No existing dataset to replace: {name}")

    if not replace:
        try:
            dataset = get_dataset(name=name)
            logger.info(
                f"Loaded existing evaluation dataset: {name} "
                f"(dataset_id={dataset.dataset_id})"
            )
        except Exception:
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
    if "eval_results" in eval_results.tables:
        results_df: pd.DataFrame = eval_results.tables["eval_results"].copy()
    elif eval_results.tables:
        first_key = next(iter(eval_results.tables))
        results_df: pd.DataFrame = eval_results.tables[first_key].copy()
    else:
        return pd.DataFrame()

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

SCORER_NAME_MAP: dict[str, type[Scorer]] = {
    "safety": Safety,
    "completeness": Completeness,
    "relevance_to_query": RelevanceToQuery,
    "tool_call_efficiency": ToolCallEfficiency,
}


def _resolve_scorer_patterns(patterns: list[str]) -> list[type[Scorer]]:
    """Resolve scorer name patterns (including globs) to scorer classes.

    Supports exact names (``"safety"``) and ``fnmatch``-style glob
    patterns (``"*"``, ``"safe*"``).
    """
    from fnmatch import fnmatch

    resolved: list[type[Scorer]] = []
    seen: set[str] = set()

    for pattern in patterns:
        if pattern in SCORER_NAME_MAP:
            if pattern not in seen:
                resolved.append(SCORER_NAME_MAP[pattern])
                seen.add(pattern)
            continue

        matched: bool = False
        for name, cls in SCORER_NAME_MAP.items():
            if fnmatch(name, pattern) and name not in seen:
                resolved.append(cls)
                seen.add(name)
                matched = True

        if not matched and pattern not in seen:
            logger.warning(
                f"Scorer pattern '{pattern}' did not match any built-in scorers. "
                f"Available: {list(SCORER_NAME_MAP.keys())}"
            )

    return resolved


def _ensure_scorer_running(
    scorer: Scorer,
    name: str,
    desired_rate: float,
    existing_scorers: dict[str, Scorer],
) -> Scorer:
    """Register a new scorer or update an existing one to match the desired sample rate.

    If the scorer already exists, its sampling config is updated to match
    the config-declared rate so that every deploy converges to the YAML state.
    """
    if name in existing_scorers:
        existing: Scorer = existing_scorers[name]
        current_rate: float = getattr(existing, "sample_rate", -1)
        if current_rate == desired_rate:
            logger.info(f"Scorer already running at desired rate, no change: {name}")
            return existing
        updated: Scorer = existing.update(
            sampling_config=ScorerSamplingConfig(sample_rate=desired_rate),
        )
        logger.info(
            f"Updated scorer sample_rate: {name} ({current_rate} -> {desired_rate})"
        )
        return updated

    registered: Scorer = scorer.register(name=name)
    started: Scorer = registered.start(
        sampling_config=ScorerSamplingConfig(sample_rate=desired_rate),
    )
    logger.info(f"Registered and started scorer: {name} (sample_rate={desired_rate})")
    return started


def register_monitoring_scorers(
    monitoring_config: MonitoringModel,
    experiment_id: str,
    sql_warehouse_id: str | None = None,
) -> list[Scorer]:
    """
    Register and start evaluation scorers for production monitoring.

    Registers the configured built-in scorers, any Guidelines scorers,
    and any ``GuardrailModel`` entries against the given MLflow experiment
    so they continuously evaluate production traces.

    Existing scorers are updated to match the configured sample rate so
    that every deploy converges to the YAML-declared state.

    Args:
        monitoring_config: ``MonitoringModel`` with scorer selection,
            sampling rates, and optional guidelines.
        experiment_id: MLflow experiment ID where production traces are
            logged.
        sql_warehouse_id: Optional SQL warehouse ID for production monitoring
            when traces are stored in Unity Catalog. If provided,
            ``set_databricks_monitoring_sql_warehouse_id`` is called to
            enable the monitoring service to query UC trace tables.

    Returns:
        List of registered (and started/updated) scorer instances.
    """
    import mlflow

    from dao_ai.config import GuardrailModel

    mlflow.set_experiment(experiment_id=experiment_id)

    if sql_warehouse_id:
        from mlflow.tracing.databricks import set_databricks_monitoring_sql_warehouse_id

        set_databricks_monitoring_sql_warehouse_id(
            sql_warehouse_id=sql_warehouse_id,
            experiment_id=experiment_id,
        )
        logger.info(
            "Configured monitoring SQL warehouse for UC traces",
            sql_warehouse_id=sql_warehouse_id,
        )

    existing_scorers: dict[str, Scorer] = {s.name: s for s in list_scorers()}

    result: list[Scorer] = []

    guardrail_entries: list[GuardrailModel] = []

    if monitoring_config.scorers is not None:
        builtin_patterns: list[str] = []
        for scorer_entry in monitoring_config.scorers:
            if isinstance(scorer_entry, GuardrailModel):
                guardrail_entries.append(scorer_entry)
            else:
                builtin_patterns.append(scorer_entry)

        scorer_classes: list[type[Scorer]] = _resolve_scorer_patterns(builtin_patterns)
    else:
        scorer_classes = _BUILTIN_SCORER_CLASSES

    for scorer_cls in scorer_classes:
        name: str = scorer_cls.__name__
        scorer: Scorer = _ensure_scorer_running(
            scorer=scorer_cls(),
            name=name,
            desired_rate=monitoring_config.sample_rate,
            existing_scorers=existing_scorers,
        )
        result.append(scorer)

    if monitoring_config.guidelines:
        guideline_scorers: list[Guidelines] = create_guidelines_scorers(
            monitoring_config.guidelines
        )
        for gs in guideline_scorers:
            scorer = _ensure_scorer_running(
                scorer=gs,
                name=gs.name,
                desired_rate=monitoring_config.guidelines_sample_rate,
                existing_scorers=existing_scorers,
            )
            result.append(scorer)

    for guardrail in guardrail_entries:
        guardrail_scorer: Scorer = guardrail.as_scorer()
        scorer = _ensure_scorer_running(
            scorer=guardrail_scorer,
            name=guardrail.name,
            desired_rate=monitoring_config.guidelines_sample_rate,
            existing_scorers=existing_scorers,
        )
        result.append(scorer)

    logger.info(f"Production monitoring: {len(result)} scorers active")
    return result


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
