"""
DAO AI Evaluation Module

Provides reusable utilities for MLflow GenAI evaluation using built-in
MLflow 3.10+ judges and scorers.
"""

import json
from typing import Any, Optional

import pandas as pd
from loguru import logger
from mlflow.genai.datasets import create_dataset, get_dataset
from mlflow.genai.scorers import (
    Completeness,
    Guidelines,
    RelevanceToQuery,
    Safety,
    ToolCallEfficiency,
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
    guidelines_config: list[Any],
    judge_model: Optional[str] = None,
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


def build_scorers(evaluation_config: Any) -> list[Any]:
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
    scorers: list[Any] = [
        Safety(),
        Completeness(),
        RelevanceToQuery(),
        ToolCallEfficiency(),
    ]

    if evaluation_config.guidelines:
        guideline_scorers = create_guidelines_scorers(evaluation_config.guidelines)
        scorers.extend(guideline_scorers)
        logger.info(f"Added {len(guideline_scorers)} Guidelines scorers")

    return scorers


def create_or_get_eval_dataset(
    name: str,
    experiment_id: str,
    source_df: pd.DataFrame,
    tags: Optional[dict[str, str]] = None,
) -> Any:
    """
    Create or load a versioned MLflow evaluation dataset and populate it.

    On the first call for a given ``name``, a new dataset is created in the
    experiment.  Subsequent calls load the existing dataset.  In both cases
    the rows in ``source_df`` are merged into the dataset -- MLflow
    deduplicates records by input hash automatically.

    Args:
        name: Human-readable dataset name.  A good convention is to derive
            it from the evaluation table (e.g.
            ``"catalog_schema_evaluation"``).
        experiment_id: MLflow experiment ID to associate the dataset with.
        source_df: DataFrame with ``inputs`` (and optionally
            ``expectations``) columns that will be merged into the dataset.
        tags: Optional key-value tags for organising and filtering datasets.

    Returns:
        An ``mlflow.entities.EvaluationDataset`` that can be passed directly
        to ``mlflow.genai.evaluate(data=...)``.
    """
    try:
        dataset = get_dataset(name=name)
        logger.info(f"Loaded existing evaluation dataset: {name}")
    except Exception:
        kwargs: dict[str, Any] = {
            "name": name,
            "experiment_id": [experiment_id],
        }
        if tags:
            kwargs["tags"] = tags
        dataset = create_dataset(**kwargs)
        logger.info(f"Created new evaluation dataset: {name}")

    dataset = dataset.merge_records(source_df)
    logger.info(f"Merged {len(source_df)} records into dataset {name}")
    return dataset


def prepare_eval_results_for_display(eval_results: Any) -> pd.DataFrame:
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
