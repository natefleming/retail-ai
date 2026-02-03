# Databricks notebook source
# MAGIC %pip install --quiet --upgrade -r ../requirements.txt
# MAGIC %pip uninstall --quiet -y databricks-connect pyspark pyspark-connect
# MAGIC %pip install --quiet --upgrade databricks-connect
# MAGIC %restart_python

# COMMAND ----------

from typing import Sequence
import os


def find_yaml_files_os_walk(base_path: str) -> Sequence[str]:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    if not os.path.isdir(base_path):
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")

    yaml_files = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith((".yaml", ".yml")):
                yaml_files.append(os.path.join(root, file))

    return sorted(yaml_files)


# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")

config_files: Sequence[str] = find_yaml_files_os_walk("../config")
dbutils.widgets.dropdown(
    name="config-paths",
    choices=config_files,
    defaultValue=next(iter(config_files), ""),
)

config_path: str | None = dbutils.widgets.get("config-path") or None
project_path: str = dbutils.widgets.get("config-paths") or None

config_path: str = config_path or project_path

print(config_path)

# COMMAND ----------

import sys
from typing import Sequence
from importlib.metadata import version

sys.path.insert(0, "../src")

pip_requirements: Sequence[str] = (
    f"databricks-agents=={version('databricks-agents')}",
    f"mlflow=={version('mlflow')}",
    f"optuna=={version('optuna')}",
)
print("\n".join(pip_requirements))

# COMMAND ----------

import nest_asyncio

nest_asyncio.apply()

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from loguru import logger

from dao_ai.logging import configure_logging

configure_logging(level="DEBUG")

# COMMAND ----------

# MAGIC %md
# MAGIC # Context-Aware Cache Threshold Optimization
# MAGIC
# MAGIC This notebook optimizes the thresholds for the Genie context-aware cache using Optuna
# MAGIC Bayesian optimization. The optimizer tunes:
# MAGIC
# MAGIC - **similarity_threshold**: Minimum similarity for question matching (0.5-0.99)
# MAGIC - **context_similarity_threshold**: Minimum similarity for context matching (0.5-0.99)
# MAGIC - **question_weight**: Weight for question vs context in combined score (0.1-0.9)
# MAGIC
# MAGIC ## How It Works
# MAGIC
# MAGIC 1. **Evaluation Dataset**: The optimizer uses pairs of questions/contexts with known labels
# MAGIC    (should match or should not match)
# MAGIC 2. **Bayesian Optimization**: Optuna's TPE sampler efficiently searches the parameter space
# MAGIC 3. **LLM-as-Judge**: For unlabeled data, an LLM determines semantic equivalence
# MAGIC 4. **Metrics**: Optimizes for F1 score (balances precision and recall) by default
# MAGIC
# MAGIC ## Important: Context Window Size
# MAGIC
# MAGIC This notebook optimizes **thresholds** for cache matching. The `context_window_size`
# MAGIC parameter (which controls how many previous prompts are included in context embeddings)
# MAGIC should be configured separately in your cache parameters.
# MAGIC
# MAGIC If you change `context_window_size`, consider:
# MAGIC 1. Re-running this optimization with the new window size
# MAGIC 2. Rebuilding the cache using `from_space()` to re-embed entries

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 1: Use Configuration File
# MAGIC
# MAGIC If you have a configuration file with threshold optimization settings:

# COMMAND ----------

from dao_ai.config import AppConfig

# Load configuration if available
try:
    config: AppConfig = AppConfig.from_file(path=config_path)
    print(f"Loaded configuration from: {config_path}")
except Exception as e:
    config = None
    print(f"No configuration loaded: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 2: Generate Dataset from Existing Cache
# MAGIC
# MAGIC If you have an existing semantic cache with entries, you can generate an
# MAGIC evaluation dataset automatically using paraphrasing:

# COMMAND ----------

from dao_ai.genie.cache.optimization import (
    ContextAwareCacheEvalDataset,
    ContextAwareCacheEvalEntry,
    ThresholdOptimizationResult,
    generate_eval_dataset_from_cache,
    optimize_context_aware_cache_thresholds,
)

# Example: Generate dataset from cache entries
# Uncomment and modify to use with your cache data

# cache_entries = [
#     {
#         "question": "What are total sales for Q1?",
#         "conversation_context": "Previous: Show me revenue",
#         "question_embedding": [...],  # Your embeddings
#         "context_embedding": [...],
#     },
#     # ... more entries
# ]
#
# eval_dataset = generate_eval_dataset_from_cache(
#     cache_entries=cache_entries,
#     embedding_model="databricks-gte-large-en",
#     num_positive_pairs=50,
#     num_negative_pairs=50,
#     dataset_name="my_cache_eval",
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 3: Create Dataset Manually
# MAGIC
# MAGIC Create an evaluation dataset with known semantic equivalence labels:

# COMMAND ----------

from dao_ai.config import LLMModel

# Create embeddings model for generating embeddings
embedding_model = LLMModel(name="databricks-gte-large-en")
embeddings = embedding_model.as_embeddings_model()

# Define test pairs with known labels
test_pairs = [
    # Positive pairs (should match)
    {
        "question1": "What are total sales?",
        "question2": "Show me total sales",
        "context1": "",
        "context2": "",
        "expected_match": True,
    },
    {
        "question1": "How much revenue did we make?",
        "question2": "What's our total revenue?",
        "context1": "Previous: Show Q1 results",
        "context2": "Previous: Show Q1 results",
        "expected_match": True,
    },
    # Negative pairs (should not match)
    {
        "question1": "What are total sales?",
        "question2": "What is the inventory count?",
        "context1": "",
        "context2": "",
        "expected_match": False,
    },
    {
        "question1": "Show revenue by region",
        "question2": "Show expenses by category",
        "context1": "",
        "context2": "",
        "expected_match": False,
    },
]

# Generate embeddings and create dataset
entries = []
for pair in test_pairs:
    q1_emb = embeddings.embed_query(pair["question1"])
    q2_emb = embeddings.embed_query(pair["question2"])
    c1_emb = embeddings.embed_query(pair["context1"]) if pair["context1"] else []
    c2_emb = embeddings.embed_query(pair["context2"]) if pair["context2"] else []

    entries.append(
        ContextAwareCacheEvalEntry(
            question=pair["question1"],
            question_embedding=q1_emb,
            context=pair["context1"],
            context_embedding=c1_emb if c1_emb else [0.0] * len(q1_emb),
            cached_question=pair["question2"],
            cached_question_embedding=q2_emb,
            cached_context=pair["context2"],
            cached_context_embedding=c2_emb if c2_emb else [0.0] * len(q2_emb),
            expected_match=pair["expected_match"],
        )
    )

manual_dataset = ContextAwareCacheEvalDataset(
    name="manual_eval_dataset",
    entries=entries,
    description="Manually created evaluation dataset",
)

print(f"Created dataset with {len(manual_dataset)} entries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Optimization
# MAGIC
# MAGIC Run the threshold optimization with the evaluation dataset:

# COMMAND ----------

import mlflow

# Set MLflow registry for logging
mlflow.set_registry_uri("databricks-uc")

# Run optimization
# Note: For production use, increase n_trials to 50+ for better results
result: ThresholdOptimizationResult = optimize_context_aware_cache_thresholds(
    dataset=manual_dataset,
    judge_model="databricks-meta-llama-3-3-70b-instruct",  # For unlabeled entries
    n_trials=20,  # Increase for better optimization (50+ recommended)
    metric="f1",  # Options: f1, precision, recall, fbeta
    register_if_improved=True,
    show_progress_bar=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

import pandas as pd

# Display results
print("=" * 80)
print("OPTIMIZATION RESULTS")
print("=" * 80)

print(f"\nStudy Name: {result.study_name}")
print(f"Total Trials: {result.n_trials}")
print(f"Best Trial: #{result.best_trial_number}")

print(f"\n{'Original Thresholds:':<30}")
for key, value in result.original_thresholds.items():
    print(f"  {key}: {value:.4f}")

print(f"\n{'Optimized Thresholds:':<30}")
for key, value in result.optimized_thresholds.items():
    print(f"  {key}: {value:.4f}")

print(f"\n{'Performance:':<30}")
print(f"  Original Score: {result.original_score:.4f}")
print(f"  Optimized Score: {result.optimized_score:.4f}")
print(f"  Improvement: {result.improvement:.1%}")

if result.improved:
    print("\n✅ Optimization IMPROVED thresholds!")
else:
    print("\n⚠️ Optimization did not improve over original thresholds")

# COMMAND ----------

# Create comparison DataFrame
comparison_data = []
for key in result.original_thresholds.keys():
    original = result.original_thresholds[key]
    optimized = result.optimized_thresholds[key]
    change = optimized - original
    comparison_data.append({
        "Parameter": key,
        "Original": f"{original:.4f}",
        "Optimized": f"{optimized:.4f}",
        "Change": f"{change:+.4f}",
    })

comparison_df = pd.DataFrame(comparison_data)
display(comparison_df)

# COMMAND ----------

# Display detailed metrics from metadata
if "confusion_matrix" in result.metadata:
    print("\nConfusion Matrix:")
    confusion = result.metadata["confusion_matrix"]
    print(f"  True Positives: {confusion.get('true_positives', 0)}")
    print(f"  False Positives: {confusion.get('false_positives', 0)}")
    print(f"  True Negatives: {confusion.get('true_negatives', 0)}")
    print(f"  False Negatives: {confusion.get('false_negatives', 0)}")

print(f"\nDetailed Metrics:")
print(f"  Precision: {result.metadata.get('optimized_precision', 0):.4f}")
print(f"  Recall: {result.metadata.get('optimized_recall', 0):.4f}")
print(f"  F1 Score: {result.metadata.get('optimized_f1', 0):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Optimized Thresholds
# MAGIC
# MAGIC To use the optimized thresholds in your configuration:

# COMMAND ----------

# Generate configuration snippet
print("Add these values to your configuration:")
print()
print("```yaml")
print("cache:")
print("  type: semantic")
print("  parameters:")
print(f"    similarity_threshold: {result.optimized_thresholds['similarity_threshold']:.4f}")
print(f"    context_similarity_threshold: {result.optimized_thresholds['context_similarity_threshold']:.4f}")
print(f"    question_weight: {result.optimized_thresholds['question_weight']:.4f}")
print("    # ... other parameters")
print("```")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC After optimizing thresholds:
# MAGIC
# MAGIC 1. **Review results in MLflow**: Check the logged metrics and parameters
# MAGIC 2. **Update your configuration**: Apply the optimized thresholds to your cache config
# MAGIC 3. **Test in staging**: Validate the new thresholds with real queries
# MAGIC 4. **Monitor performance**: Track cache hit rates and accuracy in production
# MAGIC
# MAGIC ### Tips for Better Optimization
# MAGIC
# MAGIC - **More trials**: Use `n_trials=100+` for better convergence
# MAGIC - **More data**: Include diverse question pairs in your evaluation dataset
# MAGIC - **Balance dataset**: Equal positive and negative pairs help avoid bias
# MAGIC - **Use real data**: Generate dataset from actual cache usage patterns

# COMMAND ----------

print("\n" + "=" * 80)
print("Context-Aware Cache Threshold Optimization Complete")
print("=" * 80)
