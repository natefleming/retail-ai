"""
Instructed retriever for query decomposition and result fusion.

This module provides functions for decomposing user queries into multiple
subqueries with metadata filters and merging results using Reciprocal Rank Fusion.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import mlflow
import yaml
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from loguru import logger
from mlflow.entities import SpanType
from pydantic import BaseModel, ConfigDict, Field

from dao_ai.config import (
    ColumnInfo,
    DecomposedQueries,
    FilterItem,
    LLMModel,
    SearchQuery,
)

# Module-level cache for LLM clients
_llm_cache: dict[str, BaseChatModel] = {}

# Load prompt template
_PROMPT_PATH = (
    Path(__file__).parent.parent / "prompts" / "instructed_retriever_decomposition.yaml"
)


def _load_prompt_template() -> dict[str, Any]:
    """Load the decomposition prompt template from YAML."""
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)


def _get_cached_llm(model_config: LLMModel) -> BaseChatModel:
    """
    Get or create cached LLM client for decomposition.

    Uses full config as cache key to avoid collisions when same model name
    has different parameters (temperature, API keys, etc.).
    """
    cache_key = model_config.model_dump_json()
    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = model_config.as_chat_model()
        logger.debug(
            "Created new LLM client for decomposition", model=model_config.name
        )
    return _llm_cache[cache_key]


def _format_constraints(constraints: list[str] | None) -> str:
    """Format constraints list for prompt injection."""
    if not constraints:
        return "No additional constraints."
    return "\n".join(f"- {c}" for c in constraints)


def _format_examples(examples: list[dict[str, Any]] | None) -> str:
    """Format few-shot examples for prompt injection.

    Converts dict-style filters from config to FilterItem array format
    to match the expected JSON schema output.
    """
    if not examples:
        return "No examples provided."

    formatted = []
    for i, ex in enumerate(examples, 1):
        query = ex.get("query", "")
        filters = ex.get("filters", {})
        # Convert dict to FilterItem array format
        filter_items = [{"key": k, "value": v} for k, v in filters.items()]
        formatted.append(
            f'Example {i}:\n  Query: "{query}"\n  Filters: {json.dumps(filter_items)}'
        )
    return "\n".join(formatted)


def create_decomposition_schema(
    columns: list[ColumnInfo] | None = None,
) -> type[BaseModel]:
    """Create schema-aware DecomposedQueries model with dynamic descriptions.

    When columns are provided, the column names and valid operators are embedded
    directly into the JSON schema that with_structured_output sends to the LLM.
    This improves accuracy by making valid filter keys explicit in the schema.

    Args:
        columns: List of column metadata for dynamic schema generation

    Returns:
        A DecomposedQueries-compatible Pydantic model class
    """
    if not columns:
        # Fall back to generic models
        return DecomposedQueries

    # Build column info with types for the schema description
    column_info = ", ".join(f"{c.name} ({c.type})" for c in columns)

    # Build operator list from column definitions (union of all column operators)
    all_operators: set[str] = set()
    for col in columns:
        all_operators.update(col.operators)
    # Remove empty string (equality) and sort for consistent output
    named_operators = sorted(all_operators - {""})
    operator_list = ", ".join(named_operators) if named_operators else "equality only"

    # Build valid key examples with operators
    key_examples: list[str] = []
    for col in columns[:3]:  # Show examples for first 3 columns
        key_examples.append(f"'{col.name}'")
        if "<" in col.operators:
            key_examples.append(f"'{col.name} <'")
        if "NOT" in col.operators:
            key_examples.append(f"'{col.name} NOT'")

    # Create dynamic FilterItem with schema-aware description
    class SchemaFilterItem(BaseModel):
        """A metadata filter for vector search with schema-specific columns."""

        model_config = ConfigDict(extra="forbid")
        key: str = Field(
            description=(
                f"Column name with optional operator suffix. "
                f"Valid columns: {column_info}. "
                f"Operators: (none) for equality, {operator_list}. "
                f"Examples: {', '.join(key_examples[:5])}"
            )
        )
        value: Union[str, int, float, bool, list[Union[str, int, float, bool]]] = Field(
            description="The filter value matching the column type."
        )

    # Create dynamic SearchQuery using SchemaFilterItem
    class SchemaSearchQuery(BaseModel):
        """A search query with schema-aware filters."""

        model_config = ConfigDict(extra="forbid")
        text: str = Field(
            description=(
                "Natural language search query text optimized for semantic similarity. "
                "Should be focused on a single search intent. "
                "Do NOT include filter criteria in the text; use the filters field instead."
            )
        )
        filters: Optional[list[SchemaFilterItem]] = Field(
            default=None,
            description=(
                f"Metadata filters to constrain search results. "
                f"Valid filter columns: {column_info}. "
                f"Set to null if no filters apply."
            ),
        )

    # Create dynamic DecomposedQueries using SchemaSearchQuery
    class SchemaDecomposedQueries(BaseModel):
        """Decomposed search queries with schema-aware filters."""

        model_config = ConfigDict(extra="forbid")
        queries: list[SchemaSearchQuery] = Field(
            description=(
                "List of search queries extracted from the user request. "
                "Each query should target a distinct search intent. "
                "Order queries by importance, with the most relevant first."
            )
        )

    return SchemaDecomposedQueries


@mlflow.trace(name="decompose_query", span_type=SpanType.LLM)
def decompose_query(
    llm: BaseChatModel,
    query: str,
    schema_description: str,
    constraints: list[str] | None = None,
    max_subqueries: int = 3,
    examples: list[dict[str, Any]] | None = None,
    previous_feedback: str | None = None,
    columns: list[ColumnInfo] | None = None,
) -> list[SearchQuery]:
    """
    Decompose a user query into multiple search queries with filters.

    Uses structured output for reliable parsing and injects current time
    for resolving relative date references. When columns are provided,
    schema-aware Pydantic models are used for improved filter accuracy.

    Args:
        llm: Language model for decomposition
        query: User's search query
        schema_description: Column names, types, and valid filter syntax
        constraints: Default constraints to apply
        max_subqueries: Maximum number of subqueries to generate
        examples: Few-shot examples for domain-specific filter translation
        previous_feedback: Feedback from failed verification (for retry)
        columns: Structured column info for dynamic schema generation

    Returns:
        List of SearchQuery objects with text and optional filters
    """
    current_time = datetime.now().isoformat()

    # Load and format prompt
    prompt_config = _load_prompt_template()
    prompt_template = prompt_config["template"]

    # Add previous feedback section if provided (for retry)
    feedback_section = ""
    if previous_feedback:
        feedback_section = f"\n\n## Previous Attempt Feedback\nThe previous search attempt failed verification: {previous_feedback}\nAdjust your filters to address this feedback."

    prompt = (
        prompt_template.format(
            current_time=current_time,
            schema_description=schema_description,
            constraints=_format_constraints(constraints),
            examples=_format_examples(examples),
            max_subqueries=max_subqueries,
            query=query,
        )
        + feedback_section
    )

    logger.trace(
        "Decomposing query",
        query=query[:100],
        max_subqueries=max_subqueries,
        dynamic_schema=columns is not None,
    )

    # Create schema-aware model when columns are provided
    DecompositionSchema: type[BaseModel] = create_decomposition_schema(columns)

    # Use LangChain's with_structured_output for automatic strategy selection
    # (JSON schema vs tool calling based on model capabilities)
    try:
        structured_llm: Runnable[str, BaseModel] = llm.with_structured_output(
            DecompositionSchema
        )
        result: BaseModel = structured_llm.invoke(prompt)
    except Exception as e:
        logger.warning("Query decomposition failed", error=str(e))
        raise

    # Extract queries from result (works with both static and dynamic schemas)
    subqueries: list[SearchQuery] = []
    for query_obj in result.queries[:max_subqueries]:
        # Convert dynamic schema objects to SearchQuery for consistent return type
        filters: list[FilterItem] | None = None
        if query_obj.filters:
            filters = [FilterItem(key=f.key, value=f.value) for f in query_obj.filters]
        subqueries.append(SearchQuery(text=query_obj.text, filters=filters))

    # Log for observability
    mlflow.set_tag("num_subqueries", len(subqueries))
    mlflow.log_text(
        json.dumps([sq.model_dump() for sq in subqueries], indent=2),
        "decomposition.json",
    )

    logger.debug(
        "Query decomposed",
        num_subqueries=len(subqueries),
        queries=[sq.text[:50] for sq in subqueries],
    )

    return subqueries


def rrf_merge(
    results_lists: list[list[Document]],
    k: int = 60,
    primary_key: str | None = None,
) -> list[Document]:
    """
    Merge results from multiple queries using Reciprocal Rank Fusion.

    RRF is safer than raw score sorting because Databricks Vector Search
    scores aren't normalized across query types (HYBRID vs ANN).

    RRF Score = Σ 1 / (k + rank_i) for each result list

    Args:
        results_lists: List of document lists from different subqueries
        k: RRF constant (lower values weight top ranks more heavily)
        primary_key: Metadata key for document identity (for deduplication)

    Returns:
        Merged and deduplicated documents sorted by RRF score
    """
    if not results_lists:
        return []

    # Filter empty lists first
    non_empty = [r for r in results_lists if r]
    if not non_empty:
        return []

    # Single list optimization (still add RRF scores for consistency)
    if len(non_empty) == 1:
        docs_with_scores: list[Document] = []
        for rank, doc in enumerate(non_empty[0]):
            rrf_score = 1.0 / (k + rank + 1)
            docs_with_scores.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "rrf_score": rrf_score},
                )
            )
        return docs_with_scores

    # Calculate RRF scores
    # Key: document identifier, Value: (total_rrf_score, Document)
    doc_scores: dict[str, tuple[float, Document]] = {}

    def get_doc_id(doc: Document) -> str:
        """Get unique identifier for document."""
        if primary_key and primary_key in doc.metadata:
            return str(doc.metadata[primary_key])
        # Fallback to content hash
        return str(hash(doc.page_content))

    for result_list in non_empty:
        for rank, doc in enumerate(result_list):
            doc_id = get_doc_id(doc)
            rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed

            if doc_id in doc_scores:
                # Accumulate RRF score for duplicates
                existing_score, existing_doc = doc_scores[doc_id]
                doc_scores[doc_id] = (existing_score + rrf_score, existing_doc)
            else:
                doc_scores[doc_id] = (rrf_score, doc)

    # Sort by RRF score descending
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)

    # Add RRF score to metadata
    merged_docs: list[Document] = []
    for rrf_score, doc in sorted_docs:
        merged_doc = Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rrf_score": rrf_score},
        )
        merged_docs.append(merged_doc)

    logger.debug(
        "RRF merge complete",
        input_lists=len(results_lists),
        total_docs=sum(len(r) for r in results_lists),
        unique_docs=len(merged_docs),
    )

    return merged_docs
