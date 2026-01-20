"""
Instructed retriever for query decomposition and result fusion.

This module provides functions for decomposing user queries into multiple
subqueries with metadata filters and merging results using Reciprocal Rank Fusion.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import yaml
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from loguru import logger
from mlflow.entities import SpanType

from dao_ai.config import DecomposedQueries, LLMModel, SearchQuery


def _get_databricks_compatible_schema() -> dict[str, Any]:
    """Generate JSON schema for query decomposition structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "decomposed_queries",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "description": "List of decomposed search queries with filters",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The search query text",
                                },
                                "filters": {
                                    "type": "object",
                                    "description": "Filters to apply in Databricks Vector Search syntax",
                                },
                            },
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["queries"],
                "additionalProperties": False,
            },
        },
    }

# Module-level cache for LLM clients
_llm_cache: dict[str, BaseChatModel] = {}

# Load prompt template
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "instructed_retriever_decomposition.yaml"


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
        logger.debug("Created new LLM client for decomposition", model=model_config.name)
    return _llm_cache[cache_key]


def _format_constraints(constraints: list[str] | None) -> str:
    """Format constraints list for prompt injection."""
    if not constraints:
        return "No additional constraints."
    return "\n".join(f"- {c}" for c in constraints)


def _format_examples(examples: list[dict[str, Any]] | None) -> str:
    """Format few-shot examples for prompt injection."""
    if not examples:
        return "No examples provided."

    formatted = []
    for i, ex in enumerate(examples, 1):
        query = ex.get("query", "")
        filters = ex.get("filters", {})
        formatted.append(
            f"Example {i}:\n"
            f"  Query: \"{query}\"\n"
            f"  Filters: {json.dumps(filters)}"
        )
    return "\n".join(formatted)


@mlflow.trace(name="decompose_query", span_type=SpanType.LLM)
def decompose_query(
    llm: BaseChatModel,
    query: str,
    schema_description: str,
    constraints: list[str] | None = None,
    max_subqueries: int = 3,
    examples: list[dict[str, Any]] | None = None,
    previous_feedback: str | None = None,
) -> list[SearchQuery]:
    """
    Decompose a user query into multiple search queries with filters.

    Uses structured output for reliable parsing and injects current time
    for resolving relative date references.

    Args:
        llm: Language model for decomposition
        query: User's search query
        schema_description: Column names, types, and valid filter syntax
        constraints: Default constraints to apply
        max_subqueries: Maximum number of subqueries to generate
        examples: Few-shot examples for domain-specific filter translation
        previous_feedback: Feedback from failed verification (for retry)

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

    prompt = prompt_template.format(
        current_time=current_time,
        schema_description=schema_description,
        constraints=_format_constraints(constraints),
        examples=_format_examples(examples),
        max_subqueries=max_subqueries,
        query=query,
    ) + feedback_section

    logger.trace("Decomposing query", query=query[:100], max_subqueries=max_subqueries)

    response_format = _get_databricks_compatible_schema()
    bound_llm = llm.bind(response_format=response_format)

    try:
        response = bound_llm.invoke(prompt)
        # Parse JSON response into Pydantic model
        result_dict = json.loads(response.content)
        parsed = DecomposedQueries.model_validate(result_dict)
        subqueries = parsed.queries[:max_subqueries]

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

    except Exception as e:
        logger.warning("Query decomposition failed", error=str(e))
        raise


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
