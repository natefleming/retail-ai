"""
Instruction-aware reranker for constraint-based document reordering.

Runs after FlashRank to apply user instructions and constraints to the ranking.
General-purpose component usable with any retrieval strategy.
"""

from pathlib import Path
from typing import Any

import mlflow
import yaml
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from loguru import logger
from mlflow.entities import SpanType
from pydantic import ValidationError

from dao_ai.config import RankingResult
from dao_ai.utils import invoke_with_structured_output

# Load prompt template
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "instruction_reranker.yaml"


def _load_prompt_template() -> dict[str, Any]:
    """Load the instruction reranker prompt template from YAML."""
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)


def _format_documents(documents: list[Document]) -> str:
    """Format documents for the reranking prompt."""
    if not documents:
        return "No documents to rerank."

    formatted = []
    for i, doc in enumerate(documents):
        metadata_str = ", ".join(
            f"{k}: {v}"
            for k, v in doc.metadata.items()
            if not k.startswith("_") and k not in ("rrf_score", "reranker_score")
        )
        content_preview = (
            doc.page_content[:300] + "..."
            if len(doc.page_content) > 300
            else doc.page_content
        )
        formatted.append(
            f"[{i}] Content: {content_preview}\n    Metadata: {metadata_str}"
        )

    return "\n\n".join(formatted)


@mlflow.trace(name="instruction_aware_rerank", span_type=SpanType.LLM)
def instruction_aware_rerank(
    llm: BaseChatModel,
    query: str,
    documents: list[Document],
    instructions: str | None = None,
    schema_description: str | None = None,
    top_n: int | None = None,
) -> list[Document]:
    """
    Rerank documents based on user instructions and constraints.

    Args:
        llm: Language model for reranking
        query: User's search query
        documents: Documents to rerank (typically FlashRank output)
        instructions: Custom reranking instructions
        schema_description: Column names and types for context
        top_n: Number of documents to return (None = all scored documents)

    Returns:
        Reranked documents with instruction_rerank_score in metadata
    """
    if not documents:
        return []

    prompt_config = _load_prompt_template()
    prompt_template = prompt_config["template"]

    default_instructions = (
        "Prioritize results that best match the user's explicit constraints "
        "(price, brand, category, date). Prefer more specific matches over general results."
    )

    prompt = prompt_template.format(
        query=query,
        schema_description=schema_description or "No schema information provided.",
        instructions=instructions or default_instructions,
        documents=_format_documents(documents),
    )

    logger.trace("Instruction reranking", query=query[:100], num_docs=len(documents))

    # Use Databricks-compatible structured output with proper response_format
    try:
        result = invoke_with_structured_output(llm, prompt, RankingResult)
    except (ValidationError, Exception) as e:
        logger.warning(
            "Failed to get structured output from reranker, returning original order",
            error=str(e),
        )
        result = None

    # Handle None result (can happen if LLM doesn't produce valid output)
    if result is None:
        logger.warning(
            "Reranker returned None, returning original order",
        )
        return [
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "instruction_rerank_score": 1.0 - (i / len(documents)),
                    "instruction_rerank_reason": "fallback: no valid response",
                },
            )
            for i, doc in enumerate(documents[:top_n] if top_n else documents)
        ]

    # Build reranked document list
    reranked: list[Document] = []
    for ranking in result.rankings:
        if ranking.index < 0 or ranking.index >= len(documents):
            logger.warning("Invalid document index from reranker", index=ranking.index)
            continue

        original_doc = documents[ranking.index]
        reranked_doc = Document(
            page_content=original_doc.page_content,
            metadata={
                **original_doc.metadata,
                "instruction_rerank_score": ranking.score,
                "instruction_rerank_reason": ranking.reason,
            },
        )
        reranked.append(reranked_doc)

    # Apply top_n limit
    if top_n is not None and len(reranked) > top_n:
        reranked = reranked[:top_n]

    # Calculate and log average score
    if reranked:
        avg_score = sum(
            d.metadata.get("instruction_rerank_score", 0) for d in reranked
        ) / len(reranked)
        mlflow.set_tag("reranker.instruction_avg_score", f"{avg_score:.3f}")

    logger.debug(
        "Instruction reranking complete",
        input_count=len(documents),
        output_count=len(reranked),
    )

    return reranked
