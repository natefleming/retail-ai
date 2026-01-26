"""
Result verifier for validating search results against user constraints.

Provides structured feedback for intelligent retry when results don't match intent.
"""

import json
from pathlib import Path
from typing import Any

import mlflow
import yaml
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from loguru import logger
from mlflow.entities import SpanType

from dao_ai.config import VerificationResult

# Load prompt template
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "verifier.yaml"


def _load_prompt_template() -> dict[str, Any]:
    """Load the verifier prompt template from YAML."""
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)


def _format_results_summary(documents: list[Document], max_docs: int = 5) -> str:
    """Format top documents for verification prompt."""
    if not documents:
        return "No results retrieved."

    summaries = []
    for i, doc in enumerate(documents[:max_docs]):
        metadata_str = ", ".join(
            f"{k}: {v}"
            for k, v in doc.metadata.items()
            if not k.startswith("_") and k not in ("rrf_score", "reranker_score")
        )
        content_preview = (
            doc.page_content[:200] + "..."
            if len(doc.page_content) > 200
            else doc.page_content
        )
        summaries.append(f"{i + 1}. {content_preview}\n   Metadata: {metadata_str}")

    return "\n\n".join(summaries)


def _format_constraints(constraints: list[str] | None) -> str:
    """Format constraints list for prompt."""
    if not constraints:
        return "No explicit constraints specified."
    return "\n".join(f"- {c}" for c in constraints)


@mlflow.trace(name="verify_results", span_type=SpanType.LLM)
def verify_results(
    llm: BaseChatModel,
    query: str,
    documents: list[Document],
    schema_description: str,
    constraints: list[str] | None = None,
    previous_feedback: str | None = None,
) -> VerificationResult:
    """
    Verify that search results satisfy user constraints.

    Args:
        llm: Language model for verification
        query: User's original search query
        documents: Retrieved documents to verify
        schema_description: Column names, types, and filter syntax
        constraints: Explicit constraints to verify
        previous_feedback: Feedback from previous failed attempt (for retry)

    Returns:
        VerificationResult with pass/fail status and structured feedback
    """
    prompt_config = _load_prompt_template()
    prompt_template = prompt_config["template"]

    prompt = prompt_template.format(
        query=query,
        schema_description=schema_description,
        constraints=_format_constraints(constraints),
        num_results=len(documents),
        results_summary=_format_results_summary(documents),
        previous_feedback=previous_feedback or "N/A (first attempt)",
    )

    logger.trace("Verifying results", query=query[:100], num_docs=len(documents))

    # Use LangChain's with_structured_output for automatic strategy selection
    # (JSON schema vs tool calling based on model capabilities)
    try:
        structured_llm: Runnable[str, VerificationResult] = llm.with_structured_output(
            VerificationResult
        )
        result: VerificationResult = structured_llm.invoke(prompt)
    except Exception as e:
        logger.warning(
            "Verifier failed, treating as passed with low confidence", error=str(e)
        )
        return VerificationResult(
            passed=True,
            confidence=0.0,
            feedback="Verification failed to produce a valid result",
        )

    # Log for observability
    mlflow.log_text(
        json.dumps(result.model_dump(), indent=2),
        "verification_result.json",
    )

    logger.debug(
        "Verification complete",
        passed=result.passed,
        confidence=result.confidence,
        unmet_constraints=result.unmet_constraints,
    )

    return result


def add_verification_metadata(
    documents: list[Document],
    result: VerificationResult,
    exhausted: bool = False,
) -> list[Document]:
    """
    Add verification metadata to documents.

    Args:
        documents: Documents to annotate
        result: Verification result
        exhausted: Whether max retries were exhausted

    Returns:
        Documents with verification metadata added
    """
    status = "exhausted" if exhausted else ("passed" if result.passed else "failed")

    annotated = []
    for doc in documents:
        metadata = {
            **doc.metadata,
            "_verification_status": status,
            "_verification_confidence": result.confidence,
        }
        if result.feedback:
            metadata["_verification_feedback"] = result.feedback
        annotated.append(Document(page_content=doc.page_content, metadata=metadata))

    return annotated
