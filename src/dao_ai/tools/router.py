"""
Query router for selecting execution mode based on query characteristics.

Routes to internal execution modes within the same retriever instance:
- standard: Single similarity_search for simple queries
- instructed: Decompose -> Parallel Search -> RRF for constrained queries
"""

from pathlib import Path
from typing import Any, Literal

import mlflow
import yaml
from langchain_core.language_models import BaseChatModel
from loguru import logger
from mlflow.entities import SpanType
from pydantic import BaseModel, ConfigDict, Field

from dao_ai.utils import invoke_with_structured_output

# Load prompt template
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "router.yaml"


def _load_prompt_template() -> dict[str, Any]:
    """Load the router prompt template from YAML."""
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)


class RouterDecision(BaseModel):
    """Structured output for router decision."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["standard", "instructed"] = Field(
        description="Execution mode: 'standard' for simple queries, 'instructed' for constrained queries"
    )


@mlflow.trace(name="route_query", span_type=SpanType.LLM)
def route_query(
    llm: BaseChatModel,
    query: str,
    schema_description: str,
) -> Literal["standard", "instructed"]:
    """
    Determine the execution mode for a search query.

    Args:
        llm: Language model for routing decision
        query: User's search query
        schema_description: Column names, types, and filter syntax

    Returns:
        "standard" for simple queries, "instructed" for constrained queries
    """
    prompt_config = _load_prompt_template()
    prompt_template = prompt_config["template"]

    prompt = prompt_template.format(
        schema_description=schema_description,
        query=query,
    )

    logger.trace("Routing query", query=query[:100])

    # Use Databricks-compatible structured output with proper response_format
    decision = invoke_with_structured_output(llm, prompt, RouterDecision)

    # Handle None result (can happen if LLM doesn't produce valid output)
    if decision is None:
        logger.warning("Router returned None, defaulting to standard mode")
        return "standard"

    logger.debug("Router decision", mode=decision.mode, query=query[:50])
    mlflow.set_tag("router.mode", decision.mode)

    return decision.mode
