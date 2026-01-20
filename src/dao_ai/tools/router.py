"""
Query router for selecting execution mode based on query characteristics.

Routes to internal execution modes within the same retriever instance:
- standard: Single similarity_search for simple queries
- instructed: Decompose -> Parallel Search -> RRF for constrained queries
"""

import json
from pathlib import Path
from typing import Any, Literal

import mlflow
import yaml
from langchain_core.language_models import BaseChatModel
from loguru import logger
from mlflow.entities import SpanType
from pydantic import BaseModel, ConfigDict, Field

# Load prompt template
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "router.yaml"


def _load_prompt_template() -> dict[str, Any]:
    """Load the router prompt template from YAML."""
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)


def _get_router_schema() -> dict[str, Any]:
    """Generate JSON schema for router decision structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "router_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["standard", "instructed"],
                        "description": "Execution mode: 'standard' for simple queries, 'instructed' for constrained queries",
                    }
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        },
    }


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

    response_format = _get_router_schema()
    bound_llm = llm.bind(response_format=response_format)
    response = bound_llm.invoke(prompt)
    result_dict = json.loads(response.content)
    decision = RouterDecision.model_validate(result_dict)

    logger.debug("Router decision", mode=decision.mode, query=query[:50])
    mlflow.set_tag("router.mode", decision.mode)

    return decision.mode
