"""
DAO AI Evaluation Module

Provides reusable scorers and helper functions for MLflow GenAI evaluation.
Implements MLflow 3.8+ best practices for trace linking and scorer patterns.
"""

from typing import Any, Callable, Optional, TypedDict, Union

import mlflow
from loguru import logger
from mlflow.entities import Feedback, SpanStatus, SpanStatusCode, Trace
from mlflow.entities.span import Span
from mlflow.genai.scorers import Guidelines, Safety, scorer

# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------


class ResponseOutput(TypedDict, total=False):
    """Expected output format with response content."""

    response: str


class NestedOutput(TypedDict, total=False):
    """Nested output format from some MLflow configurations."""

    outputs: ResponseOutput


# Union type for all possible output formats from MLflow
ScorerOutputs = Union[str, ResponseOutput, NestedOutput, dict[str, Any]]


# -----------------------------------------------------------------------------
# Helper for extracting response content
# -----------------------------------------------------------------------------


def _extract_response_content(outputs: ScorerOutputs) -> tuple[str, Optional[str]]:
    """
    Extract response content from various output formats.

    Args:
        outputs: Model outputs in any supported format

    Returns:
        Tuple of (content_string, error_message). If error_message is not None,
        content_string will be empty and the error should be returned.
    """
    if isinstance(outputs, str):
        return outputs, None

    if isinstance(outputs, dict):
        # Check for nested format first: {"outputs": {"response": "..."}}
        nested_outputs = outputs.get("outputs")
        if isinstance(nested_outputs, dict):
            content = nested_outputs.get("response", "")
            return str(content) if content else "", None

        # Flat format: {"response": "..."}
        content = outputs.get("response", "")
        return str(content) if content else "", None

    return "", f"Unexpected output type: {type(outputs).__name__}"


# -----------------------------------------------------------------------------
# Custom Scorers
# -----------------------------------------------------------------------------


@scorer
def response_completeness(outputs: ScorerOutputs) -> Feedback:
    """
    Evaluate if the response appears complete and meaningful.

    This scorer checks:
    - Response is not too short (< 10 characters)
    - Response doesn't end with incomplete markers

    Args:
        outputs: Model outputs - can be a string, dict with "response" key,
                 or nested dict with "outputs.response".

    Returns:
        Feedback: Pass/Fail feedback with rationale
    """
    content, error = _extract_response_content(outputs)

    if error:
        return Feedback(value=False, rationale=error)

    if not content:
        return Feedback(value=False, rationale="No response content found in outputs")

    if len(content.strip()) < 10:
        return Feedback(value=False, rationale="Response too short to be meaningful")

    incomplete_markers = ("...", "etc", "and so on", "to be continued")
    if content.lower().rstrip().endswith(incomplete_markers):
        return Feedback(value=False, rationale="Response appears incomplete")

    return Feedback(value=True, rationale="Response appears complete")


@scorer
def tool_call_efficiency(trace: Optional[Trace]) -> Feedback:
    """
    Evaluate how effectively the agent uses tools.

    This trace-based scorer checks:
    - Presence of tool calls
    - Redundant tool calls (same tool called multiple times)
    - Failed tool calls

    Args:
        trace: MLflow Trace object containing span information.
               May be None if tracing is not enabled.

    Returns:
        Feedback: Pass/Fail feedback with rationale, or error if trace unavailable
    """
    if trace is None:
        # MLflow 3.8+ requires error instead of value=None
        return Feedback(error=Exception("No trace available for tool call analysis"))

    try:
        # Retrieve all tool call spans from the trace
        tool_calls: list[Span] = trace.search_spans(span_type="TOOL")
    except Exception as e:
        logger.warning(f"Error searching trace spans: {e}")
        return Feedback(error=Exception(f"Error accessing trace spans: {str(e)}"))

    if not tool_calls:
        # No tools used is valid but not evaluable - return True with note
        return Feedback(
            value=True,
            rationale="No tool usage to evaluate - agent responded without tools",
        )

    # Check for redundant calls (same tool name called multiple times)
    tool_names: list[str] = [span.name for span in tool_calls]
    if len(tool_names) != len(set(tool_names)):
        # Count duplicates for better feedback
        duplicates = [name for name in set(tool_names) if tool_names.count(name) > 1]
        return Feedback(
            value=False, rationale=f"Redundant tool calls detected: {duplicates}"
        )

    # Check for failed tool calls using typed SpanStatus
    failed_calls: list[str] = []
    for span in tool_calls:
        span_status: SpanStatus = span.status
        if span_status.status_code != SpanStatusCode.OK:
            failed_calls.append(span.name)

    if failed_calls:
        return Feedback(
            value=False,
            rationale=f"{len(failed_calls)} tool calls failed: {failed_calls}",
        )

    return Feedback(
        value=True,
        rationale=f"Efficient tool usage: {len(tool_calls)} successful calls",
    )


# -----------------------------------------------------------------------------
# Response Clarity Scorer
# -----------------------------------------------------------------------------

# Instructions for the response clarity judge
RESPONSE_CLARITY_INSTRUCTIONS = """Evaluate the clarity and readability of the response in {{ outputs }}.

Consider:
- Is the response easy to understand?
- Is the information well-organized?
- Does it avoid unnecessary jargon or explain technical terms when used?
- Is the sentence structure clear and coherent?
- Does the response directly address what was asked in {{ inputs }}?

Return "clear" if the response is clear and readable, "unclear" if it is confusing or poorly structured."""


def create_response_clarity_scorer(
    judge_model: str,
    name: str = "response_clarity",
) -> Any:
    """
    Create a response clarity scorer using MLflow's make_judge.

    This scorer evaluates whether a response is clear, well-organized,
    and easy to understand. It uses an LLM judge for nuanced assessment
    of qualities like:
    - Sentence structure and coherence
    - Information organization
    - Appropriate use of technical language
    - Overall readability

    Args:
        judge_model: The model endpoint to use for evaluation.
                     Example: "databricks:/databricks-claude-3-7-sonnet"
        name: Name for this scorer instance.

    Returns:
        A judge scorer created by make_judge

    Example:
        ```python
        from dao_ai.evaluation import create_response_clarity_scorer

        # Create the scorer with a judge model
        clarity_scorer = create_response_clarity_scorer(
            judge_model="databricks:/databricks-claude-3-7-sonnet",
        )

        # Use in evaluation
        mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=[clarity_scorer],
        )
        ```
    """
    from mlflow.genai.judges import make_judge

    return make_judge(
        name=name,
        instructions=RESPONSE_CLARITY_INSTRUCTIONS,
        # No feedback_value_type - avoids response_schema parameter
        # which Databricks endpoints don't support
        model=judge_model,
    )


# -----------------------------------------------------------------------------
# Agent Routing Scorer
# -----------------------------------------------------------------------------

# Instructions for the agent routing judge
# Note: When using {{ trace }}, it must be the ONLY template variable per MLflow make_judge rules
AGENT_ROUTING_INSTRUCTIONS = """Evaluate whether the agent routing was appropriate for the user's request.

Analyze the {{ trace }} to determine:
1. What was the user's original query or request
2. Which agents, chains, or components were invoked to handle it
3. Whether the routing sequence was logical and appropriate

Consider:
- **Relevance**: Based on the names of the components invoked, do they seem appropriate for the question type?
- **Logical Flow**: Does the sequence of invocations make sense for answering the query?
- **Completeness**: Were the right types of components invoked to fully address the query?

Note: You may not know all possible agents in the system. Focus on whether the components that WERE invoked seem reasonable given the user's query found in the trace.

Return "appropriate" if routing was appropriate, "inappropriate" if it was clearly wrong."""


def create_agent_routing_scorer(
    judge_model: str,
    name: str = "agent_routing",
) -> Any:
    """
    Create an agent routing scorer using MLflow's make_judge.

    This scorer analyzes the execution trace to evaluate whether the routing
    decisions were appropriate for the user's query. It uses MLflow's built-in
    trace-based judge functionality.

    The scorer is general-purpose and does not require knowledge of specific
    agent names - it relies on the LLM to interpret whether the components
    invoked (based on their names and context) were suitable for the query.

    Args:
        judge_model: The model endpoint to use for evaluation.
                     Example: "databricks:/databricks-claude-3-7-sonnet"
        name: Name for this scorer instance.

    Returns:
        A judge scorer created by make_judge

    Example:
        ```python
        from dao_ai.evaluation import create_agent_routing_scorer

        # Create the scorer with a judge model
        agent_routing = create_agent_routing_scorer(
            judge_model="databricks:/databricks-claude-3-7-sonnet",
        )

        # Use in evaluation
        mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=[agent_routing],
        )
        ```
    """
    from mlflow.genai.judges import make_judge

    return make_judge(
        name=name,
        instructions=AGENT_ROUTING_INSTRUCTIONS,
        # No feedback_value_type - avoids response_schema parameter
        # which Databricks endpoints don't support
        model=judge_model,
    )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def create_traced_predict_fn(
    predict_callable: Callable[[dict[str, Any]], dict[str, Any]],
    span_name: str = "predict",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Wrap a predict function with MLflow tracing.

    This ensures traces are created for each prediction, allowing
    trace-based scorers to access span information.

    Args:
        predict_callable: The original prediction function
        span_name: Name for the trace span

    Returns:
        Wrapped function with MLflow tracing enabled
    """

    @mlflow.trace(name=span_name, span_type="CHAIN")
    def traced_predict(inputs: dict[str, Any]) -> dict[str, Any]:
        result = predict_callable(inputs)
        # Normalize output format to flat structure
        if "outputs" in result and isinstance(result["outputs"], dict):
            # Extract from nested format
            return result["outputs"]
        return result

    return traced_predict


def create_guidelines_scorers(
    guidelines_config: list[Any],
    judge_model: str,
) -> list[Guidelines]:
    """
    Create Guidelines scorers from configuration with proper judge model.

    Args:
        guidelines_config: List of guideline configurations with name and guidelines
        judge_model: The model endpoint to use for evaluation (e.g., "databricks:/model-name")

    Returns:
        List of configured Guidelines scorers
    """
    scorers = []
    for guideline in guidelines_config:
        scorer_instance = Guidelines(
            name=guideline.name,
            guidelines=guideline.guidelines,
            model=judge_model,
        )
        scorers.append(scorer_instance)
        logger.debug(
            f"Created Guidelines scorer: {guideline.name} with model {judge_model}"
        )

    return scorers


def get_default_scorers(
    include_trace_scorers: bool = True,
    include_agent_routing: bool = False,
    judge_model: Optional[str] = None,
) -> list[Any]:
    """
    Get the default set of scorers for evaluation.

    Args:
        include_trace_scorers: Whether to include trace-based scorers like tool_call_efficiency
        include_agent_routing: Whether to include the agent routing scorer
        judge_model: The model endpoint to use for LLM-based scorers (Safety, clarity, routing).
                     Example: "databricks-gpt-5-2"

    Returns:
        List of scorer instances
    """
    # Safety requires a judge model for LLM-based evaluation
    if judge_model:
        safety_scorer = Safety(model=judge_model)
    else:
        safety_scorer = Safety()
        logger.warning(
            "No judge_model provided for Safety scorer. "
            "This may cause errors if no default model is configured."
        )

    scorers: list[Any] = [
        safety_scorer,
        response_completeness,
    ]

    # TODO: Re-enable when Databricks endpoints support make_judge
    # if judge_model:
    #     scorers.append(create_response_clarity_scorer(judge_model=judge_model))

    if include_trace_scorers:
        scorers.append(tool_call_efficiency)

    # TODO: Re-enable when Databricks endpoints support make_judge
    # if include_agent_routing and judge_model:
    #     scorers.append(create_agent_routing_scorer(judge_model=judge_model))

    return scorers


def setup_evaluation_tracking(
    experiment_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> None:
    """
    Set up MLflow tracking for evaluation.

    Configures:
    - Registry URI to databricks-uc
    - Experiment context
    - Autologging for LangChain

    Args:
        experiment_id: Optional experiment ID to use
        experiment_name: Optional experiment name to use (creates if doesn't exist)
    """
    mlflow.set_registry_uri("databricks-uc")

    if experiment_id:
        mlflow.set_experiment(experiment_id=experiment_id)
    elif experiment_name:
        mlflow.set_experiment(experiment_name=experiment_name)

    # Enable autologging with trace support
    mlflow.langchain.autolog(log_traces=True)
    logger.debug("MLflow evaluation tracking configured")


def run_evaluation(
    data: Any,
    predict_fn: Callable,
    model_id: Optional[str] = None,
    scorers: Optional[list[Any]] = None,
    judge_model: Optional[str] = None,
    guidelines: Optional[list[Any]] = None,
) -> Any:
    """
    Run MLflow GenAI evaluation with proper configuration.

    This is a convenience wrapper around mlflow.genai.evaluate() that:
    - Wraps predict_fn with tracing
    - Configures default scorers
    - Sets up Guidelines with judge model

    Args:
        data: Evaluation dataset (DataFrame or list of dicts)
        predict_fn: Function to generate predictions
        model_id: Optional model ID for linking
        scorers: Optional list of scorers (uses defaults if not provided)
        judge_model: Model endpoint for LLM-based scorers
        guidelines: Optional list of guideline configurations

    Returns:
        EvaluationResult from mlflow.genai.evaluate()
    """
    # Wrap predict function with tracing
    traced_fn = create_traced_predict_fn(predict_fn)

    # Build scorer list
    if scorers is None:
        scorers = get_default_scorers(include_trace_scorers=True)

    # Add Guidelines scorers if provided
    if guidelines and judge_model:
        guideline_scorers = create_guidelines_scorers(guidelines, judge_model)
        scorers.extend(guideline_scorers)
    elif guidelines:
        logger.warning(
            "Guidelines provided but no judge_model specified - Guidelines scorers will not be created"
        )

    # Run evaluation
    eval_kwargs: dict[str, Any] = {
        "data": data,
        "predict_fn": traced_fn,
        "scorers": scorers,
    }

    if model_id:
        eval_kwargs["model_id"] = model_id

    return mlflow.genai.evaluate(**eval_kwargs)


def prepare_eval_results_for_display(eval_results: Any) -> Any:
    """
    Prepare evaluation results DataFrame for display in Databricks.

    The 'assessments' column and other complex object columns can't be
    directly converted to Arrow format for display. This function converts
    them to string representation.

    Args:
        eval_results: EvaluationResult from mlflow.genai.evaluate()

    Returns:
        DataFrame copy with complex columns converted to strings
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available, returning original results")
        return eval_results.tables.get("eval_results")

    results_df: pd.DataFrame = eval_results.tables["eval_results"].copy()

    # Convert complex columns to string for display compatibility
    if "assessments" in results_df.columns:
        results_df["assessments"] = results_df["assessments"].astype(str)

    # Convert any other object columns that might cause Arrow conversion issues
    for col in results_df.columns:
        if results_df[col].dtype == "object":
            try:
                # Try to keep as-is first, only convert if it fails
                results_df[col].to_list()
            except Exception:
                results_df[col] = results_df[col].astype(str)

    return results_df
