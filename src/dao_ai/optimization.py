"""
Prompt optimization using GEPA (Generative Evolution of Prompts and Agents).

This module provides prompt optimization for DAO AI agents using the GEPA
optimizer, which uses reflective mutation to evolve prompts based on
evaluation feedback.

GEPA is an evolutionary optimizer that:
1. Takes a seed prompt (initial template)
2. Evaluates it against training examples
3. Uses a reflection LM to propose improvements
4. Iteratively evolves the prompt to maximize the metric

Usage:
    from dao_ai.optimization import optimize_prompt

    result = optimize_prompt(
        prompt=my_prompt_model,
        agent=my_agent_model,
        dataset=my_training_dataset,
        num_candidates=50,
    )

    if result.improved:
        print(f"Improved by {result.improvement:.1%}")
        print(f"New template: {result.optimized_template}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence, Union

import mlflow
from gepa import EvaluationBatch, GEPAAdapter, GEPAResult, optimize
from loguru import logger
from mlflow.entities.model_registry import PromptVersion
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.types.responses_helpers import Message

from dao_ai.config import (
    AgentModel,
    ChatPayload,
    EvaluationDatasetEntryModel,
    EvaluationDatasetModel,
    PromptModel,
)
from dao_ai.utils import dao_ai_version

# Type alias for metric function
MetricFn = Callable[[str, "_TrainingExample"], float]

__all__ = [
    "OptimizationResult",
    "optimize_prompt",
]


@dataclass
class OptimizationResult:
    """Result of prompt optimization.

    Attributes:
        optimized_prompt: The optimized PromptModel with new template
        optimized_template: The optimized template string
        original_score: Score of the original prompt
        optimized_score: Score of the optimized prompt
        improvement: Percentage improvement
        num_evaluations: Number of metric evaluations performed
        registered_version: MLflow prompt version if registered
        metadata: Additional optimization metadata
    """

    optimized_prompt: PromptModel
    optimized_template: str
    original_score: float
    optimized_score: float
    improvement: float
    num_evaluations: int
    registered_version: Optional[PromptVersion] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        """Whether the optimization improved the prompt."""
        return self.optimized_score > self.original_score


@dataclass
class _TrainingExample:
    """Internal training example format for GEPA."""

    question: str
    expected_facts: Optional[list[str]] = None
    expected_response: Optional[str] = None
    custom_inputs: Optional[dict[str, Any]] = None


@dataclass
class _Trajectory:
    """Trajectory data for reflection."""

    question: str
    response: str
    expected: Any
    score: float
    error: Optional[str] = None


class DAOAgentAdapter(GEPAAdapter[_TrainingExample, _Trajectory, str]):
    """GEPA adapter for DAO AI agents.

    This adapter bridges GEPA's optimization loop with DAO AI's
    ResponsesAgent interface.
    """

    agent_model: AgentModel
    metric_fn: MetricFn
    _agent: Optional[Any]
    _original_prompt: Optional[Union[PromptModel, str]]

    def __init__(
        self,
        agent_model: AgentModel,
        metric_fn: Optional[MetricFn] = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            agent_model: The DAO AI agent model to optimize
            metric_fn: Optional custom metric function (response, example) -> score
        """
        self.agent_model = agent_model
        self.metric_fn = metric_fn or self._default_metric
        self._agent = None
        self._original_prompt = None

    def _get_agent(self) -> Any:
        """Lazily create the ResponsesAgent.

        Returns:
            The ResponsesAgent instance for the configured agent model.
        """
        if self._agent is None:
            self._agent = self.agent_model.as_responses_agent()
        return self._agent

    def _default_metric(self, response: str, example: _TrainingExample) -> float:
        """Default metric: check if expected facts are present in response."""
        if example.expected_facts:
            facts_found = sum(
                1 for fact in example.expected_facts if fact.lower() in response.lower()
            )
            return facts_found / len(example.expected_facts)
        elif example.expected_response:
            expected_words = set(example.expected_response.lower().split())
            response_words = set(response.lower().split())
            overlap = len(expected_words & response_words)
            return overlap / len(expected_words) if expected_words else 0.0
        return 0.0

    def evaluate(
        self,
        batch: list[_TrainingExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[_Trajectory, str]:
        """Evaluate a candidate prompt on a batch of examples.

        Args:
            batch: List of training examples to evaluate
            candidate: Dict mapping component names to text (e.g., {"prompt": "..."})
            capture_traces: Whether to capture trajectories for reflection

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories
        """
        prompt_template = candidate.get("prompt", "")

        # Create agent with the candidate prompt
        original_prompt = self.agent_model.prompt
        try:
            # Update agent's prompt template
            if isinstance(original_prompt, PromptModel):
                self.agent_model.prompt = PromptModel(
                    name=original_prompt.name,
                    schema=original_prompt.schema_model,
                    default_template=prompt_template,
                    description=original_prompt.description,
                    tags=original_prompt.tags,
                )
            else:
                self.agent_model.prompt = prompt_template

            # Recreate agent with new prompt
            self._agent = None
            agent = self._get_agent()

            outputs: list[str] = []
            scores: list[float] = []
            trajectories: list[_Trajectory] = []

            for example in batch:
                try:
                    # Build request
                    messages = [Message(role="user", content=example.question)]
                    request = ResponsesAgentRequest(
                        input=messages,
                        custom_inputs=example.custom_inputs or {},
                    )

                    # Get response
                    response: ResponsesAgentResponse = agent.predict(request)

                    # Extract response text
                    response_text = ""
                    if response.output and len(response.output) > 0:
                        content = response.output[0].content
                        if isinstance(content, str):
                            response_text = content
                        elif isinstance(content, list):
                            response_text = "".join(
                                item.get("text", str(item))
                                if isinstance(item, dict)
                                else str(item)
                                for item in content
                            )
                        else:
                            response_text = str(content)

                    # Calculate score
                    score = self.metric_fn(response_text, example)

                    outputs.append(response_text)
                    scores.append(score)

                    if capture_traces:
                        trajectories.append(
                            _Trajectory(
                                question=example.question,
                                response=response_text,
                                expected=example.expected_facts
                                or example.expected_response,
                                score=score,
                            )
                        )

                except Exception as e:
                    logger.warning("Error evaluating example", error=str(e))
                    outputs.append("")
                    scores.append(0.0)

                    if capture_traces:
                        trajectories.append(
                            _Trajectory(
                                question=example.question,
                                response="",
                                expected=example.expected_facts
                                or example.expected_response,
                                score=0.0,
                                error=str(e),
                            )
                        )

            return EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=trajectories if capture_traces else None,
            )

        finally:
            # Restore original prompt
            self.agent_model.prompt = original_prompt
            self._agent = None

    def make_reflective_dataset(
        self,
        batch: list[_TrainingExample],
        trajectories: list[_Trajectory],
        component_name: str,
    ) -> list[dict[str, str]]:
        """Create a reflective dataset for the optimizer.

        Args:
            batch: Original batch of examples
            trajectories: Trajectories from evaluation
            component_name: Name of component to reflect on

        Returns:
            List of dicts with inputs, outputs, and feedback
        """
        reflective_data: list[dict[str, str]] = []

        for example, trajectory in zip(batch, trajectories):
            feedback_parts: list[str] = []
            feedback_parts.append(f"Input: {trajectory.question}")
            feedback_parts.append(f"Output: {trajectory.response[:500]}")
            feedback_parts.append(f"Expected: {trajectory.expected}")
            feedback_parts.append(f"Score: {trajectory.score:.2f}")

            if trajectory.score < 1.0 and example.expected_facts:
                missing = [
                    f
                    for f in example.expected_facts
                    if f.lower() not in trajectory.response.lower()
                ]
                if missing:
                    feedback_parts.append(f"Missing facts: {missing}")

            if trajectory.error:
                feedback_parts.append(f"Error: {trajectory.error}")

            reflective_data.append(
                {
                    "input": trajectory.question,
                    "output": trajectory.response,
                    "feedback": "\n".join(feedback_parts),
                }
            )

        return reflective_data


def _convert_dataset(
    dataset: EvaluationDatasetModel | Sequence[EvaluationDatasetEntryModel],
) -> list[_TrainingExample]:
    """Convert DAO dataset to internal training examples.

    Args:
        dataset: EvaluationDatasetModel or list of entries

    Returns:
        List of training examples
    """
    entries: Sequence[EvaluationDatasetEntryModel]
    if isinstance(dataset, EvaluationDatasetModel):
        entries = dataset.data
    else:
        entries = dataset

    examples: list[_TrainingExample] = []

    for entry in entries:
        payload: ChatPayload = entry.inputs
        messages = payload.messages

        # Get the user's question from messages
        question = ""
        for msg in messages:
            if msg.role == "user":
                question = msg.content
                break

        example = _TrainingExample(
            question=question,
            expected_facts=entry.expectations.expected_facts
            if entry.expectations
            else None,
            expected_response=entry.expectations.expected_response
            if entry.expectations
            else None,
            custom_inputs=payload.custom_inputs,
        )
        examples.append(example)

    logger.debug(
        "Converted dataset entries to training examples", examples_count=len(examples)
    )
    return examples


def _register_optimized_prompt(
    prompt: PromptModel,
    optimized_template: str,
    improvement: float,
    original_score: float,
    optimized_score: float,
    model_name: str,
    agent_name: str,
    num_evaluations: int,
    train_size: int,
    val_size: int,
) -> PromptVersion:
    """Register the optimized prompt in MLflow.

    Args:
        prompt: Original prompt model
        optimized_template: Optimized template string
        improvement: Improvement percentage
        original_score: Original evaluation score
        optimized_score: Optimized evaluation score
        model_name: Model used for reflection/optimization
        agent_name: Name of the agent being optimized
        num_evaluations: Number of metric evaluations performed
        train_size: Size of training dataset
        val_size: Size of validation dataset

    Returns:
        Registered PromptVersion
    """
    mlflow.set_registry_uri("databricks-uc")

    prompt_name: str = prompt.full_name
    optimization_timestamp: str = datetime.now(timezone.utc).isoformat()

    logger.info("Registering optimized prompt", prompt_name=prompt_name)

    # Build comprehensive tags for the prompt registry
    tags: dict[str, str] = {
        # DAO AI metadata
        "dao_ai_version": dao_ai_version(),
        "created_by": "dao_ai.optimization",
        # Optimization metadata
        "optimizer": "gepa",
        "optimization_timestamp": optimization_timestamp,
        "target_model": model_name,
        "target_agent": agent_name,
        # Performance metrics
        "original_score": f"{original_score:.4f}",
        "optimized_score": f"{optimized_score:.4f}",
        "improvement": f"{improvement:.4f}",
        "improvement_percent": f"{improvement:.1%}",
        # Dataset info
        "num_evaluations": str(num_evaluations),
        "train_size": str(train_size),
        "val_size": str(val_size),
    }

    # Preserve original prompt tags if present
    if prompt.tags:
        for key, value in prompt.tags.items():
            if key not in tags:  # Don't override optimization tags
                tags[f"original_{key}"] = str(value)

    # Register new version with comprehensive metadata
    version: PromptVersion = mlflow.genai.register_prompt(
        name=prompt_name,
        template=optimized_template,
        commit_message=(
            f"Optimized with GEPA for agent '{agent_name}' "
            f"(improvement: {improvement:.1%}, "
            f"score: {original_score:.3f} -> {optimized_score:.3f}, "
            f"model: {model_name})"
        ),
        tags=tags,
    )

    logger.success(
        "Registered optimized prompt version",
        prompt_name=prompt_name,
        version=version.version,
    )

    # Set 'latest' alias for most recently optimized version
    mlflow.genai.set_prompt_alias(
        name=prompt_name,
        alias="latest",
        version=version.version,
    )
    logger.info("Set 'latest' alias", prompt_name=prompt_name, version=version.version)

    # Set 'champion' alias if there was actual improvement
    if improvement > 0:
        mlflow.genai.set_prompt_alias(
            name=prompt_name,
            alias="champion",
            version=version.version,
        )
        logger.success(
            "Set 'champion' alias", prompt_name=prompt_name, version=version.version
        )

    return version


def optimize_prompt(
    prompt: PromptModel,
    agent: AgentModel,
    dataset: EvaluationDatasetModel | Sequence[EvaluationDatasetEntryModel],
    reflection_model: Optional[str] = None,
    num_candidates: int = 50,
    metric: Optional[Callable[[str, _TrainingExample], float]] = None,
    register_if_improved: bool = True,
    min_improvement: float = 0.0,
) -> OptimizationResult:
    """
    Optimize a prompt using GEPA.

    GEPA (Generative Evolution of Prompts and Agents) is an evolutionary
    optimizer that uses reflective mutation to improve prompts based on
    evaluation feedback.

    Args:
        prompt: The PromptModel to optimize
        agent: The AgentModel that uses this prompt
        dataset: Training data for optimization
        reflection_model: LLM for reflection (defaults to agent's model)
        num_candidates: Maximum metric calls / candidate evaluations
        metric: Optional custom metric function (response, example) -> score
        register_if_improved: Register optimized prompt in MLflow if improved
        min_improvement: Minimum improvement required to register

    Returns:
        OptimizationResult with optimization details

    Example:
        from dao_ai.config import AgentModel, PromptModel, LLMModel
        from dao_ai.optimization import optimize_prompt

        prompt = PromptModel(
            name="my_prompt",
            default_template="Answer the question: {question}"
        )
        agent = AgentModel(
            name="my_agent",
            model=LLMModel(name="databricks-meta-llama-3-3-70b-instruct"),
            prompt=prompt,
        )

        result = optimize_prompt(
            prompt=prompt,
            agent=agent,
            dataset=training_data,
            num_candidates=50,
        )

        if result.improved:
            print(f"Improved by {result.improvement:.1%}")
    """
    logger.info("Starting GEPA optimization", prompt_name=prompt.name)

    # Get the original template
    original_template = prompt.template
    if not original_template:
        raise ValueError(f"Prompt '{prompt.name}' has no template to optimize")

    # Convert dataset
    examples = _convert_dataset(dataset)
    if not examples:
        raise ValueError("Dataset is empty")

    # Split into train/val
    split_idx = max(1, len(examples) * 4 // 5)
    trainset = examples[:split_idx]
    valset = examples[split_idx:] if split_idx < len(examples) else examples

    logger.info("Dataset split", train_size=len(trainset), val_size=len(valset))

    # Get reflection model
    reflection_model_name = reflection_model or agent.model.uri
    logger.info("Using reflection model", model=reflection_model_name)

    # Create adapter
    adapter = DAOAgentAdapter(agent_model=agent, metric_fn=metric)

    # Seed candidate
    seed_candidate = {"prompt": original_template}

    # Run GEPA optimization
    logger.info("Running GEPA optimization", max_evaluations=num_candidates)

    try:
        result: GEPAResult = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_model_name,
            max_metric_calls=num_candidates,
            display_progress_bar=True,
            skip_perfect_score=True,
        )
    except Exception as e:
        logger.error("GEPA optimization failed", error=str(e))
        return OptimizationResult(
            optimized_prompt=prompt,
            optimized_template=original_template,
            original_score=0.0,
            optimized_score=0.0,
            improvement=0.0,
            num_evaluations=0,
            metadata={"error": str(e)},
        )

    # Extract results from GEPAResult
    # GEPAResult has:
    # - candidates: list of candidate dicts
    # - val_aggregate_scores: list of scores (index 0 is seed)
    # - best_idx: index of best candidate
    # - best_candidate: dict for best candidate
    # - total_metric_calls: number of metric evaluations
    best_candidate: dict[str, str] = result.best_candidate
    optimized_template: str = best_candidate.get("prompt", original_template)

    # Get scores from result - val_aggregate_scores[0] is the seed candidate score
    val_scores: list[float] = result.val_aggregate_scores
    original_score: float = val_scores[0] if val_scores else 0.0
    best_idx: int = result.best_idx
    optimized_score: float = val_scores[best_idx] if val_scores else 0.0
    num_evaluations: int = result.total_metric_calls or num_candidates

    improvement: float = (
        (optimized_score - original_score) / original_score
        if original_score > 0
        else 0.0
    )

    logger.success(
        "Optimization complete",
        original_score=f"{original_score:.3f}",
        optimized_score=f"{optimized_score:.3f}",
        improvement=f"{improvement:.1%}",
    )

    # Register if improved
    registered_version: Optional[PromptVersion] = None
    if (
        register_if_improved
        and improvement >= min_improvement
        and optimized_score > original_score
        and optimized_template != original_template
    ):
        try:
            registered_version = _register_optimized_prompt(
                prompt=prompt,
                optimized_template=optimized_template,
                improvement=improvement,
                original_score=original_score,
                optimized_score=optimized_score,
                model_name=reflection_model_name,
                agent_name=agent.name,
                num_evaluations=num_evaluations,
                train_size=len(trainset),
                val_size=len(valset),
            )
        except Exception as e:
            logger.error("Failed to register optimized prompt", error=str(e))

    # Build optimized prompt model with comprehensive tags
    optimized_tags: dict[str, str] = {
        **(prompt.tags or {}),
        "dao_ai_version": dao_ai_version(),
        "optimizer": "gepa",
        "target_model": reflection_model_name,
        "target_agent": agent.name,
        "original_score": f"{original_score:.4f}",
        "optimized_score": f"{optimized_score:.4f}",
        "improvement": f"{improvement:.4f}",
        "num_evaluations": str(num_evaluations),
    }

    optimized_prompt = PromptModel(
        name=prompt.name,
        schema=prompt.schema_model,
        default_template=optimized_template,
        description=f"Optimized with GEPA for agent '{agent.name}' (improvement: {improvement:.1%})",
        alias="champion" if improvement > min_improvement else "latest",
        tags=optimized_tags,
    )

    return OptimizationResult(
        optimized_prompt=optimized_prompt,
        optimized_template=optimized_template,
        original_score=original_score,
        optimized_score=optimized_score,
        improvement=improvement,
        num_evaluations=num_evaluations,
        registered_version=registered_version,
        metadata={
            "optimizer": "gepa",
            "reflection_model": reflection_model_name,
            "train_size": len(trainset),
            "val_size": len(valset),
        },
    )
