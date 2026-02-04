"""
Tool call observability middleware for DAO AI agents.

This middleware logs detailed information about tool call patterns to help
diagnose whether tools are being called in parallel or sequentially.

Example YAML config::

    middleware:
      - name: dao_ai.middleware.tool_call_observability.\
              create_tool_call_observability_middleware
        args:
          log_level: INFO
          include_args: false
          track_timing: true
"""

from __future__ import annotations

import time
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.runtime import Runtime
from loguru import logger

from dao_ai.state import AgentState, Context

__all__ = [
    "ToolCallObservabilityMiddleware",
    "create_tool_call_observability_middleware",
]


class ToolCallObservabilityMiddleware(AgentMiddleware[AgentState, Context]):
    """
    Middleware that observes and logs tool call patterns.

    Tracks:
    - Number of tool calls per model response
    - Whether tools are called in parallel (multiple per response) or sequentially
    - Tool execution timing
    - Cumulative statistics across the conversation
    """

    def __init__(
        self,
        log_level: str = "INFO",
        include_args: bool = False,
        track_timing: bool = True,
    ):
        """
        Initialize the observability middleware.

        Args:
            log_level: Logging level ("DEBUG", "INFO", "WARNING")
            include_args: Whether to log tool call arguments (may be verbose)
            track_timing: Whether to track execution timing
        """
        self.log_level = log_level.upper()
        self.include_args = include_args
        self.track_timing = track_timing

        # Statistics tracking (per-run, reset on before_agent)
        self._total_model_calls = 0
        self._total_tool_calls = 0
        self._parallel_batches = 0  # Responses with 2+ tool calls
        self._sequential_calls = 0  # Responses with exactly 1 tool call
        self._tool_execution_times: dict[str, list[float]] = {}
        self._run_start_time: float | None = None

    def _log(self, message: str, **kwargs: Any) -> None:
        """Log at the configured level."""
        log_fn = getattr(logger, self.log_level.lower(), logger.info)
        log_fn(message, **kwargs)

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime[Context],
    ) -> dict[str, Any] | None:
        """Reset statistics at the start of each agent run."""
        self._total_model_calls = 0
        self._total_tool_calls = 0
        self._parallel_batches = 0
        self._sequential_calls = 0
        self._tool_execution_times = {}
        self._run_start_time = time.time()

        self._log(
            "Tool call observability: Agent run started",
            thread_id=runtime.context.thread_id if runtime.context else None,
        )
        return None

    def after_model(
        self,
        state: AgentState,
        runtime: Runtime[Context],
    ) -> dict[str, Any] | None:
        """Analyze tool calls in model response."""
        self._total_model_calls += 1

        # Extract tool calls from state messages (last message is model response)
        messages: list[BaseMessage] = state.get("messages", [])

        # Look at the last message which should be the model's response
        for msg in messages[-1:]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                num_tool_calls = len(msg.tool_calls)
                self._total_tool_calls += num_tool_calls

                if num_tool_calls > 1:
                    self._parallel_batches += 1
                    self._log(
                        "PARALLEL tool calls detected",
                        num_tools=num_tool_calls,
                        tool_names=[tc["name"] for tc in msg.tool_calls],
                        model_call_number=self._total_model_calls,
                    )

                    if self.include_args:
                        for tc in msg.tool_calls:
                            self._log(
                                f"  Tool: {tc['name']}",
                                args=tc.get("args", {}),
                            )
                elif num_tool_calls == 1:
                    self._sequential_calls += 1
                    tc = msg.tool_calls[0]
                    log_kwargs: dict[str, Any] = {
                        "tool_name": tc["name"],
                        "model_call_number": self._total_model_calls,
                    }
                    if self.include_args:
                        log_kwargs["args"] = tc.get("args", {})
                    self._log("Sequential tool call", **log_kwargs)

        return None

    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime[Context],
    ) -> dict[str, Any] | None:
        """Log final statistics."""
        total_time = time.time() - self._run_start_time if self._run_start_time else 0

        # Calculate parallelism ratio
        total_responses_with_tools = self._parallel_batches + self._sequential_calls
        parallelism_ratio = (
            self._parallel_batches / total_responses_with_tools * 100
            if total_responses_with_tools > 0
            else 0
        )

        # Calculate average tool times
        avg_times = {
            name: round(sum(times) / len(times) * 1000, 2)
            for name, times in self._tool_execution_times.items()
        }

        self._log(
            "Tool Call Observability Summary",
            total_model_calls=self._total_model_calls,
            total_tool_calls=self._total_tool_calls,
            parallel_batches=self._parallel_batches,
            sequential_calls=self._sequential_calls,
            parallelism_ratio=f"{parallelism_ratio:.1f}%",
            total_time_ms=round(total_time * 1000, 2),
            avg_tool_times_ms=avg_times,
        )

        # Log verdict
        if self._parallel_batches > 0:
            logger.success(
                f"Parallel tool calling IS happening: "
                f"{self._parallel_batches} batches with multiple tools"
            )
        elif self._sequential_calls > 0:
            logger.warning(
                f"All tool calls are SEQUENTIAL: "
                f"{self._sequential_calls} single-tool responses. "
                f"Consider prompt engineering to encourage parallel calls."
            )

        return None


def create_tool_call_observability_middleware(
    log_level: str = "INFO",
    include_args: bool = False,
    track_timing: bool = True,
) -> ToolCallObservabilityMiddleware:
    """
    Factory function to create tool call observability middleware.

    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING")
        include_args: Whether to log tool call arguments
        track_timing: Whether to track execution timing

    Returns:
        ToolCallObservabilityMiddleware instance

    Example YAML config::

        middleware:
          - name: dao_ai.middleware.tool_call_observability.\
                  create_tool_call_observability_middleware
            args:
              log_level: INFO
              include_args: false
              track_timing: true
    """
    logger.debug(
        "Creating tool call observability middleware",
        log_level=log_level,
        include_args=include_args,
        track_timing=track_timing,
    )
    return ToolCallObservabilityMiddleware(
        log_level=log_level,
        include_args=include_args,
        track_timing=track_timing,
    )
