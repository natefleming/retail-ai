"""Logging configuration for DAO AI."""

import logging
import sys
from typing import Any

from loguru import logger

# Re-export logger for convenience
__all__ = ["logger", "configure_logging", "suppress_autolog_context_warnings"]


class _AutologWarningFilter(logging.Filter):
    """Drops two classes of noisy but harmless MLflow autolog warnings.

    1. ``was created in a different Context`` — MLflow's autologging emits
       these when it tries to reset a ContextVar token across async
       boundaries (nest_asyncio).

    2. ``Span for run_id ... not found`` — emitted by
       ``mlflow.langchain.langchain_tracer._get_span_by_run_id`` and
       swallowed by MLflow's autolog safety wrapper as an
       ``Encountered unexpected error during autologging: ...`` warning.
       This fires on every LLM call made from the memory-extraction
       background executor (langmem's ``LocalReflectionExecutor``): that
       executor internally uses ``get_executor_for_config(config)`` to
       spawn a nested thread pool for parallel ``store.search`` calls,
       and MLflow's autolog installs its tracer via a monkey-patch of
       ``BaseCallbackManager.__init__``. Each new callback manager on
       those sub-threads creates a fresh ``MlflowLangchainTracer`` whose
       instance-state ``_run_span_mapping`` doesn't have the parent
       tracer's run_ids. Passing ``callbacks=[]`` in the RunnableConfig
       does not silence this because autolog is injected at the callback
       manager level, not through the caller's config. And
       ``mlflow.utils.autologging_utils.disable_autologging`` toggles a
       process-global flag that would race with concurrent main-pipeline
       requests. So we filter the warning at the log-record level:
       memory-extraction still writes to the store correctly, and main
       pipeline traces continue to land in MLflow — the only thing this
       change silences is the log noise.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg: str = record.getMessage()
        if "was created in a different Context" in msg:
            return False
        if "Span for run_id" in msg and "not found" in msg:
            return False
        return True


class _SpanEntitiesFilter(logging.Filter):
    """Drops ``mlflow.entities.span: Failed to end span`` warnings.

    Same root cause as the ``Span for run_id ... not found`` filter above:
    a memory-extraction sub-worker tries to end a span whose parent tracer
    doesn't know about it. Non-fatal.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Failed to end span" not in record.getMessage()


def suppress_autolog_context_warnings() -> None:
    """Suppress noisy MLflow autolog warnings that occur on cross-thread paths.

    Call this after ``mlflow.langchain.autolog()`` in entry-point modules
    (e.g., ``model_serving.py``, ``handlers.py``).

    Silences:
    - ``mlflow.utils.autologging_utils``:
      - ``was created in a different Context`` (nest_asyncio ContextVar resets)
      - ``Span for run_id ... not found`` (memory-extraction background thread)
    - ``mlflow.entities.span``:
      - ``Failed to end span`` (same memory-extraction path)

    Defense-in-depth: the primary MLflow-side fix for the first warning is
    ``mlflow.langchain.autolog(run_tracer_inline=True)``, which keeps
    LangChain callbacks on the main async task so the active-span
    ``ContextVar`` doesn't get reset across thread-pool boundaries. The
    second/third warnings do not have a runtime fix — MLflow's autolog
    installs itself via a monkey-patch of ``BaseCallbackManager.__init__``
    and its tracer state is not ContextVar-based, so the memory-extraction
    background thread will always emit them. They are harmless: memory
    writes still succeed and main-pipeline traces are unaffected.
    """
    logging.getLogger("mlflow.utils.autologging_utils").addFilter(
        _AutologWarningFilter()
    )
    logging.getLogger("mlflow.entities.span").addFilter(_SpanEntitiesFilter())


def format_extra(record: dict[str, Any]) -> str:
    """Format extra fields as key=value pairs."""
    extra: dict[str, Any] = record["extra"]
    if not extra:
        return ""

    formatted_pairs: list[str] = []
    for key, value in extra.items():
        # Handle different value types
        if isinstance(value, str):
            formatted_pairs.append(f"{key}={value}")
        elif isinstance(value, (list, tuple)):
            formatted_pairs.append(f"{key}={','.join(str(v) for v in value)}")
        else:
            formatted_pairs.append(f"{key}={value}")

    return " | ".join(formatted_pairs)


def configure_logging(level: str = "INFO") -> None:
    """
    Configure loguru logging with structured output.

    Args:
        level: The log level (e.g., "INFO", "DEBUG", "WARNING")
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
            "{extra}"
        ),
    )

    # Add custom formatter for extra fields
    logger.configure(
        patcher=lambda record: record.update(
            extra=" | " + format_extra(record) if record["extra"] else ""
        )
    )
