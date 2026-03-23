"""Logging configuration for DAO AI."""

import logging
import sys
from typing import Any

from loguru import logger

# Re-export logger for convenience
__all__ = ["logger", "configure_logging", "suppress_autolog_context_warnings"]


class _ContextVarWarningFilter(logging.Filter):
    """Drops the noisy 'was created in a different Context' warnings.

    MLflow's autologging emits these when it tries to reset a ContextVar
    token across async boundaries (nest_asyncio). They are harmless but
    extremely noisy in Model Serving logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "was created in a different Context" not in record.getMessage()


def suppress_autolog_context_warnings() -> None:
    """Suppress ``mlflow.utils.autologging_utils`` ContextVar warnings.

    Call this after ``mlflow.langchain.autolog()`` in entry-point modules
    (e.g., ``model_serving.py``, ``handlers.py``).
    """
    logging.getLogger("mlflow.utils.autologging_utils").addFilter(
        _ContextVarWarningFilter()
    )


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
