"""Logging configuration for DAO AI."""

import sys
from typing import Any

from loguru import logger

# Re-export logger for convenience
__all__ = ["logger", "configure_logging"]


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
