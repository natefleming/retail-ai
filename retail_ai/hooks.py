from typing import Any

from loguru import logger


def null_hook(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    logger.debug("Executing null hook")
    return {}
