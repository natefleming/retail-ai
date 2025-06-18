from typing import Any

from loguru import logger


def null_hook(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    logger.debug("Executing null hook")
    return {}


def require_user_id_hook(
    state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    logger.debug("Executing user_id validation hook")

    config = config.get("custom_inputs", config)

    configurable: dict[str, Any] = config.get("configurable", {})

    if "user_id" not in configurable or not configurable["user_id"]:
        logger.error("User ID is required but not provided in the configuration.")
        raise ValueError("User ID is required but not provided in the configuration.")

    return {}
