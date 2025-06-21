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

        error_message = """
## Authentication Required

A **user_id** is required to process your request. Please provide your user ID in the configuration.

### Required Configuration Format

Please include the following JSON in your request configuration:

```json
{
  "configurable": {
    "thread_id": "1",
    "user_id": "my_user_id", 
    "store_num": 87887
  }
}
```

### Field Descriptions
- **user_id**: Your unique user identifier (required)
- **thread_id**: Conversation thread identifier (optional)
- **store_num**: Store number for retail context (optional)

Please update your configuration and try again.
        """.strip()

        raise ValueError(error_message)

    return {}
