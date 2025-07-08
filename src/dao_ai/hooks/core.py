from typing import Any, Callable, Sequence

from loguru import logger

from dao_ai.config import AppConfig, FunctionHook, PythonFunctionModel


def create_hooks(
    function_hooks: FunctionHook | list[FunctionHook] | None,
) -> Sequence[Callable[..., Any]]:
    logger.debug(f"Creating hooks from: {function_hooks}")
    hooks: Sequence[Callable[..., Any]] = []
    if not function_hooks:
        return []
    if not isinstance(function_hooks, (list, tuple, set)):
        function_hooks = [function_hooks]
    for function_hook in function_hooks:
        if isinstance(function_hook, str):
            function_hook = PythonFunctionModel(name=function_hook)
        hooks.extend(function_hook.as_tools())
    logger.debug(f"Created hooks: {hooks}")
    return hooks


def null_hook(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    logger.debug("Executing null hook")
    return {}


def null_initialization_hook(config: AppConfig) -> None:
    logger.debug("Executing null initialization hook")


def null_shutdown_hook(config: AppConfig) -> None:
    logger.debug("Executing null shutdown hook")


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
  }
}
```

### Field Descriptions
- **user_id**: Your unique user identifier (required)
- **thread_id**: Conversation thread identifier (optional)

Please update your configuration and try again.
        """.strip()

        raise ValueError(error_message)

    return {}


def require_thread_id_hook(
    state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    logger.debug("Executing thread_id validation hook")

    config = config.get("custom_inputs", config)

    configurable: dict[str, Any] = config.get("configurable", {})

    if "thread_id" not in configurable or not configurable["thread_id"]:
        logger.error("Thread ID is required but not provided in the configuration.")

        error_message = """
## Authentication Required

A **thread_id** is required to process your request. Please provide your user ID in the configuration.

### Required Configuration Format

Please include the following JSON in your request configuration:

```json
{
  "configurable": {
    "thread_id": "1",
    "user_id": "my_user_id"
  }
}
```

### Field Descriptions
- **user_id**: Your unique user identifier (required)
- **thread_id**: Conversation thread identifier (optional)

Please update your configuration and try again.
        """.strip()

        raise ValueError(error_message)

    return {}
