import json
from typing import Any, Callable, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langgraph.runtime import Runtime
from loguru import logger

from dao_ai.config import AppConfig, FunctionHook, PythonFunctionModel
from dao_ai.messages import last_human_message
from dao_ai.state import Context


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


def null_hook(state: dict[str, Any], runtime: Runtime[Context]) -> dict[str, Any]:
    logger.debug("Executing null hook")
    return {}


def null_initialization_hook(config: AppConfig) -> None:
    logger.debug("Executing null initialization hook")


def null_shutdown_hook(config: AppConfig) -> None:
    logger.debug("Executing null shutdown hook")


def filter_last_human_message_hook(
    state: dict[str, Any], runtime: Runtime[Context]
) -> dict[str, Any]:
    """
    Filter messages to keep only the last human message.

    This hook removes all messages except for the most recent human message,
    which can be useful for scenarios where you want to process only the
    latest user input without conversation history.

    Args:
        state: The current state containing messages
        runtime: The runtime context (unused in this hook)

    Returns:
        Updated state with filtered messages
    """
    logger.debug("Executing filter_last_human_message hook")

    messages: list[BaseMessage] = state.get("messages", [])

    if not messages:
        logger.debug("No messages found in state")
        return state

    # Use the helper function to find the last human message
    last_message: HumanMessage = last_human_message(messages)

    if last_message is None:
        logger.debug("No human messages found in state")
        # Return empty messages if no human message found
        updated_state = state.copy()
        updated_state["messages"] = []
        return updated_state

    logger.debug(f"Filtered {len(messages)} messages down to 1 (last human message)")

    removed_messages: Sequence[BaseMessage] = [
        RemoveMessage(id=message.id)
        for message in messages
        if message.id != last_message.id
    ]

    updated_state: dict[str, Sequence[BaseMessage]] = {"messages": removed_messages}

    return updated_state


def require_user_id_hook(
    state: dict[str, Any], runtime: Runtime[Context]
) -> dict[str, Any]:
    logger.debug("Executing user_id validation hook")

    context: Context = runtime.context or Context()

    user_id: str | None = context.user_id

    if not user_id:
        logger.error("User ID is required but not provided in the configuration.")

        # Create corrected configuration using any provided context parameters
        corrected_config = {
            "configurable": {
                "thread_id": context.thread_id or "1",
                "user_id": "my_user_id",
                "store_num": context.store_num or 87887,
            }
        }

        # Format as JSON for copy-paste
        corrected_config_json = json.dumps(corrected_config, indent=2)

        error_message = f"""
## Authentication Required

A **user_id** is required to process your request. Please provide your user ID in the configuration.

### Required Configuration Format

Please include the following JSON in your request configuration:

```json
{corrected_config_json}
```

### Field Descriptions
- **user_id**: Your unique user identifier (required)
- **thread_id**: Conversation thread identifier (required)
- **store_num**: Your store number (required)

Please update your configuration and try again.
        """.strip()

        raise ValueError(error_message)

    if "." in user_id:
        logger.error(f"User ID '{user_id}' contains invalid character '.'")

        # Create a corrected version of the user_id
        corrected_user_id = user_id.replace(".", "_")

        # Create corrected configuration for the error message

        # Corrected config with fixed user_id
        corrected_config = {
            "configurable": {
                "thread_id": context.thread_id or "1",
                "user_id": corrected_user_id,
                "store_num": context.store_num or 87887,
            }
        }

        # Format as JSON for copy-paste
        corrected_config_json = json.dumps(corrected_config, indent=2)

        error_message = f"""
## Invalid User ID Format

The **user_id** cannot contain a dot character ('.'). Please provide a valid user ID without dots.

### Corrected Configuration (Copy & Paste This)
```json
{corrected_config_json}
```

Please update your user_id and try again.
        """.strip()

        raise ValueError(error_message)

    return {}


def require_thread_id_hook(
    state: dict[str, Any], runtime: Runtime[Context]
) -> dict[str, Any]:
    logger.debug("Executing thread_id validation hook")

    context: Context = runtime.context or Context()

    thread_id: str | None = context.thread_id

    if not thread_id:
        logger.error("Thread ID is required but not provided in the configuration.")

        # Create corrected configuration using any provided context parameters
        corrected_config = {
            "configurable": {
                "thread_id": "1",
                "user_id": context.user_id or "my_user_id",
                "store_num": context.store_num or 87887,
            }
        }

        # Format as JSON for copy-paste
        corrected_config_json = json.dumps(corrected_config, indent=2)

        error_message = f"""
## Authentication Required

A **thread_id** is required to process your request. Please provide your thread ID in the configuration.

### Required Configuration Format

Please include the following JSON in your request configuration:

```json
{corrected_config_json}
```

### Field Descriptions
- **thread_id**: Conversation thread identifier (required)
- **user_id**: Your unique user identifier (required)
- **store_num**: Your store number (required)

Please update your configuration and try again.
        """.strip()

        raise ValueError(error_message)

    return {}
