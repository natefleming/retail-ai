import json
from typing import Any

from langgraph.runtime import Runtime
from loguru import logger

from dao_ai.state import Context


def require_store_num_hook(
    state: dict[str, Any], runtime: Runtime[Context]
) -> dict[str, Any]:
    logger.debug("Executing validation hook for required fields")

    context: Context = runtime.context or Context()

    # Check for missing required fields
    thread_id: str | None = context.thread_id
    user_id: str | None = context.user_id
    store_num: int | None = context.store_num

    required_fields = []
    if not thread_id:
        required_fields.append("thread_id")
    if not user_id:
        required_fields.append("user_id")
    if not store_num:
        required_fields.append("store_num")

    if required_fields:
        logger.error(f"Required fields are missing: {', '.join(required_fields)}")

        # Create corrected configuration using any provided context parameters
        corrected_config = {
            "configurable": {
                "thread_id": thread_id or "1",
                "user_id": user_id or "my_user_id",
                "store_num": store_num or 87887,
            }
        }

        # Format as JSON for copy-paste
        corrected_config_json = json.dumps(corrected_config, indent=2)

        error_message = f"""
## Authentication Required

The following required fields are missing: **{", ".join(required_fields)}**

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
