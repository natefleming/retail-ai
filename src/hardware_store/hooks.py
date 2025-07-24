import json
from typing import Any

from loguru import logger


def require_store_num_hook(
    state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    logger.debug("Executing validation hook for required fields")

    config = config.get("custom_inputs", config)

    configurable: dict[str, Any] = config.get("configurable", {})

    # Check for missing required fields
    required_fields = ["thread_id", "user_id", "store_num"]
    missing_fields = []

    for field in required_fields:
        if field not in configurable or not configurable[field]:
            missing_fields.append(field)

    if missing_fields:
        logger.error(f"Required fields are missing: {', '.join(missing_fields)}")

        corrected_config = {
            "configurable": {
                "thread_id": configurable.get("thread_id", "1"),
                "user_id": configurable.get("user_id", "my_user_id"),
                "store_num": configurable.get("store_num", 87887),
            }
        }

        # Format as JSON for copy-paste
        corrected_config_json = json.dumps(corrected_config, indent=2)

        error_message = f"""
## Authentication Required

The following required fields are missing: **{", ".join(missing_fields)}**

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
