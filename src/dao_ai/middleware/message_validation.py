"""
Message validation middleware for DAO AI agents.

These middleware implementations validate incoming messages and context
before agent processing begins.

Factory functions are provided for consistent configuration via the
DAO AI middleware factory pattern.
"""

import json
from typing import Any

from langchain.agents.middleware import hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.runtime import Runtime
from loguru import logger

from dao_ai.messages import last_human_message
from dao_ai.middleware.base import AgentMiddleware
from dao_ai.state import AgentState, Context

__all__ = [
    "MessageValidationMiddleware",
    "UserIdValidationMiddleware",
    "ThreadIdValidationMiddleware",
    "CustomFieldValidationMiddleware",
    "RequiredField",
    "FilterLastHumanMessageMiddleware",
    "create_user_id_validation_middleware",
    "create_thread_id_validation_middleware",
    "create_custom_field_validation_middleware",
    "create_filter_last_human_message_middleware",
]


class MessageValidationMiddleware(AgentMiddleware[AgentState, Context]):
    """
    Base middleware for message validation.

    Subclasses should implement the validate method to perform
    specific validation logic.
    """

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Validate messages before agent processing."""
        try:
            return self.validate(state, runtime)
        except ValueError as e:
            logger.error("Message validation failed", error=str(e))
            return {
                "is_valid": False,
                "message_error": str(e),
                "messages": [AIMessage(content=str(e))],
                "jump_to": "end",
            }

    def validate(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """
        Perform validation logic.

        Override this method in subclasses to implement specific validation.
        Raise ValueError to indicate validation failure.

        Args:
            state: The current agent state
            runtime: The LangGraph runtime context

        Returns:
            Optional dict with state updates

        Raises:
            ValueError: If validation fails
        """
        return None


class UserIdValidationMiddleware(MessageValidationMiddleware):
    """
    Middleware that validates the presence and format of user_id.

    Ensures that:
    - user_id is provided in the context
    - user_id does not contain invalid characters (like dots)
    """

    def validate(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Validate user_id is present and properly formatted."""
        logger.trace("Executing user_id validation")

        context: Context = runtime.context or Context()
        user_id: str | None = context.user_id

        if not user_id:
            logger.error("User ID is required but not provided in configuration")

            thread_val = context.thread_id or "<your_thread_id>"
            # Get extra fields from context (excluding user_id and thread_id)
            context_dict = context.model_dump()
            extra_fields = {
                k: v
                for k, v in context_dict.items()
                if k not in {"user_id", "thread_id"} and v is not None
            }

            corrected_config: dict[str, Any] = {
                "configurable": {
                    "thread_id": thread_val,
                    "user_id": "<your_user_id>",
                    **extra_fields,
                },
                "session": {
                    "conversation_id": thread_val,
                },
            }
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
- **thread_id**: Thread identifier (required in configurable)
- **conversation_id**: Alias of thread_id (in session)
- **user_id**: Your unique user identifier (required)

Please update your configuration and try again.
            """.strip()

            raise ValueError(error_message)

        if "." in user_id:
            logger.error("User ID contains invalid character '.'", user_id=user_id)

            corrected_user_id = user_id.replace(".", "_")
            thread_val = context.thread_id or "<your_thread_id>"
            # Get extra fields from context (excluding user_id and thread_id)
            context_dict = context.model_dump()
            extra_fields = {
                k: v
                for k, v in context_dict.items()
                if k not in {"user_id", "thread_id"} and v is not None
            }

            corrected_config: dict[str, Any] = {
                "configurable": {
                    "thread_id": thread_val,
                    "user_id": corrected_user_id,
                    **extra_fields,
                },
                "session": {
                    "conversation_id": thread_val,
                },
            }
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

        return None


class ThreadIdValidationMiddleware(MessageValidationMiddleware):
    """
    Middleware that validates the presence of thread_id/conversation_id.

    Note: thread_id and conversation_id are interchangeable in configurable.
    """

    def validate(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Validate thread_id/conversation_id is present."""
        logger.trace("Executing thread_id/conversation_id validation")

        context: Context = runtime.context or Context()
        thread_id: str | None = context.thread_id

        if not thread_id:
            logger.error("Thread ID / Conversation ID is required but not provided")

            # Get extra fields from context (excluding user_id and thread_id)
            context_dict = context.model_dump()
            extra_fields = {
                k: v
                for k, v in context_dict.items()
                if k not in {"user_id", "thread_id"} and v is not None
            }

            corrected_config: dict[str, Any] = {
                "configurable": {
                    "thread_id": "<your_thread_id>",
                    "user_id": context.user_id or "<your_user_id>",
                    **extra_fields,
                },
                "session": {
                    "conversation_id": "<your_thread_id>",
                },
            }
            corrected_config_json = json.dumps(corrected_config, indent=2)

            error_message = f"""
## Configuration Required

A **thread_id** is required to process your request (or **conversation_id** as an alias).

### Required Configuration Format

Please include the following JSON in your request configuration:

```json
{corrected_config_json}
```

### Field Descriptions
- **thread_id**: Thread identifier (required in configurable)
- **conversation_id**: Alias of thread_id (in session)
- **user_id**: Your unique user identifier (required)

Please update your configuration and try again.
            """.strip()

            raise ValueError(error_message)

        return None


class RequiredField:
    """Definition of a field for validation.

    Fields are marked as required or optional via the `required` flag:
    - required=True (default): Field must be provided, validated
    - required=False: Field is optional, not validated

    For required fields, an `example_value` can be provided to show in error
    messages, making it easy for users to copy-paste the configuration.

    Args:
        name: The field name (e.g., "store_num", "user_id")
        description: Human-readable description for error messages
        required: Whether this field is required (default: True)
        example_value: Example value to show in error messages for missing fields
    """

    def __init__(
        self,
        name: str,
        description: str | None = None,
        required: bool = True,
        example_value: Any = None,
    ):
        self.name = name
        self.description = description or f"Your {name}"
        self.required = required
        self.example_value = example_value

    @property
    def is_required(self) -> bool:
        """A field is required based on the required flag."""
        return self.required


class CustomFieldValidationMiddleware(MessageValidationMiddleware):
    """
    Middleware that validates the presence of required custom fields.

    This is a generic validation middleware that can check for multiple
    required fields in the context object.

    Fields are defined in the `fields` list. Each field can have:
    - name: The field name (required)
    - description: Human-readable description for error messages
    - required: Whether field is required (default: True)
    - example_value: Example value to show in error messages

    Required fields (required=True) will be validated.
    The example_value is used in error messages to help users copy-paste
    the correct configuration format.

    Args:
        fields: List of fields to validate/show. Each can be a RequiredField
            or a dict with 'name', 'description', 'required', and 'example_value' keys.
    """

    def __init__(
        self,
        fields: list[RequiredField | dict[str, Any]],
    ):
        super().__init__()

        # Convert fields to RequiredField objects
        self.fields: list[RequiredField] = []
        for field in fields:
            if isinstance(field, RequiredField):
                self.fields.append(field)
            elif isinstance(field, dict):
                self.fields.append(RequiredField(**field))

    def validate(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Validate that all required fields are present.

        Generates error messages with the new input structure:
            configurable:
                conversation_id: "abc-123"
                user_id: "nate.fleming"
                <field_name>: <example_value>
            session: {}
        """
        logger.trace("Executing custom field validation")

        context: Context = runtime.context or Context()

        # Find all missing required fields
        missing_fields: list[RequiredField] = []
        for field in self.fields:
            if field.is_required:
                field_value: Any = getattr(context, field.name, None)
                if field_value is None:
                    missing_fields.append(field)

        if not missing_fields:
            return None

        # Log the missing fields
        missing_names = [f.name for f in missing_fields]
        logger.error("Required fields missing", fields=missing_names)

        # Build the configurable dict preserving provided values
        # and using example_value for missing required fields
        # Note: only thread_id is in configurable (conversation_id goes in session)
        configurable: dict[str, Any] = {}

        thread_val = context.thread_id or "<your_thread_id>"
        configurable["thread_id"] = thread_val

        if context.user_id:
            configurable["user_id"] = context.user_id
        else:
            configurable["user_id"] = "<your_user_id>"

        # Add all extra values the user already provided
        context_dict = context.model_dump()
        for k, v in context_dict.items():
            if k not in {"user_id", "thread_id"} and v is not None:
                configurable[k] = v

        # Then add our defined fields (provided values take precedence)
        for field in self.fields:
            if field.name in configurable:
                # Field was provided by user - keep their value
                continue

            if field.is_required:
                # Missing required field - use example_value or placeholder
                configurable[field.name] = (
                    field.example_value
                    if field.example_value is not None
                    else f"<your_{field.name}>"
                )
            else:
                # Optional field not provided - use example_value if available
                if field.example_value is not None:
                    configurable[field.name] = field.example_value

        # Build the corrected config with new structure
        # Note: conversation_id is in session as an alias of thread_id
        corrected_config: dict[str, Any] = {
            "configurable": configurable,
            "session": {
                "conversation_id": thread_val,
            },
        }
        corrected_config_json = json.dumps(corrected_config, indent=2)

        # Build field descriptions
        field_descriptions: list[str] = [
            "- **thread_id**: Thread identifier (required in configurable)",
            "- **conversation_id**: Alias of thread_id (in session)",
        ]

        # Add user_id if not in custom fields
        has_user_id_field = any(f.name == "user_id" for f in self.fields)
        if not has_user_id_field:
            field_descriptions.append(
                "- **user_id**: Your unique user identifier (required)"
            )

        # Add custom field descriptions
        for field in self.fields:
            required_text = "(required)" if field.is_required else "(optional)"
            field_descriptions.append(
                f"- **{field.name}**: {field.description} {required_text}"
            )

        field_descriptions_text = "\n".join(field_descriptions)

        # Build the list of missing field names for the error message
        missing_names_formatted = ", ".join(f"**{f.name}**" for f in missing_fields)

        error_message = f"""
## Configuration Required

The following required fields are missing: {missing_names_formatted}

### Required Configuration Format

Please include the following JSON in your request configuration:

```json
{corrected_config_json}
```

### Field Descriptions
{field_descriptions_text}

Please update your configuration and try again.
        """.strip()

        raise ValueError(error_message)


class FilterLastHumanMessageMiddleware(AgentMiddleware[AgentState, Context]):
    """
    Middleware that filters messages to keep only the last human message.

    This is useful for scenarios where you want to process only the
    latest user input without conversation history.
    """

    def before_model(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Filter messages to keep only the last human message."""
        logger.trace("Executing filter_last_human_message middleware")

        messages: list[BaseMessage] = state.get("messages", [])

        if not messages:
            logger.trace("No messages found in state")
            return None

        last_message: HumanMessage | None = last_human_message(messages)

        if last_message is None:
            logger.trace("No human messages found in state")
            return {"messages": []}

        logger.trace(
            "Filtered messages to last human message", original_count=len(messages)
        )

        removed_messages = [
            RemoveMessage(id=message.id)
            for message in messages
            if message.id != last_message.id
        ]

        return {"messages": removed_messages}


# =============================================================================
# Factory Functions
# =============================================================================


def create_user_id_validation_middleware() -> UserIdValidationMiddleware:
    """
    Create a UserIdValidationMiddleware instance.

    Factory function for creating middleware that validates the presence
    and format of user_id in the runtime context.

    Returns:
        List containing UserIdValidationMiddleware instance

    Example:
        middleware = create_user_id_validation_middleware()
    """
    logger.trace("Creating user_id validation middleware")
    return UserIdValidationMiddleware()


def create_thread_id_validation_middleware() -> ThreadIdValidationMiddleware:
    """
    Create a ThreadIdValidationMiddleware instance.

    Factory function for creating middleware that validates the presence
    of thread_id in the runtime context.

    Returns:
        List containing ThreadIdValidationMiddleware instance

    Example:
        middleware = create_thread_id_validation_middleware()
    """
    logger.trace("Creating thread_id validation middleware")
    return ThreadIdValidationMiddleware()


def create_custom_field_validation_middleware(
    fields: list[dict[str, Any]],
) -> CustomFieldValidationMiddleware:
    """
    Create a CustomFieldValidationMiddleware instance.

    Factory function for creating middleware that validates the presence
    of required custom fields in the context object.

    Each field in the list should have:
    - name: The field name (required)
    - description: Human-readable description for error messages (optional)
    - required: Whether field is required (default: True)
    - example_value: Example value to show in error messages (optional)

    Required fields (required=True or not specified) will be validated.
    The example_value is used in error messages to help users copy-paste.

    Args:
        fields: List of field definitions. Each dict should have 'name', and
            optionally 'description', 'required', and 'example_value' keys.

    Returns:
        List containing CustomFieldValidationMiddleware configured with the specified fields

    Example:
        middleware = create_custom_field_validation_middleware(
            fields=[
                # Required field with example value for easy copy-paste
                {"name": "store_num", "description": "Your store number", "example_value": "12345"},
                # Optional fields (required=False)
                {"name": "thread_id", "description": "Thread ID", "required": False, "example_value": "1"},
                {"name": "user_id", "description": "User ID", "required": False, "example_value": "my_user_id"},
            ],
        )
    """
    field_names = [f.get("name", "unknown") for f in fields]
    logger.trace("Creating custom field validation middleware", fields=field_names)
    return CustomFieldValidationMiddleware(fields=fields)


def create_filter_last_human_message_middleware() -> FilterLastHumanMessageMiddleware:
    """
    Create a FilterLastHumanMessageMiddleware instance.

    Factory function for creating middleware that filters messages to keep
    only the last human message, useful for scenarios where you want to
    process only the latest user input without conversation history.

    Returns:
        List containing FilterLastHumanMessageMiddleware instance

    Example:
        middleware = create_filter_last_human_message_middleware()
    """
    logger.trace("Creating filter_last_human_message middleware")
    return FilterLastHumanMessageMiddleware()
