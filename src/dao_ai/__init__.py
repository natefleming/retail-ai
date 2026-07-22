"""
DAO AI - A framework for building AI agents with Databricks.

This module configures package-level settings including warning filters
for expected runtime warnings that don't indicate actual problems.
"""

import warnings

# Suppress Pydantic serialization warnings for Context objects during checkpointing.
# This warning occurs because LangGraph's checkpointer serializes the context_schema
# and Pydantic reports that serialization may not be as expected. This is benign
# since Context is only used at runtime and doesn't need to be persisted.
#
# The warning looks like:
# PydanticSerializationUnexpectedValue(Expected `none` - serialized value may not
# be as expected [field_name='context', input_value=Context(...), input_type=Context])
warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic serializer warnings.*",
    category=UserWarning,
)

# Also filter the specific PydanticSerializationUnexpectedValue warning
warnings.filterwarnings(
    "ignore",
    message=r".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)
