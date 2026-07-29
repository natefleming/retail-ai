"""Background agent support for dao-ai.

Provides Responses-API-compatible kickoff / poll / cancel semantics
backed by a Lakebase instance so agents can exceed Model Serving's
~5 min worker-thread timeout and Databricks Apps' ~120 s proxy timeout.

See ``docs/background_agents.md`` for the wire protocol and
``examples/19_background_agents/background_research.yaml`` for an
end-to-end example.
"""

from dao_ai.background.agent import (
    CUSTOM_INPUT_CURSOR,
    CUSTOM_INPUT_OPERATION,
    CUSTOM_INPUT_RESPONSE_ID,
    ERROR_TYPE_NOT_FOUND,
    OPERATION_CANCEL,
    OPERATION_RETRIEVE,
    BackgroundResponsesAgent,
    is_not_found_response,
)
from dao_ai.background.store import BackgroundStore, ResponseRecord, ResponseStatus

__all__ = [
    "CUSTOM_INPUT_CURSOR",
    "CUSTOM_INPUT_OPERATION",
    "CUSTOM_INPUT_RESPONSE_ID",
    "ERROR_TYPE_NOT_FOUND",
    "BackgroundResponsesAgent",
    "BackgroundStore",
    "OPERATION_CANCEL",
    "OPERATION_RETRIEVE",
    "ResponseRecord",
    "ResponseStatus",
    "is_not_found_response",
]
