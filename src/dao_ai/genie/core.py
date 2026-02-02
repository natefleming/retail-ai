"""
Core Genie service implementation.

This module provides:
- Extended Genie and GenieResponse classes that capture message_id
- GenieService: Concrete implementation of GenieServiceBase

The extended classes wrap the databricks_ai_bridge versions to add message_id
support, which is needed for sending feedback to the Genie API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieFeedbackRating
from databricks_ai_bridge.genie import Genie as DatabricksGenie
from databricks_ai_bridge.genie import GenieResponse as DatabricksGenieResponse
from loguru import logger

from dao_ai.genie.cache import CacheResult, GenieServiceBase
from dao_ai.genie.cache.base import get_latest_message_id

if TYPE_CHECKING:
    from typing import Optional


# =============================================================================
# Extended Genie Classes with message_id Support
# =============================================================================


@dataclass
class GenieResponse(DatabricksGenieResponse):
    """
    Extended GenieResponse that includes message_id.

    This extends the databricks_ai_bridge GenieResponse to capture the message_id
    from API responses, which is required for sending feedback to the Genie API.

    Attributes:
        result: The query result as string or DataFrame
        query: The generated SQL query
        description: Description of the query
        conversation_id: The conversation ID
        message_id: The message ID (NEW - enables feedback without extra API call)
    """

    result: Union[str, pd.DataFrame] = ""
    query: Optional[str] = ""
    description: Optional[str] = ""
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None


class Genie(DatabricksGenie):
    """
    Extended Genie that captures message_id in responses.

    This extends the databricks_ai_bridge Genie to return GenieResponse objects
    that include the message_id from the API response. This enables sending
    feedback without requiring an additional API call to look up the message ID.

    Usage:
        genie = Genie(space_id="my-space")
        response = genie.ask_question("What are total sales?")
        print(response.message_id)  # Now available!

    The original databricks_ai_bridge classes are available as:
        - DatabricksGenie
        - DatabricksGenieResponse
    """

    def ask_question(
        self, question: str, conversation_id: str | None = None
    ) -> GenieResponse:
        """
        Ask a question and return response with message_id.

        This overrides the parent method to capture the message_id from the
        API response and include it in the returned GenieResponse.

        Args:
            question: The question to ask
            conversation_id: Optional conversation ID for follow-up questions

        Returns:
            GenieResponse with message_id populated
        """
        with mlflow.start_span(name="ask_question"):
            # Start or continue conversation
            if not conversation_id:
                resp = self.start_conversation(question)
            else:
                resp = self.create_message(conversation_id, question)

            # Capture message_id from the API response
            message_id = resp.get("message_id")

            # Poll for the result using parent's method
            genie_response = self.poll_for_result(resp["conversation_id"], message_id)

            # Ensure conversation_id is set
            if not genie_response.conversation_id:
                genie_response.conversation_id = resp["conversation_id"]

            # Return our extended response with message_id
            return GenieResponse(
                result=genie_response.result,
                query=genie_response.query,
                description=genie_response.description,
                conversation_id=genie_response.conversation_id,
                message_id=message_id,
            )


# =============================================================================
# GenieService Implementation
# =============================================================================


class GenieService(GenieServiceBase):
    """
    Concrete implementation of GenieServiceBase using the extended Genie.

    This service wraps the extended Genie class and provides the GenieServiceBase
    interface for use with cache layers.
    """

    genie: Genie
    _workspace_client: WorkspaceClient | None

    def __init__(
        self,
        genie: Genie | DatabricksGenie,
        workspace_client: WorkspaceClient | None = None,
    ) -> None:
        """
        Initialize the GenieService.

        Args:
            genie: The Genie instance for asking questions. Can be either our
                extended Genie or the original DatabricksGenie.
            workspace_client: Optional WorkspaceClient for feedback API.
                If not provided, one will be created lazily when needed.
        """
        self.genie = genie  # type: ignore[assignment]
        self._workspace_client = workspace_client

    @property
    def workspace_client(self) -> WorkspaceClient:
        """
        Get or create a WorkspaceClient for API calls.

        Lazily creates a WorkspaceClient using default credentials if not provided.
        """
        if self._workspace_client is None:
            self._workspace_client = WorkspaceClient()
        return self._workspace_client

    @mlflow.trace(name="genie_ask_question")
    def ask_question(
        self, question: str, conversation_id: str | None = None
    ) -> CacheResult:
        """
        Ask question to Genie and return CacheResult.

        No caching at this level - returns cache miss with fresh response.
        If using our extended Genie, the message_id will be captured in the response.
        """
        response = self.genie.ask_question(question, conversation_id=conversation_id)

        # Extract message_id if available (from our extended GenieResponse)
        message_id = getattr(response, "message_id", None)

        # No caching at this level - return cache miss
        return CacheResult(
            response=response,
            cache_hit=False,
            served_by=None,
            message_id=message_id,
        )

    @property
    def space_id(self) -> str:
        return self.genie.space_id

    @mlflow.trace(name="genie_send_feedback")
    def send_feedback(
        self,
        conversation_id: str,
        rating: GenieFeedbackRating,
        message_id: str | None = None,
        was_cache_hit: bool = False,
    ) -> None:
        """
        Send feedback for a Genie message.

        For the core GenieService, this always sends feedback to the Genie API
        (the was_cache_hit parameter is ignored here - it's used by cache wrappers).

        Args:
            conversation_id: The conversation containing the message
            rating: The feedback rating (POSITIVE, NEGATIVE, or NONE)
            message_id: Optional message ID. If None, looks up the most recent message.
            was_cache_hit: Ignored by GenieService. Cache wrappers use this to decide
                whether to forward feedback to the underlying service.
        """
        # Look up message_id if not provided
        if message_id is None:
            message_id = get_latest_message_id(
                workspace_client=self.workspace_client,
                space_id=self.space_id,
                conversation_id=conversation_id,
            )
            if message_id is None:
                logger.warning(
                    "Could not find message_id for feedback, skipping",
                    space_id=self.space_id,
                    conversation_id=conversation_id,
                    rating=rating.value if rating else None,
                )
                return

        logger.info(
            "Sending feedback to Genie",
            space_id=self.space_id,
            conversation_id=conversation_id,
            message_id=message_id,
            rating=rating.value if rating else None,
        )

        try:
            self.workspace_client.genie.send_message_feedback(
                space_id=self.space_id,
                conversation_id=conversation_id,
                message_id=message_id,
                rating=rating,
            )
            logger.debug(
                "Feedback sent successfully",
                space_id=self.space_id,
                conversation_id=conversation_id,
                message_id=message_id,
            )
        except Exception as e:
            logger.error(
                "Failed to send feedback to Genie",
                space_id=self.space_id,
                conversation_id=conversation_id,
                message_id=message_id,
                rating=rating.value if rating else None,
                error=str(e),
                exc_info=True,
            )
