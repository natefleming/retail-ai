"""
Core Genie service implementation.

This module provides the concrete implementation of GenieServiceBase
that wraps the Databricks Genie SDK.
"""

import mlflow
from databricks_ai_bridge.genie import Genie, GenieResponse

from dao_ai.genie.cache import CacheResult, GenieServiceBase


class GenieService(GenieServiceBase):
    """Concrete implementation of GenieServiceBase using the Genie SDK."""

    genie: Genie

    def __init__(self, genie: Genie) -> None:
        self.genie = genie

    @mlflow.trace(name="genie_ask_question")
    def ask_question(
        self, question: str, conversation_id: str | None = None
    ) -> CacheResult:
        """Ask question to Genie and return CacheResult (no caching at this level)."""
        response: GenieResponse = self.genie.ask_question(
            question, conversation_id=conversation_id
        )
        # No caching at this level - return cache miss
        return CacheResult(response=response, cache_hit=False, served_by=None)

    @property
    def space_id(self) -> str:
        return self.genie.space_id
