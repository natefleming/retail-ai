"""The agent-node build log must name the model id that reaches the wire.

Before UC-securable model names existed, ``InferenceEndpointModel.name`` *was*
the id sent to the serving layer, so logging ``.name`` was accurate by
construction. Now ``full_name`` can differ from ``name`` — a schema-qualified
model logs ``gpt-5-4-mini`` while calling ``system.ai.gpt-5-4-mini`` — and this
line is the only build-time record of which model an agent is wired to. A log
that names a different model than the request is worse than no log: it sends
anyone debugging a 404 or a permissions error looking at the wrong securable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from dao_ai.config import AgentModel, InferenceEndpointModel


def _capture(fn) -> list[tuple[str, str]]:
    """Run ``fn`` with a temporary INFO sink; return (message, extra) pairs.

    dao-ai logs structured kwargs and its ``configure_logging`` patcher renders
    them into ``record["extra"]`` as a ``" | key=value"`` string, so the kwargs
    are not in the message text — assert against ``extra``.
    """
    records: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda m: records.append((m.record["message"], str(m.record["extra"]))),
        level="INFO",
    )
    try:
        fn()
    finally:
        logger.remove(sink_id)
    return records


def _build(model: InferenceEndpointModel) -> list[tuple[str, str]]:
    """Build an agent node, stubbing the chat client and the agent factory so
    only the logging path runs."""
    from dao_ai import nodes

    agent = AgentModel(
        name="echo",
        description="test agent",
        model=model,
        tools=[],
        prompt="You are echo.",
    )
    with (
        patch.object(type(model), "as_chat_model", return_value=MagicMock()),
        patch.object(nodes, "create_agent", return_value=MagicMock()),
    ):
        return _capture(lambda: nodes.create_agent_node(agent=agent))


@pytest.mark.unit
def test_build_log_names_the_qualified_model() -> None:
    """A schema-qualified model must be logged by its three-level id — that is
    what the request body carries."""
    model = InferenceEndpointModel.model_validate(
        {
            "schema": {"catalog_name": "system", "schema_name": "ai"},
            "name": "gpt-5-4-mini",
            "use_ai_gateway": True,
        }
    )
    extras = [e for msg, e in _build(model) if msg == "Agent configuration"]
    assert extras, "expected an 'Agent configuration' log line"
    assert "model=system.ai.gpt-5-4-mini" in extras[0]


@pytest.mark.unit
def test_build_log_unchanged_for_a_plain_endpoint_name() -> None:
    """The regression guard: with no schema, ``full_name`` is ``name``, so
    every existing config logs exactly what it logged before."""
    model = InferenceEndpointModel(name="databricks-claude-sonnet-4-5")
    extras = [e for msg, e in _build(model) if msg == "Agent configuration"]
    assert extras
    assert "model=databricks-claude-sonnet-4-5" in extras[0]
