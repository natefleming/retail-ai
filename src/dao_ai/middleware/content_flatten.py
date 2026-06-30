"""Content-flatten middleware for DAO AI agents.

Belt-and-suspenders: flattens every AIMessage's content shape to a plain
text string in ``before_model``, ensuring the LLM call sees ``content``
as a ``str`` regardless of what upstream paths produced it.

Why this exists
---------------
Databricks Foundation-Model-API chat-completions endpoints (gpt-oss-120b
et al.) reject AIMessages whose ``content`` is array-shaped with:

    Bad request: field 'messages.content' expects 'string' but got 'array'

Upstream content can land in array shape via several paths:

- ChatDatabricks returns ``str`` whose value is the JSON-encoded
  array of content blocks (``'[{"type":"reasoning",...}, ...]'``).
- Claude / Responses-API native returns ``list[dict]``.
- Memory middleware injects context that may contain block fragments.
- Handoff Commands escape via ``ParentCommand`` and bypass
  ``extract_agent_response``'s normalization step.

``extract_agent_response`` flattens at the subgraph-exit boundary, the
handoff tool flattens its trigger AIMessage, and
``filter_messages_for_agent`` flattens at the per-agent message-filter
boundary. This middleware is the LAST line of defense: it sits directly
in the ``before_model`` hook on every agent, so any AIMessage in the
state that still has array content gets normalized one final time
before reaching the LLM.

Stateless, zero-cost when content is already a string.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.runtime import Runtime
from loguru import logger

from dao_ai.orchestration.core import _flatten_message_content
from dao_ai.state import AgentState, Context

__all__ = [
    "ContentFlattenMiddleware",
    "create_content_flatten_middleware",
]


class ContentFlattenMiddleware(AgentMiddleware[AgentState, Context]):
    """Flattens AIMessage.content to plain text before every LLM call.

    Runs in ``before_model``. Scans every AIMessage in ``state["messages"]``
    and replaces ``content`` with the flattened text form via
    :func:`_flatten_message_content`. Other message types are untouched.
    """

    def before_model(
        self,
        state: AgentState,
        runtime: Runtime[Context],
    ) -> dict[str, Any] | None:
        messages: list[BaseMessage] = state.get("messages", []) or []
        if not messages:
            return None
        rewritten: list[BaseMessage] = []
        changed: int = 0
        for msg in messages:
            if isinstance(msg, AIMessage):
                flat: object = _flatten_message_content(msg.content)
                if flat is msg.content:
                    rewritten.append(msg)
                    continue
                rewritten.append(msg.model_copy(update={"content": flat}))
                changed += 1
            else:
                rewritten.append(msg)
        if changed == 0:
            return None
        logger.trace(
            "ContentFlattenMiddleware: normalized AIMessage content",
            messages_changed=changed,
            total=len(messages),
        )
        return {"messages": rewritten}


def create_content_flatten_middleware() -> ContentFlattenMiddleware:
    """Factory matching the dao-ai middleware FQN pattern."""
    return ContentFlattenMiddleware()
