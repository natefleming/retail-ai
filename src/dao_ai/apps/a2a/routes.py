"""Mount A2A protocol routes onto an existing FastAPI app.

This module exposes the single public entry point
:func:`mount_a2a_routes`, which is called from
:mod:`dao_ai.apps.server` after the long-running Responses routes have
been mounted. It is a no-op when ``config.app.a2a.enabled`` is False, so
opting out is one config line.

The routes added are:

* ``GET  /.well-known/agent-card.json`` — public Agent Card.
* ``POST /a2a``                          — A2A JSON-RPC 2.0 entrypoint.

a2a-sdk's :class:`A2AFastAPIApplication.add_routes_to_app` does the
heavy lifting; we just plug in our :class:`A2AAgentExecutor`,
:class:`TaskStore`, and the derived Agent Card.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from loguru import logger

from dao_ai.apps.a2a.agent_card import (
    DEFAULT_A2A_RPC_PATH,
    DEFAULT_AGENT_CARD_PATH,
    build_agent_card,
    effective_a2a,
)
from dao_ai.apps.a2a.executor import A2AAgentExecutor
from dao_ai.apps.a2a.task_store import build_task_store

if TYPE_CHECKING:
    from fastapi import FastAPI

    from dao_ai.config import AppConfig


def mount_a2a_routes(app: "FastAPI", config: "AppConfig") -> bool:
    """Mount the A2A discovery + JSON-RPC routes onto ``app``.

    Honors the ``app.a2a.enabled`` opt-out: when False, this is a no-op
    and returns ``False``. Otherwise wires the executor, task store, and
    Agent Card, and returns ``True``.

    Idempotency: a2a-sdk's ``add_routes_to_app`` registers concrete paths
    and will collide if called twice. Callers should call this once at
    app startup.
    """
    a2a = effective_a2a(config)
    if not a2a.enabled:
        logger.info("A2A routes disabled via app.a2a.enabled=false")
        return False

    agent_card = build_agent_card(config)
    executor = A2AAgentExecutor(config)
    task_store = build_task_store(config)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    application = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=handler,
    )
    application.add_routes_to_app(
        app,
        agent_card_url=DEFAULT_AGENT_CARD_PATH,
        rpc_url=DEFAULT_A2A_RPC_PATH,
    )

    logger.success(
        "A2A routes mounted",
        agent_card_url=DEFAULT_AGENT_CARD_PATH,
        rpc_url=DEFAULT_A2A_RPC_PATH,
        skills_count=len(agent_card.skills),
        task_store_type=type(task_store).__name__,
        security_schemes=list((agent_card.security_schemes or {}).keys()),
    )
    return True
