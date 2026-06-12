"""Inbound messaging channels for dao-ai.

A channel takes platform-specific webhook deliveries (WhatsApp, etc.),
authenticates and deduplicates them, then dispatches them into the same
agent invocation path used by ``/v1/responses`` and ``/invocations``. The
public entry point is :func:`mount_channel_routes`, called once at app
startup from :mod:`dao_ai.apps.server`.

See ``config/examples/21_channels/`` for runnable configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from fastapi import FastAPI

    from dao_ai.config import AppConfig


def mount_channel_routes(app: "FastAPI", config: "AppConfig") -> list[str]:
    """Mount every configured inbound channel on ``app``.

    No-op when ``config.app.channels`` is unset. Returns the list of
    channel names that were mounted (useful for startup logging and tests).
    """
    channels = config.app.channels if config.app else None
    if channels is None:
        return []

    mounted: list[str] = []

    if channels.whatsapp is not None:
        from dao_ai.apps.channels.whatsapp import mount_whatsapp_routes

        mount_whatsapp_routes(app, config, channels.whatsapp)
        mounted.append("whatsapp")

    if mounted:
        logger.success("Inbound channel routes mounted", channels=mounted)
    return mounted


__all__ = ["mount_channel_routes"]
