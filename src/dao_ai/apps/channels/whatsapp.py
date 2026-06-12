"""Meta WhatsApp Cloud API inbound channel.

The single public entry point is :func:`mount_whatsapp_routes`, invoked from
:func:`dao_ai.apps.channels.mount_channel_routes` at app startup. Two routes
are added to the existing FastAPI app:

* ``GET  {webhook_path}`` — Meta verification handshake. Returns the
  ``hub.challenge`` query value when the ``hub.verify_token`` matches.
* ``POST {webhook_path}`` — Inbound message delivery. Verifies the
  ``X-Hub-Signature-256`` HMAC, dedupes on Meta's message id, resolves the
  conversation thread, returns 200 immediately, and dispatches the agent
  call in a fire-and-forget background task.

The agent invocation reuses :func:`dao_ai.apps.handlers.non_streaming`, so
every guardrail, middleware, tracing, OBO header injection, memory, and
long-running offloading the agent already has applies unchanged. The text
of the agent's response is then POSTed back to ``graph.facebook.com``,
chunked at the configured character limit.

Meta retries failed deliveries; the dedup table makes the handler idempotent.
The handler always ACKs 200 in <20s — Meta's threshold for re-delivery —
regardless of how long the agent takes.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from typing import TYPE_CHECKING, Any, Optional

import httpx
import mlflow
from fastapi import Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from dao_ai.apps.channels.store import ChannelStore
from dao_ai.config import WhatsAppChannelModel, value_of

if TYPE_CHECKING:
    from fastapi import FastAPI

    from dao_ai.config import AppConfig


CHANNEL_NAME = "whatsapp"
SIGNATURE_HEADER = "X-Hub-Signature-256"
SIGNATURE_PREFIX = "sha256="
GRAPH_API_BASE = "https://graph.facebook.com"


def _hash_phone(wa_id: str) -> str:
    """Return a stable opaque identifier for a wa_id (for trace attrs)."""
    return hashlib.sha256(wa_id.encode("utf-8")).hexdigest()[:16]


def verify_signature(*, raw_body: bytes, header_value: str, app_secret: str) -> bool:
    """Constant-time check of the ``X-Hub-Signature-256`` header.

    Meta sends ``X-Hub-Signature-256: sha256=<hex>``. Per Meta's docs the
    HMAC is computed over the raw request body using the App Secret. We
    compare with :func:`hmac.compare_digest` to avoid timing leaks.
    """
    if not header_value or not header_value.startswith(SIGNATURE_PREFIX):
        return False
    expected = hmac.new(
        app_secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    provided = header_value[len(SIGNATURE_PREFIX):]
    return hmac.compare_digest(expected, provided)


def _thread_key(
    config: WhatsAppChannelModel, *, wa_id: str
) -> str:
    """Derive the per-conversation key per the configured strategy."""
    if config.default_thread_strategy == "static":
        # Validated in WhatsAppChannelModel.validate_static_thread
        assert config.static_thread_id is not None
        return f"static:{value_of(config.static_thread_id)}"
    if config.default_thread_strategy == "wa_id+phone_number_id":
        return f"{value_of(config.phone_number_id)}:{wa_id}"
    return wa_id


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into pieces no longer than ``max_chars``.

    Prefers paragraph boundaries, then newlines, then whitespace. Single
    runs that exceed ``max_chars`` are hard-wrapped — WhatsApp's limit is
    a hard server-side reject so we cannot send anything longer.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        for sep in ("\n\n", "\n", ". ", " "):
            cut = window.rfind(sep)
            if cut > max_chars // 2:
                chunks.append(remaining[:cut].rstrip())
                remaining = remaining[cut + len(sep):]
                break
        else:
            chunks.append(window)
            remaining = remaining[max_chars:]
    if remaining:
        chunks.append(remaining)
    return chunks


def _extract_reply_text(response: ResponsesAgentResponse) -> str:
    """Pull the final assistant text out of a ResponsesAgentResponse.

    Mirrors the extraction pattern used in :mod:`dao_ai.optimization`.
    Items whose ``content`` is a list of ``{"type": "output_text", "text": ...}``
    dicts are joined; string content is taken verbatim.
    """
    if not response.output:
        return ""
    pieces: list[str] = []
    for item in response.output:
        # MLflow's typed items expose .content as str | list[dict] | None
        content = getattr(item, "content", None)
        if content is None:
            continue
        if isinstance(content, str):
            pieces.append(content)
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
    return "\n".join(p for p in pieces if p).strip()


@mlflow.trace(name="whatsapp.outbound.send_text", span_type="CHAIN")
async def send_text(
    *,
    client: httpx.AsyncClient,
    config: WhatsAppChannelModel,
    access_token: str,
    to_wa_id: str,
    text: str,
) -> None:
    """POST one or more text messages to the WhatsApp Cloud API."""
    graph_api_version = str(value_of(config.graph_api_version))
    phone_number_id = str(value_of(config.phone_number_id))
    url = f"{GRAPH_API_BASE}/{graph_api_version}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    max_chunk_chars = int(value_of(config.max_outbound_chunk_chars))
    for chunk in _chunk_text(text, max_chunk_chars):
        body = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_wa_id,
            "type": "text",
            "text": {"body": chunk, "preview_url": False},
        }
        resp = await client.post(url, headers=headers, json=body, timeout=30.0)
        if resp.status_code >= 400:
            # Don't raise — surface for monitoring but allow remaining chunks
            logger.error(
                "WhatsApp outbound failed",
                status=resp.status_code,
                body=resp.text[:500],
                to_hash=_hash_phone(to_wa_id),
            )
            return
        logger.debug(
            "WhatsApp outbound sent",
            to_hash=_hash_phone(to_wa_id),
            chunk_chars=len(chunk),
        )


def _parse_inbound_text(message: dict[str, Any]) -> Optional[str]:
    """Return the user-facing text from a Meta inbound message, if any.

    Initial scope: text messages. Other types (image/audio/etc.) are
    acknowledged but not forwarded — callers can extend by passing
    media references through ``custom_inputs``.
    """
    msg_type = message.get("type")
    if msg_type == "text":
        return (message.get("text") or {}).get("body")
    if msg_type == "interactive":
        interactive = message.get("interactive") or {}
        kind = interactive.get("type")
        if kind == "button_reply":
            return (interactive.get("button_reply") or {}).get("title")
        if kind == "list_reply":
            return (interactive.get("list_reply") or {}).get("title")
    return None


def mount_whatsapp_routes(
    app: "FastAPI",
    config: "AppConfig",
    whatsapp_config: WhatsAppChannelModel,
) -> None:
    """Register the GET/POST WhatsApp webhook routes on ``app``.

    Idempotent at module level — caller (``mount_channel_routes``) ensures
    a single invocation per startup.
    """
    # Resolve database with fallback to app.long_running.database
    database = whatsapp_config.database
    if database is None and config.app and config.app.long_running:
        database = config.app.long_running.database

    # Resolve once at mount time: webhook_path is a route decorator argument,
    # table names go into the ChannelStore constructor. Per-request resolution
    # is reserved for credentials (verify_token / app_secret / access_token) and
    # outbound-format values (graph_api_version / phone_number_id) so secret
    # rotation works without an app restart.
    webhook_path = str(value_of(whatsapp_config.webhook_path))
    if not webhook_path.startswith("/"):
        raise ValueError(
            f"webhook_path must start with '/' (resolved value: {webhook_path!r})"
        )
    dedup_table = str(value_of(whatsapp_config.dedup_table_name))
    threads_table = str(value_of(whatsapp_config.threads_table_name))

    store = ChannelStore(
        database,
        dedup_table_name=dedup_table,
        threads_table_name=threads_table,
    )

    # Single shared HTTP client for outbound Graph API calls
    http_client = httpx.AsyncClient(timeout=30.0)

    async def _dispatch_one(
        *,
        wa_id: str,
        text: str,
        thread_id: str,
        message_id: str,
    ) -> None:
        """Run the agent and send its reply back over WhatsApp.

        Runs as a fire-and-forget background task. Failures are logged but
        not re-raised — Meta has already been ACKed with 200.
        """
        # Import lazily so module import doesn't pull in the agent eagerly
        from dao_ai.apps.handlers import non_streaming

        wa_hash = (
            _hash_phone(wa_id)
            if bool(value_of(whatsapp_config.redact_phone_in_traces))
            else wa_id
        )

        with mlflow.start_span(
            name="whatsapp.inbound.dispatch",
            attributes={
                "channel": CHANNEL_NAME,
                "wa_id_hash": wa_hash,
                "thread_id": thread_id,
                "message_id": message_id,
            },
        ):
            try:
                request = ResponsesAgentRequest(
                    input=[{"role": "user", "content": text}],
                    custom_inputs={
                        "configurable": {
                            "thread_id": thread_id,
                            "channel": CHANNEL_NAME,
                            "wa_id": wa_id,
                            "phone_number_id": str(value_of(whatsapp_config.phone_number_id)),
                            "message_id": message_id,
                        }
                    },
                )
                response: ResponsesAgentResponse = await non_streaming(request)
                reply = _extract_reply_text(response)
                if not reply:
                    logger.warning(
                        "Agent produced no text for WhatsApp reply",
                        wa_id_hash=wa_hash,
                    )
                    return

                access_token = str(value_of(whatsapp_config.access_token))
                await send_text(
                    client=http_client,
                    config=whatsapp_config,
                    access_token=access_token,
                    to_wa_id=wa_id,
                    text=reply,
                )
            except Exception as exc:  # noqa: BLE001 — fire-and-forget background
                logger.exception(
                    "WhatsApp dispatch failed",
                    wa_id_hash=wa_hash,
                    error=str(exc),
                )

    @app.get(webhook_path)
    async def whatsapp_verify(
        hub_mode: str = Query(default="", alias="hub.mode"),
        hub_verify_token: str = Query(default="", alias="hub.verify_token"),
        hub_challenge: str = Query(default="", alias="hub.challenge"),
    ) -> PlainTextResponse:
        expected_token = str(value_of(whatsapp_config.verify_token))
        if hub_mode == "subscribe" and hmac.compare_digest(
            hub_verify_token, expected_token
        ):
            logger.info("WhatsApp webhook verified")
            return PlainTextResponse(content=hub_challenge, status_code=200)
        logger.warning(
            "WhatsApp webhook verification rejected", mode=hub_mode
        )
        raise HTTPException(status_code=403, detail="Verification failed")

    @app.post(webhook_path)
    async def whatsapp_inbound(
        request: Request,
        x_hub_signature_256: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        raw = await request.body()

        app_secret = str(value_of(whatsapp_config.app_secret))
        if not verify_signature(
            raw_body=raw,
            header_value=x_hub_signature_256 or "",
            app_secret=app_secret,
        ):
            logger.warning("WhatsApp signature verification failed")
            raise HTTPException(status_code=403, detail="Invalid signature")

        await store.ensure_schema()

        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        for entry in payload.get("entry", []) or []:
            for change in entry.get("changes", []) or []:
                value = change.get("value") or {}
                messages = value.get("messages") or []
                for message in messages:
                    message_id = message.get("id")
                    wa_id = message.get("from")
                    text = _parse_inbound_text(message)

                    if not message_id or not wa_id:
                        logger.warning(
                            "Skipping WhatsApp message missing id or from",
                            keys=list(message.keys()),
                        )
                        continue
                    if text is None:
                        logger.info(
                            "Skipping non-text WhatsApp message",
                            message_id=message_id,
                            type=message.get("type"),
                        )
                        continue

                    dedup = await store.record_message(
                        message_id=message_id, channel=CHANNEL_NAME
                    )
                    if not dedup.inserted:
                        logger.debug(
                            "Skipping duplicate WhatsApp delivery",
                            message_id=message_id,
                        )
                        continue

                    thread_id = await store.get_or_create_thread(
                        thread_key=_thread_key(whatsapp_config, wa_id=wa_id),
                        channel=CHANNEL_NAME,
                    )

                    # Fire and forget — Meta gets the ACK before the agent runs
                    asyncio.create_task(
                        _dispatch_one(
                            wa_id=wa_id,
                            text=text,
                            thread_id=thread_id,
                            message_id=message_id,
                        )
                    )

        return JSONResponse({"status": "received"}, status_code=200)

    logger.success(
        "WhatsApp channel routes mounted",
        webhook_path=webhook_path,
        phone_number_id=str(value_of(whatsapp_config.phone_number_id)),
        store=("lakebase" if database is not None else "in-memory"),
    )


__all__ = [
    "CHANNEL_NAME",
    "mount_whatsapp_routes",
    "verify_signature",
    "send_text",
]
