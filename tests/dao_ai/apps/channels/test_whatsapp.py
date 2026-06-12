"""Unit tests for the WhatsApp inbound channel.

Covers the parts of :mod:`dao_ai.apps.channels.whatsapp` that don't need a
real Meta or Lakebase: signature verification, text chunking, payload
parsing, response extraction, dedup via the in-memory ChannelStore mode,
and the FastAPI route surface using TestClient with a mocked agent
handler.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlflow.types.responses import OutputItem, ResponsesAgentResponse

from dao_ai.apps.channels.store import ChannelStore
from dao_ai.apps.channels.whatsapp import (
    _chunk_text,
    _extract_reply_text,
    _parse_inbound_text,
    _thread_key,
    mount_whatsapp_routes,
    verify_signature,
)
from dao_ai.config import SecretVariableModel, WhatsAppChannelModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubSecret(SecretVariableModel):
    """A SecretVariableModel that resolves to a known constant in tests."""

    _resolved: str = ""

    def as_value(self) -> Any:
        return self._resolved


def _make_secret(value: str) -> SecretVariableModel:
    """Build a SecretVariableModel whose `.as_value()` returns ``value`` without hitting Databricks."""
    secret = SecretVariableModel(scope="test", secret="test")
    # SecretVariableModel is frozen — patch the bound method on the instance
    # by replacing the resolution path. We monkeypatch as_value via __dict__.
    object.__setattr__(
        secret, "as_value", lambda v=value: v  # type: ignore[arg-type]
    )
    return secret


def _make_config(
    *,
    phone_number_id: str = "123456789",
    verify: str = "v-token",
    secret: str = "app-secret",
    access: str = "access-token",
    redact: bool = True,
    strategy: str = "wa_id",
    static_thread_id: str | None = None,
) -> WhatsAppChannelModel:
    cfg = WhatsAppChannelModel(
        verify_token=_make_secret(verify),
        app_secret=_make_secret(secret),
        access_token=_make_secret(access),
        phone_number_id=phone_number_id,
        default_thread_strategy=strategy,  # type: ignore[arg-type]
        static_thread_id=static_thread_id,
        redact_phone_in_traces=redact,
    )
    return cfg


def _sign(body: bytes, app_secret: str) -> str:
    return "sha256=" + hmac.new(
        app_secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()


def _inbound_payload(*, message_id: str, from_wa_id: str, text: str) -> dict[str, Any]:
    """Minimal Meta webhook delivery shape for a text message."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "1234567890",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "123456789",
                            },
                            "messages": [
                                {
                                    "from": from_wa_id,
                                    "id": message_id,
                                    "timestamp": "1700000000",
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# verify_signature
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_verify_signature_accepts_valid_hmac() -> None:
    body = b'{"hello": "world"}'
    sig = _sign(body, "secret")
    assert verify_signature(raw_body=body, header_value=sig, app_secret="secret")


@pytest.mark.unit
def test_verify_signature_rejects_tampered_body() -> None:
    body = b'{"hello": "world"}'
    sig = _sign(body, "secret")
    assert not verify_signature(
        raw_body=b'{"hello": "tampered"}', header_value=sig, app_secret="secret"
    )


@pytest.mark.unit
def test_verify_signature_rejects_wrong_secret() -> None:
    body = b'{"hello": "world"}'
    sig = _sign(body, "right-secret")
    assert not verify_signature(
        raw_body=body, header_value=sig, app_secret="wrong-secret"
    )


@pytest.mark.unit
def test_verify_signature_rejects_missing_prefix() -> None:
    body = b'{"hello": "world"}'
    bare_hex = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
    assert not verify_signature(
        raw_body=body, header_value=bare_hex, app_secret="secret"
    )


@pytest.mark.unit
def test_verify_signature_rejects_empty_header() -> None:
    assert not verify_signature(raw_body=b"", header_value="", app_secret="secret")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chunk_text_returns_original_under_limit() -> None:
    assert _chunk_text("short", max_chars=100) == ["short"]


@pytest.mark.unit
def test_chunk_text_prefers_paragraph_break() -> None:
    # 60 chars before split point, 60 chars after
    text = ("a" * 60) + "\n\n" + ("b" * 60)
    chunks = _chunk_text(text, max_chars=70)
    assert len(chunks) == 2
    assert chunks[0] == "a" * 60
    assert chunks[1] == "b" * 60


@pytest.mark.unit
def test_chunk_text_hard_wraps_overlong_runs() -> None:
    text = "x" * 250
    chunks = _chunk_text(text, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


@pytest.mark.unit
def test_parse_inbound_text_text_type() -> None:
    message = {"type": "text", "text": {"body": "hello"}}
    assert _parse_inbound_text(message) == "hello"


@pytest.mark.unit
def test_parse_inbound_text_returns_none_for_unsupported_types() -> None:
    assert _parse_inbound_text({"type": "image", "image": {"id": "x"}}) is None
    assert _parse_inbound_text({"type": "audio", "audio": {"id": "x"}}) is None


@pytest.mark.unit
def test_parse_inbound_text_interactive_button_reply() -> None:
    message = {
        "type": "interactive",
        "interactive": {
            "type": "button_reply",
            "button_reply": {"id": "btn_yes", "title": "Yes"},
        },
    }
    assert _parse_inbound_text(message) == "Yes"


@pytest.mark.unit
def test_thread_key_wa_id_strategy() -> None:
    cfg = _make_config()
    assert _thread_key(cfg, wa_id="15551234567") == "15551234567"


@pytest.mark.unit
def test_thread_key_compound_strategy_includes_phone_number_id() -> None:
    cfg = _make_config(strategy="wa_id+phone_number_id")
    key = _thread_key(cfg, wa_id="15551234567")
    assert key == "123456789:15551234567"


@pytest.mark.unit
def test_thread_key_static_strategy() -> None:
    cfg = _make_config(strategy="static", static_thread_id="thread-a")
    assert _thread_key(cfg, wa_id="anyone") == "static:thread-a"


@pytest.mark.unit
def test_whatsapp_config_static_requires_thread_id() -> None:
    with pytest.raises(ValueError, match="static_thread_id is required"):
        WhatsAppChannelModel(
            verify_token=_make_secret("v"),
            app_secret=_make_secret("s"),
            access_token=_make_secret("a"),
            phone_number_id="x",
            default_thread_strategy="static",
        )


@pytest.mark.unit
def test_whatsapp_config_rejects_webhook_path_without_slash() -> None:
    with pytest.raises(ValueError, match="webhook_path must start with"):
        WhatsAppChannelModel(
            verify_token=_make_secret("v"),
            app_secret=_make_secret("s"),
            access_token=_make_secret("a"),
            phone_number_id="x",
            webhook_path="channels/whatsapp/webhook",
        )


# ---------------------------------------------------------------------------
# _extract_reply_text
# ---------------------------------------------------------------------------


def _message_item(text: str, *, item_id: str = "msg-1") -> OutputItem:
    """Build an MLflow OutputItem mimicking a finished assistant message."""
    return OutputItem(
        type="message",
        id=item_id,
        role="assistant",
        content=[{"type": "output_text", "text": text, "annotations": []}],
    )


@pytest.mark.unit
def test_extract_reply_text_single_message() -> None:
    response = ResponsesAgentResponse(output=[_message_item("hello world")])
    assert _extract_reply_text(response) == "hello world"


@pytest.mark.unit
def test_extract_reply_text_concatenates_multiple_blocks() -> None:
    item = OutputItem(
        type="message",
        id="m1",
        role="assistant",
        content=[
            {"type": "output_text", "text": "first chunk", "annotations": []},
            {"type": "output_text", "text": "second chunk", "annotations": []},
        ],
    )
    response = ResponsesAgentResponse(output=[item])
    assert _extract_reply_text(response) == "first chunk\nsecond chunk"


@pytest.mark.unit
def test_extract_reply_text_empty_output_returns_empty_string() -> None:
    response = ResponsesAgentResponse(output=[])
    assert _extract_reply_text(response) == ""


# ---------------------------------------------------------------------------
# ChannelStore (in-memory)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_channel_store_dedup_blocks_replays() -> None:
    async def go() -> None:
        store = ChannelStore(database=None)
        await store.ensure_schema()
        first = await store.record_message(message_id="msg-1", channel="whatsapp")
        second = await store.record_message(message_id="msg-1", channel="whatsapp")
        assert first.inserted is True
        assert second.inserted is False

    asyncio.run(go())


@pytest.mark.unit
def test_channel_store_thread_is_stable_per_key() -> None:
    async def go() -> None:
        store = ChannelStore(database=None)
        t1 = await store.get_or_create_thread(thread_key="wa-1", channel="whatsapp")
        t2 = await store.get_or_create_thread(thread_key="wa-1", channel="whatsapp")
        t3 = await store.get_or_create_thread(thread_key="wa-2", channel="whatsapp")
        assert t1 == t2
        assert t1 != t3

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Full route surface via FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_routes() -> TestClient:
    """Mount the WhatsApp routes on a fresh FastAPI app and patch out the agent."""
    fastapi_app = FastAPI()

    # Build a minimal AppConfig stub — mount_whatsapp_routes only reads
    # config.app.long_running.database (we leave it None so the store falls
    # back to in-memory mode).
    class _AppStub:
        long_running = None
        channels = None

    class _ConfigStub:
        app = _AppStub()

    wa_config = _make_config()
    mount_whatsapp_routes(fastapi_app, _ConfigStub(), wa_config)  # type: ignore[arg-type]

    return TestClient(fastapi_app)


@pytest.mark.unit
def test_verify_handshake_returns_challenge(client_with_routes: TestClient) -> None:
    resp = client_with_routes.get(
        "/channels/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "v-token",
            "hub.challenge": "1234567890",
        },
    )
    assert resp.status_code == 200
    assert resp.text == "1234567890"


@pytest.mark.unit
def test_verify_handshake_rejects_bad_token(client_with_routes: TestClient) -> None:
    resp = client_with_routes.get(
        "/channels/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong",
            "hub.challenge": "1234567890",
        },
    )
    assert resp.status_code == 403


@pytest.mark.unit
def test_inbound_rejects_invalid_signature(client_with_routes: TestClient) -> None:
    body = json.dumps(
        _inbound_payload(message_id="m1", from_wa_id="15551234567", text="hi")
    ).encode("utf-8")
    resp = client_with_routes.post(
        "/channels/whatsapp/webhook",
        content=body,
        headers={"X-Hub-Signature-256": "sha256=deadbeef"},
    )
    assert resp.status_code == 403


@pytest.mark.unit
def test_inbound_accepts_valid_signature_and_dispatches() -> None:
    """End-to-end: valid signature -> 200 ACK + agent dispatched + outbound called."""
    fastapi_app = FastAPI()

    class _AppStub:
        long_running = None
        channels = None

    class _ConfigStub:
        app = _AppStub()

    wa_config = _make_config()

    # Patch the agent handler and the outbound HTTP sender so we don't need
    # a real LangGraph or graph.facebook.com.
    fake_response = ResponsesAgentResponse(
        output=[_message_item("agent reply text")]
    )
    fake_handler = AsyncMock(return_value=fake_response)
    fake_send = AsyncMock(return_value=None)

    with patch(
        "dao_ai.apps.channels.whatsapp.send_text", new=fake_send
    ), patch.dict(
        "sys.modules",
        {
            # Stub dao_ai.apps.handlers so the lazy import inside _dispatch_one
            # resolves without loading real config / autolog.
            "dao_ai.apps.handlers": type(
                "stub", (), {"non_streaming": fake_handler}
            )(),
        },
    ):
        mount_whatsapp_routes(fastapi_app, _ConfigStub(), wa_config)  # type: ignore[arg-type]

        body = json.dumps(
            _inbound_payload(
                message_id="msg-abc", from_wa_id="15551234567", text="hi"
            )
        ).encode("utf-8")
        sig = _sign(body, "app-secret")

        with TestClient(fastapi_app) as client:
            resp = client.post(
                "/channels/whatsapp/webhook",
                content=body,
                headers={
                    "X-Hub-Signature-256": sig,
                    "Content-Type": "application/json",
                },
            )

        assert resp.status_code == 200
        # The TestClient context-manager runs lifespan and gives the
        # asyncio.create_task time to finish before exit.
        assert fake_handler.await_count == 1
        sent_request = fake_handler.await_args.args[0]
        assert sent_request.custom_inputs["configurable"]["channel"] == "whatsapp"
        assert sent_request.custom_inputs["configurable"]["thread_id"]
        assert fake_send.await_count == 1
        assert fake_send.await_args.kwargs["to_wa_id"] == "15551234567"
        assert fake_send.await_args.kwargs["text"] == "agent reply text"


@pytest.mark.unit
def test_inbound_dedups_meta_redelivery() -> None:
    """Replayed delivery with the same message_id only dispatches once."""
    fastapi_app = FastAPI()

    class _AppStub:
        long_running = None
        channels = None

    class _ConfigStub:
        app = _AppStub()

    wa_config = _make_config()

    fake_response = ResponsesAgentResponse(output=[_message_item("reply")])
    fake_handler = AsyncMock(return_value=fake_response)
    fake_send = AsyncMock(return_value=None)

    with patch(
        "dao_ai.apps.channels.whatsapp.send_text", new=fake_send
    ), patch.dict(
        "sys.modules",
        {
            "dao_ai.apps.handlers": type(
                "stub", (), {"non_streaming": fake_handler}
            )(),
        },
    ):
        mount_whatsapp_routes(fastapi_app, _ConfigStub(), wa_config)  # type: ignore[arg-type]

        body = json.dumps(
            _inbound_payload(
                message_id="msg-dup", from_wa_id="15551234567", text="hi"
            )
        ).encode("utf-8")
        sig = _sign(body, "app-secret")
        headers = {"X-Hub-Signature-256": sig, "Content-Type": "application/json"}

        with TestClient(fastapi_app) as client:
            r1 = client.post("/channels/whatsapp/webhook", content=body, headers=headers)
            r2 = client.post("/channels/whatsapp/webhook", content=body, headers=headers)

        assert r1.status_code == 200
        assert r2.status_code == 200
        # First delivery dispatches; replay is dropped at the dedup step.
        assert fake_handler.await_count == 1
        assert fake_send.await_count == 1
