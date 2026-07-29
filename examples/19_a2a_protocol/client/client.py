"""End-to-end A2A client for any dao-ai agent deployed to Databricks Apps.

Exercises the full A2A capability surface against a live deployment:

  1. Fetches the Agent Card at ``/.well-known/agent-card.json``.
  2. Sends a single ``message/send`` and prints the artifact text.
  3. Streams a ``message/stream`` and prints each SSE event as it arrives.
  4. Demonstrates the HITL contract: when the agent enters
     ``input-required``, supplies a resume payload via ``DataPart``
     ``{"decisions": [...]}``.

Run against a deployed dao-ai Apps URL:

  uv run python examples/19_a2a_protocol/client/client.py \\
      --base-url https://<app-host>.cloud.databricksapps.com \\
      --bearer-token "$(databricks auth token --profile fevm | jq -r .access_token)"

Or against a local dev server:

  DAO_AI_CONFIG_PATH=examples/19_a2a_protocol/a2a_minimal.yaml \\
      uv run python -m dao_ai.apps.server &
  uv run python examples/19_a2a_protocol/client/client.py --base-url http://localhost:8000

No Databricks SDK needed; only ``httpx`` (already a transitive dep of dao-ai).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

import httpx


def _jsonrpc(method: str, params: dict[str, Any], request_id: str | None = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id or uuid.uuid4().hex[:8],
        "method": method,
        "params": params,
    }


def _text_message(text: str, *, context_id: str | None = None) -> dict:
    msg: dict[str, Any] = {
        "messageId": uuid.uuid4().hex,
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
    }
    if context_id is not None:
        msg["contextId"] = context_id
    return {"message": msg}


def _data_resume(decisions: list[dict], *, context_id: str, task_id: str) -> dict:
    return {
        "message": {
            "messageId": uuid.uuid4().hex,
            "role": "user",
            "parts": [{"kind": "data", "data": {"decisions": decisions}}],
            "contextId": context_id,
            "taskId": task_id,
        }
    }


def fetch_agent_card(client: httpx.Client, base_url: str) -> dict:
    print(f"\n[1/4] GET {base_url}/.well-known/agent-card.json")
    resp = client.get(f"{base_url}/.well-known/agent-card.json")
    resp.raise_for_status()
    card = resp.json()
    print(f"  name        : {card['name']}")
    print(f"  description : {card.get('description', '')}")
    print(f"  url         : {card.get('url', '')}")
    print(f"  version     : {card.get('version', '')}")
    print(f"  skills      : {[s['id'] for s in card.get('skills', [])]}")
    print(
        f"  security    : {list((card.get('securitySchemes') or {}).keys())}"
    )
    print(f"  capabilities: {card.get('capabilities', {})}")
    return card


def message_send(
    client: httpx.Client, base_url: str, text: str, *, context_id: str | None = None
) -> dict:
    label = "[2/4]"
    print(f"\n{label} POST {base_url}/a2a  (method=message/send, text={text!r})")
    payload = _jsonrpc("message/send", _text_message(text, context_id=context_id))
    resp = client.post(f"{base_url}/a2a", json=payload)
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        print(f"  ❌ {body['error']}")
        return body
    result = body["result"]
    state = result.get("status", {}).get("state")
    print(f"  task id  : {result.get('id')}")
    print(f"  context  : {result.get('contextId')}")
    print(f"  state    : {state}")
    for artifact in result.get("artifacts") or []:
        for part in artifact.get("parts", []):
            kind = part.get("kind")
            if kind == "text":
                print(f"  text     : {part.get('text')}")
            elif kind == "data":
                print(f"  data     : {json.dumps(part.get('data'), indent=2)}")
    return body


def message_stream(client: httpx.Client, base_url: str, text: str) -> list[dict]:
    print(f"\n[3/4] POST {base_url}/a2a  (method=message/stream, text={text!r})")
    payload = _jsonrpc("message/stream", _text_message(text))
    events: list[dict] = []
    with client.stream(
        "POST",
        f"{base_url}/a2a",
        json=payload,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw or not raw.startswith("data:"):
                continue
            chunk = raw[len("data:") :].strip()
            if not chunk:
                continue
            event = json.loads(chunk)
            events.append(event)
            result = event.get("result", {})
            kind = result.get("kind")
            if kind == "status-update":
                print(
                    f"  STATUS   {result.get('status', {}).get('state'):14s}  "
                    f"final={result.get('final')}"
                )
            elif kind == "artifact-update":
                parts = result.get("artifact", {}).get("parts", [])
                text_parts = [p["text"] for p in parts if p.get("kind") == "text"]
                if text_parts:
                    print(f"  ARTIFACT {' '.join(text_parts)[:120]!r}")
                else:
                    print(f"  ARTIFACT (no text, {len(parts)} parts)")
            else:
                print(f"  EVENT    {kind}")
    return events


def hitl_round_trip(client: httpx.Client, base_url: str, prompt: str | None = None) -> None:
    """Run a HITL flow: send → if input-required, resume via DataPart decisions.

    Whether your agent actually emits an interrupt depends on its tools and
    middleware; this demo just shows the wire shape. If the first
    ``message/send`` completes immediately, we print that and skip the
    resume step.

    For agents whose tools are tagged ``human_in_the_loop`` (see
    ``examples/19_a2a_protocol/a2a_hitl_obo.yaml``), supply a
    prompt that forces a tool call — e.g. ``--hitl-message "What time is it?"``
    on the demo HITL+OBO agent.
    """
    text = prompt or "Run a task that requires my approval before continuing."
    print(f"\n[4/4] HITL demo (interrupt → input-required → resume)")
    print(f"      prompt: {text!r}")
    payload = _jsonrpc(
        "message/send",
        _text_message(text),
    )
    resp = client.post(f"{base_url}/a2a", json=payload)
    resp.raise_for_status()
    body = resp.json()
    result = body.get("result", {})
    state = result.get("status", {}).get("state")
    print(f"  initial state: {state}")

    if state != "input-required":
        print("  (agent did not request input — done.)")
        return

    task_id = result["id"]
    context_id = result["contextId"]
    interrupts = result.get("status", {}).get("message", {}).get("parts", [])
    print(f"  agent asked for input; resuming with decisions=[{{'type': 'approve'}}]")

    resume_payload = _jsonrpc(
        "message/send",
        _data_resume(
            decisions=[{"type": "approve"}],
            context_id=context_id,
            task_id=task_id,
        ),
    )
    resp = client.post(f"{base_url}/a2a", json=resume_payload)
    resp.raise_for_status()
    final = resp.json().get("result", {})
    print(f"  resumed state: {final.get('status', {}).get('state')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        required=True,
        help="Deployed Apps URL or local dev server (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--bearer-token",
        default=None,
        help="Bearer token to forward (typically a Databricks workspace token)",
    )
    parser.add_argument(
        "--skip-stream", action="store_true", help="Skip the message/stream demo"
    )
    parser.add_argument(
        "--skip-hitl", action="store_true", help="Skip the HITL demo"
    )
    parser.add_argument(
        "--message",
        default="Say hi in one sentence.",
        help="Text payload for the message/send and message/stream calls",
    )
    parser.add_argument(
        "--hitl-message",
        default=None,
        help=(
            "Prompt for the HITL probe. Set to something that forces the "
            "agent to call a HITL-tagged tool, e.g. 'What time is it?' for "
            "the a2a_hitl_obo demo. Defaults to a generic 'requires my "
            "approval' prompt."
        ),
    )
    args = parser.parse_args()

    headers: dict[str, str] = {}
    if args.bearer_token:
        headers["Authorization"] = f"Bearer {args.bearer_token}"

    base_url = args.base_url.rstrip("/")

    with httpx.Client(headers=headers, timeout=60.0) as client:
        fetch_agent_card(client, base_url)
        message_send(client, base_url, args.message)
        if not args.skip_stream:
            message_stream(client, base_url, args.message)
        if not args.skip_hitl:
            hitl_round_trip(client, base_url, prompt=args.hitl_message)

    print("\n✓ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
