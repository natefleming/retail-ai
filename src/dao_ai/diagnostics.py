"""Runtime diagnostics for verifying dao-ai deployment plumbing.

Stage-1 probes for the MS-trace-persistence investigation: capture the
env-var surface that a deployed model actually sees, and redact anything
that looks like a credential before it lands in logs or ``custom_outputs``.

Off by default. Set ``DAO_AI_TRACE_ENV_DUMP=1`` in the endpoint's env vars
to enable the in-model probe.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Mapping

DIAGNOSTIC_ENV_FLAG: str = "DAO_AI_TRACE_ENV_DUMP"

# Keys whose *values* must be redacted. Match on the key name — value
# heuristics (looking for JWT-looking strings, base64 blobs, etc.) are
# too noisy to be trustworthy at boot time.
#
#
# ``AUTHORIZATION`` and ``BEARER`` cover config and env keys that name a bearer
# directly. Note the deliberate absence of a bare ``AUTH``: ``TRACE_RELEVANT_KEYS``
# below surfaces ``DATABRICKS_AUTH_TYPE`` verbatim on purpose, and ``AUTHORIZATION``
# does not match ``AUTH_TYPE``.
#
# This is a *substring* match, which over-matches by design: over-redacting an
# env var costs an operator one puzzled moment, while under-redacting prints a
# credential. Surfaces that cannot absorb a false positive — anything the caller
# has to get back in order to resend it — want :func:`is_secret_field_name`.
_SECRET_KEY_PATTERN: re.Pattern[str] = re.compile(
    r"(SECRET|TOKEN|PASSWORD|KEY|CREDENTIAL|COOKIE|SESSION|AUTHORIZATION|BEARER)",
    re.IGNORECASE,
)

# Whole-segment credential markers for *field-name* surfaces. A name is split on
# separators and camelCase humps, then each segment is matched exactly, so
# ``api_key`` and ``apiKey`` are caught while ``monkey_wrench`` and ``session_id``
# survive — both of which the substring pattern above eats.
#
# ``session`` is absent on purpose: a caller-supplied ``session_id`` is metadata
# that has to round-trip, and a session *credential* is caught by ``token`` or
# ``cookie`` instead.
_SECRET_NAME_SEGMENTS: frozenset[str] = frozenset(
    {
        "apikey",
        "authorization",
        "bearer",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "key",
        "keys",
        "passwd",
        "password",
        "passwords",
        "pwd",
        "secret",
        "secrets",
        "token",
        "tokens",
    }
)

# Credential names that carry no separator to split on, so segment matching alone
# misses them. Only unambiguous compounds belong here — the point of this list is
# to stay precise, so no ``monkey``-catching prefix or suffix matching.
_SECRET_NAME_COMPOUNDS: frozenset[str] = frozenset(
    {
        "accesstoken",
        "apikeys",
        "authtoken",
        "bearertoken",
        "clientsecret",
        "idtoken",
        "privatekey",
        "refreshtoken",
        "secretkey",
        "sessiontoken",
    }
)

# Separators and camelCase humps, so ``x-forwarded-access-token``, ``api_key``
# and ``apiKey`` all split into their words.
_NAME_SEGMENT_BOUNDARY: re.Pattern[str] = re.compile(
    r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])"
)

# Keys we care about surfacing verbatim for the MS-trace investigation.
# Everything else falls back to the redact-or-echo decision. Keeping this
# explicit set means the probe output is scannable — the operator doesn't
# have to grep 300 env vars to find what they came for.
TRACE_RELEVANT_KEYS: tuple[str, ...] = (
    "DATABRICKS_HOST",
    "DATABRICKS_CLIENT_ID",
    "DATABRICKS_CLIENT_SECRET",
    "DATABRICKS_TOKEN",
    "MLFLOW_EXPERIMENT_ID",
    "MLFLOW_TRACING_DESTINATION",
    "MLFLOW_TRACING_SQL_WAREHOUSE_ID",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_REGISTRY_URI",
    "DATABRICKS_AUTH_TYPE",
    "DAO_AI_TRACE_ENV_DUMP",
)


def is_enabled() -> bool:
    """Whether the in-model env probe is enabled for this process."""
    return os.environ.get(DIAGNOSTIC_ENV_FLAG, "").lower() in ("1", "true", "yes")


def is_secret_key(key: str) -> bool:
    """Whether ``key``'s *name* marks its value as credential-shaped.

    Substring match, tuned for env-var and config-key dumps where redacting one
    key too many is harmless. For a field the caller must be able to read back,
    use :func:`is_secret_field_name`.
    """
    return bool(_SECRET_KEY_PATTERN.search(key))


def is_secret_field_name(name: str) -> bool:
    """Whether ``name`` names a credential, matching whole words only.

    For surfaces where a false positive silently deletes a field the caller
    needs: the ``configurable`` block echoed back by
    ``dao_ai.state.context_configurable_fields``, and span payloads redacted by
    ``dao_ai._tracing.redaction``. ``api_key``, ``apiKey`` and
    ``x-forwarded-access-token`` match; ``session_id``, ``monkey_wrench`` and
    ``idempotency_key``'s sibling ``store_num`` do not.

    Note that a whole-segment ``key`` *is* treated as credential-shaped, so
    ``cache_key``-style names are filtered too — catching ``api_key`` and
    ``openai_key`` is worth that much over-matching, and a caller who needs a
    cache key echoed can rename it.
    """
    segments: list[str] = [
        segment.lower() for segment in _NAME_SEGMENT_BOUNDARY.split(name) if segment
    ]
    if any(segment in _SECRET_NAME_SEGMENTS for segment in segments):
        return True
    return "".join(segments) in _SECRET_NAME_COMPOUNDS


def redact_value(key: str, value: str) -> str:
    """Return ``value`` unchanged, or a redacted placeholder if ``key`` looks
    like a credential.

    Preserves length + first/last two chars so the operator can eyeball
    "did the value round-trip" without leaking the secret itself.
    """
    if not is_secret_key(key):
        return value
    if not value:
        return "<empty>"
    if len(value) <= 6:
        return f"<redacted len={len(value)}>"
    return f"{value[:2]}…{value[-2:]} <redacted len={len(value)}>"


def env_snapshot(
    keys: Iterable[str] | None = None,
    include_all: bool = False,
) -> dict[str, str]:
    """Return a redacted view of ``os.environ``.

    Args:
        keys: Iterable of env-var names to include. If ``None``,
            defaults to :data:`TRACE_RELEVANT_KEYS`.
        include_all: If ``True``, also include every other env var
            present in the process (redacted per :func:`redact_value`).
            Off by default — the trace-relevant subset is what the MS
            investigation needs.

    Absent keys are emitted as ``"<unset>"`` so the operator can
    distinguish "stripped by platform" from "never set by dao-ai".
    """
    target_keys: tuple[str, ...] = (
        tuple(keys) if keys is not None else TRACE_RELEVANT_KEYS
    )
    snapshot: dict[str, str] = {}
    for k in target_keys:
        raw: str | None = os.environ.get(k)
        snapshot[k] = "<unset>" if raw is None else redact_value(k, raw)

    if include_all:
        for k, raw in os.environ.items():
            if k in snapshot:
                continue
            snapshot[k] = redact_value(k, raw)

    return snapshot


def redacted_env_var_map(
    env_vars: Mapping[str, str] | None,
) -> dict[str, str]:
    """Redact secret-shaped values in a ``{name: value}`` env-var map.

    Used by the post-deploy reflection probe to log
    ``serving_endpoints.get(...).config.served_entities[0].environment_vars``
    without spilling secrets. Absent input returns an empty dict.
    """
    if not env_vars:
        return {}
    return {k: redact_value(k, str(v)) for k, v in env_vars.items()}
