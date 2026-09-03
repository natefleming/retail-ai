"""Discover the ``custom_inputs.configurable`` fields an agent's config requires.

The dao-ai Console prepopulates its custom-inputs editor with the required and
example values an agent needs. Those are declared on the config's
``CustomFieldValidationMiddleware`` specs (the
``create_custom_field_validation_middleware`` factory). This walks the parsed
config for those specs and returns their fields — config-agnostic, read-only,
non-secret (names / descriptions / example values only).
"""

from __future__ import annotations

from typing import Any, Iterator

from pydantic import BaseModel

# thread_id / user_id are managed by the runtime, never prompted for.
_RUNTIME_MANAGED = frozenset({"thread_id", "user_id"})


def _walk_middleware(obj: Any, seen: set[int]) -> Iterator[Any]:
    """Yield every ``MiddlewareModel`` reachable from ``obj`` (cycle-safe)."""
    from dao_ai.config import MiddlewareModel

    if obj is None or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, MiddlewareModel):
        yield obj
    if isinstance(obj, BaseModel):
        for field_name in type(obj).model_fields:
            yield from _walk_middleware(getattr(obj, field_name, None), seen)
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_middleware(value, seen)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from _walk_middleware(value, seen)


def discover_custom_input_fields(config: Any) -> list[dict[str, Any]]:
    """Return the configurable fields declared by custom-field-validation
    middleware in ``config``.

    Each entry is ``{name, description, required, example_value}``. Fields are
    de-duplicated by name (first wins) and ``thread_id`` / ``user_id`` are
    excluded (runtime-managed). Returns ``[]`` when none are configured.
    """
    fields: dict[str, dict[str, Any]] = {}
    for middleware in _walk_middleware(config, set()):
        if "custom_field_validation" not in middleware.name:
            continue
        for spec in middleware.args.get("fields", []) or []:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            if not name or name in _RUNTIME_MANAGED or name in fields:
                continue
            fields[name] = {
                "name": name,
                "description": spec.get("description"),
                "required": spec.get("required", True),
                "example_value": spec.get("example_value"),
            }
    return list(fields.values())
