"""Load-time parameter substitution for DAO AI YAML configs.

Configs may declare typed input parameters at the top level::

    parameters:
      catalog:
        description: Unity Catalog catalog name
        default: main
      module_id:
        description: Workshop module identifier
        # no default => required

and reference them inline anywhere in any string value, using either
``${param.NAME}`` or ``${var.NAME}`` (the two prefixes are interchangeable
aliases - ``${var.…}`` matches the Databricks Asset Bundle convention,
``${param.…}`` matches the ``parameters:`` block name)::

    schemas:
      workshop_schema:
        catalog_name: ${var.catalog}
        schema_name: ${param.schema:-dao_ai}

Substitution happens once, before the YAML is handed to MLflow's
``ModelConfig``. Resolution precedence for each reference:

    1. ``cli_vars[NAME]`` (CLI ``--var name=value`` / ``AppConfig.from_file(params=...)``)
    2. Process environment variable, with ``NAME`` upper-cased and ``.``/``-``
       normalized to ``_`` (e.g. ``${var.catalog.name}`` -> ``CATALOG_NAME``)
    3. ``parameters[NAME].default``
    4. Inline ``${var.NAME:-fallback}`` on the reference itself
    5. Error - reference is required and unresolved

This deliberately stays a load-time, text-level mechanism. The existing typed
``EnvironmentVariableModel`` (``env: FOO``) and ``SecretVariableModel``
(``scope:/secret:``) remain the right tools for runtime resolution inside a
deployed model or app.

YAML quoting caveat
-------------------
Substitution is purely text-level - the substituted value is spliced into
the YAML before the YAML parser runs. If a value contains YAML-special
characters (``:`` followed by space, leading ``#``, ``-``, ``[``, ``{``,
embedded newlines, quotes), an unquoted reference can produce invalid YAML.

When a reference may receive arbitrary user input, quote it in the YAML::

    prompt: "${var.user_prompt}"   # safe regardless of value content
    label: ${var.label}            # OK only for plain alphanumeric values

Substitution does not recurse: if a substituted value happens to contain
``${var.x}`` literally, it is preserved as-is and not re-resolved.
"""

from __future__ import annotations

import os
import re
from typing import Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

PARAM_PATTERN: re.Pattern[str] = re.compile(
    r"\$\{(?:param|var)\.(?P<name>[a-zA-Z_][a-zA-Z0-9_.\-]*)"
    r"(?::-(?P<default>[^}]*))?\}"
)


class ParameterDeclarationModel(BaseModel):
    """Declared input parameter for a DAO AI config file.

    Declared in the top-level ``parameters:`` block and referenced inline
    with ``${param.NAME}`` or its alias ``${var.NAME}``. See
    :mod:`dao_ai.config_vars` for resolution rules.
    """

    model_config = ConfigDict(frozen=True, use_enum_values=True, extra="forbid")

    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of what this parameter controls.",
    )
    default: Optional[str] = Field(
        default=None,
        description="Default value if not provided at load time. Omit to make required.",
    )


class ConfigVariableError(ValueError):
    """Raised when load-time parameter substitution fails.

    Carries structured ``missing_required`` and ``undeclared`` lists so
    callers (e.g. ``dao-ai validate``) can render a combined report.
    """

    def __init__(
        self,
        *,
        path: str,
        missing_required: Optional[list[str]] = None,
        undeclared: Optional[list[str]] = None,
    ) -> None:
        self.path: str = path
        self.missing_required: list[str] = sorted(set(missing_required or []))
        self.undeclared: list[str] = sorted(set(undeclared or []))

        parts: list[str] = [f"Config parameter error in {path}:"]
        if self.missing_required:
            parts.append(
                f"  missing required: {', '.join(self.missing_required)}. "
                "Pass with --var name=value or set the equivalent env var."
            )
        if self.undeclared:
            parts.append(
                f"  undeclared ${{param.NAME}} / ${{var.NAME}} references: "
                f"{', '.join(self.undeclared)}. "
                "Add them to the top-level parameters: block."
            )
        super().__init__("\n".join(parts))


def find_param_references(text: str) -> set[str]:
    """Return the distinct names referenced via ``${param.NAME}`` or ``${var.NAME}``."""
    return {m.group("name") for m in PARAM_PATTERN.finditer(text)}


def _env_key(name: str) -> str:
    """Normalize a parameter name to its env-var lookup key.

    ``${var.catalog.name}`` (or ``${param.catalog.name}``) -> ``CATALOG_NAME``.
    """
    return name.upper().replace(".", "_").replace("-", "_")


class ResolvedParameter(BaseModel):
    """Result of resolving a single declared parameter, used by ``vars list``."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: Optional[str]
    source: str
    required: bool
    default: Optional[str]
    description: Optional[str]


def substitute_params(
    text: str,
    *,
    declarations: Optional[Mapping[str, ParameterDeclarationModel]] = None,
    cli_vars: Optional[Mapping[str, str]] = None,
    env: Optional[Mapping[str, str]] = None,
    source: str = "<string>",
) -> str:
    """Render ``${param.NAME}`` and ``${var.NAME}`` references to literal values.

    Both prefixes are interchangeable aliases for the same parameter namespace.

    Raises :class:`ConfigVariableError` when references are undeclared
    (when ``declarations`` is provided) or when required references cannot
    be resolved.
    """
    decls: Mapping[str, ParameterDeclarationModel] = declarations or {}
    overrides: Mapping[str, str] = cli_vars or {}
    env_map: Mapping[str, str] = env if env is not None else os.environ

    references: set[str] = find_param_references(text)
    undeclared: list[str] = (
        sorted(n for n in references if n not in decls) if decls else []
    )

    missing: list[str] = []

    def _resolve(match: re.Match[str]) -> str:
        name: str = match.group("name")
        inline_default: Optional[str] = match.group("default")
        if name in overrides:
            return str(overrides[name])
        env_name: str = _env_key(name)
        if env_name in env_map:
            return env_map[env_name]
        decl: Optional[ParameterDeclarationModel] = decls.get(name)
        if decl is not None and decl.default is not None:
            return decl.default
        if inline_default is not None:
            return inline_default
        # Only count declared references as "missing required". Undeclared
        # references are reported separately via `undeclared` so we never
        # double-count them in the error message.
        if not decls or name in decls:
            missing.append(name)
        return match.group(0)

    rendered: str = PARAM_PATTERN.sub(_resolve, text)
    if undeclared or missing:
        raise ConfigVariableError(
            path=source,
            missing_required=missing,
            undeclared=undeclared,
        )
    return rendered


def resolve_parameters(
    declarations: Mapping[str, ParameterDeclarationModel],
    *,
    cli_vars: Optional[Mapping[str, str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> list[ResolvedParameter]:
    """Resolve every declared parameter and report where its value came from.

    Used by ``dao-ai vars list`` to render a discoverability table.
    Does not raise; missing required parameters appear with ``source='MISSING'``.
    """
    overrides: Mapping[str, str] = cli_vars or {}
    env_map: Mapping[str, str] = env if env is not None else os.environ

    results: list[ResolvedParameter] = []
    for name, decl in declarations.items():
        value: Optional[str] = None
        source: str
        if name in overrides:
            value = str(overrides[name])
            source = "--var"
        elif _env_key(name) in env_map:
            value = env_map[_env_key(name)]
            source = "env"
        elif decl.default is not None:
            value = decl.default
            source = "default"
        else:
            source = "MISSING"

        results.append(
            ResolvedParameter(
                name=name,
                value=value,
                source=source,
                required=decl.default is None,
                default=decl.default,
                description=decl.description,
            )
        )
    return results
