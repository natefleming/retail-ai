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
from typing import Any, Callable, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

PARAM_PATTERN: re.Pattern[str] = re.compile(
    r"\$\{(?:param|var)\.(?P<name>[a-zA-Z_][a-zA-Z0-9_.\-]*)"
    r"(?::-(?P<default>[^}]*))?\}"
)

WORKSPACE_PATTERN: re.Pattern[str] = re.compile(
    r"\$\{workspace\.(?P<path>[a-zA-Z_][a-zA-Z0-9_.]*)\}"
)


def _yaml_comment_spans(text: str) -> list[tuple[int, int]]:
    """Return ``[start, end)`` offsets of YAML ``#`` comment ranges in *text*.

    A ``#`` starts a comment when it appears at the start of a line
    (after optional whitespace) or is preceded by whitespace on the
    same line, and is not inside single- or double-quoted strings. The
    comment extends to (but does not include) the next newline.

    This is a best-effort text scan; it does not track YAML block scalars
    (``|`` / ``>``). A ``#``-prefixed content line inside a block scalar
    can be misclassified as a comment. If that becomes a problem for a
    specific config, declare the parameter or move the ``${var.X}``
    reference to a line whose first non-space character is not ``#``.
    """
    spans: list[tuple[int, int]] = []
    n: int = len(text)
    line_start: int = 0
    while line_start <= n:
        line_end: int = text.find("\n", line_start)
        if line_end == -1:
            line_end = n
        line: str = text[line_start:line_end]
        in_squote: bool = False
        in_dquote: bool = False
        j: int = 0
        while j < len(line):
            ch: str = line[j]
            if in_squote:
                if ch == "'":
                    if j + 1 < len(line) and line[j + 1] == "'":
                        j += 2
                        continue
                    in_squote = False
            elif in_dquote:
                if ch == "\\" and j + 1 < len(line):
                    j += 2
                    continue
                if ch == '"':
                    in_dquote = False
            else:
                if ch == "'":
                    in_squote = True
                elif ch == '"':
                    in_dquote = True
                elif ch == "#" and (j == 0 or line[j - 1] in " \t"):
                    spans.append((line_start + j, line_end))
                    break
            j += 1
        if line_end == n:
            break
        line_start = line_end + 1
    return spans


def _in_any_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    """Return True if ``pos`` falls inside any ``[start, end)`` span."""
    for start, end in spans:
        if start <= pos < end:
            return True
        if start > pos:
            return False
    return False


_SUPPORTED_WORKSPACE_PATHS: frozenset[str] = frozenset(
    {
        "host",
        "current_user.userName",
        "current_user.short_name",
        "current_user.domain_friendly_name",
    }
)


class WorkspaceVariableError(ValueError):
    """Raised when ``${workspace.*}`` substitution fails (unknown path, missing client, etc.)."""


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
    """Return the distinct names referenced via ``${param.NAME}`` or ``${var.NAME}``.

    References that fall inside YAML ``#`` comments are ignored, so a
    commented-out example line like ``# schema: ${var.schema}`` does
    not require the parameter to be declared.
    """
    spans: list[tuple[int, int]] = _yaml_comment_spans(text)
    return {
        m.group("name")
        for m in PARAM_PATTERN.finditer(text)
        if not _in_any_span(m.start(), spans)
    }


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
    defer: Optional[set[str]] = None,
    source: str = "<string>",
) -> str:
    """Render ``${param.NAME}`` and ``${var.NAME}`` references to literal values.

    Both prefixes are interchangeable aliases for the same parameter namespace.

    Raises :class:`ConfigVariableError` when references are undeclared
    (when ``declarations`` is provided) or when required references cannot
    be resolved.

    ``defer`` names are left in place as their literal ``${var.NAME}`` reference
    and never counted as missing — even when they have no value and no default.
    Used only by the workflow staging path (see
    :func:`dao_ai.pipeline.bundle.write_pipeline_bundle`) to preserve a Genie
    room's ``space_id: ${var.X}`` binding so the provisioning notebook can detect
    it via :func:`is_parameter` and create/forward the space at run time. Inert
    (no behavior change) when ``None``.
    """
    decls: Mapping[str, ParameterDeclarationModel] = declarations or {}
    overrides: Mapping[str, str] = cli_vars or {}
    env_map: Mapping[str, str] = env if env is not None else os.environ
    deferred: set[str] = defer or set()

    comment_spans: list[tuple[int, int]] = _yaml_comment_spans(text)
    references: set[str] = find_param_references(text)
    undeclared: list[str] = (
        sorted(n for n in references if n not in decls) if decls else []
    )

    missing: list[str] = []

    def _resolve(match: re.Match[str]) -> str:
        # References inside YAML comments are documentation — leave the
        # literal ${var.X} in place and don't count them toward missing.
        if _in_any_span(match.start(), comment_spans):
            return match.group(0)
        name: str = match.group("name")
        inline_default: Optional[str] = match.group("default")
        # Deferred names survive as their literal ${var.NAME} reference (the
        # workflow provisioning path resolves them at run time) and are never
        # counted missing, even with no value/default.
        if name in deferred:
            return match.group(0)
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
            source = "--param"
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


def find_workspace_references(text: str) -> set[str]:
    """Return the distinct paths referenced via ``${workspace.PATH}``.

    References inside YAML ``#`` comments are ignored.
    """
    spans: list[tuple[int, int]] = _yaml_comment_spans(text)
    return {
        m.group("path")
        for m in WORKSPACE_PATTERN.finditer(text)
        if not _in_any_span(m.start(), spans)
    }


def _email_short_name(email: str) -> str:
    """Email prefix before ``@`` (dots intact). Matches the DABs convention."""
    return email.split("@", 1)[0] if email else ""


def _email_domain(email: str) -> str:
    """Email domain after ``@`` (e.g. ``databricks.com``). Matches the DABs convention."""
    parts: list[str] = email.split("@", 1) if email else []
    return parts[1] if len(parts) == 2 else ""


def substitute_workspace_refs(
    text: str,
    *,
    workspace_client_factory: Optional[Callable[[], Any]] = None,
    source: str = "<string>",
) -> str:
    """Render ``${workspace.*}`` references to literal values.

    Mirrors the Databricks Asset Bundles substitution namespace. Supported paths:

        ``${workspace.host}``                                 — workspace URL (no trailing slash)
        ``${workspace.current_user.userName}``                — full email
        ``${workspace.current_user.short_name}``              — email prefix (dots intact)
        ``${workspace.current_user.domain_friendly_name}``    — email domain (``databricks.com``)

    The ``workspace_client_factory`` is invoked lazily (only when the text
    contains at least one ``${workspace.*}`` reference). When omitted, a
    default ``WorkspaceClient()`` is constructed - it will honour the active
    profile / environment exactly like ``databricks bundle``.

    Raises:
        WorkspaceVariableError: if the text references an unsupported path,
            or if constructing the WorkspaceClient / fetching user info fails.
    """
    comment_spans: list[tuple[int, int]] = _yaml_comment_spans(text)
    refs: set[str] = find_workspace_references(text)
    if not refs:
        return text

    unknown: list[str] = sorted(r for r in refs if r not in _SUPPORTED_WORKSPACE_PATHS)
    if unknown:
        raise WorkspaceVariableError(
            f"Unsupported ${{workspace.*}} reference(s) in {source}: "
            f"{', '.join(unknown)}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_WORKSPACE_PATHS))}."
        )

    def _default_factory() -> Any:
        from databricks.sdk import WorkspaceClient

        return WorkspaceClient()

    factory: Callable[[], Any] = workspace_client_factory or _default_factory

    cache: dict[str, str] = {}
    client_box: dict[str, Any] = {}
    email_box: dict[str, str] = {}

    def _client() -> Any:
        if "c" not in client_box:
            try:
                client_box["c"] = factory()
            except Exception as exc:
                raise WorkspaceVariableError(
                    f"Failed to build a WorkspaceClient while resolving "
                    f"${{workspace.*}} in {source}: {exc}"
                ) from exc
        return client_box["c"]

    def _email() -> str:
        if "e" not in email_box:
            try:
                user: Any = _client().current_user.me()
            except WorkspaceVariableError:
                raise
            except Exception as exc:
                raise WorkspaceVariableError(
                    f"Failed to fetch current user while resolving "
                    f"${{workspace.current_user.*}} in {source}: {exc}"
                ) from exc
            email_box["e"] = str(getattr(user, "user_name", "") or "")
        return email_box["e"]

    def _resolve(path: str) -> str:
        if path in cache:
            return cache[path]
        try:
            if path == "host":
                host: str = str(_client().config.host or "")
                value: str = host.rstrip("/")
            elif path == "current_user.userName":
                value = _email()
            elif path == "current_user.short_name":
                value = _email_short_name(_email())
            elif path == "current_user.domain_friendly_name":
                value = _email_domain(_email())
            else:  # pragma: no cover - guarded by _SUPPORTED_WORKSPACE_PATHS
                raise WorkspaceVariableError(f"Unsupported workspace path: {path}")
        except WorkspaceVariableError:
            raise
        except Exception as exc:
            raise WorkspaceVariableError(
                f"Failed to resolve ${{workspace.{path}}} in {source}: {exc}"
            ) from exc
        cache[path] = value
        return value

    def _sub(match: re.Match[str]) -> str:
        # Leave comment-embedded references untouched (documentation, not config).
        if _in_any_span(match.start(), comment_spans):
            return match.group(0)
        return _resolve(match.group("path"))

    return WORKSPACE_PATTERN.sub(_sub, text)
