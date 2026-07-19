"""Emit a deployable Databricks Apps bundle that runs the dao-ai MCP server.

Invoked by ``dao-ai generate-mcp`` (see :mod:`dao_ai.cli`). Delegates the
App + experiment + resources DAB shape to
:func:`dao_ai.apps.bundle.generate_databricks_yaml` /
:func:`dao_ai.apps.bundle.generate_resources_app_yaml`, passing an
MCP-flavored runtime command and disabling the chat-proxy UI env vars.
The result is byte-for-byte the same App-resource pattern that
``generate-bundle`` emits, so the deployed MCP server inherits the same
service-principal credentials, resource grants, experiment binding, and
trace-location wiring.

Files produced:

* ``databricks.yaml`` — bundle metadata, ``include: [resources/*.yml]``,
  targets, and (in ``--development`` mode) ``sync: [dist/*.whl]``.
* ``resources/app.yml`` — the App + experiment block (embedded
  ``config.command`` / ``config.env``, no standalone ``app.yaml``).
* ``pyproject.toml`` — build metadata; runtime deps live in requirements.txt.
* ``requirements.txt`` — unbounded ``dao-ai[mcp]`` (prod) or the bundled
  wheel with ``[mcp]`` extras (dev). Apps' build phase installs from
  this directly.
* ``dao_ai.yaml`` — the rendered (param-substituted) config, stripped of
  its top-level ``parameters:`` block.
* ``.gitignore`` / ``.python-version`` — bundle-parity scaffolding.
* ``README.md`` — deploy snippet + the single tool name to point clients at.
"""

from __future__ import annotations

import shutil
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

from loguru import logger

from dao_ai.apps.bundle import (
    _GITIGNORE_CONTENT,
    _GITIGNORE_DEV_CONTENT,
    _strip_parameters_block,
    generate_databricks_yaml,
    generate_resources_app_yaml,
)
from dao_ai.config import AppConfig
from dao_ai.mcp.agent_tool import _slugify

DEFAULT_CONFIG_FILENAME = "dao_ai.yaml"

# MCP-flavored runtime command. Forwarded to the shared
# :func:`_build_app_block` via ``app_command=`` so everything else — env
# vars, resource bindings, experiment binding, trace_location wiring —
# is reused verbatim from ``generate-bundle``.
_MCP_APP_COMMAND: list[str] = ["python", "-m", "dao_ai.mcp.server"]


def write_mcp_bundle(
    config: AppConfig,
    output_dir: Path,
    *,
    overwrite: bool = False,
    development: bool = False,
) -> None:
    """Write an MCP-server deploy bundle into ``output_dir``.

    Requires ``config.app.name`` (used as both the Databricks App name and
    the single MCP tool's name) and encourages ``config.app.description``
    (surfaced as the tool description to MCP clients).

    When ``development=True``, builds the local dao-ai wheel and bundles it
    into ``output_dir/dist/``; the generated requirements.txt then installs
    from that wheel instead of pulling ``dao-ai[mcp]`` from PyPI.
    """
    if config.app is None or not config.app.name:
        raise ValueError(
            "generate-mcp requires config.app.name (used as the deployed "
            "Databricks App name and the MCP tool name). Add an `app:` "
            "block with `name:` to your dao-ai config."
        )
    if not config.app.description:
        logger.warning(
            "mcp.generate.missing_description — config.app.description is "
            "unset; a generic placeholder will be used as the MCP tool "
            "description. Add `app.description` for a better client UX."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    app_name = _derive_app_name(config)
    tool_name = _slugify(str(config.app.name))
    source_config_path: str | None = getattr(config, "_source_config_path", None)
    config_filename = (
        Path(source_config_path).name if source_config_path else DEFAULT_CONFIG_FILENAME
    )

    written: list[str] = []
    skipped: list[str] = []

    def _write(path: Path, content: str) -> None:
        if path.exists() and not overwrite:
            skipped.append(str(path.relative_to(output_dir)))
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        written.append(str(path.relative_to(output_dir)))

    # Reuse the shared bundle helpers so the App + experiment + resource
    # bindings match ``generate-bundle`` byte-for-byte. MCP disables the
    # chat-proxy UI env vars (no chat UI in the MCP server) and skips the
    # ``artifacts:`` block (MCP doesn't build a user wheel — it installs
    # ``dao-ai[mcp]`` from PyPI via requirements.txt).
    _write(
        output_dir / "databricks.yaml",
        generate_databricks_yaml(
            config,
            development=development,
            config_filename=config_filename,
            app_command=_MCP_APP_COMMAND,
            include_chat_ui=False,
            include_artifacts=False,
        ),
    )

    resources_dir = output_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    _write(
        resources_dir / "app.yml",
        generate_resources_app_yaml(
            config,
            config_filename=config_filename,
            app_command=_MCP_APP_COMMAND,
            include_chat_ui=False,
        ),
    )

    wheel_filename: str | None = None
    if development:
        wheel_filename = _bundle_local_wheel(output_dir, written=written)

    _write(
        output_dir / "pyproject.toml",
        _render_pyproject(app_name=app_name, wheel_filename=wheel_filename),
    )
    _write(
        output_dir / "requirements.txt",
        _make_requirements_txt(development=development, wheel_filename=wheel_filename),
    )
    _write(
        output_dir / ".gitignore",
        _GITIGNORE_DEV_CONTENT if development else _GITIGNORE_CONTENT,
    )
    _write(output_dir / ".python-version", "3.11\n")

    rendered: str | None = getattr(config, "_rendered_yaml", None)
    if rendered is not None:
        _write(output_dir / config_filename, _strip_parameters_block(rendered))
    elif source_config_path is not None:
        _write(output_dir / config_filename, Path(source_config_path).read_text())
    else:
        logger.warning("mcp.generate.no_source_config — skipping config copy")

    _write(
        output_dir / "README.md",
        _render_readme(config, app_name=app_name, tool_name=tool_name),
    )

    print(f"\nMCP bundle generated in {output_dir}/\n")
    for name in written:
        print(f"  {name:<32s} (created)")
    for name in skipped:
        print(f"  {name:<32s} (skipped — re-run with --overwrite to overwrite)")

    print(f"\nMCP tool exposed: {tool_name}")
    print("\nNext steps:")
    print(f"  cd {output_dir}")
    print("  databricks bundle deploy --target dev")
    if config.app.trace_location:
        # Idempotent — safe on every deploy — but load-bearing on
        # re-deploys and after trace_location changes. See
        # `dao-ai link-trace-destination --help` for the full story.
        print(
            f"  dao-ai link-trace-destination -c {config_filename}"
            f"   # links UC trace destination for {app_name}"
        )
    print(f"  databricks bundle run {app_name} --target dev")
    print()
    print("  # Apps' build phase installs deps directly from requirements.txt;")
    print("  # no uv sync or URL rewrite required.")
    print()


def _derive_app_name(config: AppConfig) -> str:
    """Return the Databricks App name — parity with ``generate-bundle``."""
    assert config.app is not None and config.app.name  # enforced by caller
    return str(config.app.name).lower().replace("_", "-")


def _render_pyproject(*, app_name: str, wheel_filename: str | None = None) -> str:
    package_name = app_name.replace("-", "_")
    if wheel_filename:
        return _PYPROJECT_DEV_TEMPLATE.format(
            name=app_name, package_name=package_name, wheel_filename=wheel_filename
        )
    return _PYPROJECT_TEMPLATE.format(
        name=app_name,
        package_name=package_name,
        dao_ai_version=_get_dao_ai_version(),
    )


def _make_requirements_txt(
    *,
    development: bool,
    wheel_filename: str | None = None,
) -> str:
    """Build the requirements.txt content for an MCP app bundle.

    The published pin is intentionally *unbounded* (``dao-ai[mcp]``) — see
    ``dao_ai.apps.bundle._make_requirements_txt`` for the reasoning. We
    install the ``[mcp]`` extra so the server module's dependencies
    (fastmcp, uvicorn, etc.) resolve.
    """
    if development:
        if not wheel_filename:
            raise ValueError(
                "_make_requirements_txt: wheel_filename is required in development mode."
            )
        return f"./dist/{wheel_filename}[mcp]\n"
    return "dao-ai[mcp]\n"


def _bundle_local_wheel(output_dir: Path, *, written: list[str]) -> str:
    """Build (or reuse) a local dao-ai wheel and copy it under ``output_dir/dist/``."""
    project_root = Path(__file__).parents[3]
    source_dir = project_root / "src" / "dao_ai"

    if not source_dir.is_dir():
        raise RuntimeError(
            f"--development requested but dao-ai source tree not found at {source_dir}"
        )

    dist_src = project_root / "dist"
    dist_src.mkdir(parents=True, exist_ok=True)
    for stale in dist_src.glob("dao_ai-*.whl"):
        stale.unlink()

    logger.info("mcp.generate.dev.build_wheel", project_root=str(project_root))
    result = subprocess.run(
        ["uv", "build", "--wheel"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv build --wheel failed: {result.stderr}")

    wheels = sorted(dist_src.glob("dao_ai-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise RuntimeError(f"No wheel produced under {dist_src}")
    wheel_path = wheels[-1]

    dist_dst = output_dir / "dist"
    dist_dst.mkdir(parents=True, exist_ok=True)
    dst_path = dist_dst / wheel_path.name
    shutil.copy2(wheel_path, dst_path)
    written.append(f"dist/{wheel_path.name}")
    logger.info("mcp.generate.dev.copy_wheel", wheel=wheel_path.name)
    return wheel_path.name


def _render_readme(config: AppConfig, *, app_name: str, tool_name: str) -> str:
    parameters = list((config.parameters or {}).items())
    required_vars = [name for name, decl in parameters if decl.default is None]
    optional_vars = [name for name, decl in parameters if decl.default is not None]

    description = (
        str(config.app.description)
        if config.app is not None and config.app.description
        else f"dao-ai MCP agent server: {app_name}"
    )

    lines: list[str] = [f"# {app_name}", ""]
    lines.append(
        "MCP server that exposes a dao-ai agent as a single MCP tool. "
        "Generated by `dao-ai generate-mcp`."
    )
    lines.append("")
    lines.append("## Exposed tool")
    lines.append("")
    lines.append(f"- `{tool_name}` — {description}")
    lines.append("")
    lines.append(
        "MCP clients see one high-level tool; the deployed agent handles "
        "orchestration and tool routing internally. OBO tokens forwarded by "
        "the MCP client (via `x-forwarded-access-token`) flow into the "
        "agent's Context, so downstream Genie / Vector Search / UC-function "
        "calls run as the caller — not the MCP App's service principal."
    )
    lines.append("")
    lines.append("## Deploy")
    lines.append("")
    lines.append("```bash")
    deploy_cmd = ["databricks bundle deploy -t dev"]
    for var in required_vars:
        deploy_cmd.append(f'  --var "{var}=<value>"')
    lines.append(" \\\n".join(deploy_cmd))
    lines.append("```")
    lines.append("")
    if optional_vars:
        joined = ", ".join(f"`{v}`" for v in optional_vars)
        lines.append(f"Optional overrides (have defaults): {joined}")
        lines.append("")
    lines.append("Look up the app URL:")
    lines.append("")
    lines.append("```bash")
    lines.append(f"databricks apps get {app_name} -o json | jq .url")
    lines.append("```")
    lines.append("")
    lines.append(
        "Point any MCP client (Claude Desktop, Cursor, MAS, etc.) at "
        "`https://<app-url>/` over Streamable HTTP."
    )
    lines.append("")
    return "\n".join(lines)


def _get_dao_ai_version() -> str:
    try:
        return _pkg_version("dao-ai")
    except PackageNotFoundError:
        return "0.1.0"


_PYPROJECT_TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"
description = "Databricks Apps MCP server generated from a dao-ai config."
requires-python = ">=3.11,<3.12"
# Runtime deps live in requirements.txt (installed by Apps' build phase).
# Kept declared here too so local `uv sync` works for iterative dev.
dependencies = [
    "dao-ai[mcp]>={dao_ai_version}",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{package_name}"]
"""

_PYPROJECT_DEV_TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"
description = "Databricks Apps MCP server (development build with bundled wheel)."
requires-python = ">=3.11,<3.12"
# Runtime deps live in requirements.txt (installed by Apps' build phase).
# Kept declared here too so local `uv sync` works for iterative dev.
dependencies = [
    "dao-ai[mcp]",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{package_name}"]

[tool.uv.sources]
dao-ai = {{ path = "dist/{wheel_filename}" }}
"""


__all__ = ["write_mcp_bundle"]
