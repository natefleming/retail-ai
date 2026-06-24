"""Emit a deployable Databricks Apps bundle that runs the dao-ai MCP server.

Invoked by ``dao-ai generate-mcp`` (see :mod:`dao_ai.cli`). Mirrors the shape
of :func:`dao_ai.apps.bundle.write_bundle` but produces an MCP-only artifact:

* ``databricks.yml`` — DAB with ``bundle.engine: direct`` and the App resource
  bindings derived from ``config.tools`` (filtered to entries with registered
  MCP adapters).
* ``app.yaml`` — ``command: ["dao-ai-mcp-server"]`` plus env vars matching the
  App resource bindings.
* ``pyproject.toml`` — single dep ``dao-ai[mcp]>=<current-version>``; declares
  ``dao-ai-mcp-server`` as a re-exported script.
* ``dao_ai.yaml`` — the rendered (param-substituted) config, stripped of its
  top-level ``parameters:`` block.
* ``README.md`` — generated deploy snippet listing required ``--var`` flags.
"""

from __future__ import annotations

import shutil
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from loguru import logger
from ruamel.yaml import YAML

from dao_ai.apps.bundle import _convert_to_bundle_resources, _strip_parameters_block
from dao_ai.apps.resources import generate_app_resources
from dao_ai.config import AppConfig, FactoryFunctionModel

# Side-effect imports so adapter registry is populated before we filter.
from dao_ai.mcp.adapters import genie as _genie_adapter  # noqa: F401
from dao_ai.mcp.adapters import get_adapter
from dao_ai.mcp.adapters import vector_search as _vector_search_adapter  # noqa: F401

DEFAULT_CONFIG_FILENAME = "dao_ai.yaml"
# Bare console-script entry. Apps' native uv support runs `uv sync --locked
# --no-dev` at BUILD phase, which installs `dao-ai-mcp-server` into
# .venv/bin/ and puts that on PATH for the runtime.
APP_COMMAND: list[str] = ["dao-ai-mcp-server"]


def write_mcp_bundle(
    config: AppConfig,
    output_dir: Path,
    *,
    force: bool = False,
    development: bool = False,
) -> None:
    """Write an MCP-server deploy bundle into ``output_dir``.

    When ``development=True``, builds the local dao-ai wheel and bundles it
    into ``output_dir/dist/``; the generated pyproject.toml then installs from
    that wheel instead of pulling ``dao-ai[mcp]`` from PyPI. This is the only
    way to deploy MCP server code that hasn't shipped to PyPI yet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    mcp_tool_names = _mcp_recognized_tool_names(config)
    if not mcp_tool_names:
        raise ValueError(
            "No MCP-registerable tools found in AppConfig.tools. Provide at "
            "least one entry whose function.name matches a registered MCP "
            "adapter (e.g. 'dao_ai.tools.create_genie_toolkit' or "
            "'dao_ai.tools.create_vector_search_tool')."
        )

    app_name = _derive_app_name(config)
    source_config_path: str | None = getattr(config, "_source_config_path", None)
    config_filename = (
        Path(source_config_path).name if source_config_path else DEFAULT_CONFIG_FILENAME
    )

    written: list[str] = []
    skipped: list[str] = []

    def _write(path: Path, content: str) -> None:
        if path.exists() and not force:
            skipped.append(path.name)
            return
        path.write_text(content)
        written.append(path.name)

    _write(
        output_dir / "databricks.yml",
        _render_databricks_yml(
            config, app_name=app_name, config_filename=config_filename
        ),
    )
    _write(
        output_dir / "app.yaml",
        _render_app_yaml(config, config_filename=config_filename),
    )

    wheel_filename: str | None = None
    if development:
        wheel_filename = _bundle_local_wheel(output_dir, written=written)

    _write(
        output_dir / "pyproject.toml",
        _render_pyproject(app_name=app_name, wheel_filename=wheel_filename),
    )

    rendered: str | None = getattr(config, "_rendered_yaml", None)
    if rendered is not None:
        _write(output_dir / config_filename, _strip_parameters_block(rendered))
    elif source_config_path is not None:
        _write(output_dir / config_filename, Path(source_config_path).read_text())
    else:
        logger.warning("mcp.generate.no_source_config — skipping config copy")

    _write(
        output_dir / "README.md",
        _render_readme(config, app_name=app_name, mcp_tool_names=mcp_tool_names),
    )

    print(f"\nMCP bundle generated in {output_dir}/\n")
    for name in written:
        print(f"  {name:<20s} (created)")
    for name in skipped:
        print(f"  {name:<20s} (skipped — re-run with --force to overwrite)")

    print("\nMCP tools that will be exposed:")
    for name in sorted(mcp_tool_names):
        print(f"  - {name}")

    print("\nNext steps:")
    print(f"  cd {output_dir}")
    print("  uv sync                              # generate uv.lock against your env")
    print("  # Databricks-internal users only: rewrite internal-proxy URLs in the lock")
    print("  # so Apps containers can fetch from public PyPI (see README).")
    print("  databricks bundle validate -t dev -p <profile>")
    print("  databricks bundle deploy -t dev -p <profile>")
    print()


def _mcp_recognized_tool_names(config: AppConfig) -> set[str]:
    """Return YAML tool names whose factory has an adapter registered."""
    recognized: set[str] = set()
    for name, tool_def in config.tools.items():
        fn = tool_def.function
        if not isinstance(fn, FactoryFunctionModel):
            continue
        if get_adapter(fn.name) is not None:
            recognized.add(name)
    return recognized


def _derive_app_name(config: AppConfig) -> str:
    """Pick a Databricks App name.

    Prefer ``config.app.name`` for parity with ``generate-bundle``, fall back
    to ``mcp-dao-ai``. The ``mcp-`` prefix is a discovery signal for
    Databricks Multi-Agent Supervisor (MAS), which pattern-matches it when
    enumerating MCP-hosted Apps across an account.
    """
    if config.app is not None and config.app.name:
        return str(config.app.name).lower().replace("_", "-")
    return "mcp-dao-ai"


def _render_databricks_yml(
    config: AppConfig, *, app_name: str, config_filename: str
) -> str:
    """Emit a direct-engine DAB with the App + its bound resources.

    Reuses ``_convert_to_bundle_resources`` from ``dao_ai.apps.bundle`` so the
    binding shape (genie_space, sql_warehouse, postgres, uc_securable, etc.)
    matches what ``generate-bundle`` emits and what the Databricks CLI accepts.

    The Lakebase ``database_instance`` is *not* auto-provisioned here — operators
    pre-provision (or declare it themselves in databricks.yml). This keeps the
    generated bundle minimal and avoids accidentally re-creating instances
    across deploys.
    """
    bundle_resources = _convert_to_bundle_resources(generate_app_resources(config))

    bundle: dict[str, Any] = {
        "bundle": {"name": app_name, "engine": "direct"},
        "resources": {
            "apps": {
                app_name.replace("-", "_"): {
                    "name": app_name,
                    "description": _app_description(config, app_name),
                    "source_code_path": "${workspace.file_path}",
                    "resources": bundle_resources,
                }
            }
        },
        "targets": {
            "dev": {"mode": "development", "default": True},
            "prod": {"mode": "production"},
        },
    }

    return _dump_yaml(
        bundle, header=_DATABRICKS_YML_HEADER.format(filename=config_filename)
    )


def _render_app_yaml(config: AppConfig, *, config_filename: str) -> str:
    """Emit app.yaml — command + env vars.

    The MCP server reads its config exclusively from ``DAO_AI_MCP_CONFIG_PATH``
    (which points at the rendered ``dao_ai.yaml`` co-located in the bundle).
    Because the bundle CLI substitutes ``${param.NAME}`` references at deploy
    time and the rendered config is committed alongside, the server doesn't
    need additional env var bindings for ``${var.NAME}`` resolution at runtime
    — every reference is already resolved literally in the shipped YAML.
    """
    env: list[dict[str, str]] = [
        {"name": "DAO_AI_MCP_CONFIG_PATH", "value": config_filename},
        {"name": "MLFLOW_TRACKING_URI", "value": "databricks"},
        {"name": "MLFLOW_REGISTRY_URI", "value": "databricks-uc"},
        {"name": "UV_SYSTEM_PYTHON", "value": "1"},
    ]

    return _dump_yaml(
        {"command": APP_COMMAND, "env": env},
        header=_APP_YAML_HEADER,
    )


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


def _render_readme(
    config: AppConfig, *, app_name: str, mcp_tool_names: set[str]
) -> str:
    parameters = list((config.parameters or {}).items())
    required_vars = [name for name, decl in parameters if decl.default is None]
    optional_vars = [name for name, decl in parameters if decl.default is not None]

    lines: list[str] = [f"# {app_name}", ""]
    lines.append(
        "MCP server deploying the dao-ai-configured tools below as an MCP "
        "Streamable-HTTP endpoint on Databricks Apps. Generated by "
        "`dao-ai generate-mcp`."
    )
    lines.append("")
    lines.append("## Exposed tools")
    lines.append("")
    for name in sorted(mcp_tool_names):
        lines.append(f"- `{name}`")
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
        "Point any MCP client (Claude Desktop, Cursor, etc.) at "
        "`https://<app-url>/` over Streamable HTTP."
    )
    lines.append("")
    return "\n".join(lines)


def _app_description(config: AppConfig, app_name: str) -> str:
    if config.app is not None and config.app.description:
        return str(config.app.description)
    return f"dao-ai MCP server: {app_name}"


def _dump_yaml(data: dict[str, Any], *, header: str | None = None) -> str:
    """ruamel-quality dump that preserves key order without anchor mangling."""
    rt = YAML(typ="rt")
    rt.preserve_quotes = True
    rt.width = 4096
    rt.indent(mapping=2, sequence=4, offset=2)

    import io

    buf = io.StringIO()
    if header:
        buf.write(header)
        if not header.endswith("\n"):
            buf.write("\n")
    rt.dump(data, buf)
    return buf.getvalue()


def _get_dao_ai_version() -> str:
    try:
        return _pkg_version("dao-ai")
    except PackageNotFoundError:
        return "0.1.0"


_DATABRICKS_YML_HEADER = """\
# Generated by `dao-ai generate-mcp`.
# Source config: {filename}
# Edit by re-running generate-mcp against an updated config.
"""

_APP_YAML_HEADER = """\
# Generated by `dao-ai generate-mcp`.
# Databricks Apps reads this file to launch the MCP server.
"""

_PYPROJECT_TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"
description = "Databricks Apps MCP server generated from a dao-ai config."
requires-python = ">=3.11,<3.12"
dependencies = [
    "dao-ai[mcp]>={dao_ai_version}",
]

[project.scripts]
dao-ai-mcp-server = "dao_ai.mcp.server:main"

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
dependencies = [
    "dao-ai[mcp]",
]

[project.scripts]
dao-ai-mcp-server = "dao_ai.mcp.server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{package_name}"]

[tool.uv.sources]
dao-ai = {{ path = "dist/{wheel_filename}" }}
"""


__all__ = ["write_mcp_bundle"]
