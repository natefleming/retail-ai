"""Emit a deployable Databricks Apps bundle that runs the dao-ai MCP server.

Invoked by ``dao-ai generate-mcp`` (see :mod:`dao_ai.cli`). Mirrors the
shape of :func:`dao_ai.apps.bundle.write_bundle` but produces a bundle
whose runtime is the FastMCP server in :mod:`dao_ai.mcp.server`. The
emitted server exposes exactly one MCP tool: the full dao-ai agent graph
(named after ``config.app.name``, described by ``config.app.description``).

Files produced:

* ``databricks.yml`` — DAB with ``bundle.engine: direct`` and the App
  resource bindings derived from :func:`generate_app_resources`.
* ``app.yaml`` — ``command: ["python", "-m", "dao_ai.mcp.server"]`` plus env vars.
* ``pyproject.toml`` — build metadata; runtime deps live in requirements.txt.
* ``requirements.txt`` — ``dao-ai[mcp]>=<current-version>`` (prod) or the
  bundled wheel with ``[mcp]`` extras (dev). Apps' build phase installs
  from this directly — no ``uv sync`` or URL-rewrite step required.
* ``dao_ai.yaml`` — the rendered (param-substituted) config, stripped of
  its top-level ``parameters:`` block.
* ``README.md`` — deploy snippet + the single tool name to point clients at.
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
from dao_ai.apps.resources import (
    _extract_raw_trace_location_resources,
    generate_app_resources,
)
from dao_ai.config import AppConfig, value_of
from dao_ai.mcp.agent_tool import _slugify

DEFAULT_CONFIG_FILENAME = "dao_ai.yaml"
# Invoke the MCP server via ``python -m`` so we don't depend on ``.venv/bin``
# being on PATH inside the Apps runtime container. Parallel to how
# ``generate-bundle`` invokes ``python -m dao_ai.apps.server`` — see
# ``dao_ai/apps/bundle.py::_build_app_block``.
APP_COMMAND: list[str] = ["python", "-m", "dao_ai.mcp.server"]


def write_mcp_bundle(
    config: AppConfig,
    output_dir: Path,
    *,
    force: bool = False,
    development: bool = False,
) -> None:
    """Write an MCP-server deploy bundle into ``output_dir``.

    Requires ``config.app.name`` (used as both the Databricks App name and
    the single MCP tool's name) and encourages ``config.app.description``
    (surfaced as the tool description to MCP clients).

    When ``development=True``, builds the local dao-ai wheel and bundles it
    into ``output_dir/dist/``; the generated pyproject.toml then installs
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
    _write(
        output_dir / "requirements.txt",
        _make_requirements_txt(development=development, wheel_filename=wheel_filename),
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
        _render_readme(config, app_name=app_name, tool_name=tool_name),
    )

    print(f"\nMCP bundle generated in {output_dir}/\n")
    for name in written:
        print(f"  {name:<20s} (created)")
    for name in skipped:
        print(f"  {name:<20s} (skipped — re-run with --force to overwrite)")

    # `databricks bundle run` addresses resources by their DAB key, which
    # ``_render_databricks_yml`` derives as ``app_name.replace("-", "_")``.
    # Keep this in sync with that mapping so the printed command is
    # copy-pasteable when ``app.name`` contains hyphens (e.g. ``mcp-foo``).
    bundle_key = app_name.replace("-", "_")

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
    print(f"  databricks bundle run {bundle_key} --target dev")
    print()
    print("  # Apps' build phase installs deps directly from requirements.txt;")
    print("  # no uv sync or URL rewrite required.")
    print()


def _derive_app_name(config: AppConfig) -> str:
    """Return the Databricks App name — parity with ``generate-bundle``.

    The ``mcp-`` prefix is a discovery signal for Databricks Multi-Agent
    Supervisor (MAS), which pattern-matches it when enumerating
    MCP-hosted Apps across an account. Callers who don't want that
    prefix can set ``config.app.name`` to any value they like.
    """
    assert config.app is not None and config.app.name  # enforced by caller
    return str(config.app.name).lower().replace("_", "-")


def _render_databricks_yml(
    config: AppConfig, *, app_name: str, config_filename: str
) -> str:
    """Emit a direct-engine DAB with the App + its bound resources.

    Mirrors :func:`dao_ai.apps.bundle.generate_resources_app_yaml`'s
    experiment + trace-location wiring so the MCP server can materialize
    its own OTEL trace tables at boot. See
    ``src/dao_ai/apps/bundle.py:355-513`` for the reference implementation.
    """
    assert config.app is not None and config.app.name  # enforced by caller
    app_resources = generate_app_resources(config)
    bundle_resources = _convert_to_bundle_resources(app_resources)

    # When trace_location is configured, attach the trace warehouse as an
    # App resource so the platform grants the App SP CAN_USE on it (needed
    # for MLflow to materialize OTEL Delta tables at first-trace-write).
    # OTEL table uc_securable resources are NOT emitted because the tables
    # don't exist yet at deploy time — operators must GRANT USE_SCHEMA +
    # CREATE_TABLE + MODIFY + SELECT on the trace schema to the App SP
    # manually (same limitation as generate-bundle).
    if config.app.trace_location:
        existing_wh_ids: set[str] = set()
        if config.resources:
            for _wh in config.resources.warehouses.values():
                try:
                    existing_wh_ids.add(value_of(_wh.warehouse_id))
                except Exception:
                    pass
        bundle_resources.extend(
            _extract_raw_trace_location_resources(
                config.app.trace_location,
                existing_warehouse_ids=existing_wh_ids,
            )
        )

    # Always bind an MLflow experiment as an App resource so the runtime
    # SP can read/write traces via auto-auth. Two variants (mirroring
    # bundle.py:443-463):
    #   * config.app.experiment set → bind by literal id (admin-owned;
    #     ``manage_permissions=false`` downgrades to CAN_READ).
    #   * otherwise → declare in top-level ``experiments:`` block and
    #     bind via ${resources.experiments.<key>.id} so DABs materializes
    #     + grants CAN_EDIT to the App SP.
    experiment_key = f"{app_name}-experiment"
    experiments_block: dict[str, Any] = {}
    external_experiment_id: str | None = None
    if config.app.experiment is not None:
        # Reuse dao-ai's existing resolution — same call generate-bundle uses.
        config.app.experiment.create()
        external_experiment_id = config.app.experiment.resolved_id

    if external_experiment_id:
        experiment_app_resource: dict[str, Any] = {
            "name": "experiment",
            "experiment": {
                "experiment_id": external_experiment_id,
                "permission": (
                    "CAN_EDIT" if config.app.manage_permissions else "CAN_READ"
                ),
            },
        }
    else:
        experiments_block[experiment_key] = {
            "name": f"/Users/${{workspace.current_user.userName}}/{app_name}",
        }
        experiment_app_resource = {
            "name": "experiment",
            "experiment": {
                "experiment_id": f"${{resources.experiments.{experiment_key}.id}}",
                "permission": "CAN_EDIT",
            },
        }
    # Insert the experiment first so it's the visually-obvious binding.
    bundle_resources.insert(0, experiment_app_resource)

    resources_section: dict[str, Any] = {}
    if experiments_block:
        resources_section["experiments"] = experiments_block
    resources_section["apps"] = {
        app_name.replace("-", "_"): {
            "name": app_name,
            "description": _app_description(config, app_name),
            "source_code_path": "${workspace.file_path}",
            "resources": bundle_resources,
        }
    }

    bundle: dict[str, Any] = {
        "bundle": {"name": app_name, "engine": "direct"},
        "resources": resources_section,
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

    Injects ``MLFLOW_EXPERIMENT_ID`` via ``value_from: experiment`` so DABs
    binds it to the resolved experiment id at deploy time, plus
    ``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` when ``config.app.trace_location``
    is set (required by MLflow's OTEL exporter to materialize / query
    the UC Delta trace tables).
    """
    # NOTE: keys are camelCase (``valueFrom``) — this file is consumed
    # directly by the Databricks Apps runtime, not by DABs. DABs
    # ``resources.apps.*.config`` uses snake_case (``value_from``) and DABs
    # rewrites to camelCase before the Apps platform sees it.
    env: list[dict[str, Any]] = [
        {"name": "DAO_AI_MCP_CONFIG_PATH", "value": config_filename},
        {"name": "MLFLOW_TRACKING_URI", "value": "databricks"},
        {"name": "MLFLOW_REGISTRY_URI", "value": "databricks-uc"},
        {"name": "MLFLOW_EXPERIMENT_ID", "valueFrom": "experiment"},
        {"name": "UV_SYSTEM_PYTHON", "value": "1"},
    ]
    if config.app and config.app.trace_location:
        env.append(
            {
                "name": "MLFLOW_TRACING_SQL_WAREHOUSE_ID",
                "value": value_of(config.app.trace_location.warehouse_id),
            }
        )

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


def _make_requirements_txt(
    *,
    development: bool,
    wheel_filename: str | None = None,
) -> str:
    """Build the requirements.txt content for an MCP app bundle.

    The published pin is left *unbounded* (``dao-ai[mcp]``) rather than
    floor-pinned to the locally-installed version. ``_get_dao_ai_version()``
    reflects whatever is checked out in the developer's tree, which may
    be an unreleased pre-publish build (e.g. ``0.1.112`` before it lands
    on PyPI). Baking that floor into requirements.txt would cause Apps
    to fail with ``Could not find a version that satisfies …``. Users
    who need reproducibility can tighten the pin by hand.

    We install the ``[mcp]`` extra so the server module's dependencies
    (fastmcp, uvicorn, etc.) resolve. Development mode installs the
    bundled wheel with that extra so transitive MCP deps still resolve
    from public PyPI.
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

    description = _app_description(config, app_name)

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


def _app_description(config: AppConfig, app_name: str) -> str:
    if config.app is not None and config.app.description:
        return str(config.app.description)
    return f"dao-ai MCP agent server: {app_name}"


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
