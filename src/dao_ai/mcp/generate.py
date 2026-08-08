"""Emit a deployable Databricks Apps bundle that runs the dao-ai MCP server.

Invoked by ``dao-ai agent build --as-mcp`` (see :mod:`dao_ai.cli`). Delegates the
App + experiment + resources DAB shape to
:func:`dao_ai.apps.bundle.generate_databricks_yaml` /
:func:`dao_ai.apps.bundle.generate_resources_app_yaml`, passing an
MCP-flavored runtime command and disabling the chat-proxy UI env vars.
The result is byte-for-byte the same App-resource pattern that
``generate-agent`` emits, so the deployed MCP server inherits the same
service-principal credentials, resource grants, experiment binding, and
trace-location wiring.

Files produced:

* ``databricks.yaml`` — bundle metadata, ``include: [resources/*.yml]``,
  targets, and (in ``--development`` mode) ``sync: [dist/*.whl]``.
* ``resources/app.yml`` — the App + experiment block (embedded
  ``config.command`` / ``config.env``, no standalone ``app.yaml``).
* ``pyproject.toml`` — the sole ``dao-ai[mcp]`` dep (dev redirects it to the
  bundled wheel via ``[tool.uv.sources]``); build metadata.
* ``uv.lock`` — the portable, public-CDN-pinned dependency closure. Apps' build
  phase runs ``uv sync --locked --no-dev`` from pyproject.toml + uv.lock.
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

from dao_ai._locking import generate_bundle_lock
from dao_ai.apps.bundle import (
    _GITIGNORE_CONTENT,
    _GITIGNORE_DEV_CONTENT,
    _strip_parameters_block,
    generate_databricks_yaml,
    generate_resources_app_yaml,
)
from dao_ai.config import AppConfig, app_name_for
from dao_ai.mcp.agent_tool import _slugify

DEFAULT_CONFIG_FILENAME = "dao_ai.yaml"

# MCP-flavored runtime command. Forwarded to the shared
# :func:`_build_app_block` via ``app_command=`` so everything else — env
# vars, resource bindings, experiment binding, trace_location wiring —
# is reused verbatim from ``generate-agent``.
_MCP_APP_COMMAND: list[str] = ["python", "-m", "dao_ai.mcp.server"]


def write_mcp_bundle(
    config: AppConfig,
    staging_dir: Path,
    *,
    overwrite: bool = False,
    development: bool = False,
) -> None:
    """Write an MCP-server deploy bundle into ``staging_dir``.

    Requires ``config.app.name`` (used as both the Databricks App name and
    the single MCP tool's name) and encourages ``config.app.description``
    (surfaced as the tool description to MCP clients).

    When ``development=True``, builds the local dao-ai wheel and bundles it
    into ``staging_dir/dist/``; the generated requirements.txt then installs
    from that wheel instead of pulling ``dao-ai[mcp]`` from PyPI.

    The staging dir is ephemeral build output: dao-ai-generated files
    (databricks.yaml, resources/app.yml, pyproject.toml, uv.lock, scaffold,
    README.md) are (re)written every build, while user-owned content (the
    rendered config, code_paths, src/<pkg>, ``resources/`` overlays) is copied
    once and never overwritten unless ``overwrite``.
    """
    if config.app is None or not config.app.name:
        raise ValueError(
            "agent build --as-mcp requires config.app.name (used as the deployed "
            "Databricks App name and the MCP tool name). Add an `app:` "
            "block with `name:` to your dao-ai config."
        )
    if not config.app.description:
        logger.warning(
            "mcp.generate.missing_description — config.app.description is "
            "unset; a generic placeholder will be used as the MCP tool "
            "description. Add `app.description` for a better client UX."
        )

    staging_dir.mkdir(parents=True, exist_ok=True)

    app_name = _derive_app_name(config)
    tool_name = _slugify(str(config.app.name))
    source_config_path: str | None = config._source_config_path
    config_filename = (
        Path(source_config_path).name if source_config_path else DEFAULT_CONFIG_FILENAME
    )

    written: list[str] = []
    skipped: list[str] = []
    # User code (src/ + code_paths + source config) preserved as-is.
    preserved: list[str] = []

    def _write(path: Path, content: str) -> None:
        if path.exists() and not overwrite:
            skipped.append(str(path.relative_to(staging_dir)))
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        written.append(str(path.relative_to(staging_dir)))

    # Reuse the shared bundle helpers so the App + experiment + resource
    # bindings match ``generate-agent`` byte-for-byte. MCP disables the
    # chat-proxy UI env vars (no chat UI in the MCP server) and skips the
    # ``artifacts:`` block (MCP doesn't build a user wheel — it installs
    # ``dao-ai[mcp]`` from PyPI via requirements.txt).
    _write(
        staging_dir / "databricks.yaml",
        generate_databricks_yaml(
            config,
            development=development,
            config_filename=config_filename,
            app_command=_MCP_APP_COMMAND,
            include_chat_ui=False,
            include_artifacts=False,
            as_mcp=True,
        ),
    )

    resources_dir = staging_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    _write(
        resources_dir / "app.yml",
        generate_resources_app_yaml(
            config,
            config_filename=config_filename,
            app_command=_MCP_APP_COMMAND,
            include_chat_ui=False,
            as_mcp=True,
        ),
    )

    wheel_filename: str | None = None
    if development:
        wheel_filename = _bundle_local_wheel(staging_dir, written=written)

    # The MCP server always needs the ``mcp`` extra (fastapi/uvicorn); merge in
    # whatever optional-feature extras the config exercises so ``uv sync``
    # installs them (e.g. "mcp,a2a,rerank").
    from dao_ai._extras import expand_all, resolve_required_extras_or_all

    merged_extras: str = ",".join(
        sorted(
            {"mcp", *expand_all(resolve_required_extras_or_all(config, target="mcp"))}
        )
    )

    # User-declared extra pip packages (config.app.pip_requirements) folded
    # into the pyproject deps so uv.lock captures them. config.app.code_paths
    # files are copied into the bundle below (next to the config) so they import
    # at runtime via ``add_code_paths_to_sys_path``; the manual ``src/<package>``
    # hatch-wheel route still works for hand-packaged code.
    from dao_ai.apps.bundle import _format_extra_deps

    user_pip_requirements: list[str] = (
        list(config.app.pip_requirements) if config.app else []
    )
    extra_deps: str = _format_extra_deps(user_pip_requirements)

    _write(
        staging_dir / "pyproject.toml",
        _render_pyproject(
            app_name=app_name,
            wheel_filename=wheel_filename,
            extras=merged_extras,
            extra_deps=extra_deps,
        ),
    )

    # Stub package so uv can build the local project when locking (the
    # pyproject declares ``packages = ["src/<pkg>"]``).
    package_name = app_name.replace("-", "_")
    _write(staging_dir / "src" / package_name / "__init__.py", "")

    _write(
        staging_dir / ".gitignore",
        _GITIGNORE_DEV_CONTENT if development else _GITIGNORE_CONTENT,
    )
    _write(staging_dir / ".python-version", "3.11\n")

    # Generate the portable uv.lock (Apps runs ``uv sync --locked --no-dev``
    # from pyproject.toml + uv.lock; no requirements.txt). In dev mode the lock
    # references the bundled wheel via ``[tool.uv.sources]``; published mode
    # locks the full ``dao-ai[mcp]`` public-PyPI closure.
    generate_bundle_lock(staging_dir)
    written.append("uv.lock")

    config_dest = staging_dir / config_filename
    if source_config_path is not None and (
        config_dest.resolve() == Path(source_config_path).resolve()
    ):
        # Never write over the user's ORIGINAL config in place (would strip
        # their parameters: block). Belt-and-suspenders behind the CLI guard.
        preserved.append(config_filename)
    else:
        rendered: str | None = config._rendered_yaml
        if rendered is not None:
            _write(config_dest, _strip_parameters_block(rendered))
        elif source_config_path is not None:
            _write(config_dest, Path(source_config_path).read_text())
        else:
            logger.warning("mcp.generate.no_source_config — skipping config copy")

    # Copy the config's custom code (app.code_paths) next to the config so it is
    # importable at runtime (bundle root is the app CWD; add_code_paths_to_sys_path
    # inserts each entry's parent onto sys.path). Shared with generate-agent.
    from dao_ai.code_paths import (
        _SRC_DIRNAME,
        discover_src_packages,
        iter_code_path_stagings,
        iter_resource_path_stagings,
        walk_code_path_files,
    )

    for src, code_dest in iter_code_path_stagings(config):
        for file_src, file_dest in walk_code_path_files(src, code_dest):
            out = staging_dir / file_dest
            # User code is sacred: never overwrite; never copy onto itself.
            if file_src.resolve() == out.resolve() or out.exists():
                preserved.append(file_dest)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_src, out)
            written.append(file_dest)

    # Convention: copy colocated ``src/<pkg>`` packages into the bundle's ``src/``
    # so hatch (``packages=["src"]``) builds them prefix-free (``foo.bar``).
    for pkg_dir in discover_src_packages(config):
        for file_src, file_dest in walk_code_path_files(
            pkg_dir, f"{_SRC_DIRNAME}/{pkg_dir.name}"
        ):
            out = staging_dir / file_dest
            if file_src.resolve() == out.resolve() or out.exists():
                preserved.append(file_dest)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_src, out)
            written.append(file_dest)

    # DAB resource overlays (app.resource_paths + the colocated resources/
    # convention) → resources/, merged by the generated databricks.yaml's
    # ``include: [resources/*.yml]``. Copied once; an existing staged copy is
    # refreshed only under --overwrite (matching the field's documented contract),
    # and never copied onto itself.
    for res_src, res_dest in iter_resource_path_stagings(config):
        out = staging_dir / res_dest
        if res_src.resolve() == out.resolve():
            preserved.append(res_dest)
            continue
        if out.exists() and not overwrite:
            preserved.append(res_dest)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(res_src, out)
        written.append(res_dest)

    _write(
        staging_dir / "README.md",
        _render_readme(config, app_name=app_name, tool_name=tool_name),
    )

    print(f"\nMCP bundle generated in {staging_dir}/\n")
    for name in written:
        print(f"  {name:<32s} (created)")
    for name in skipped:
        print(f"  {name:<32s} (skipped — re-run with --overwrite to overwrite)")
    for name in preserved:
        print(f"  {name:<32s} (preserved — your code, not overwritten)")

    print(f"\nMCP tool exposed: {tool_name}")
    print("\nNext steps:")
    print(f"  cd {staging_dir}")
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
    print("  # Apps' build phase installs deps via `uv sync --locked` from")
    print("  # pyproject.toml + uv.lock (portable, public-CDN URLs).")
    print()


def _derive_app_name(config: AppConfig) -> str:
    """Return the deployed Databricks App name for the MCP server.

    ``mcp-`` prefixed (via :func:`dao_ai.config.app_name_for`) so an MCP server
    and a chat App generated from the same config deploy to DIFFERENT Apps
    instead of replacing one another. The prefix also matches what Databricks
    Multi-Agent Supervisor pattern-matches on when auto-discovering MCP Apps.
    """
    assert config.app is not None and config.app.name  # enforced by caller
    return app_name_for(config.app.name, as_mcp=True)


def _render_pyproject(
    *,
    app_name: str,
    wheel_filename: str | None = None,
    extras: str = "mcp",
    extra_deps: str = "",
) -> str:
    package_name = app_name.replace("-", "_")
    if wheel_filename:
        return _PYPROJECT_DEV_TEMPLATE.format(
            name=app_name,
            package_name=package_name,
            wheel_filename=wheel_filename,
            extras=extras,
            extra_deps=extra_deps,
        )
    return _PYPROJECT_TEMPLATE.format(
        name=app_name,
        package_name=package_name,
        dao_ai_version=_get_dao_ai_version(),
        extras=extras,
        extra_deps=extra_deps,
    )


def _bundle_local_wheel(staging_dir: Path, *, written: list[str]) -> str:
    """Build (or reuse) a local dao-ai wheel and copy it under ``staging_dir/dist/``."""
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
    # Stamp a unique local version so the dev wheel always out-ranks the
    # same-base-version published package in the Apps container. See
    # ``dev_local_version``.
    from dao_ai.utils import dev_local_version

    with dev_local_version(project_root / "pyproject.toml"):
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

    dist_dst = staging_dir / "dist"
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
        "Generated by `dao-ai agent build --as-mcp`."
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
# Runtime deps are pinned in uv.lock (Apps' build phase runs `uv sync`).
dependencies = [
    "dao-ai[{extras}]=={dao_ai_version}",{extra_deps}
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
# ``packages = ["src"]`` + ``sources = ["src"]`` auto-discovers every top-level
# package under ``src/`` and installs it prefix-free (``src/foo`` -> ``foo``).
packages = ["src"]
sources = ["src"]
"""

_PYPROJECT_DEV_TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"
description = "Databricks Apps MCP server (development build with bundled wheel)."
requires-python = ">=3.11,<3.12"
# Runtime deps are pinned in uv.lock (Apps' build phase runs `uv sync`).
dependencies = [
    "dao-ai[{extras}]",{extra_deps}
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
sources = ["src"]

[tool.uv.sources]
dao-ai = {{ path = "dist/{wheel_filename}" }}
"""


__all__ = ["write_mcp_bundle"]
