"""Generate a portable ``uv.lock`` for a staged Apps bundle.

Databricks Apps' build phase runs ``uv sync --locked --no-dev`` when a bundle
ships ``pyproject.toml`` + ``uv.lock`` (and no ``requirements.txt``). The lock
must be generated at bundle-generation time (there is no deploy-time lock hook)
and must be **portable** — i.e. free of the internal PyPI mirror host, whose
absolute wheel URLs are unreachable from the Apps container and from customers.

Behind the corp mirror a plain ``uv lock`` bakes ``pypi-proxy.dev.databricks.com``
URLs into the lock. The mirror is a transparent passthrough of the public CDN
(identical package paths, hashes, and upload-times — only the host differs), so
we rewrite the recorded host back to the public CDN after locking. The result is
byte-equivalent to a clean-room (public-index) lock. See the ADR on Apps
dependency management and the repo Makefile ``lock-local`` target for the same
technique applied to the dao-ai repo's own lock.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

# The internal mirror host and its public-CDN equivalent. The mirror mirrors the
# public CDN's ``/packages/<hash-path>/<wheel>`` layout verbatim, so a host swap
# yields a portable lock without changing any package, version, or hash.
_MIRROR_HOST = "pypi-proxy.dev.databricks.com"
_PUBLIC_HOST = "files.pythonhosted.org"


def generate_bundle_lock(bundle_dir: Path) -> None:
    """Generate a portable ``uv.lock`` in ``bundle_dir``.

    Runs ``uv lock`` against ``bundle_dir/pyproject.toml``, rewrites any internal
    mirror host in the resulting lock to the public CDN, and asserts the lock is
    clean. Raises ``RuntimeError`` if ``uv lock`` fails or if a mirror reference
    survives the rewrite (which would break the Apps ``uv sync`` at deploy time).

    Args:
        bundle_dir: Staged bundle directory containing the generated
            ``pyproject.toml`` (and, for development builds, the local wheel
            under ``dist/`` referenced via ``[tool.uv.sources]``).
    """
    result = subprocess.run(
        ["uv", "lock"],
        cwd=bundle_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr
        # Published bundles pin the current dao-ai version. Before that version
        # is published to PyPI, `uv lock` cannot resolve it. This is expected
        # pre-release: published-mode locks are generated at release/CI time,
        # once the version exists on the index. Give an actionable message
        # rather than uv's raw resolver output.
        if "No solution found" in stderr and "dao-ai" in stderr:
            raise RuntimeError(
                "uv lock could not resolve the pinned dao-ai version — it is not "
                "yet published to the index. Published-mode bundle locks are "
                "generated at release time (in CI, after dao-ai is published). "
                "For local/pre-release use, generate the bundle with "
                "`--development` (locks against the bundled local wheel).\n\n"
                f"uv output:\n{stderr}"
            )
        raise RuntimeError(f"uv lock failed in {bundle_dir}: {stderr}")

    lock_path = bundle_dir / "uv.lock"
    if not lock_path.exists():
        raise RuntimeError(f"uv lock did not produce {lock_path}")

    original = lock_path.read_text()
    rewritten = original.replace(
        f"https://{_MIRROR_HOST}/", f"https://{_PUBLIC_HOST}/"
    )
    if rewritten != original:
        lock_path.write_text(rewritten)
        logger.info(
            "Rewrote internal mirror host in bundle uv.lock to public CDN",
            mirror=_MIRROR_HOST,
            public=_PUBLIC_HOST,
        )

    if _MIRROR_HOST in lock_path.read_text():
        raise RuntimeError(
            f"{lock_path} still references the internal mirror ({_MIRROR_HOST}) "
            "after rewrite; the lock would not resolve in the Apps container or "
            "for customers. Aborting."
        )

    logger.info("Generated portable bundle uv.lock", path=str(lock_path))


def render_portable_lock(pyproject_content: str, wheel_path: Path | None = None) -> str:
    """Return a portable ``uv.lock`` for the given ``pyproject.toml`` content.

    For the direct-deploy path (``deploy_apps_agent``), which uploads files to a
    workspace source path rather than staging a local bundle dir. Locks in a
    throwaway temp dir and returns the lock text. When the pyproject references a
    local wheel via ``[tool.uv.sources]``, pass ``wheel_path`` so it can be copied
    into the temp dir's ``dist/`` for resolution.

    Args:
        pyproject_content: The bundle ``pyproject.toml`` text.
        wheel_path: Local dao-ai wheel to stage under ``dist/`` (dev mode only).

    Returns:
        The portable (public-CDN) ``uv.lock`` contents.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "pyproject.toml").write_text(pyproject_content)
        # Stub package so uv can build the local project during locking. dao-ai
        # templates use ``packages = ["src"]`` (auto-discover every package under
        # ``src/``); older bundles used ``packages = ["src/<pkg>"]``. Either way
        # hatch needs at least one package present under ``src/`` — create a stub.
        import re

        legacy = re.search(r'packages\s*=\s*\["src/([^"]+)"\]', pyproject_content)
        if legacy:
            pkg_dir = tmp_dir / "src" / legacy.group(1)
        elif re.search(r'packages\s*=\s*\["src"\]', pyproject_content):
            pkg_dir = tmp_dir / "src" / "_daoai_lockstub"
        else:
            pkg_dir = None
        if pkg_dir is not None:
            pkg_dir.mkdir(parents=True, exist_ok=True)
            (pkg_dir / "__init__.py").write_text("")
        if wheel_path is not None:
            dist_dir = tmp_dir / "dist"
            dist_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wheel_path, dist_dir / wheel_path.name)
        generate_bundle_lock(tmp_dir)
        return (tmp_dir / "uv.lock").read_text()
