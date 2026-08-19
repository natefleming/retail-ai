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

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

# A portable bundle lock may reference only PUBLIC package infrastructure — the
# public simple index (``pypi.org``) in ``source`` registry fields and the
# public CDN (``files.pythonhosted.org``) in wheel/sdist download URLs. Any
# other host is an internal mirror/proxy, unreachable from the Apps build
# container and from customers, so ``uv sync`` there fails.
#
# Two distinct poisoning shapes appear, depending on which internal index the
# locking machine is configured against:
#   * corp mirror (``pypi-proxy.{dev,cloud,...}.databricks.com``) — a transparent
#     passthrough of the public CDN, so it bakes its host into the wheel/sdist
#     ``url`` fields. Fix: swap the host back to the public CDN (identical paths
#     and hashes).
#   * serverless proxy (``node.host.local:8184/pypi/v1/simple/``) — a simple
#     index that leaves the wheel URLs pointing at the public CDN but records
#     itself in the ``source = { registry = "..." }`` field. Fix: normalize the
#     registry to the canonical public index.
#
# Rather than enumerate proxy hostnames (the original bug hardcoded only the corp
# ``.dev.`` subdomain and silently shipped ``.cloud.`` and serverless URLs), any
# non-public host is treated as an internal mirror generically.
_PUBLIC_CDN_HOST = "files.pythonhosted.org"
_PUBLIC_INDEX = "https://pypi.org/simple"
_PUBLIC_HOSTS = frozenset({"files.pythonhosted.org", "pypi.org"})

# ``source = { registry = "<index-url>" }`` and ``url = "<download-url>"`` — the
# only two places a host is recorded in a uv.lock.
_REGISTRY_RE = re.compile(r'registry = "([^"]+)"')
_URL_RE = re.compile(r'url = "([^"]+)"')


def _host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _is_public(url: str) -> bool:
    return _host_of(url) in _PUBLIC_HOSTS


def _make_lock_portable(lock_text: str) -> str:
    """Rewrite internal-mirror references to their public equivalents.

    Non-public ``source`` registries become the canonical public index; wheel/
    sdist ``url`` hosts on an internal CDN passthrough are swapped back to the
    public CDN (path and hash unchanged). URLs with no network host (local
    ``file://`` wheel sources used by dev builds) are left untouched.
    """

    def _fix_registry(m: "re.Match[str]") -> str:
        if _is_public(m.group(1)):
            return m.group(0)
        return f'registry = "{_PUBLIC_INDEX}"'

    def _fix_url(m: "re.Match[str]") -> str:
        url = m.group(1)
        if not _host_of(url) or _is_public(url):
            return m.group(0)
        swapped = urlsplit(url)._replace(scheme="https", netloc=_PUBLIC_CDN_HOST)
        return f'url = "{urlunsplit(swapped)}"'

    text = _REGISTRY_RE.sub(_fix_registry, lock_text)
    text = _URL_RE.sub(_fix_url, text)
    return text


def _first_non_public_ref(lock_text: str) -> str | None:
    """Return the first non-public index/host reference, or ``None`` if clean."""
    for m in _REGISTRY_RE.finditer(lock_text):
        if not _is_public(m.group(1)):
            return m.group(1)
    for m in _URL_RE.finditer(lock_text):
        host = _host_of(m.group(1))
        if host and host not in _PUBLIC_HOSTS:
            return m.group(1)
    return None


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
    rewritten = _make_lock_portable(original)
    if rewritten != original:
        lock_path.write_text(rewritten)
        logger.info(
            "Rewrote internal mirror reference(s) in bundle uv.lock to public PyPI",
            index=_PUBLIC_INDEX,
            cdn=_PUBLIC_CDN_HOST,
        )

    # Clean-check: assert no non-public index/host survived the rewrite (an
    # independent scan, defense-in-depth) rather than shipping an unresolvable
    # lock.
    surviving = _first_non_public_ref(lock_path.read_text())
    if surviving:
        raise RuntimeError(
            f"{lock_path} still references a non-public package index / internal "
            f"mirror ({surviving}) after rewrite; the lock would not resolve in "
            "the Apps container or for customers. Aborting."
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
