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

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

# A portable bundle lock must not reference an INTERNAL Databricks package proxy
# — those hosts are unreachable from the Apps build container and from customers,
# so ``uv sync`` there fails. Two distinct proxies occur, depending on where the
# lock was generated, and they poison different fields:
#   * corp CDN mirror (``pypi-proxy.{dev,cloud,<region>}.databricks.com``) — a
#     transparent passthrough of the public CDN (identical ``/packages/<hash>``
#     paths and hashes); it bakes its host into the wheel/sdist ``url`` fields.
#     Fix: host-swap back to the public CDN.
#   * serverless build proxy (``node.host.local:<port>/pypi/v1/simple/``) — a
#     simple index that leaves wheel URLs on the public CDN but records itself in
#     the ``source = { registry = "..." }`` field. Fix: normalize the registry to
#     the canonical public index.
#
# Only these internal mirror hosts are rewritten. Legitimate PUBLIC alternate
# indexes and direct-URL/git sources (e.g. ``download.pytorch.org``) are reachable
# from the Apps container and must be left untouched.
_PUBLIC_CDN_HOST = "files.pythonhosted.org"
_PUBLIC_INDEX = "https://pypi.org/simple"

# Internal Databricks proxy hosts, matched ANYWHERE in the lock text (independent
# of TOML field/whitespace formatting) — both by the rewrite and, critically, by
# the survivor guard, so an unusual layout the field-aware rewrite misses still
# fails loudly instead of shipping an unresolvable lock. Broadened beyond the
# original ``pypi-proxy*.databricks.com``-only match (which silently shipped the
# serverless ``node.host.local`` proxy).
_MIRROR_HOST_RE = re.compile(
    r"pypi-proxy[\w.-]*\.databricks\.com|node\.host\.local(?::\d+)?"
)

# The two lock fields that carry a host: the registry index and download URLs.
_REGISTRY_RE = re.compile(r'registry = "([^"]*)"')
_URL_RE = re.compile(r'url = "([^"]*)"')

# The serverless build proxy's non-transparent URL PATH: it rewrites downloads to
# ``/pypi/vN/packages/<name>/<version>/<file>`` (not the public CDN's
# ``/packages/<hash>`` layout). A host-swap of such a URL yields a valid host with
# an invalid path that 404s at ``uv sync``, so it must be caught rather than
# shipped — the only correct fix is resolving against public PyPI.
_PROXY_PATH_RE = re.compile(r"/pypi/v\d+/(?:packages|simple)/")


def _make_lock_portable(lock_text: str) -> str:
    """Rewrite internal-mirror references to their public equivalents.

    A ``source`` registry on an internal mirror becomes the canonical public
    index; a wheel/sdist ``url`` on the corp CDN mirror is host-swapped back to
    the public CDN (path and hash unchanged). Only internal Databricks mirror
    hosts are touched — public alternate indexes and direct URLs are left as-is.
    """

    def _fix_registry(m: "re.Match[str]") -> str:
        if _MIRROR_HOST_RE.search(m.group(1)):
            return f'registry = "{_PUBLIC_INDEX}"'
        return m.group(0)

    def _fix_url(m: "re.Match[str]") -> str:
        url = m.group(1)
        if not _MIRROR_HOST_RE.search(url):
            return m.group(0)
        swapped = urlsplit(url)._replace(scheme="https", netloc=_PUBLIC_CDN_HOST)
        return f'url = "{urlunsplit(swapped)}"'

    text = _REGISTRY_RE.sub(_fix_registry, lock_text)
    text = _URL_RE.sub(_fix_url, text)
    return text


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

    # Resolve against PUBLIC PyPI so wheel/sdist URLs are the real, portable
    # ``files.pythonhosted.org/packages/<hash>`` paths. Resolving against an
    # internal proxy and host-swapping only works for the transparent corp CDN
    # mirror (identical ``/packages/<hash>`` paths); the serverless build proxy
    # (``node.host.local``) rewrites the URL PATH to
    # ``/pypi/vN/packages/<name>/<version>/<file>``, which a host-swap turns into
    # an invalid public-CDN URL that 404s at ``uv sync``. Fall back to the ambient
    # index (e.g. the corp mirror on a laptop where public PyPI is blocked), whose
    # transparent URLs the host-swap below can fix.
    def _uv_lock(force_public: bool) -> "subprocess.CompletedProcess[str]":
        env = dict(os.environ)
        if force_public:
            env["UV_INDEX_URL"] = _PUBLIC_INDEX
            env["UV_DEFAULT_INDEX"] = _PUBLIC_INDEX
        return subprocess.run(
            ["uv", "lock"],
            cwd=bundle_dir,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    result = _uv_lock(force_public=True)
    if result.returncode != 0 and "No solution found" not in result.stderr:
        # Public PyPI unreachable (e.g. corp laptop) — fall back to the ambient
        # index; a transparent corp mirror's URLs are host-swapped below.
        logger.warning(
            "uv lock against public PyPI failed; retrying with the ambient index",
            stderr=(result.stderr or "")[-400:],
        )
        result = _uv_lock(force_public=False)
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

    final = lock_path.read_text()
    # Guard 1: assert no internal mirror HOST survived the rewrite. Scans the whole
    # lock text (format-independent), so it fails loudly even if the field-aware
    # rewrite missed an unusual layout.
    surviving = _MIRROR_HOST_RE.search(final)
    if surviving:
        raise RuntimeError(
            f"{lock_path} still references an internal package proxy "
            f"({surviving.group()}) after rewrite; the lock would not resolve in "
            "the Apps container or for customers. Aborting."
        )
    # Guard 2: assert no serverless-proxy URL PATH survived. A host-swap leaves
    # ``files.pythonhosted.org/pypi/vN/packages/...`` — a valid host but an invalid
    # CDN path that 404s. Only resolving against public PyPI yields the real
    # ``/packages/<hash>`` layout, so a surviving fingerprint means the lock is
    # unresolvable and must not ship.
    proxied = _PROXY_PATH_RE.search(final)
    if proxied:
        raise RuntimeError(
            f"{lock_path} contains a non-portable serverless-proxy URL path "
            f"({proxied.group()!r}); it must be generated against public PyPI. "
            "Aborting rather than ship a lock that 404s in the Apps container."
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
