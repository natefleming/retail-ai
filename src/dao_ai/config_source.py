"""Loading dao-ai configs from somewhere other than the local filesystem.

Small, pure helpers so ``AppConfig.from_file`` can accept an ``http(s)`` URL —
a config shared from a GitHub repo, for instance — without the rest of the
loader caring where the YAML came from.

Only YAML is parsed. Nothing fetched here is executed, and no credentials are
sent to third-party hosts.
"""

from __future__ import annotations

from os import PathLike
from urllib.parse import urlparse, urlunparse

from loguru import logger

#: Hosts whose "view this file" URLs point at an HTML page rather than raw bytes.
_GITHUB_HOST = "github.com"
_GITHUB_RAW_HOST = "raw.githubusercontent.com"

#: Default timeout for fetching a remote config, in seconds.
DEFAULT_FETCH_TIMEOUT: float = 30.0


def is_remote_config(path: str | PathLike[str]) -> bool:
    """True when ``path`` is an ``http``/``https`` URL rather than a file path.

    Deliberately narrow: only http(s) counts. A ``file://`` URL would bypass the
    filesystem branch's path handling for no benefit, and other schemes
    (``ftp://``, ``s3://``) are not supported, so they are better reported as an
    unreadable path than half-handled here.
    """
    text = str(path)
    # A bare Windows drive letter ("C:\...") parses as a scheme, so require the
    # scheme to be one we actually handle.
    return urlparse(text).scheme in ("http", "https")


def normalize_config_url(url: str) -> str:
    """Rewrite a GitHub file-viewer URL to the raw content URL; pass others through.

    ``https://github.com/<owner>/<repo>/blob/<ref>/<path>`` serves an HTML page,
    so fetching it yields markup rather than YAML. The raw equivalent is
    ``https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>``.

    Idempotent: a URL that is already raw, or that points at another host, is
    returned unchanged.
    """
    parsed = urlparse(url)
    if parsed.netloc not in (_GITHUB_HOST, f"www.{_GITHUB_HOST}"):
        return url

    # /<owner>/<repo>/(blob|raw)/<rest...>
    parts = parsed.path.lstrip("/").split("/")
    if len(parts) < 5 or parts[2] not in ("blob", "raw"):
        return url

    owner, repo, _, *rest = parts
    raw_path = "/".join([owner, repo, *rest])
    rewritten = urlunparse(
        ("https", _GITHUB_RAW_HOST, f"/{raw_path}", "", parsed.query, "")
    )
    logger.debug(f"Rewrote GitHub URL to raw content: {url} -> {rewritten}")
    return rewritten


def fetch_config_text(url: str, *, timeout: float = DEFAULT_FETCH_TIMEOUT) -> str:
    """Fetch a remote YAML config and return its text.

    Args:
        url: An ``http(s)`` URL. Pass it through :func:`normalize_config_url`
            first if it may be a GitHub file-viewer link.
        timeout: Seconds to wait for the request.

    Returns:
        The response body as text — parsed as YAML by the caller, never executed.

    Raises:
        ValueError: on any non-2xx response or transport failure, with a message
            naming the URL. A private GitHub repo answers 404 rather than 401, so
            a missing file and an unauthorized one are indistinguishable; the
            error says so instead of guessing.
    """
    import httpx

    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError as e:
        raise ValueError(f"Could not fetch config from {url}: {e}") from e

    if response.status_code in (401, 403, 404):
        raise ValueError(
            f"Config not found at {url} (HTTP {response.status_code}). If it "
            "lives in a private repository, download it locally and pass the "
            "path — dao-ai does not send credentials to third-party hosts."
        )
    if response.status_code >= 400:
        detail = response.text.strip()[:200]
        raise ValueError(
            f"Could not fetch config from {url} (HTTP {response.status_code})"
            + (f": {detail}" if detail else "")
        )

    return response.text
