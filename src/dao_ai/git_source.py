"""Loading a dao-ai config from a git repository, with its whole project tree.

``UrlSource`` fetches one YAML, so a config declaring ``ddl: data/products.sql``
cannot be loaded from a URL at all — there is no directory behind it. Most real
dao-ai projects are a config *plus* colocated assets (``data/``, ``functions/``,
``src/``, ``skills/``, ``resources/``), so sharing one means sharing a tree.

This materializes the repo into a local cache and hands the loader a real path
inside the checkout. Every path convention then resolves exactly as it does for a
local project, because they all anchor on the config's own directory.

Resolution is strictly client-side: the generated bundle is self-contained, so
``git`` is never needed on a cluster.

Trust: a git locator runs the repo's code, exactly as ``git clone`` followed by
``dao-ai agent up`` would — a config can ship Python via ``code_paths`` / ``src/``
and inline tool code. The resolved commit SHA is always reported. Pin a tag or SHA
for repos you do not control.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from loguru import logger

from dao_ai.sources import ConfigSource, ResolvedConfig

__all__ = ["GitLocator", "GitSource", "is_git_locator", "parse_git_locator"]

#: Scheme prefix marking a git locator, mirroring pip/uv (``git+https://...``).
_GIT_PREFIX = "git+"

#: Shorthand host aliases: ``gh:owner/repo`` -> GitHub.
_HOST_ALIASES: dict[str, str] = {
    "gh": "github.com",
    "github": "github.com",
}

#: A full 40-hex commit SHA — immutable, so it never needs re-resolving.
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

#: Filenames that identify the config in a directory without a scan.
_CONVENTIONAL_NAMES: tuple[str, ...] = ("dao-ai.yaml", "dao-ai.yml", "dao_ai.yaml")


@dataclass(frozen=True)
class GitLocator:
    """A parsed git locator.

    Attributes:
        remote_url: The URL to hand ``git`` (``git+`` stripped, no userinfo).
        ref: Branch, tag, or commit SHA; ``None`` means the remote's default HEAD.
        in_repo_path: Config path within the repo, or ``None`` to auto-discover.
        original: The locator as the user typed it, for error messages.
    """

    remote_url: str
    ref: str | None
    in_repo_path: str | None
    original: str

    @property
    def is_immutable_ref(self) -> bool:
        """True when ``ref`` is a full commit SHA, which can never move.

        Tags are deliberately NOT treated as immutable: git tags are mutable by
        design (``git tag -f``), and a moved tag silently serving a stale checkout
        is a worse failure than one extra ``ls-remote``.
        """
        return self.ref is not None and bool(_FULL_SHA.match(self.ref))


def is_git_locator(spec: str | PathLike[str]) -> bool:
    """True when ``spec`` is a git locator rather than a path or plain URL.

    Two accepted spellings::

        git+https://github.com/owner/repo@v1.0#path/to/agent.yaml
        gh:owner/repo@main#path/to/agent.yaml

    ``git+https`` parses with scheme ``git+https``, so it can never be mistaken
    for an ``http(s)`` URL by :func:`dao_ai.config_source.is_remote_config`.
    """
    text = str(spec)
    if text.startswith(_GIT_PREFIX):
        return True
    scheme: str = urlparse(text).scheme
    # `gh:owner/repo` — require a path so a bare `gh:` isn't claimed.
    return scheme in _HOST_ALIASES and bool(text[len(scheme) + 1 :].strip("/"))


def parse_git_locator(spec: str | PathLike[str]) -> GitLocator:
    """Parse a git locator into its remote URL, ref, and in-repo config path.

    Grammar: ``<repo>[@<ref>][#<in-repo-path>]``, where ``<repo>`` is either
    ``git+<scheme>://<host>/<owner>/<repo>`` or a ``<alias>:<owner>/<repo>``
    shorthand.

    The ``@`` separating the ref is the **first** one after the final path
    segment, so refs containing ``/`` work (``@feature/foo``). A ref containing
    ``@`` is not supported. ``git+ssh://git@host/...`` is handled: the ``git@``
    there is userinfo in the netloc, not a ref separator.

    Raises:
        ValueError: if ``spec`` is not a locator, carries inline credentials, or
            names no repository.
    """
    original = str(spec)
    if not is_git_locator(original):
        raise ValueError(
            f"Not a git locator: {original!r}. Expected "
            "'git+https://host/owner/repo[@ref][#path]' or "
            "'gh:owner/repo[@ref][#path]'."
        )

    # Split the fragment first: '#' can only introduce the in-repo path, and a
    # ref never contains one.
    body, _, fragment = original.partition("#")
    in_repo_path: str | None = fragment.strip("/") or None

    if body.startswith(_GIT_PREFIX):
        parsed = urlparse(body[len(_GIT_PREFIX) :])
        scheme, netloc, path = parsed.scheme, parsed.netloc, parsed.path
    else:
        alias, _, rest = body.partition(":")
        scheme, netloc, path = "https", _HOST_ALIASES[alias], f"/{rest.lstrip('/')}"

    # An inline token would land in the cache path and in git's stderr. Refuse it
    # and point at the env var, which never touches disk. `git@host` (ssh, no
    # password) is the one legitimate userinfo form.
    if "@" in netloc:
        userinfo, _, host = netloc.rpartition("@")
        if ":" in userinfo or userinfo not in ("git", ""):
            raise ValueError(
                f"Git locator carries inline credentials: {original!r}. Remove "
                "them and set DAO_AI_GIT_TOKEN (or GITHUB_TOKEN) instead — it is "
                "passed to git without being written to disk."
            )
        netloc = host if userinfo == "" else netloc

    # The repo is always exactly <owner>/<repo>, so the ref begins at the first
    # '@' that appears after the SECOND path segment. Scanning from the last '/'
    # would break on a ref containing one ('@feature/foo'), since that slash then
    # becomes the last.
    ref: str | None = None
    segments: list[str] = path.strip("/").split("/", 2)
    if len(segments) >= 2 and "@" in segments[1]:
        segments[1], _, ref = segments[1].partition("@")
        # A ref with '/' swallowed the remaining segments; give them back.
        if len(segments) > 2:
            ref = f"{ref}/{segments[2]}"
            del segments[2]
        path = "/".join(segments[:2])
        ref = ref or None

    repo_path: str = path.strip("/")
    if not repo_path:
        raise ValueError(f"Git locator names no repository: {original!r}")

    return GitLocator(
        remote_url=urlunparse((scheme, netloc, f"/{repo_path}", "", "", "")),
        ref=ref,
        in_repo_path=in_repo_path,
        original=original,
    )


class GitSource(ConfigSource):
    """A config in a git repository, materialized with its whole tree."""

    def __init__(
        self,
        locator: str | PathLike[str],
        *,
        token: str | None = None,
        cache_dir: Path | None = None,
        refresh: bool = False,
    ) -> None:
        self.locator: str = str(locator)
        self.token: str | None = token
        self.cache_dir: Path | None = cache_dir
        self.refresh: bool = refresh

    @staticmethod
    def handles(spec: str) -> bool:
        return is_git_locator(spec)

    def load(self) -> ResolvedConfig:
        raise NotImplementedError
