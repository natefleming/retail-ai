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

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import yaml
from loguru import logger

from dao_ai.sources import ConfigSource, ResolvedConfig

__all__ = [
    "GitLocator",
    "GitSource",
    "cache_root",
    "discover_config",
    "is_git_locator",
    "parse_git_locator",
]

#: Override for the checkout cache root.
_CACHE_ENV_VAR = "DAO_AI_GIT_CACHE"

#: Token env vars, in precedence order, for fetching a private repo over https.
_TOKEN_ENV_VARS: tuple[str, ...] = ("DAO_AI_GIT_TOKEN", "GITHUB_TOKEN")

#: Env var the inline credential helper reads the token from. Distinct from the
#: user-facing vars so the helper works whichever of them supplied the token.
_TOKEN_HANDOFF_VAR = "DAO_AI_GIT_TOKEN_INTERNAL"

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

#: Top-level keys a dao-ai config never has, which mark a YAML file as something
#: else. ``bundle``/``targets``/``artifacts`` are Databricks Asset Bundles (whose
#: top-level ``resources:`` otherwise mimics a config); ``command``/``env`` are an
#: Apps ``app.yaml``; ``client``/``dependencies`` are a serverless environment
#: spec. All of these commonly sit in a repo root next to a real config.
_FOREIGN_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"bundle", "targets", "artifacts", "command", "env", "client"}
)


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

    # The ref starts at the LAST '@' in the path. Scanning from the last '/'
    # would break a ref containing one ('@feature/foo'), since that slash then
    # becomes the last; assuming '<owner>/<repo>' would break a deep path
    # ('git+file:///home/me/repos/proj@main'). A ref containing '@' is
    # unsupported, which is what makes the last '@' unambiguous.
    ref: str | None = None
    at: int = path.rfind("@")
    if at != -1:
        path, ref = path[:at], path[at + 1 :] or None

    repo_path: str = path.strip("/")
    if not repo_path:
        raise ValueError(f"Git locator names no repository: {original!r}")

    return GitLocator(
        remote_url=urlunparse((scheme, netloc, f"/{repo_path}", "", "", "")),
        ref=ref,
        in_repo_path=in_repo_path,
        original=original,
    )


def revision_for_checkout_path(path: Path | str) -> str | None:
    """Recover the commit SHA from a path inside the checkout cache.

    The CLI materializes a locator once during ``parse_args`` and hands downstream
    consumers the plain local path, so ``AppConfig.from_file`` legitimately loads
    that path via :class:`~dao_ai.sources.FileSource` and never sees the locator.
    The revision still matters — the bundle checksum folds it in so a ref bump
    re-stages — and the cache layout already encodes it
    (``<root>/<host>/<owner>/<repo>/<sha>/...``), so read it back rather than
    threading it through every call site.

    Returns ``None`` for any path outside the cache, or whose cache-relative
    segments contain no commit SHA.
    """
    resolved: Path = Path(path).resolve()
    root: Path = cache_root().resolve()
    try:
        relative: Path = resolved.relative_to(root)
    except ValueError:
        return None
    # `<host>/<owner>/<repo>/<sha>/<in-repo path...>`: take the first segment that
    # is a full SHA. Repo path depth varies (a `file://` remote is arbitrarily
    # deep), so match on shape rather than position.
    for part in relative.parts:
        if _FULL_SHA.match(part):
            return part
    return None


def _token_from_env() -> str | None:
    """First non-empty token env var, or ``None``."""
    for var in _TOKEN_ENV_VARS:
        if value := os.environ.get(var):
            return value
    return None


def _is_populated(path: Path) -> bool:
    """True when ``path`` is a checkout with content (not a bare or partial dir)."""
    return path.is_dir() and any(child.name != ".git" for child in path.iterdir())


def _newest_cached_checkout(
    locator: GitLocator, *, root: Path
) -> tuple[Path, str] | None:
    """Most recently fetched cached checkout of this repo, if any.

    The offline fallback: better to serve a known-stale commit, loudly, than to
    fail a command outright.
    """
    repo_dir: Path = _checkout_dir(locator, "x", root=root).parent
    if not repo_dir.is_dir():
        return None
    candidates: list[Path] = [d for d in repo_dir.iterdir() if _is_populated(d)]
    if not candidates:
        return None
    newest: Path = max(candidates, key=lambda d: d.stat().st_mtime)
    return newest, newest.name


def _looks_like_dao_ai_config(path: Path) -> bool:
    """True when ``path`` parses as YAML that could be a dao-ai config.

    Used only to disambiguate a directory holding several YAML files (an example
    dir shipping both a config and an ``examples.yaml`` of sample prompts). Checks
    for a top-level key that only a config would carry, rather than attempting
    full validation: a config using ``${var.X}`` in a typed field will not
    validate until its parameters are substituted, which happens later.

    Files carrying a key that a dao-ai config never has at the top level are
    excluded outright. This matters most for a Databricks Asset Bundle
    (``databricks.yml``), which also has a top-level ``resources:`` and so would
    otherwise look like a config — and often sits in the repo root, exactly where
    a locator with no ``#path`` looks.
    """
    try:
        loaded = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError):
        return False
    if not isinstance(loaded, dict):
        return False
    if _FOREIGN_TOP_LEVEL_KEYS & loaded.keys():
        return False
    return bool({"app", "agents", "resources", "tools", "retrievers"} & loaded.keys())


def discover_config(checkout: Path, in_repo_path: str | None, *, locator: str) -> Path:
    """Locate the config file within a checkout.

    A locator may name the config exactly, name a directory, or omit the path
    entirely. For a directory (or an omitted path, meaning the repo root) a
    conventional filename wins; failing that, the directory's YAML files are
    filtered to those that look like a config and the choice must be unambiguous.

    Raises:
        ValueError: if the path does not exist, or a directory holds no
            recognizable config or more than one.
    """
    target: Path = checkout / in_repo_path if in_repo_path else checkout
    if not target.exists():
        raise ValueError(
            f"{in_repo_path!r} does not exist in {locator}. Check the path after "
            "'#' — it is relative to the repository root."
        )
    if target.is_file():
        return target

    for name in _CONVENTIONAL_NAMES:
        if (conventional := target / name).is_file():
            return conventional

    candidates: list[Path] = sorted(
        p
        for p in target.iterdir()
        if p.is_file()
        and p.suffix in (".yaml", ".yml")
        and _looks_like_dao_ai_config(p)
    )
    location: str = f"{in_repo_path!r}" if in_repo_path else "the repository root"
    if not candidates:
        raise ValueError(
            f"No dao-ai config found in {location} of {locator}. Name the config "
            "file explicitly after '#', e.g. '#path/to/agent.yaml'."
        )
    if len(candidates) > 1:
        listed: str = "".join(
            f"\n  - {p.relative_to(checkout).as_posix()}" for p in candidates
        )
        raise ValueError(
            f"{len(candidates)} dao-ai configs in {location} of {locator}:{listed}"
            "\nName the one you want after '#'."
        )
    return candidates[0]


def cache_root(override: Path | None = None) -> Path:
    """Root of the checkout cache.

    ``$DAO_AI_GIT_CACHE`` wins, then ``$XDG_CACHE_HOME/dao-ai/git``, then
    ``~/.cache/dao-ai/git``.
    """
    if override is not None:
        return override
    if env_root := os.environ.get(_CACHE_ENV_VAR):
        return Path(env_root).expanduser()
    xdg: str | None = os.environ.get("XDG_CACHE_HOME")
    base: Path = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "dao-ai" / "git"


def repo_cache_dir(locator: GitLocator, *, root: Path | None = None) -> Path:
    """Cache directory holding every checkout of one repository.

    ``<root>/<host>/<owner>/<repo>``, with per-commit subdirectories beneath.
    """
    parsed = urlparse(locator.remote_url)
    # Strip userinfo ('git@github.com') so it can't become a path segment.
    host: str = parsed.netloc.rpartition("@")[2] or "local"
    repo_path: str = parsed.path.strip("/").removesuffix(".git")
    return cache_root(root).joinpath(host, *repo_path.split("/"))


def _checkout_dir(locator: GitLocator, sha: str, *, root: Path) -> Path:
    """Cache location for one commit: ``<root>/<host>/<owner>/<repo>/<sha>``.

    Keyed by commit, so an immutable ref is a pure cache hit and two refs of the
    same repo never clobber one another.
    """
    return repo_cache_dir(locator, root=root) / sha


def _git_env(token: str | None) -> dict[str, str]:
    """Environment for a git subprocess, carrying the token if there is one."""
    env: dict[str, str] = dict(os.environ)
    # Never let git block on an interactive credential prompt: in a notebook or
    # CI that hangs forever instead of failing.
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token:
        env[_TOKEN_HANDOFF_VAR] = token
    return env


def _auth_args(token: str | None) -> list[str]:
    """``git -c`` args that authenticate via the token, if one is set.

    The token is passed through the child's environment and read by an inline
    credential helper. It is deliberately NOT interpolated into the remote URL:
    ``git remote add https://TOKEN@host/repo`` writes the token verbatim into the
    cache's ``.git/config``, where it persists indefinitely. Keeping it out of the
    URL also keeps it out of git's stderr and out of argv.
    """
    if not token:
        return []
    helper: str = (
        f'!f() {{ echo username=x; echo "password=${_TOKEN_HANDOFF_VAR}"; }}; f'
    )
    return ["-c", f"credential.helper={helper}"]


def _redact(text: str, token: str | None) -> str:
    """Blank a token out of command output before it reaches a log or exception."""
    return text.replace(token, "***") if token else text


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    token: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a git command, raising a redacted ``ValueError`` on failure.

    Raises:
        ValueError: if ``git`` is missing, or the command fails and ``check``.
    """
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            env=_git_env(token),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise ValueError(
            "git is required to load a config from a repository but was not "
            "found on PATH. Install git, or download the project and pass a "
            "local path."
        ) from e

    if check and result.returncode != 0:
        detail: str = _redact(
            (result.stderr or result.stdout or "").strip()[:500], token
        )
        raise ValueError(f"git {args[0]} failed: {detail}")
    return result


class GitSource(ConfigSource):
    """A config in a git repository, materialized with its whole tree.

    Unlike :class:`~dao_ai.sources.UrlSource`, this yields a real ``base_path``,
    so a remote project's colocated ``ddl`` / ``data`` / ``code_paths`` / ``src``
    / ``skills`` resolve exactly as a local project's do.

    Args:
        locator: A git locator (see :func:`parse_git_locator`).
        token: Token for a private repo. Defaults to ``$DAO_AI_GIT_TOKEN`` or
            ``$GITHUB_TOKEN``; ignored for ssh remotes, which use ssh-agent.
        cache_dir: Override the checkout cache root.
        refresh: Re-fetch a mutable ref even if it is already cached.
    """

    def __init__(
        self,
        locator: str | PathLike[str],
        *,
        token: str | None = None,
        cache_dir: Path | None = None,
        refresh: bool = False,
    ) -> None:
        self.locator: GitLocator = parse_git_locator(locator)
        self.token: str | None = token or _token_from_env()
        self.cache_dir: Path | None = cache_dir
        self.refresh: bool = refresh

    @staticmethod
    def handles(spec: str) -> bool:
        return is_git_locator(spec)

    def __str__(self) -> str:
        return self.locator.original

    def load(self) -> ResolvedConfig:
        checkout, sha = self._materialize()
        config_file: Path = discover_config(
            checkout, self.locator.in_repo_path, locator=self.locator.original
        )
        logger.info(
            "Loaded config from git",
            source=self.locator.remote_url,
            ref=self.locator.ref or "HEAD",
            sha=sha[:12],
            config=config_file.relative_to(checkout).as_posix(),
            cache=str(checkout),
        )
        return ResolvedConfig(
            text=config_file.read_text(),
            # The locator, not the cache path: this reaches the user in error
            # messages, and a cache path means nothing to whoever typed the locator.
            origin=self.locator.original,
            base_path=config_file.parent,
            local_path=config_file,
            revision=sha,
        )

    def _materialize(self) -> tuple[Path, str]:
        """Ensure the commit is checked out locally; return its dir and SHA."""
        root: Path = cache_root(self.cache_dir)

        # An immutable ref names its own commit, so the cache can be trusted
        # without asking the remote anything.
        if self.locator.is_immutable_ref:
            assert self.locator.ref is not None
            cached: Path = _checkout_dir(self.locator, self.locator.ref, root=root)
            if _is_populated(cached) and not self.refresh:
                logger.debug(f"Using cached checkout {cached}")
                return cached, self.locator.ref
            return self._fetch(self.locator.ref, root=root), self.locator.ref

        # A mutable ref (branch, tag, or default HEAD) may have moved: ask the
        # remote cheaply, and reuse the cache when it hasn't.
        sha: str | None = self._resolve_remote_sha()
        if sha is not None:
            cached = _checkout_dir(self.locator, sha, root=root)
            if _is_populated(cached) and not self.refresh:
                ref_name: str = self.locator.ref or "HEAD"
                logger.debug(f"Ref {ref_name} unchanged at {sha[:12]}")
                return cached, sha
            return self._fetch(self.locator.ref, root=root, expected_sha=sha), sha

        # ls-remote failed (offline, most likely). Any cached checkout of this
        # repo beats failing outright.
        if fallback := _newest_cached_checkout(self.locator, root=root):
            path, cached_sha = fallback
            logger.warning(
                "Could not reach the remote; using the most recent cached "
                f"checkout of {self.locator.remote_url} ({cached_sha[:12]}). "
                "Pass --refresh when back online."
            )
            return path, cached_sha
        raise ValueError(
            f"Could not resolve {self.locator.ref or 'HEAD'} at "
            f"{self.locator.remote_url} and no cached checkout is available. "
            "Check the repository URL, the ref, and your network access."
        )

    def _resolve_remote_sha(self) -> str | None:
        """Commit a mutable ref currently points at, or ``None`` if unreachable."""
        args: list[str] = [
            *_auth_args(self.token),
            "ls-remote",
            self.locator.remote_url,
        ]
        if self.locator.ref:
            args.append(self.locator.ref)
        result = _run(args, token=self.token, check=False)
        if result.returncode != 0:
            logger.debug(
                f"ls-remote failed: {_redact(result.stderr.strip(), self.token)}"
            )
            return None

        lines: list[str] = [ln for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            return None
        if self.locator.ref:
            # Prefer an exact tag/head match; a bare ref can match several
            # entries (refs/heads/x, refs/tags/x, and tags' ^{} peels).
            wanted: tuple[str, ...] = (
                f"refs/heads/{self.locator.ref}",
                f"refs/tags/{self.locator.ref}",
            )
            for line in lines:
                sha, _, ref_name = line.partition("\t")
                if ref_name.strip() in wanted:
                    return sha.strip()
        return lines[0].partition("\t")[0].strip()

    def _fetch(
        self, ref: str | None, *, root: Path, expected_sha: str | None = None
    ) -> Path:
        """Fetch ``ref`` into the cache and return its checkout directory.

        Fetches into a temporary sibling and moves it into place, so a concurrent
        invocation can never observe a half-populated checkout.
        """
        root.mkdir(parents=True, exist_ok=True)
        staging: Path = Path(tempfile.mkdtemp(prefix=".fetch-", dir=str(root)))
        try:
            _run(["init", "--quiet"], cwd=staging, token=self.token)
            _run(
                ["remote", "add", "origin", self.locator.remote_url],
                cwd=staging,
                token=self.token,
            )

            fetch_args: list[str] = [*_auth_args(self.token), "fetch", "--quiet"]
            target: str = ref or "HEAD"
            shallow = _run(
                [*fetch_args, "--depth", "1", "origin", target],
                cwd=staging,
                token=self.token,
                check=False,
            )
            if shallow.returncode != 0:
                # Fetching a bare SHA needs uploadpack.allowReachableSHA1InWant,
                # which many GHE/GitLab installs disable. Full history always works.
                logger.debug("Shallow fetch refused; retrying without --depth")
                _run([*fetch_args, "origin", target], cwd=staging, token=self.token)

            _run(["checkout", "--quiet", "FETCH_HEAD"], cwd=staging, token=self.token)
            sha: str = _run(
                ["rev-parse", "HEAD"], cwd=staging, token=self.token
            ).stdout.strip()

            if expected_sha and sha != expected_sha:
                logger.debug(
                    f"Ref moved mid-fetch: expected {expected_sha[:12]}, got {sha[:12]}"
                )

            destination: Path = _checkout_dir(self.locator, sha, root=root)
            if _is_populated(destination):
                # Another process won the race; its checkout is the same commit.
                return destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(staging, destination)
            except OSError:
                # Cross-device or a non-empty destination: fall back to a copy.
                shutil.copytree(staging, destination, dirs_exist_ok=True)
                return destination
            staging = destination  # moved; nothing left to clean up
            return destination
        finally:
            if staging.exists() and staging.name.startswith(".fetch-"):
                shutil.rmtree(staging, ignore_errors=True)
