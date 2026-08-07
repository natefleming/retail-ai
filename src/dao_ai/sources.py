"""Where a dao-ai config comes from, and whether a local tree backs it.

``AppConfig.from_file`` accepts a filesystem path, an ``http(s)`` URL, or a git
locator. Those differ in exactly two ways — how the YAML text is read, and
whether there is a local directory to anchor relative ``ddl`` / ``data`` /
``code_paths`` / ``skills`` against — and are identical downstream. This module
isolates that difference behind :class:`ConfigSource` so the loader itself does
not care where the bytes came from.

The ``base_path`` a source returns is the whole story for asset resolution:

* :class:`FileSource` — the config's own directory, so colocated assets resolve.
* :class:`UrlSource` — ``None``. One YAML with no tree behind it, so a config
  declaring a relative asset is rejected (see
  ``_reject_relative_assets_for_remote_config``).
* :class:`GitSource` — the config's directory *inside the checkout*, so a remote
  project's assets resolve exactly as a local one's do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Union

__all__ = [
    "ConfigSource",
    "FileSource",
    "ResolvedConfig",
    "SourceLike",
    "UrlSource",
    "resolve_source",
]


@dataclass(frozen=True)
class ResolvedConfig:
    """A config's raw text plus where (if anywhere) its relative paths anchor.

    Attributes:
        text: The unparsed YAML. Never executed by the loader.
        origin: The source as the user expressed it — a path, a URL, or a git
            locator. Stamped onto ``AppConfig._source_config_path`` and echoed in
            error messages, so it must stay recognizable to whoever typed it.
        base_path: Directory that relative assets resolve against, or ``None``
            when no local tree backs the config.
        local_path: The config's own path on disk, when it has one. For a git
            source this is the file inside the checkout, which the CLI needs so
            that consumers doing ``Path(config)`` keep working.
        revision: Immutable identifier of the fetched revision (a git commit SHA)
            when the source has one. Folded into the bundle checksum so bumping a
            ref re-stages even if the config text is byte-identical.
    """

    text: str
    origin: str
    base_path: Path | None
    local_path: Path | None = None
    revision: str | None = None


class ConfigSource(ABC):
    """A place a config can be loaded from.

    An ABC rather than a ``Protocol`` because callers dispatch on ``isinstance``
    to validate that an explicitly-constructed source is the kind a given
    ``from_*`` method accepts; a ``runtime_checkable`` Protocol would only test
    for the presence of a ``load`` attribute and let the wrong source through.

    Subclasses hold their own spec so a caller can construct one with options a
    bare string cannot express (``GitSource(spec, token=...)``).
    """

    @staticmethod
    @abstractmethod
    def handles(spec: str) -> bool:
        """True when this source claims ``spec``."""

    @abstractmethod
    def load(self) -> ResolvedConfig:
        """Read the config, returning its text and asset anchor."""


class UrlSource(ConfigSource):
    """A config fetched over ``http(s)``.

    One YAML and nothing else: there is no directory behind a URL, so
    ``base_path`` is ``None`` and the loader rejects any relative asset the config
    declares. Use :class:`GitSource` to bring a whole project along.
    """

    def __init__(self, url: str | PathLike[str]) -> None:
        self.url: str = str(url)

    @staticmethod
    def handles(spec: str) -> bool:
        from dao_ai.config_source import is_remote_config

        return is_remote_config(spec)

    def load(self) -> ResolvedConfig:
        # Imported inside `load` (not at module scope) so tests can monkeypatch
        # `dao_ai.config_source.fetch_config_text` and have this see the patch.
        from dao_ai import config_source

        url: str = config_source.normalize_config_url(self.url)
        return ResolvedConfig(
            text=config_source.fetch_config_text(url),
            origin=url,
            base_path=None,
        )


class FileSource(ConfigSource):
    """A config on the local filesystem.

    The fallback when no other source claims a spec.
    """

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path: Path = Path(path)

    @staticmethod
    def handles(spec: str) -> bool:
        """Always true: this is the last source tried, so anything unclaimed is a path.

        Deliberately not an existence check — a missing file must surface as the
        read's own ``FileNotFoundError`` naming the path, not as an
        unhelpful "no source handles this" from :func:`resolve_source`.
        """
        return True

    def load(self) -> ResolvedConfig:
        from dao_ai.git_source import revision_for_checkout_path

        # `as_posix()` matches the pre-refactor `_source_config_path` value that
        # every path convention (and its tests) already anchors on.
        return ResolvedConfig(
            text=self.path.read_text(),
            origin=self.path.as_posix(),
            base_path=self.path.parent,
            local_path=self.path,
            # A path inside the checkout cache is still a git-sourced config: the
            # CLI materializes a locator during parse_args and passes the plain
            # path downstream, so this is the common case for `dao-ai <noun>`
            # commands. Recovering the SHA here keeps the bundle checksum
            # revision-aware regardless of which door the config came through.
            revision=revision_for_checkout_path(self.path),
        )


def _sources() -> tuple[type[ConfigSource], ...]:
    """Source classes in claim order.

    ``GitSource`` first, so ``git+https://...`` is claimed as git before anything
    reads it as a URL. ``FileSource`` last, since its ``handles`` is
    unconditional. Built lazily because ``git_source`` imports from here.
    """
    from dao_ai.git_source import GitSource

    return (GitSource, UrlSource, FileSource)


SourceLike = Union[str, PathLike, ConfigSource]


def resolve_source(spec: str | PathLike[str]) -> ConfigSource:
    """The first source whose :meth:`ConfigSource.handles` claims ``spec``."""
    text: str = str(spec)
    for source_cls in _sources():
        if source_cls.handles(text):
            return source_cls(text)
    # Unreachable: FileSource.handles is unconditional.
    raise AssertionError(f"No config source claimed {text!r}")
