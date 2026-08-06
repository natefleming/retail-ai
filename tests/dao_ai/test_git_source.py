"""Tests for loading configs from a git repository.

``UrlSource`` can only serve a single self-contained YAML — a remote config
declaring ``ddl: data/x.sql`` is rejected, because a URL has no directory to
anchor it to. ``GitSource`` materializes the whole tree, so these cover the
locator grammar, the cache/ref semantics, config discovery, token safety, and the
dual spec-or-source surface on every ``from_*`` method.

The locator/parse/discovery tests are pure. The ones that fetch use a real local
repository built by the ``git_repo`` fixture rather than a mocked ``subprocess``:
the interesting behavior here IS the git interaction (shallow fetch, ref
resolution, atomic move), and a stubbed ``run`` would assert only that we call the
commands we wrote down.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dao_ai.config_source import is_remote_config
from dao_ai.git_source import (
    GitSource,
    cache_root,
    discover_config,
    is_git_locator,
    parse_git_locator,
)
from dao_ai.sources import (
    ConfigSource,
    FileSource,
    ResolvedConfig,
    UrlSource,
    resolve_source,
)

_MINIMAL_YAML = """
app:
  name: git-app
  agents:
  - name: a
    description: d
    model:
      name: databricks-gpt-5-4-mini
"""


def _git(*args: str, cwd: Path) -> None:
    """Run a git command in ``cwd``, failing the test on error."""
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": __import__("os").environ.get("PATH", ""),
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A local repository holding a config plus colocated assets.

    Mirrors the shape of a real dao-ai project: a config with relative ``ddl`` /
    ``data`` paths, which is exactly what a URL source cannot load.
    """
    repo: Path = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "nested").mkdir()
    (repo / "data" / "seed.sql").write_text("SELECT 1;\n")
    (repo / "dao-ai.yaml").write_text(_MINIMAL_YAML)
    (repo / "nested" / "agent.yaml").write_text(_MINIMAL_YAML)
    # A sibling YAML that is NOT a config, so discovery must not count it.
    (repo / "nested" / "examples.yaml").write_text("prompts:\n- hello\n")
    _git("init", "--quiet", cwd=repo)
    _git("add", "-A", cwd=repo)
    _git("commit", "--quiet", "--message", "init", cwd=repo)
    return repo


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the checkout cache at a temp dir so tests never touch the real one."""
    root: Path = tmp_path / "cache"
    monkeypatch.setenv("DAO_AI_GIT_CACHE", str(root))
    # A stray token in the ambient env would change what the fetch does.
    monkeypatch.delenv("DAO_AI_GIT_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    return root


def _locator(repo: Path, ref: str = "main", path: str | None = "dao-ai.yaml") -> str:
    suffix: str = f"#{path}" if path else ""
    return f"git+file://{repo}@{ref}{suffix}"


@pytest.mark.unit
class TestIsGitLocator:
    def test_detects_git_prefixed_urls(self) -> None:
        assert is_git_locator("git+https://github.com/o/r@main#a.yaml")
        assert is_git_locator("git+ssh://git@github.com/o/r")
        assert is_git_locator("git+file:///tmp/repo")

    def test_detects_host_shorthand(self) -> None:
        assert is_git_locator("gh:o/r")
        assert is_git_locator("github:o/r@v1#a.yaml")

    def test_rejects_paths_and_plain_urls(self) -> None:
        for spec in (
            "config.yaml",
            "/abs/config.yaml",
            "./rel/config.yaml",
            "https://example.com/c.yaml",
            "http://example.com/c.yaml",
            r"C:\repos\config.yaml",
        ):
            assert not is_git_locator(spec), spec

    def test_bare_shorthand_scheme_is_not_a_locator(self) -> None:
        """`gh:` with nothing after it names no repository."""
        assert not is_git_locator("gh:")

    def test_does_not_collide_with_the_url_source(self) -> None:
        """`git+https` must never be mistaken for an http(s) URL.

        Both loaders sniff the spec; if `is_remote_config` claimed a git locator
        it would fetch the URL verbatim and fail on HTML.
        """
        for spec in ("git+https://github.com/o/r@main#a.yaml", "gh:o/r@main"):
            assert is_git_locator(spec)
            assert not is_remote_config(spec)


@pytest.mark.unit
class TestParseGitLocator:
    def test_parses_full_locator(self) -> None:
        loc = parse_git_locator("git+https://github.com/o/r@v1.0#dir/agent.yaml")
        assert loc.remote_url == "https://github.com/o/r"
        assert loc.ref == "v1.0"
        assert loc.in_repo_path == "dir/agent.yaml"

    def test_ref_and_path_are_optional(self) -> None:
        loc = parse_git_locator("gh:o/r")
        assert (loc.remote_url, loc.ref, loc.in_repo_path) == (
            "https://github.com/o/r",
            None,
            None,
        )

    def test_expands_host_shorthand(self) -> None:
        assert parse_git_locator("gh:o/r@main").remote_url == "https://github.com/o/r"

    def test_ref_may_contain_slashes(self) -> None:
        """`@feature/foo` is a valid branch name and must survive parsing."""
        loc = parse_git_locator("gh:o/r@feature/foo#a.yaml")
        assert (loc.remote_url, loc.ref) == ("https://github.com/o/r", "feature/foo")

        deep = parse_git_locator("gh:o/r@release/2026/q3#a.yaml")
        assert deep.ref == "release/2026/q3"

    def test_ssh_userinfo_is_not_a_ref_separator(self) -> None:
        """The `git@` in an ssh remote is userinfo, not the start of a ref."""
        loc = parse_git_locator("git+ssh://git@github.com/o/r@v1#a.yaml")
        assert loc.remote_url == "ssh://git@github.com/o/r"
        assert loc.ref == "v1"

    def test_handles_a_deep_repo_path(self) -> None:
        """A `file://` remote is an arbitrarily deep path, not `<owner>/<repo>`."""
        loc = parse_git_locator("git+file:///home/me/repos/proj@main#a.yaml")
        assert loc.remote_url == "file:///home/me/repos/proj"
        assert loc.ref == "main"

    def test_preserves_a_dot_git_suffix(self) -> None:
        assert parse_git_locator("gh:o/r.git@main").remote_url == (
            "https://github.com/o/r.git"
        )

    def test_full_sha_is_immutable_but_a_tag_is_not(self) -> None:
        """Tags are mutable (`git tag -f`), so only a full SHA skips re-resolution."""
        sha: str = "8f50a25c9b1e4d7a8f50a25c9b1e4d7a8f50a25c"
        assert parse_git_locator(f"gh:o/r@{sha}").is_immutable_ref
        assert not parse_git_locator("gh:o/r@v1.0").is_immutable_ref
        assert not parse_git_locator("gh:o/r@main").is_immutable_ref
        assert not parse_git_locator("gh:o/r").is_immutable_ref

    def test_rejects_inline_credentials(self) -> None:
        """A token in the URL would land in the cache path and in git's stderr."""
        for spec in (
            "git+https://user:tok@github.com/o/r#a.yaml",
            "git+https://ghp_abc123@github.com/o/r#a.yaml",
        ):
            with pytest.raises(ValueError, match="DAO_AI_GIT_TOKEN"):
                parse_git_locator(spec)

    def test_rejects_a_non_locator(self) -> None:
        with pytest.raises(ValueError, match="Not a git locator"):
            parse_git_locator("config.yaml")

    def test_rejects_a_locator_naming_no_repository(self) -> None:
        with pytest.raises(ValueError, match="names no repository"):
            parse_git_locator("git+https://github.com")


@pytest.mark.unit
class TestResolveSource:
    def test_classifies_each_spec_shape(self) -> None:
        assert isinstance(resolve_source("gh:o/r@main"), GitSource)
        assert isinstance(resolve_source("https://x/c.yaml"), UrlSource)
        assert isinstance(resolve_source("config.yaml"), FileSource)

    def test_git_is_claimed_before_url(self) -> None:
        """Order matters: `git+https://` must not be read as a URL."""
        assert isinstance(resolve_source("git+https://github.com/o/r"), GitSource)

    def test_file_source_is_the_fallback(self) -> None:
        """A missing path must still classify, so the read reports the real error."""
        source = resolve_source("does/not/exist.yaml")
        assert isinstance(source, FileSource)
        with pytest.raises(FileNotFoundError):
            source.load()


@pytest.mark.unit
class TestTokenHandling:
    def test_token_never_appears_in_argv(self) -> None:
        """Only the env var NAME may reach the command line."""
        from dao_ai.git_source import _auth_args

        token = "ghp_SECRET"
        argv = " ".join(_auth_args(token))
        assert token not in argv
        assert "credential.helper" in argv

    def test_token_travels_by_environment(self) -> None:
        from dao_ai.git_source import _git_env

        env = _git_env("ghp_SECRET")
        assert "ghp_SECRET" in env.values()
        # An interactive prompt would hang a notebook or CI job forever.
        assert env["GIT_TERMINAL_PROMPT"] == "0"

    def test_no_auth_args_without_a_token(self) -> None:
        from dao_ai.git_source import _auth_args

        assert _auth_args(None) == []

    def test_errors_are_redacted(self) -> None:
        from dao_ai.git_source import _redact

        assert "ghp_SECRET" not in _redact("failed for ghp_SECRET", "ghp_SECRET")

    def test_explicit_token_beats_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAO_AI_GIT_TOKEN", "from_env")
        assert GitSource("gh:o/r", token="explicit").token == "explicit"
        assert GitSource("gh:o/r").token == "from_env"

    def test_falls_back_to_github_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAO_AI_GIT_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "gh_env")
        assert GitSource("gh:o/r").token == "gh_env"


@pytest.mark.unit
class TestDiscoverConfig:
    def test_explicit_file_wins(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text(_MINIMAL_YAML)
        found = discover_config(tmp_path, "a.yaml", locator="gh:o/r#a.yaml")
        assert found.name == "a.yaml"

    def test_conventional_name_in_a_directory(self, tmp_path: Path) -> None:
        (tmp_path / "dao-ai.yaml").write_text(_MINIMAL_YAML)
        (tmp_path / "other.yaml").write_text(_MINIMAL_YAML)
        # A conventional name is decisive even alongside another candidate.
        assert discover_config(tmp_path, None, locator="gh:o/r").name == "dao-ai.yaml"

    def test_single_candidate_is_used(self, tmp_path: Path) -> None:
        (tmp_path / "whatever.yaml").write_text(_MINIMAL_YAML)
        assert discover_config(tmp_path, None, locator="gh:o/r").name == (
            "whatever.yaml"
        )

    def test_non_config_yaml_is_ignored(self, tmp_path: Path) -> None:
        """An `examples.yaml` of sample prompts must not create ambiguity."""
        (tmp_path / "agent.yaml").write_text(_MINIMAL_YAML)
        (tmp_path / "examples.yaml").write_text("prompts:\n- hello\n")
        assert discover_config(tmp_path, None, locator="gh:o/r").name == "agent.yaml"

    def test_a_databricks_bundle_is_not_a_config(self, tmp_path: Path) -> None:
        """A DAB has a top-level `resources:` too, so key-presence alone is not enough.

        Found live: a repo root holding only `databricks.yml` was "discovered" as
        that file, which then failed to parse as an AppConfig.
        """
        (tmp_path / "databricks.yml").write_text(
            "bundle:\n  name: b\ntargets:\n  default: {}\nresources:\n  apps: {}\n"
        )
        with pytest.raises(ValueError, match="No dao-ai config found"):
            discover_config(tmp_path, None, locator="gh:o/r")

    def test_a_bundle_beside_a_config_does_not_create_ambiguity(
        self, tmp_path: Path
    ) -> None:
        """The common repo-root layout: a DAB and the real config side by side."""
        (tmp_path / "databricks.yml").write_text("bundle:\n  name: b\nresources: {}\n")
        (tmp_path / "agent.yaml").write_text(_MINIMAL_YAML)
        assert discover_config(tmp_path, None, locator="gh:o/r").name == "agent.yaml"

    @pytest.mark.parametrize(
        "name,body",
        [
            ("app.yaml", "command:\n- python\n- app.py\nenv: []\n"),
            ("environment.yaml", "client: '5'\ndependencies:\n- dao-ai\n"),
        ],
    )
    def test_other_databricks_yaml_files_are_not_configs(
        self, tmp_path: Path, name: str, body: str
    ) -> None:
        """An Apps `app.yaml` / serverless env spec also lives beside a config."""
        (tmp_path / name).write_text(body)
        (tmp_path / "agent.yaml").write_text(_MINIMAL_YAML)
        assert discover_config(tmp_path, None, locator="gh:o/r").name == "agent.yaml"

    def test_every_shipped_example_config_is_recognized(self) -> None:
        """Guards the exclusion list against rejecting a real config.

        The foreign-key check is a denylist, so a key that a config legitimately
        uses at the top level would silently make it undiscoverable.
        """
        from dao_ai.git_source import _looks_like_dao_ai_config

        examples: Path = Path(__file__).parents[2] / "examples"
        if not examples.is_dir():
            pytest.skip("examples/ not present in this checkout")

        not_configs = {"examples.yaml", "app.yaml", "environment.yaml"}
        rejected = [
            path
            for path in examples.rglob("*.yaml")
            if path.name not in not_configs and not _looks_like_dao_ai_config(path)
        ]
        assert not rejected, f"real configs rejected by discovery: {rejected}"

    def test_ambiguity_lists_the_candidates(self, tmp_path: Path) -> None:
        for name in ("one.yaml", "two.yaml", "three.yaml"):
            (tmp_path / name).write_text(_MINIMAL_YAML)
        with pytest.raises(ValueError, match="3 dao-ai configs") as excinfo:
            discover_config(tmp_path, None, locator="gh:o/r")
        message = str(excinfo.value)
        for name in ("one.yaml", "two.yaml", "three.yaml"):
            assert name in message

    def test_no_candidate_says_so(self, tmp_path: Path) -> None:
        (tmp_path / "notes.md").write_text("hi")
        with pytest.raises(ValueError, match="No dao-ai config found"):
            discover_config(tmp_path, None, locator="gh:o/r")

    def test_missing_path_names_the_fragment(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            discover_config(tmp_path, "nope/x.yaml", locator="gh:o/r#nope/x.yaml")


@pytest.mark.unit
class TestGitSourceLoad:
    def test_loads_a_config_and_anchors_relative_assets(
        self, git_repo: Path, cache: Path
    ) -> None:
        """The whole point: a real base_path, so colocated assets resolve.

        This is precisely what a URL source rejects.
        """
        resolved: ResolvedConfig = GitSource(_locator(git_repo)).load()
        assert resolved.base_path is not None
        assert (resolved.base_path / "data" / "seed.sql").exists()
        assert resolved.local_path is not None and resolved.local_path.is_file()
        assert len(resolved.revision or "") == 40

    def test_origin_is_the_locator_not_the_cache_path(
        self, git_repo: Path, cache: Path
    ) -> None:
        """A cache path is meaningless to whoever typed a locator."""
        locator: str = _locator(git_repo)
        assert GitSource(locator).load().origin == locator

    def test_checkout_is_keyed_by_commit(self, git_repo: Path, cache: Path) -> None:
        resolved = GitSource(_locator(git_repo)).load()
        assert resolved.revision is not None
        assert resolved.revision in str(resolved.base_path)

    def test_second_load_reuses_the_cache(self, git_repo: Path, cache: Path) -> None:
        first = GitSource(_locator(git_repo)).load()
        second = GitSource(_locator(git_repo)).load()
        assert first.base_path == second.base_path
        assert first.revision == second.revision

    def test_a_pinned_sha_needs_no_remote(self, git_repo: Path, cache: Path) -> None:
        """An immutable ref is a pure cache hit — verified by deleting the repo."""
        sha = GitSource(_locator(git_repo)).load().revision
        assert sha is not None

        import shutil

        shutil.rmtree(git_repo)
        pinned = GitSource(_locator(git_repo, ref=sha)).load()
        assert pinned.revision == sha

    def test_discovers_the_config_in_a_subdirectory(
        self, git_repo: Path, cache: Path
    ) -> None:
        """`#nested` holds agent.yaml plus a non-config examples.yaml."""
        resolved = GitSource(_locator(git_repo, path="nested")).load()
        assert resolved.local_path is not None
        assert resolved.local_path.name == "agent.yaml"

    def test_omitted_path_finds_the_conventional_root_config(
        self, git_repo: Path, cache: Path
    ) -> None:
        resolved = GitSource(_locator(git_repo, path=None)).load()
        assert resolved.local_path is not None
        assert resolved.local_path.name == "dao-ai.yaml"

    def test_unreachable_remote_is_a_clear_error(self, cache: Path) -> None:
        with pytest.raises(ValueError, match="Could not resolve"):
            GitSource("git+file:///nonexistent/repo@main#a.yaml").load()

    def test_no_token_is_written_into_the_checkout(
        self, git_repo: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`git remote add https://TOKEN@host/r` would persist it in .git/config."""
        monkeypatch.setenv("DAO_AI_GIT_TOKEN", "ghp_MUSTNOTPERSIST")
        GitSource(_locator(git_repo)).load()
        for config_file in cache.rglob(".git/config"):
            assert "ghp_MUSTNOTPERSIST" not in config_file.read_text()

    def test_cache_root_honors_the_env_override(self, cache: Path) -> None:
        assert cache_root() == cache

    def test_explicit_cache_dir_wins(self, tmp_path: Path, cache: Path) -> None:
        override: Path = tmp_path / "elsewhere"
        assert cache_root(override) == override


@pytest.mark.unit
class TestFromStarSurface:
    """Every ``from_*`` accepts a spec string or a typed source."""

    def test_from_git_accepts_both_spellings(self, git_repo: Path, cache: Path) -> None:
        from dao_ai.config import AppConfig

        locator: str = _locator(git_repo)
        from_string = AppConfig.from_git(locator, initialize=False)
        from_source = AppConfig.from_git(GitSource(locator), initialize=False)
        assert from_string.app is not None and from_source.app is not None
        assert from_string.app.name == from_source.app.name == "git-app"
        assert from_string._source_git_sha == from_source._source_git_sha

    def test_from_file_accepts_a_locator_and_any_source(
        self, git_repo: Path, cache: Path
    ) -> None:
        """`from_file` is the lenient general-purpose door."""
        from dao_ai.config import AppConfig

        locator: str = _locator(git_repo)
        assert AppConfig.from_file(locator, initialize=False).app is not None
        assert AppConfig.from_file(GitSource(locator), initialize=False).app is not None
        assert (
            AppConfig.from_file(
                FileSource(git_repo / "dao-ai.yaml"), initialize=False
            ).app
            is not None
        )

    def test_from_source_classifies_a_spec(self, git_repo: Path, cache: Path) -> None:
        from dao_ai.config import AppConfig

        assert AppConfig.from_source(_locator(git_repo), initialize=False).app
        assert AppConfig.from_source(
            str(git_repo / "dao-ai.yaml"), initialize=False
        ).app

    def test_from_git_rejects_a_local_path(self) -> None:
        from dao_ai.config import AppConfig

        with pytest.raises(ValueError, match="Not a valid GitSource spec"):
            AppConfig.from_git("config.yaml")

    def test_from_url_rejects_a_git_source(self) -> None:
        """A typed source of the wrong kind must not slip through."""
        from dao_ai.config import AppConfig

        with pytest.raises(ValueError, match="Expected UrlSource"):
            AppConfig.from_url(GitSource("gh:o/r"))

    def test_from_git_rejects_a_url_source(self) -> None:
        from dao_ai.config import AppConfig

        with pytest.raises(ValueError, match="Expected GitSource"):
            AppConfig.from_git(UrlSource("https://x/c.yaml"))

    def test_from_url_still_rejects_a_local_path(self) -> None:
        """The long-standing message is preserved."""
        from dao_ai.config import AppConfig

        with pytest.raises(ValueError, match="Not an http"):
            AppConfig.from_url("config.yaml")

    def test_git_config_records_the_revision(self, git_repo: Path, cache: Path) -> None:
        """The bundle checksum folds this in, so a ref bump re-stages."""
        from dao_ai.config import AppConfig

        config = AppConfig.from_git(_locator(git_repo), initialize=False)
        assert len(config._source_git_sha or "") == 40

    def test_local_config_has_no_revision(self, git_repo: Path) -> None:
        from dao_ai.config import AppConfig

        config = AppConfig.from_file(str(git_repo / "dao-ai.yaml"), initialize=False)
        assert config._source_git_sha is None

    def test_a_cache_path_still_carries_its_revision(
        self, git_repo: Path, cache: Path
    ) -> None:
        """Loading the materialized path directly must not lose the SHA.

        The CLI resolves a locator once in ``parse_args`` and passes the plain
        local path downstream, so this — not the locator — is what every
        ``dao-ai <noun>`` command actually loads. Without the revision the bundle
        checksum stops being ref-aware, and a ref bump whose only change is a
        ``ddl``/``data`` asset silently skips the rebuild.
        """
        from dao_ai.config import AppConfig

        resolved = GitSource(_locator(git_repo)).load()
        assert resolved.local_path is not None

        via_path = AppConfig.from_file(str(resolved.local_path), initialize=False)
        assert via_path._source_git_sha == resolved.revision

    def test_checksum_tracks_the_revision(self, git_repo: Path, cache: Path) -> None:
        """A ref bump changing only a colocated asset must move the checksum."""
        from dao_ai.cli import _config_checksum
        from dao_ai.config import AppConfig

        first = AppConfig.from_git(_locator(git_repo), initialize=False)
        before: str = _config_checksum(first, development=False)

        # Touch only an asset referenced by relative path: the config text stays
        # byte-identical, and ddl/data bytes are NOT in _custom_input_digests.
        (git_repo / "data" / "seed.sql").write_text("SELECT 2;\n")
        _git("add", "-A", cwd=git_repo)
        _git("commit", "--quiet", "--message", "seed v2", cwd=git_repo)

        second = AppConfig.from_git(_locator(git_repo), initialize=False)
        assert second._source_git_sha != first._source_git_sha
        assert _config_checksum(second, development=False) != before


@pytest.mark.unit
class TestSourceProtocol:
    def test_every_source_is_a_config_source(self) -> None:
        assert issubclass(GitSource, ConfigSource)
        assert issubclass(UrlSource, ConfigSource)
        assert issubclass(FileSource, ConfigSource)

    def test_config_source_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            ConfigSource()  # type: ignore[abstract]
