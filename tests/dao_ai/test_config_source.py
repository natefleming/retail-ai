"""Tests for loading configs from an http(s) URL.

``AppConfig.from_file`` was strictly a filesystem read — and ``Path()`` even
mangles a URL (``Path("https://x/y").as_posix()`` drops a slash), so a URL
failed at the read with a confusing error. These cover the URL branch plus the
GitHub blob→raw rewrite, and pin that local paths behave exactly as before.
"""

from __future__ import annotations

import pytest

from dao_ai.config_source import (
    fetch_config_text,
    is_remote_config,
    normalize_config_url,
)

_BLOB = (
    "https://github.com/natefleming/dao-ai-workshop/blob/main/"
    "L100-foundations/lab-01-first-agent/greeter.yaml"
)
_RAW = (
    "https://raw.githubusercontent.com/natefleming/dao-ai-workshop/"
    "refs/heads/main/L100-foundations/lab-01-first-agent/greeter.yaml"
)


@pytest.mark.unit
class TestIsRemoteConfig:
    def test_detects_http_and_https(self) -> None:
        assert is_remote_config("http://example.com/c.yaml")
        assert is_remote_config("https://example.com/c.yaml")

    def test_rejects_local_paths(self) -> None:
        assert not is_remote_config("config.yaml")
        assert not is_remote_config("/abs/path/config.yaml")
        assert not is_remote_config("./rel/config.yaml")

    def test_rejects_other_schemes(self) -> None:
        """Only http(s) is handled; anything else is better reported as a path."""
        assert not is_remote_config("file:///tmp/c.yaml")
        assert not is_remote_config("s3://bucket/c.yaml")


@pytest.mark.unit
class TestNormalizeConfigUrl:
    def test_rewrites_github_blob_to_raw(self) -> None:
        """A blob URL serves HTML, so fetching it would yield markup, not YAML."""
        assert normalize_config_url(_BLOB) == (
            "https://raw.githubusercontent.com/natefleming/dao-ai-workshop/main/"
            "L100-foundations/lab-01-first-agent/greeter.yaml"
        )

    def test_rewrites_github_raw_path_form(self) -> None:
        url = "https://github.com/o/r/raw/main/c.yaml"
        assert normalize_config_url(url) == (
            "https://raw.githubusercontent.com/o/r/main/c.yaml"
        )

    def test_leaves_an_already_raw_url_unchanged(self) -> None:
        assert normalize_config_url(_RAW) == _RAW

    def test_is_idempotent(self) -> None:
        once = normalize_config_url(_BLOB)
        assert normalize_config_url(once) == once

    def test_leaves_other_hosts_unchanged(self) -> None:
        url = "https://gitlab.example.com/o/r/blob/main/c.yaml"
        assert normalize_config_url(url) == url

    def test_leaves_short_github_paths_unchanged(self) -> None:
        """Not a file URL (no blob/raw segment) — nothing to rewrite."""
        url = "https://github.com/natefleming/dao-ai-workshop"
        assert normalize_config_url(url) == url


@pytest.mark.unit
class TestFetchConfigText:
    def test_returns_body_on_success(self, monkeypatch) -> None:
        import httpx

        monkeypatch.setattr(
            httpx, "get", lambda *a, **k: httpx.Response(200, text="app: {}")
        )
        assert fetch_config_text("https://example.com/c.yaml") == "app: {}"

    def test_404_names_the_private_repo_possibility(self, monkeypatch) -> None:
        """A private repo answers 404, not 401 — the error must not guess."""
        import httpx

        monkeypatch.setattr(
            httpx, "get", lambda *a, **k: httpx.Response(404, text="404: Not Found")
        )
        with pytest.raises(ValueError, match="private repository"):
            fetch_config_text("https://example.com/nope.yaml")

    def test_403_is_also_treated_as_unreachable(self, monkeypatch) -> None:
        import httpx

        monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(403))
        with pytest.raises(ValueError, match="private repository"):
            fetch_config_text("https://example.com/c.yaml")

    def test_other_errors_include_the_status(self, monkeypatch) -> None:
        import httpx

        monkeypatch.setattr(
            httpx, "get", lambda *a, **k: httpx.Response(500, text="boom")
        )
        with pytest.raises(ValueError, match="HTTP 500"):
            fetch_config_text("https://example.com/c.yaml")

    def test_transport_failure_names_the_url(self, monkeypatch) -> None:
        import httpx

        def _raise(*a, **k):
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr(httpx, "get", _raise)
        with pytest.raises(ValueError, match="example.com"):
            fetch_config_text("https://example.com/c.yaml")


@pytest.mark.unit
class TestFromFileWithUrl:
    _YAML = """
app:
  name: remote-app
  agents:
  - name: a
    description: d
    model:
      name: databricks-gpt-5-4-mini
"""

    def _patch_fetch(self, monkeypatch, text: str) -> None:
        monkeypatch.setattr(
            "dao_ai.config_source.fetch_config_text", lambda url, **k: text
        )

    def test_loads_a_url_without_touching_the_filesystem(self, monkeypatch) -> None:
        from dao_ai.config import AppConfig

        self._patch_fetch(monkeypatch, self._YAML)
        config = AppConfig.from_file(_RAW, initialize=False)
        assert config.app is not None and config.app.name == "remote-app"

    def test_records_the_url_as_the_source(self, monkeypatch) -> None:
        from dao_ai.config import AppConfig

        self._patch_fetch(monkeypatch, self._YAML)
        config = AppConfig.from_file(_RAW, initialize=False)
        assert config._source_config_path == _RAW

    def test_from_url_rejects_a_local_path(self) -> None:
        from dao_ai.config import AppConfig

        with pytest.raises(ValueError, match="Not an http"):
            AppConfig.from_url("config.yaml")

    def test_rejects_relative_dataset_paths(self, monkeypatch) -> None:
        """No local directory to resolve `ddl: functions/x.sql` against."""
        from dao_ai.config import AppConfig

        self._patch_fetch(
            monkeypatch,
            self._YAML
            + """
datasets:
- table:
    schema:
      catalog_name: cat
      schema_name: sch
    name: t
  ddl: functions/find_x.sql
""",
        )
        with pytest.raises(ValueError, match="relative paths"):
            AppConfig.from_file(_RAW, initialize=False)

    def test_allows_volume_paths(self, monkeypatch) -> None:
        """An absolute/Volume path is resolvable without a local directory."""
        from dao_ai.config import AppConfig

        self._patch_fetch(
            monkeypatch,
            self._YAML
            + """
datasets:
- table:
    schema:
      catalog_name: cat
      schema_name: sch
    name: t
  ddl: /Volumes/cat/sch/vol/find_x.sql
""",
        )
        config = AppConfig.from_file(_RAW, initialize=False)
        assert config.datasets

    def test_local_path_behaviour_is_unchanged(self, tmp_path) -> None:
        """The filesystem branch still stamps _base_path for relative assets."""
        from dao_ai.config import AppConfig

        cfg = tmp_path / "c.yaml"
        cfg.write_text(self._YAML)
        config = AppConfig.from_file(str(cfg), initialize=False)
        assert config.app is not None and config.app.name == "remote-app"
        assert config._source_config_path == cfg.as_posix()
