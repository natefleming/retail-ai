"""Tests for load-time ${param.NAME} / ${var.NAME} substitution in DAO AI configs."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from dao_ai.cli import _parse_var_args
from dao_ai.config import AppConfig
from dao_ai.config_vars import (
    PARAM_PATTERN,
    WORKSPACE_PATTERN,
    ConfigVariableError,
    ParameterDeclarationModel,
    WorkspaceVariableError,
    find_param_references,
    find_workspace_references,
    resolve_parameters,
    substitute_params,
    substitute_workspace_refs,
)


@pytest.fixture(autouse=True)
def _stub_bundle_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``generate_bundle_lock`` so the write_bundle tests don't invoke a
    real ``uv lock`` (needs network + a published dao-ai version). Writes a
    placeholder ``uv.lock``; the lock is covered by test_bundle_dependency_lock.
    Harmless no-op for tests that never generate a bundle.
    """
    from pathlib import Path as _Path

    def _fake_lock(bundle_dir: _Path) -> None:
        (bundle_dir / "uv.lock").write_text("# stub lock for tests\n")

    monkeypatch.setattr("dao_ai.apps.bundle.generate_bundle_lock", _fake_lock)


@pytest.mark.unit
def test_find_param_references_extracts_distinct_names() -> None:
    text = "a: ${param.foo}\nb: ${param.bar:-default}\nc: ${param.foo}"
    assert find_param_references(text) == {"foo", "bar"}


@pytest.mark.unit
def test_find_param_references_ignores_full_line_comments() -> None:
    text = dedent(
        """
        # example: schema: ${var.example_schema}
        catalog: ${var.catalog}
        """
    )
    assert find_param_references(text) == {"catalog"}


@pytest.mark.unit
def test_find_param_references_ignores_inline_comments() -> None:
    text = "catalog: ${var.catalog}  # or use ${var.override}\n"
    assert find_param_references(text) == {"catalog"}


@pytest.mark.unit
def test_find_param_references_keeps_hash_inside_double_quotes() -> None:
    # A `#` inside a double-quoted string is not a comment starter.
    text = 'label: "hash-mark #${var.tag} still counts"\n'
    assert find_param_references(text) == {"tag"}


@pytest.mark.unit
def test_find_param_references_keeps_hash_inside_single_quotes() -> None:
    text = "label: 'hash # ${var.tag} still counts'\n"
    assert find_param_references(text) == {"tag"}


@pytest.mark.unit
def test_substitute_params_ignores_undeclared_references_in_comments() -> None:
    text = dedent(
        """
        # example: schema: ${var.example_schema}
        catalog: ${var.catalog}
        """
    )
    decls = {"catalog": ParameterDeclarationModel(default="main")}
    rendered = substitute_params(text, declarations=decls)
    # Comment ${var.example_schema} is left untouched; catalog substituted.
    assert "${var.example_schema}" in rendered
    assert "catalog: main" in rendered


@pytest.mark.unit
def test_substitute_params_ignores_missing_required_in_comments() -> None:
    # A declared-but-unresolved ref inside a comment should not raise.
    text = "# hint: use ${var.optional_hint}\ncatalog: ${var.catalog}\n"
    decls = {
        "optional_hint": ParameterDeclarationModel(),  # required, no default
        "catalog": ParameterDeclarationModel(default="main"),
    }
    rendered = substitute_params(text, declarations=decls)
    assert "${var.optional_hint}" in rendered
    assert "catalog: main" in rendered


@pytest.mark.unit
def test_substitute_uses_cli_vars_first() -> None:
    decls = {"foo": ParameterDeclarationModel(default="from-default")}
    rendered = substitute_params(
        "x: ${param.foo}",
        declarations=decls,
        cli_vars={"foo": "from-cli"},
        env={"FOO": "from-env"},
    )
    assert rendered == "x: from-cli"


@pytest.mark.unit
def test_substitute_falls_back_to_env_then_default() -> None:
    decls = {"foo": ParameterDeclarationModel(default="from-default")}
    assert (
        substitute_params(
            "x: ${param.foo}", declarations=decls, env={"FOO": "from-env"}
        )
        == "x: from-env"
    )
    assert (
        substitute_params("x: ${param.foo}", declarations=decls, env={})
        == "x: from-default"
    )


@pytest.mark.unit
def test_substitute_uses_inline_default_when_nothing_else_set() -> None:
    decls = {"foo": ParameterDeclarationModel()}
    rendered = substitute_params("x: ${param.foo:-inline}", declarations=decls, env={})
    assert rendered == "x: inline"


@pytest.mark.unit
def test_env_key_normalization_lowercases_dotted_names() -> None:
    decls = {"app.catalog-name": ParameterDeclarationModel()}
    rendered = substitute_params(
        "x: ${param.app.catalog-name}",
        declarations=decls,
        env={"APP_CATALOG_NAME": "main"},
    )
    assert rendered == "x: main"


@pytest.mark.unit
def test_missing_required_raises_with_all_names() -> None:
    decls = {
        "a": ParameterDeclarationModel(),
        "b": ParameterDeclarationModel(),
    }
    with pytest.raises(ConfigVariableError) as exc:
        substitute_params(
            "x: ${param.a}\ny: ${param.b}",
            declarations=decls,
            env={},
            source="test.yaml",
        )
    assert exc.value.missing_required == ["a", "b"]
    assert exc.value.undeclared == []
    assert "a" in str(exc.value) and "b" in str(exc.value)


@pytest.mark.unit
def test_undeclared_reference_raises_with_all_names() -> None:
    with pytest.raises(ConfigVariableError) as exc:
        substitute_params(
            "x: ${param.unknown}\ny: ${param.also_unknown}",
            declarations={"known": ParameterDeclarationModel(default="x")},
            source="test.yaml",
        )
    assert exc.value.undeclared == ["also_unknown", "unknown"]
    assert exc.value.missing_required == []


@pytest.mark.unit
def test_missing_and_undeclared_reported_together() -> None:
    decls = {"a": ParameterDeclarationModel()}
    with pytest.raises(ConfigVariableError) as exc:
        substitute_params(
            "x: ${param.a}\ny: ${param.b}",
            declarations=decls,
            env={},
            source="t.yaml",
        )
    assert exc.value.missing_required == ["a"]
    assert exc.value.undeclared == ["b"]


@pytest.mark.unit
def test_substitute_does_not_touch_workspace_or_secret_tokens() -> None:
    text = "host: ${workspace.host}\nsecret: {{secrets/foo/bar}}\nvalue: ${param.x}"
    rendered = substitute_params(
        text,
        declarations={"x": ParameterDeclarationModel(default="ok")},
    )
    assert "${workspace.host}" in rendered
    assert "{{secrets/foo/bar}}" in rendered
    assert "value: ok" in rendered


@pytest.mark.unit
def test_no_declarations_means_no_undeclared_check() -> None:
    rendered = substitute_params(
        "x: ${param.foo}",
        declarations=None,
        cli_vars={"foo": "bar"},
    )
    assert rendered == "x: bar"


@pytest.mark.unit
def test_pattern_does_not_match_unprefixed_or_spaced_tokens() -> None:
    """Tokens like ${foo} or ${ var.x } (with whitespace) must be left alone."""
    text = "a: ${foo}\nb: ${ var.x }\nc: ${workspace.host}"
    assert PARAM_PATTERN.findall(text) == []


@pytest.mark.unit
def test_defer_leaves_ref_literal_and_not_missing() -> None:
    """A deferred name survives as its ${var.X} literal and is never 'missing',
    even declared with no default and no value (workflow provisioning path)."""
    decls = {"genie_space_id": ParameterDeclarationModel(default="")}
    rendered = substitute_params(
        'space_id: "${var.genie_space_id}"',
        declarations=decls,
        defer={"genie_space_id"},
    )
    assert rendered == 'space_id: "${var.genie_space_id}"'


@pytest.mark.unit
def test_defer_no_default_no_value_does_not_raise() -> None:
    """Deferred + declared with NO default + no value must NOT raise missing."""
    decls = {"genie_space_id": ParameterDeclarationModel()}
    rendered = substitute_params(
        "id: ${var.genie_space_id}",
        declarations=decls,
        defer={"genie_space_id"},
    )
    assert rendered == "id: ${var.genie_space_id}"


@pytest.mark.unit
def test_defer_is_inert_when_none() -> None:
    """defer=None must not change behavior for existing callers."""
    decls = {"catalog": ParameterDeclarationModel(default="main")}
    assert substitute_params("c: ${var.catalog}", declarations=decls) == "c: main"
    assert (
        substitute_params("c: ${var.catalog}", declarations=decls, defer=None)
        == "c: main"
    )


@pytest.mark.unit
def test_defer_only_affects_named_refs() -> None:
    """Non-deferred refs still bake; only the deferred name is left literal."""
    decls = {
        "catalog": ParameterDeclarationModel(default="main"),
        "genie_space_id": ParameterDeclarationModel(default=""),
    }
    rendered = substitute_params(
        "cat: ${var.catalog}\nid: ${var.genie_space_id}",
        declarations=decls,
        defer={"genie_space_id"},
    )
    assert "cat: main" in rendered
    assert "id: ${var.genie_space_id}" in rendered


@pytest.mark.unit
def test_provisioned_param_resolves_empty_when_unset() -> None:
    """provisioned:true, no value, no default -> resolves to '' (not missing)."""
    decls = {"gsid": ParameterDeclarationModel(provided=True)}
    assert substitute_params("s: ${var.gsid}", declarations=decls) == "s: "


@pytest.mark.unit
def test_provisioned_param_uses_default_when_present() -> None:
    """A default is an optional fallback for a provisioned param."""
    decls = {"gsid": ParameterDeclarationModel(provided=True, default="fb")}
    assert substitute_params("s: ${var.gsid}", declarations=decls) == "s: fb"


@pytest.mark.unit
def test_provisioned_param_uses_supplied_value() -> None:
    decls = {"gsid": ParameterDeclarationModel(provided=True)}
    assert (
        substitute_params("s: ${var.gsid}", declarations=decls, cli_vars={"gsid": "R"})
        == "s: R"
    )


@pytest.mark.unit
def test_non_provisioned_no_default_still_raises() -> None:
    """A plain required param (not provisioned) still raises when unset."""
    decls = {"gsid": ParameterDeclarationModel()}
    with pytest.raises(ConfigVariableError):
        substitute_params("s: ${var.gsid}", declarations=decls)


@pytest.mark.unit
def test_retain_only_parameters_keeps_whitelist() -> None:
    from dao_ai.apps.bundle import _retain_only_parameters

    doc = (
        'parameters:\n  a:\n    default: x\n  genie_space_id:\n    default: ""\n'
        "resources:\n  r: 1\n"
    )
    out = _retain_only_parameters(doc, keep={"genie_space_id"})
    assert "genie_space_id:" in out
    # 'a' is dropped from the parameters block (but the resources key stays).
    assert "  a:\n" not in out
    assert "resources:" in out


@pytest.mark.unit
def test_retain_only_parameters_empty_keep_strips_block() -> None:
    from dao_ai.apps.bundle import _retain_only_parameters, _strip_parameters_block

    doc = "parameters:\n  a:\n    default: x\nresources:\n  r: 1\n"
    # Empty keep behaves like a full strip.
    assert _retain_only_parameters(doc, keep=set()) == _strip_parameters_block(doc)


@pytest.mark.unit
def test_retain_only_parameters_drops_block_when_none_kept() -> None:
    from dao_ai.apps.bundle import _retain_only_parameters

    doc = "parameters:\n  a:\n    default: x\nresources:\n  r: 1\n"
    out = _retain_only_parameters(doc, keep={"not_present"})
    assert "parameters:" not in out
    assert "resources:" in out


@pytest.mark.unit
def test_var_prefix_alias_is_equivalent_to_param() -> None:
    """${var.NAME} and ${param.NAME} resolve identically."""
    decls = {"foo": ParameterDeclarationModel(default="d")}
    assert (
        substitute_params("x: ${var.foo}", declarations=decls)
        == substitute_params("x: ${param.foo}", declarations=decls)
        == "x: d"
    )


@pytest.mark.unit
def test_var_prefix_with_inline_default() -> None:
    rendered = substitute_params(
        "x: ${var.foo:-fallback}",
        declarations={"foo": ParameterDeclarationModel()},
        env={},
    )
    assert rendered == "x: fallback"


@pytest.mark.unit
def test_var_and_param_prefixes_in_same_document() -> None:
    """Both prefixes can appear in the same YAML and reference the same name."""
    decls = {
        "catalog": ParameterDeclarationModel(default="main"),
        "schema": ParameterDeclarationModel(default="dao_ai"),
    }
    rendered = substitute_params(
        "a: ${var.catalog}\nb: ${param.catalog}\nc: ${var.schema}",
        declarations=decls,
    )
    assert rendered == "a: main\nb: main\nc: dao_ai"


@pytest.mark.unit
def test_find_param_references_handles_both_prefixes() -> None:
    text = "a: ${var.foo}\nb: ${param.bar}\nc: ${var.foo}"
    assert find_param_references(text) == {"foo", "bar"}


@pytest.mark.unit
def test_var_prefix_undeclared_reference_raises() -> None:
    """An undeclared ${var.NAME} is reported just like ${param.NAME}."""
    with pytest.raises(ConfigVariableError) as exc:
        substitute_params(
            "x: ${var.unknown}",
            declarations={"known": ParameterDeclarationModel(default="x")},
            source="t.yaml",
        )
    assert exc.value.undeclared == ["unknown"]


@pytest.mark.unit
def test_resolve_parameters_reports_sources_correctly() -> None:
    decls = {
        "from_cli": ParameterDeclarationModel(default="d"),
        "from_env": ParameterDeclarationModel(default="d"),
        "from_default": ParameterDeclarationModel(default="d"),
        "missing": ParameterDeclarationModel(),
    }
    resolved = resolve_parameters(
        decls,
        cli_vars={"from_cli": "cli-value"},
        env={"FROM_ENV": "env-value"},
    )
    by_name = {p.name: p for p in resolved}
    assert by_name["from_cli"].source == "--param"
    assert by_name["from_cli"].value == "cli-value"
    assert by_name["from_env"].source == "env"
    assert by_name["from_env"].value == "env-value"
    assert by_name["from_default"].source == "default"
    assert by_name["from_default"].value == "d"
    assert by_name["missing"].source == "MISSING"
    assert by_name["missing"].value is None


@pytest.mark.unit
def test_resolve_parameters_reports_provided() -> None:
    """A `provided` param (no value) reports source='provided', not MISSING, and
    is not load-time required; an operator value still wins."""
    decls = {
        "runtime": ParameterDeclarationModel(provided=True),
        "runtime_supplied": ParameterDeclarationModel(provided=True),
        "runtime_default": ParameterDeclarationModel(provided=True, default="fb"),
    }
    by_name = {
        p.name: p
        for p in resolve_parameters(
            decls, cli_vars={"runtime_supplied": "given"}, env={}
        )
    }
    # Unsupplied provided param: not missing, not required, flagged provided.
    assert by_name["runtime"].source == "provided"
    assert by_name["runtime"].value is None
    assert by_name["runtime"].required is False
    assert by_name["runtime"].provided is True
    # Operator value wins over the provided placeholder.
    assert by_name["runtime_supplied"].source == "--param"
    assert by_name["runtime_supplied"].value == "given"
    # A default fallback wins over the provided placeholder too.
    assert by_name["runtime_default"].source == "default"
    assert by_name["runtime_default"].value == "fb"
    assert by_name["runtime_default"].provided is True


class _FakeUser:
    def __init__(self, user_name: str) -> None:
        self.user_name = user_name


class _FakeWorkspaceClient:
    def __init__(self, *, host: str, user_name: str) -> None:
        self.config = type("Cfg", (), {"host": host})()
        self.current_user = type("CU", (), {"me": lambda _self: _FakeUser(user_name)})()


@pytest.mark.unit
def test_workspace_pattern_only_matches_known_prefix() -> None:
    text = "a: ${workspace.host}\nb: ${param.x}\nc: ${var.y}\nd: ${other.z}"
    assert find_workspace_references(text) == {"host"}
    assert WORKSPACE_PATTERN.findall("e: ${workspace}") == []


@pytest.mark.unit
def test_substitute_workspace_refs_no_refs_skips_factory() -> None:
    def _boom() -> None:
        raise AssertionError("factory should not be called when no refs present")

    assert (
        substitute_workspace_refs("a: 1\nb: 2", workspace_client_factory=_boom)
        == "a: 1\nb: 2"
    )


@pytest.mark.unit
def test_substitute_workspace_refs_resolves_all_four_dabs_paths() -> None:
    client = _FakeWorkspaceClient(
        host="https://example.cloud.databricks.com/",
        user_name="nate.fleming@databricks.com",
    )
    text = (
        "host: ${workspace.host}\n"
        "email: ${workspace.current_user.userName}\n"
        "short: ${workspace.current_user.short_name}\n"
        "domain: ${workspace.current_user.domain_friendly_name}\n"
    )
    rendered = substitute_workspace_refs(text, workspace_client_factory=lambda: client)
    assert "host: https://example.cloud.databricks.com" in rendered
    assert "host: https://example.cloud.databricks.com/" not in rendered
    assert "email: nate.fleming@databricks.com" in rendered
    assert "short: nate.fleming" in rendered
    assert "domain: databricks.com" in rendered


@pytest.mark.unit
def test_substitute_workspace_refs_caches_user_calls() -> None:
    calls: dict[str, int] = {"me": 0}

    class CountingCurrentUser:
        def me(self) -> _FakeUser:
            calls["me"] += 1
            return _FakeUser("a.b@c.com")

    class CountingClient:
        config = type("Cfg", (), {"host": "https://x"})()
        current_user = CountingCurrentUser()

    text = (
        "${workspace.current_user.userName} "
        "${workspace.current_user.short_name} "
        "${workspace.current_user.domain_friendly_name}"
    )
    substitute_workspace_refs(text, workspace_client_factory=lambda: CountingClient())
    assert calls["me"] == 1  # cached across multiple paths derived from email


@pytest.mark.unit
def test_substitute_workspace_refs_rejects_unsupported_path() -> None:
    with pytest.raises(WorkspaceVariableError) as exc:
        substitute_workspace_refs(
            "x: ${workspace.current_user.email}",
            workspace_client_factory=lambda: None,
            source="t.yaml",
        )
    assert "current_user.email" in str(exc.value)


@pytest.mark.unit
def test_substitute_workspace_refs_wraps_client_failures() -> None:
    def _broken() -> None:
        raise RuntimeError("no auth")

    with pytest.raises(WorkspaceVariableError) as exc:
        substitute_workspace_refs(
            "x: ${workspace.host}", workspace_client_factory=_broken, source="t.yaml"
        )
    assert "no auth" in str(exc.value)


@pytest.mark.unit
def test_workspace_refs_resolve_inside_parameter_defaults(tmp_path: Path) -> None:
    """Workspace refs inside a `parameters:` default must resolve before param substitution."""
    client = _FakeWorkspaceClient(
        host="https://x", user_name="nate.fleming@databricks.com"
    )
    text = (
        "parameters:\n"
        "  genie_parent_path:\n"
        '    default: "/Users/${workspace.current_user.userName}/genie"\n'
        "value: ${var.genie_parent_path}\n"
    )
    workspace_resolved = substitute_workspace_refs(
        text, workspace_client_factory=lambda: client
    )
    parsed = yaml.safe_load(workspace_resolved) or {}
    declarations = {
        name: ParameterDeclarationModel(**(spec or {}))
        for name, spec in (parsed.get("parameters") or {}).items()
    }
    rendered = substitute_params(workspace_resolved, declarations=declarations)
    assert "value: /Users/nate.fleming@databricks.com/genie" in rendered


@pytest.mark.unit
def test_parse_var_args_handles_equals_in_value() -> None:
    parsed = _parse_var_args(["a=1", "b=hello world", "c=key=value"])
    assert parsed == {"a": "1", "b": "hello world", "c": "key=value"}


@pytest.mark.unit
def test_parse_var_args_rejects_missing_equals() -> None:
    with pytest.raises(SystemExit):
        _parse_var_args(["bad"])


@pytest.mark.unit
def test_parse_var_args_handles_none() -> None:
    assert _parse_var_args(None) == {}


@pytest.mark.unit
def test_app_config_from_file_substitutes_params(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        dedent(
            """
            parameters:
              catalog:
                description: UC catalog
                default: main
              module_id:
                description: Workshop module ID

            schemas:
              workshop_schema:
                catalog_name: ${param.catalog}
                schema_name: dao_ai_${param.module_id}
            """
        ).lstrip()
    )

    config = AppConfig.from_file(
        yaml_path, params={"module_id": "09"}, initialize=False
    )

    assert config.schemas["workshop_schema"].catalog_name == "main"
    assert config.schemas["workshop_schema"].schema_name == "dao_ai_09"
    assert config.source_config_path == yaml_path.as_posix()
    assert config.rendered_yaml is not None
    assert "${param." not in config.rendered_yaml
    assert config.substitution_vars == {"module_id": "09"}


# ---------------------------------------------------------------------------
# AppConfig.from_file(task_values=...) — pull resolved parameter values from
# an upstream Databricks job task via the dbutils.jobs.taskValues protocol.
# ---------------------------------------------------------------------------


class _FakeTaskValues:
    """In-memory stand-in for ``dbutils.jobs.taskValues`` shaped as
    :class:`dao_ai.config.TaskValuesLike`. Records every probe so tests
    can assert which parameter names were looked up."""

    def __init__(self, values: dict[tuple[str, str], object] | None = None) -> None:
        self._values: dict[tuple[str, str], object] = values or {}
        self.calls: list[tuple[str, str]] = []

    def get(
        self,
        *,
        taskKey: str,
        key: str,
        default: object = None,
        debugValue: object = None,
    ) -> object:
        self.calls.append((taskKey, key))
        return self._values.get((taskKey, key), default)


class _RaisingTaskValues:
    """TaskValuesLike whose ``get`` always raises (mimics dbutils when the
    upstream task key has not run yet)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get(
        self,
        *,
        taskKey: str,
        key: str,
        default: object = None,
        debugValue: object = None,
    ) -> object:
        self.calls.append((taskKey, key))
        raise ValueError(f"No taskValue for taskKey={taskKey!r}, key={key!r}")


def _write_genie_space_yaml(yaml_path: Path) -> None:
    """Helper: writes a YAML with two declared parameters and references."""
    yaml_path.write_text(
        dedent(
            """
            parameters:
              genie_space_id:
                description: Genie space id (provisioned upstream)
              warehouse_id:
                description: SQL warehouse id
                default: wh-default

            schemas:
              s:
                catalog_name: main
                schema_name: "space-${param.genie_space_id}-wh-${param.warehouse_id}"
            """
        ).lstrip()
    )


@pytest.mark.unit
def test_from_file_pulls_params_from_task_values(tmp_path: Path) -> None:
    """taskValues with a matching declared parameter is folded into substitution."""
    yaml_path = tmp_path / "cfg.yaml"
    _write_genie_space_yaml(yaml_path)
    tv = _FakeTaskValues({("provision-genie", "genie_space_id"): "abc-123"})

    config = AppConfig.from_file(
        yaml_path,
        task_values=tv,
        task_key="provision-genie",
        initialize=False,
    )

    assert config.schemas["s"].schema_name == "space-abc-123-wh-wh-default"
    assert config.substitution_vars == {"genie_space_id": "abc-123"}


@pytest.mark.unit
def test_from_file_task_values_requires_task_key(tmp_path: Path) -> None:
    """task_values without task_key is a programming error — raise immediately."""
    yaml_path = tmp_path / "cfg.yaml"
    _write_genie_space_yaml(yaml_path)
    tv = _FakeTaskValues({("provision-genie", "genie_space_id"): "abc-123"})

    with pytest.raises(ValueError, match="task_values requires task_key"):
        AppConfig.from_file(yaml_path, task_values=tv, initialize=False)


@pytest.mark.unit
def test_from_file_params_beat_task_values(tmp_path: Path) -> None:
    """CLI-style ``params`` overrides matching ``task_values`` per key."""
    yaml_path = tmp_path / "cfg.yaml"
    _write_genie_space_yaml(yaml_path)
    tv = _FakeTaskValues({("provision-genie", "genie_space_id"): "from-tv"})

    config = AppConfig.from_file(
        yaml_path,
        params={"genie_space_id": "from-cli"},
        task_values=tv,
        task_key="provision-genie",
        initialize=False,
    )

    assert config.schemas["s"].schema_name == "space-from-cli-wh-wh-default"
    assert config.substitution_vars == {"genie_space_id": "from-cli"}


@pytest.mark.unit
def test_from_file_task_values_empty_strings_are_skipped(tmp_path: Path) -> None:
    """An empty taskValue result must not shadow a declared default."""
    yaml_path = tmp_path / "cfg.yaml"
    _write_genie_space_yaml(yaml_path)
    tv = _FakeTaskValues(
        {
            ("provision-genie", "genie_space_id"): "abc-123",
            ("provision-genie", "warehouse_id"): "",
        }
    )

    config = AppConfig.from_file(
        yaml_path,
        task_values=tv,
        task_key="provision-genie",
        initialize=False,
    )

    assert config.schemas["s"].schema_name == "space-abc-123-wh-wh-default"


@pytest.mark.unit
def test_from_file_task_values_only_probes_declared_parameters(tmp_path: Path) -> None:
    """task_values.get must only be called for declared parameter names."""
    yaml_path = tmp_path / "cfg.yaml"
    _write_genie_space_yaml(yaml_path)
    tv = _FakeTaskValues({("provision-genie", "genie_space_id"): "abc-123"})

    AppConfig.from_file(
        yaml_path,
        task_values=tv,
        task_key="provision-genie",
        initialize=False,
    )

    probed = sorted(key for _, key in tv.calls)
    assert probed == ["genie_space_id", "warehouse_id"]


@pytest.mark.unit
def test_from_file_task_values_get_exception_is_tolerated(tmp_path: Path) -> None:
    """An exception from task_values.get must not break config loading."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        dedent(
            """
            parameters:
              with_default:
                default: fallback

            schemas:
              s:
                catalog_name: ${param.with_default}
                schema_name: x
            """
        ).lstrip()
    )
    tv = _RaisingTaskValues()

    config = AppConfig.from_file(
        yaml_path,
        task_values=tv,
        task_key="provision-genie",
        initialize=False,
    )

    assert config.schemas["s"].catalog_name == "fallback"
    # The exception was tolerated but the probe was still attempted.
    assert tv.calls == [("provision-genie", "with_default")]


@pytest.mark.unit
def test_app_config_from_file_raises_for_missing_required(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        dedent(
            """
            parameters:
              required_one:
                description: must be provided

            schemas:
              s:
                catalog_name: ${param.required_one}
                schema_name: x
            """
        ).lstrip()
    )

    with pytest.raises(ConfigVariableError) as exc:
        AppConfig.from_file(yaml_path, initialize=False)
    assert exc.value.missing_required == ["required_one"]


@pytest.mark.unit
def test_app_config_from_file_raises_for_undeclared_reference(
    tmp_path: Path,
) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        dedent(
            """
            parameters:
              declared:
                default: x

            schemas:
              s:
                catalog_name: ${param.declared}
                schema_name: ${param.not_declared}
            """
        ).lstrip()
    )

    with pytest.raises(ConfigVariableError) as exc:
        AppConfig.from_file(yaml_path, initialize=False)
    assert exc.value.undeclared == ["not_declared"]


@pytest.mark.unit
def test_app_config_from_file_without_parameters_block_is_unchanged(
    tmp_path: Path,
) -> None:
    """Configs that don't use ${param.*} or 'parameters:' must work as before."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        dedent(
            """
            schemas:
              s:
                catalog_name: main
                schema_name: dao_ai
            """
        ).lstrip()
    )

    config = AppConfig.from_file(yaml_path, initialize=False)
    assert config.schemas["s"].catalog_name == "main"
    assert config.schemas["s"].schema_name == "dao_ai"


@pytest.mark.unit
def test_env_var_resolution_via_real_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke test that the default env source (os.environ) is consulted."""
    monkeypatch.setenv("DAO_AI_TEST_CATALOG", "from-real-env")

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        dedent(
            """
            parameters:
              dao_ai_test_catalog:
                description: test only

            schemas:
              s:
                catalog_name: ${param.dao_ai_test_catalog}
                schema_name: x
            """
        ).lstrip()
    )

    config = AppConfig.from_file(yaml_path, initialize=False)
    assert config.schemas["s"].catalog_name == "from-real-env"


# ---------------------------------------------------------------------------
# Edge cases: anchors, multi-line scalars, repeated refs, coercion, recursion,
# duplicate declarations, empty defaults, error formatting, YAML safety.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_substitution_works_inside_yaml_anchor(tmp_path: Path) -> None:
    """Anchors are textual constructs - substitution happens before YAML parses."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        dedent(
            """
            parameters:
              catalog:
                default: main

            schemas:
              base: &base
                catalog_name: ${var.catalog}
                schema_name: dao_ai
              also_uses_anchor:
                <<: *base
            """
        ).lstrip()
    )
    config = AppConfig.from_file(yaml_path, initialize=False)
    assert config.schemas["base"].catalog_name == "main"
    assert config.schemas["also_uses_anchor"].catalog_name == "main"


@pytest.mark.unit
def test_substitution_works_inside_multiline_block_scalar(tmp_path: Path) -> None:
    """${var.NAME} in 'prompt: |' style blocks must be substituted."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        dedent(
            """
            parameters:
              brand:
                default: AcmeStore

            schemas:
              s:
                catalog_name: main
                schema_name: |
                  greeting from ${var.brand}
                  on multiple lines.
            """
        ).lstrip()
    )
    config = AppConfig.from_file(yaml_path, initialize=False)
    assert "greeting from AcmeStore" in config.schemas["s"].schema_name
    assert "on multiple lines." in config.schemas["s"].schema_name


@pytest.mark.unit
def test_multiple_references_on_same_line_all_substitute() -> None:
    decls = {
        "x": ParameterDeclarationModel(default="X"),
        "y": ParameterDeclarationModel(default="Y"),
    }
    rendered = substitute_params(
        "value: ${var.x}-${var.y}-${param.x}", declarations=decls
    )
    assert rendered == "value: X-Y-X"


@pytest.mark.unit
def test_repeated_reference_resolves_each_occurrence() -> None:
    decls = {"foo": ParameterDeclarationModel(default="bar")}
    rendered = substitute_params(
        "a: ${var.foo}\nb: ${var.foo}\nc: ${var.foo}", declarations=decls
    )
    assert rendered.count("bar") == 3
    assert "${var.foo}" not in rendered


@pytest.mark.unit
def test_empty_string_default_is_a_valid_value(tmp_path: Path) -> None:
    """default: '' (empty string) is distinct from no default and means optional."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        dedent(
            """
            parameters:
              prefix:
                default: ""

            schemas:
              s:
                catalog_name: main
                schema_name: dao_ai${var.prefix}
            """
        ).lstrip()
    )
    config = AppConfig.from_file(yaml_path, initialize=False)
    assert config.schemas["s"].schema_name == "dao_ai"


@pytest.mark.unit
def test_omitted_default_makes_parameter_required() -> None:
    """No 'default:' key at all (vs explicit empty string) means required."""
    decls = {"foo": ParameterDeclarationModel()}
    assert decls["foo"].default is None
    with pytest.raises(ConfigVariableError) as exc:
        substitute_params("x: ${var.foo}", declarations=decls, env={})
    assert exc.value.missing_required == ["foo"]


@pytest.mark.unit
def test_substitution_does_not_recurse() -> None:
    """A substituted value containing ${var.…} is preserved literally."""
    decls = {
        "outer": ParameterDeclarationModel(default="${var.inner}"),
        "inner": ParameterDeclarationModel(default="resolved"),
    }
    rendered = substitute_params("a: ${var.outer}", declarations=decls)
    # Should be the literal text of the default, NOT 'resolved'.
    assert rendered == "a: ${var.inner}"


@pytest.mark.unit
def test_numeric_string_value_is_coerced_by_pydantic(tmp_path: Path) -> None:
    """--var threshold=0.7 produces '0.7' which Pydantic coerces to float."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        dedent(
            """
            parameters:
              max_tokens:
                default: "1024"

            resources:
              llms:
                m:
                  name: databricks-test-llm
                  max_tokens: ${var.max_tokens}
                  temperature: 0.1
            """
        ).lstrip()
    )
    config = AppConfig.from_file(
        yaml_path, params={"max_tokens": "4096"}, initialize=False
    )
    assert config.resources.llms["m"].max_tokens == 4096
    assert isinstance(config.resources.llms["m"].max_tokens, int)


@pytest.mark.unit
def test_duplicate_parameter_name_uses_last_silently() -> None:
    """PyYAML silently uses the last occurrence on duplicate mapping keys.

    We document this rather than try to outsmart the YAML parser - if the
    user wants strict duplicate detection they can run a YAML linter.
    """
    raw = dedent(
        """
        parameters:
          foo:
            default: first
          foo:
            default: second
        """
    ).lstrip()
    parsed = yaml.safe_load(raw)
    assert parsed["parameters"]["foo"]["default"] == "second"


@pytest.mark.unit
def test_config_variable_error_str_includes_path_and_both_lists() -> None:
    """The user-facing error message format is part of the contract."""
    err = ConfigVariableError(
        path="/tmp/my_cfg.yaml",
        missing_required=["b", "a"],  # purposely unsorted
        undeclared=["d", "c"],
    )
    msg = str(err)
    assert "/tmp/my_cfg.yaml" in msg
    # Sorted alphabetically and joined with ", "
    assert "a, b" in msg
    assert "c, d" in msg
    assert err.missing_required == ["a", "b"]
    assert err.undeclared == ["c", "d"]
    # Bare instantiation with no errors is allowed but still has the path header.
    bare = ConfigVariableError(path="x.yaml")
    assert "x.yaml" in str(bare)


@pytest.mark.unit
def test_yaml_safety_quoted_reference_handles_special_chars(
    tmp_path: Path,
) -> None:
    """A quoted "${var.NAME}" reference safely accepts values with YAML colons."""
    yaml_path = tmp_path / "cfg.yaml"
    # Note the surrounding quotes - this is the documented best practice.
    yaml_path.write_text(
        dedent(
            """
            parameters:
              note:
                default: "default note"

            schemas:
              s:
                catalog_name: main
                schema_name: "${var.note}"
            """
        ).lstrip()
    )
    config = AppConfig.from_file(
        yaml_path, params={"note": "key: value with colon"}, initialize=False
    )
    assert config.schemas["s"].schema_name == "key: value with colon"


@pytest.mark.unit
def test_yaml_safety_unquoted_reference_with_special_chars_raises(
    tmp_path: Path,
) -> None:
    """An UNquoted reference with a YAML-special value produces a YAML parse error.

    This is the documented failure mode - users should quote risky references.
    """
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        dedent(
            """
            parameters:
              note:
                default: ok

            schemas:
              s:
                catalog_name: main
                schema_name: ${var.note}
            """
        ).lstrip()
    )
    with pytest.raises(yaml.YAMLError):
        AppConfig.from_file(yaml_path, params={"note": "key: bad"}, initialize=False)


# ---------------------------------------------------------------------------
# Bundle baking: write_bundle should write the rendered YAML with the
# parameters block dropped, so the deployed app does not need --var.
# ---------------------------------------------------------------------------


@pytest.fixture
def parameterised_config_path(tmp_path: Path) -> Path:
    """Minimal but bundle-shaped config that exercises ${var.…} in many places."""
    p = tmp_path / "cfg.yaml"
    p.write_text(
        dedent(
            """
            parameters:
              catalog:
                description: UC catalog
                default: main
              module_id:
                description: workshop module id

            schemas:
              s: &s
                catalog_name: ${var.catalog}
                schema_name: dao_ai_${var.module_id}

            resources:
              llms:
                default_llm: &default_llm
                  name: databricks-test-llm

            agents:
              hello: &hello
                name: hello
                description: hello
                model: *default_llm
                prompt: |
                  Welcome to module ${var.module_id}.

            app:
              name: dao_ai_test_${var.module_id}
              description: "test"
              agents:
                - *hello
            """
        ).lstrip()
    )
    return p


@pytest.mark.unit
def test_write_bundle_bakes_resolved_values_and_drops_parameters(
    parameterised_config_path: Path, tmp_path: Path
) -> None:
    from dao_ai.apps.bundle import write_bundle

    config = AppConfig.from_file(
        parameterised_config_path,
        params={"module_id": "09", "catalog": "nfleming"},
        initialize=False,
    )
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()

    write_bundle(config, out_dir, overwrite=True, development=False)

    rendered_path = out_dir / "cfg.yaml"
    assert rendered_path.exists()
    parsed = yaml.safe_load(rendered_path.read_text())

    assert "parameters" not in parsed, (
        "parameters: block must be stripped from the emitted bundle YAML"
    )
    assert parsed["schemas"]["s"]["catalog_name"] == "nfleming"
    assert parsed["schemas"]["s"]["schema_name"] == "dao_ai_09"
    assert parsed["app"]["name"] == "dao_ai_test_09"
    assert "Welcome to module 09" in parsed["agents"]["hello"]["prompt"]
    # No ${var.…} or ${param.…} tokens leak through.
    raw = rendered_path.read_text()
    assert "${var." not in raw
    assert "${param." not in raw


@pytest.mark.unit
def test_write_bundle_ships_pyproject_and_uv_lock_no_requirements(
    parameterised_config_path: Path, tmp_path: Path
) -> None:
    """The generated bundle MUST ship pyproject.toml + uv.lock (Apps' build
    phase runs ``uv sync --locked --no-dev``) and MUST NOT ship a
    requirements.txt (it would take precedence over the uv path and force pip).
    """
    from dao_ai.apps.bundle import write_bundle

    config = AppConfig.from_file(
        parameterised_config_path,
        params={"module_id": "09", "catalog": "nfleming"},
        initialize=False,
    )
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()

    write_bundle(config, out_dir, overwrite=True, development=False)

    assert (out_dir / "pyproject.toml").exists(), (
        "pyproject.toml must be present — it (with uv.lock) drives Apps' "
        "``uv sync --locked`` build."
    )
    assert (out_dir / "uv.lock").exists(), (
        "Generated bundle must ship uv.lock — Apps' build phase runs "
        "``uv sync --locked`` from pyproject.toml + uv.lock."
    )
    assert not (out_dir / "requirements.txt").exists(), (
        "Generated bundle must NOT ship requirements.txt — it takes precedence "
        "over pyproject.toml + uv.lock and forces the pip path."
    )

    # The trimmed databricks.yaml carries bundle/include/targets/artifacts only;
    # the App + experiment block lives in resources/app.yml so users can drop
    # sibling resources/*.yml files (jobs, pipelines, etc.) without conflicting
    # with regen.
    db_yaml = yaml.safe_load((out_dir / "databricks.yaml").read_text())
    assert db_yaml["include"] == ["resources/*.yml"], (
        f"databricks.yaml must wildcard-include resources/*.yml; got {db_yaml.get('include')!r}"
    )
    assert "resources" not in db_yaml or "apps" not in db_yaml.get("resources", {}), (
        "databricks.yaml must not carry the App block; it lives in resources/app.yml"
    )

    app_yaml = yaml.safe_load((out_dir / "resources" / "app.yml").read_text())
    app_name = next(iter(app_yaml["resources"]["apps"]))
    cmd = app_yaml["resources"]["apps"][app_name]["config"]["command"]
    assert cmd == ["python", "-m", "dao_ai.apps.start_app"], (
        f"App command must be bare python (no uv run wrapper); got {cmd!r}"
    )


@pytest.mark.unit
def test_write_bundle_folds_user_pip_requirements_into_pyproject(
    parameterised_config_path: Path, tmp_path: Path
) -> None:
    """``config.app.pip_requirements`` must be folded into the generated
    pyproject ``dependencies`` array so ``uv lock`` captures the deployer's
    custom-code deps (regression guard: the uv switch briefly dropped them).
    ``code_paths`` is NOT copied for Apps — custom code lives under ``src/``.
    """
    from dao_ai.apps.bundle import write_bundle

    config = AppConfig.from_file(
        parameterised_config_path,
        params={"module_id": "09", "catalog": "nfleming"},
        initialize=False,
    )
    config.app.pip_requirements = ["cowsay==6.1", "my-internal-lib>=1.0"]
    config.app.code_paths = ["some_module.py"]
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()

    write_bundle(config, out_dir, overwrite=True, development=False)

    pyproject = (out_dir / "pyproject.toml").read_text()
    assert '"cowsay==6.1"' in pyproject, (
        f"pip_requirements must be folded into pyproject deps; got:\n{pyproject}"
    )
    assert '"my-internal-lib>=1.0"' in pyproject
    # code_paths is a Model-Serving concern; Apps custom code lives under src/.
    assert not (out_dir / "some_module.py").exists(), (
        "code_paths must NOT be copied into an Apps bundle"
    )


@pytest.mark.unit
def test_write_bundle_attaches_trace_location_resources(
    parameterised_config_path: Path, tmp_path: Path
) -> None:
    """When app.trace_location is set, the generated bundle must attach the
    SQL warehouse as an app-resource (so the App SP gets CAN_USE on it) and
    must add MLFLOW_TRACING_SQL_WAREHOUSE_ID to the App's env vars. The
    OTEL trace tables themselves are auto-created by MLflow at first trace
    write — dao-ai does NOT emit per-table securables for them because the
    Apps platform validates table existence at deploy time and the tables
    don't exist yet. Users must manually grant the App SP USE_SCHEMA +
    CREATE_TABLE + MODIFY + SELECT on the trace schema after deploy (one-
    time setup, documented in README "Trace persistence on Apps").

    End-to-end verified on fevm 2026-06-22: traces persist to UC OTEL
    tables, search_traces returns them, and the 9-span agent trace is
    fully queryable.
    """
    import yaml as _yaml

    from dao_ai.apps.bundle import write_bundle
    from dao_ai.config import SchemaModel, TraceLocationModel

    config = AppConfig.from_file(
        parameterised_config_path,
        params={"module_id": "09", "catalog": "nfleming"},
        initialize=False,
    )
    config.app.trace_location = TraceLocationModel(
        schema=SchemaModel(catalog_name="nfleming", schema_name="traces"),
        warehouse="148ccb90800933a1",
    )

    out_dir = tmp_path / "bundle"
    out_dir.mkdir()
    write_bundle(config, out_dir, overwrite=True, development=False)

    app_yaml = _yaml.safe_load((out_dir / "resources" / "app.yml").read_text())
    app_name = next(iter(app_yaml["resources"]["apps"]))
    app_block = app_yaml["resources"]["apps"][app_name]

    resource_names = {r.get("name") for r in app_block["resources"]}
    assert "trace_warehouse" in resource_names, (
        "Missing trace_warehouse app-resource (App SP won't have CAN_USE on "
        "the warehouse and UC OTEL trace writes will fail silently)."
    )

    # The OTEL tables must NOT appear as TABLE securables — they don't exist
    # at deploy time (MLflow auto-creates them at first trace write) and the
    # Apps platform rejects securables that point at non-existent tables.
    forbidden_table_resources = {
        "trace_otel_spans",
        "trace_otel_logs",
        "trace_otel_metrics",
    }
    leaked = forbidden_table_resources & resource_names
    assert not leaked, (
        f"Bundle must not emit TABLE securables for the OTEL trace tables — "
        f"they're auto-created by MLflow at runtime, so the Apps deploy "
        f"validator will reject them as non-existent. Found: {leaked}"
    )

    warehouse_resource = next(
        r for r in app_block["resources"] if r["name"] == "trace_warehouse"
    )
    assert warehouse_resource["sql_warehouse"]["permission"] == "CAN_USE"

    env_var_names = {e["name"] for e in app_block["config"]["env"]}
    assert "MLFLOW_TRACING_SQL_WAREHOUSE_ID" in env_var_names, (
        "App's env must include MLFLOW_TRACING_SQL_WAREHOUSE_ID so handlers.py "
        "can route trace export through the configured warehouse."
    )


@pytest.mark.unit
def test_write_bundle_preserves_user_resources_yml(
    parameterised_config_path: Path, tmp_path: Path
) -> None:
    """A user-authored resources/jobs.yml must survive a re-run of write_bundle.

    The whole point of the wildcard include is that users can drop sibling
    resources/*.yml files into the bundle without `--overwrite` clobbering them.
    """
    from dao_ai.apps.bundle import write_bundle

    config = AppConfig.from_file(
        parameterised_config_path,
        params={"module_id": "09", "catalog": "nfleming"},
        initialize=False,
    )
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()

    write_bundle(config, out_dir, overwrite=True, development=False)

    user_resource = out_dir / "resources" / "jobs.yml"
    user_payload = (
        "resources:\n  jobs:\n    user_job:\n      name: user_job\n      tasks: []\n"
    )
    user_resource.write_text(user_payload)

    write_bundle(config, out_dir, overwrite=True, development=False)

    assert user_resource.exists(), (
        "User-authored resources/jobs.yml must not be deleted by write_bundle"
    )
    assert user_resource.read_text() == user_payload, (
        "User-authored resources/jobs.yml must not be modified by write_bundle"
    )

    # And the regen-owned files are still present + correct.
    db_yaml = yaml.safe_load((out_dir / "databricks.yaml").read_text())
    assert db_yaml["include"] == ["resources/*.yml"]
    app_yaml = yaml.safe_load((out_dir / "resources" / "app.yml").read_text())
    assert "apps" in app_yaml["resources"]


# ---------------------------------------------------------------------------
# Bundle config round-trip: ruamel-based _strip_parameters_block must
# preserve descriptive anchor names, aliases, comments, key order, and
# merge keys, while still dropping the parameters: block and baking in
# ${param.NAME} substitutions.
# ---------------------------------------------------------------------------


@pytest.fixture
def anchored_config_path(tmp_path: Path) -> Path:
    """Config exercising anchors, aliases, merge keys, comments, and params.

    Anchors land only in pydantic-valid fields:
    - `&hardware_store_schema` on a schemas entry, referenced by another
      schemas entry via a YAML merge key
    - `&default_llm` on an LLM resource, referenced by `agents.hello.model`
    - `&hello` on an agent, referenced by `app.agents[0]`
    """
    p = tmp_path / "anchored.yaml"
    p.write_text(
        dedent(
            """
            # Top-of-file comment that should survive round-trip
            parameters:
              catalog:
                description: UC catalog
                default: main
              module_id:
                description: workshop module id

            schemas:  # shared schema definitions reused across resources
              hardware: &hardware_store_schema
                catalog_name: ${var.catalog}
                schema_name: dao_ai_${var.module_id}
              clothing:
                <<: *hardware_store_schema
                schema_name: dao_ai_clothing_${var.module_id}

            resources:
              llms:
                default_llm: &default_llm
                  name: databricks-test-llm
                  temperature: 0.0

            agents:
              hello: &hello  # primary greeter agent
                name: hello
                description: hello agent for module ${var.module_id}
                model: *default_llm
                prompt: |
                  Welcome to module ${var.module_id}.

            app:
              name: dao_ai_test_${var.module_id}
              description: "test"
              agents:
                - *hello
            """
        ).lstrip()
    )
    return p


def _emitted_config_text(
    config_path: Path,
    out_dir: Path,
    params: dict[str, str],
) -> str:
    """Helper: run write_bundle and return the emitted config YAML text."""
    from dao_ai.apps.bundle import write_bundle

    config = AppConfig.from_file(config_path, params=params, initialize=False)
    out_dir.mkdir(exist_ok=True)
    write_bundle(config, out_dir, overwrite=True, development=False)
    return (out_dir / config_path.name).read_text()


@pytest.mark.unit
def test_round_trip_preserves_descriptive_anchor_names(
    anchored_config_path: Path, tmp_path: Path
) -> None:
    """Source uses `&hardware_store_schema`; emitted YAML must keep that
    literal name, not auto-generate `&id001`."""
    text = _emitted_config_text(
        anchored_config_path,
        tmp_path / "bundle",
        params={"module_id": "09", "catalog": "nfleming"},
    )
    assert "&hardware_store_schema" in text, (
        f"descriptive anchor name lost — got:\n{text}"
    )
    assert "&default_llm" in text
    assert "&hello" in text
    # No PyYAML-style auto-generated anchor noise.
    assert "&id001" not in text
    assert "&id002" not in text


@pytest.mark.unit
def test_round_trip_preserves_aliases_pointing_at_descriptive_anchors(
    anchored_config_path: Path, tmp_path: Path
) -> None:
    """Aliases like `*hardware_store_schema` must stay textual aliases (not
    inlined copies), and they must point at the original anchor name."""
    text = _emitted_config_text(
        anchored_config_path,
        tmp_path / "bundle",
        params={"module_id": "09", "catalog": "nfleming"},
    )
    assert "*hardware_store_schema" in text, (
        f"alias was inlined or renamed — got:\n{text}"
    )
    assert "*default_llm" in text
    assert "*hello" in text


@pytest.mark.unit
def test_round_trip_preserves_merge_keys(
    anchored_config_path: Path, tmp_path: Path
) -> None:
    """The clothing schema uses `<<: *hardware_store_schema` — both the merge
    syntax and the alias name must survive."""
    text = _emitted_config_text(
        anchored_config_path,
        tmp_path / "bundle",
        params={"module_id": "09", "catalog": "nfleming"},
    )
    assert "<<: *hardware_store_schema" in text, (
        f"merge key did not survive — got:\n{text}"
    )


@pytest.mark.unit
def test_round_trip_aliases_resolve_to_same_data_after_load(
    anchored_config_path: Path, tmp_path: Path
) -> None:
    """Semantic check: parse the emitted YAML and confirm the alias target
    equals the anchor source. Anchor preservation is cosmetic; this proves
    we didn't accidentally fork the references."""
    text = _emitted_config_text(
        anchored_config_path,
        tmp_path / "bundle",
        params={"module_id": "09", "catalog": "nfleming"},
    )
    parsed = yaml.safe_load(text)
    # `&hello` is referenced by `app.agents[0]` via `*hello`.
    assert parsed["app"]["agents"][0] == parsed["agents"]["hello"], (
        "alias resolved to a different value than the anchor source"
    )
    # `&default_llm` is referenced by `agents.hello.model` via `*default_llm`.
    assert (
        parsed["agents"]["hello"]["model"] == parsed["resources"]["llms"]["default_llm"]
    )
    # Merge key: clothing schema inherits catalog_name from hardware but
    # overrides schema_name.
    assert parsed["schemas"]["clothing"]["catalog_name"] == "nfleming"
    assert parsed["schemas"]["clothing"]["schema_name"] == "dao_ai_clothing_09"


@pytest.mark.unit
def test_round_trip_drops_parameters_block_with_anchors_present(
    anchored_config_path: Path, tmp_path: Path
) -> None:
    """The parameters: removal path must coexist with anchors elsewhere."""
    text = _emitted_config_text(
        anchored_config_path,
        tmp_path / "bundle",
        params={"module_id": "09", "catalog": "nfleming"},
    )
    parsed = yaml.safe_load(text)
    assert "parameters" not in parsed, (
        "parameters: block must be stripped from the emitted bundle YAML"
    )


@pytest.mark.unit
def test_round_trip_bakes_in_param_substitutions_with_anchors(
    anchored_config_path: Path, tmp_path: Path
) -> None:
    """`${var.…}` and `${param.…}` references must be substituted, even
    inside anchored mappings."""
    text = _emitted_config_text(
        anchored_config_path,
        tmp_path / "bundle",
        params={"module_id": "09", "catalog": "nfleming"},
    )
    assert "${var." not in text
    assert "${param." not in text
    parsed = yaml.safe_load(text)
    assert parsed["schemas"]["hardware"]["catalog_name"] == "nfleming"
    assert parsed["schemas"]["hardware"]["schema_name"] == "dao_ai_09"
    assert parsed["app"]["name"] == "dao_ai_test_09"


@pytest.mark.unit
def test_round_trip_preserves_top_level_key_order(
    anchored_config_path: Path, tmp_path: Path
) -> None:
    """ruamel rt mode preserves key order; PyYAML safe_dump did not."""
    text = _emitted_config_text(
        anchored_config_path,
        tmp_path / "bundle",
        params={"module_id": "09", "catalog": "nfleming"},
    )
    # Source order (after parameters: removed): schemas, resources, agents, app
    expected_order = ["schemas", "resources", "agents", "app"]
    actual_order = [
        line.split(":", 1)[0]
        for line in text.splitlines()
        if line and not line.startswith((" ", "\t", "#"))
    ]
    actual_order = [k for k in actual_order if k in expected_order]
    assert actual_order == expected_order, (
        f"key order changed: expected {expected_order}, got {actual_order}"
    )


@pytest.mark.unit
def test_round_trip_preserves_comments(
    anchored_config_path: Path, tmp_path: Path
) -> None:
    """ruamel rt mode preserves comments (PyYAML does not)."""
    text = _emitted_config_text(
        anchored_config_path,
        tmp_path / "bundle",
        params={"module_id": "09", "catalog": "nfleming"},
    )
    assert "# Top-of-file comment that should survive round-trip" in text
    # Inline-on-key comments are the most robustly-anchored kind; both
    # should survive even though the parameters: block (a sibling of
    # schemas:) was removed.
    assert "# shared schema definitions reused across resources" in text
    assert "# primary greeter agent" in text


# ---------------------------------------------------------------------------
# Direct unit tests for the round-trip helper.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_strip_parameters_block_drops_top_level_parameters() -> None:
    from dao_ai.apps.bundle import _strip_parameters_block

    src = dedent(
        """
        parameters:
          foo:
            default: bar
        app:
          name: x
        """
    ).lstrip()
    out = _strip_parameters_block(src)
    parsed = yaml.safe_load(out)
    assert "parameters" not in parsed
    assert parsed["app"]["name"] == "x"


@pytest.mark.unit
def test_strip_parameters_block_no_op_when_parameters_absent() -> None:
    """If no top-level parameters: key, output is semantically identical."""
    from dao_ai.apps.bundle import _strip_parameters_block

    src = dedent(
        """
        # leading comment
        schemas:
          s: &s
            name: a
        agents:
          a:
            schema: *s
        """
    ).lstrip()
    out = _strip_parameters_block(src)
    assert "&s" in out
    assert "*s" in out
    assert "# leading comment" in out
    parsed = yaml.safe_load(out)
    assert parsed["agents"]["a"]["schema"] == parsed["schemas"]["s"]


@pytest.mark.unit
def test_strip_parameters_block_returns_input_on_parse_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Malformed YAML must not corrupt: returned unchanged so the caller
    isn't silently handed broken output."""
    from dao_ai.apps.bundle import _strip_parameters_block

    bad = "key: [unclosed\n"
    assert _strip_parameters_block(bad) == bad


@pytest.mark.unit
def test_strip_parameters_block_handles_empty_string() -> None:
    from dao_ai.apps.bundle import _strip_parameters_block

    assert _strip_parameters_block("") == ""


@pytest.mark.unit
def test_strip_parameters_block_handles_top_level_list() -> None:
    """If the document root is a list (no top-level mapping), pop is a no-op
    and the input must come back functionally intact."""
    from dao_ai.apps.bundle import _strip_parameters_block

    src = "- one\n- two\n"
    out = _strip_parameters_block(src)
    assert yaml.safe_load(out) == ["one", "two"]


# ---------------------------------------------------------------------------
# CLI integration: parse_args + handler call against tmp configs.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_validate_handler_pretty_prints_missing_required(
    parameterised_config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """validate without --var module_id should exit 1 and print a useful error."""
    from dao_ai.cli import handle_validate_command, parse_args

    options = parse_args(["validate", "-c", str(parameterised_config_path)])
    with pytest.raises(SystemExit) as exc:
        handle_validate_command(options)
    assert exc.value.code == 1

    captured = capsys.readouterr()
    assert "Missing required parameters" in captured.err
    assert "module_id" in captured.err


@pytest.mark.unit
def test_cli_vars_list_handler_prints_table(
    parameterised_config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dao_ai.cli import handle_vars_command, parse_args

    options = parse_args(
        [
            "vars",
            "list",
            "-c",
            str(parameterised_config_path),
            "--var",
            "module_id=09",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        handle_vars_command(options)
    assert exc.value.code == 0  # all required resolved

    out = capsys.readouterr().out
    assert "NAME" in out and "REQUIRED" in out and "RESOLVED" in out
    assert "catalog" in out and "main" in out  # default
    assert "module_id" in out and "09" in out  # --var


@pytest.mark.unit
def test_cli_vars_list_handler_exits_nonzero_when_required_missing(
    parameterised_config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dao_ai.cli import handle_vars_command, parse_args

    options = parse_args(["vars", "list", "-c", str(parameterised_config_path)])
    with pytest.raises(SystemExit) as exc:
        handle_vars_command(options)
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "MISSING" in out
    assert "module_id" in out


@pytest.mark.unit
def test_cli_var_flag_appears_in_subparser_help(
    parameterised_config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every config-aware subparser must surface --var in its help."""
    from dao_ai.cli import parse_args

    simple_help = (["validate"], ["graph"], ["chat"], ["vars"])
    for argv in (*simple_help, ["monitor", "scorers"], ["monitor", "logs"]):
        with pytest.raises(SystemExit):
            # --help triggers a clean SystemExit(0) after printing.
            parse_args([*argv, "--help"])
        captured = capsys.readouterr()
        assert "--var" in captured.out, (
            f"--var flag missing from `{' '.join(argv)} --help` output"
        )

    # Bundle-noun verbs get --var from the shared _add_bundle_common_args helper.
    for argv in (["agent", "build", "--help"], ["workflow", "sync", "--help"]):
        with pytest.raises(SystemExit):
            parse_args(argv)
        captured = capsys.readouterr()
        assert "--var" in captured.out, (
            f"--var flag missing from `{' '.join(argv)}` output"
        )


@pytest.mark.unit
def test_cli_run_databricks_command_forwards_only_declared_vars(
    parameterised_config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dao-ai --param is forwarded to `databricks bundle` as --var ONLY when the
    staged databricks.yaml declares a matching bundle variable.

    dao-ai config params are baked into the staged config; forwarding a param the
    bundle doesn't declare makes `databricks bundle` hard-fail ("variable X has
    not been defined"). So `module_id` (declared here) forwards; `undeclared_x`
    (not a bundle var) must be dropped. Regression guard for the crash-fix.
    """
    from dao_ai import cli

    captured: dict[str, list[str]] = {}

    # Pre-staged bundle dir whose databricks.yaml declares module_id + catalog as
    # bundle variables (but NOT undeclared_x). stage=False → use it in place.
    out = tmp_path / "staged"
    out.mkdir()
    (out / "databricks.yaml").write_text(
        "variables:\n  module_id:\n    description: d\n  catalog:\n    description: d\n"
    )

    monkeypatch.setattr(cli, "_apply_profile_context", lambda profile: None)
    monkeypatch.setattr(cli, "detect_cloud_provider", lambda profile: "aws")

    class _FakeStdout:
        def readline(self) -> str:
            return ""

    class _FakeProcess:
        returncode = 0
        stdout = _FakeStdout()

        def wait(self) -> int:
            return 0

    def _fake_popen(cmd, **_kwargs):
        captured["cmd"] = cmd
        return _FakeProcess()

    import subprocess as _sp

    monkeypatch.setattr(_sp, "Popen", _fake_popen)

    cli.run_databricks_command(
        ["bundle", "deploy"],
        config=str(parameterised_config_path),
        config_vars={"module_id": "09", "catalog": "nfleming", "undeclared_x": "z"},
        staging_dir=str(out),
        development=False,
        stage=False,
    )

    cmd = captured["cmd"]
    cmd_str = " ".join(cmd)
    assert '--var="module_id=09"' in cmd_str
    assert '--var="catalog=nfleming"' in cmd_str
    assert "undeclared_x" not in cmd_str


@pytest.mark.unit
def test_run_databricks_command_hard_fails_when_cloud_undetectable(
    parameterised_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real run must stop (exit 1) when the cloud can't be auto-detected and
    no --cloud was passed, rather than silently guessing a cloud."""
    from dao_ai import cli

    monkeypatch.setattr(cli, "_apply_profile_context", lambda profile: None)
    monkeypatch.setattr(cli, "detect_cloud_provider", lambda profile: None)

    import dao_ai.pipeline.bundle as _pb

    monkeypatch.setattr(_pb, "write_pipeline_bundle", lambda *a, **k: None)

    # The bundle executor must never be reached — the cloud gate should exit first.
    def _fail_exec(*_a: object, **_k: object) -> None:  # pragma: no cover
        raise AssertionError("bundle command should not run when cloud is undetectable")

    monkeypatch.setattr(cli, "_exec_bundle_command", _fail_exec)

    with pytest.raises(SystemExit) as exc:
        cli.run_databricks_command(
            ["bundle", "deploy"],
            config=str(parameterised_config_path),
            config_vars={"module_id": "09", "catalog": "nfleming"},
            cloud=None,
            dry_run=False,
            development=False,
        )
    assert exc.value.code == 1


@pytest.mark.unit
def test_run_databricks_command_dry_run_survives_undetectable_cloud(
    parameterised_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--dry-run must still render its preview when the cloud can't be detected,
    falling back to a visible placeholder in the bundle target instead of
    exiting (real runs still hard-fail — see the sibling test)."""
    from dao_ai import cli

    monkeypatch.setattr(cli, "_apply_profile_context", lambda profile: None)
    monkeypatch.setattr(cli, "detect_cloud_provider", lambda profile: None)

    import dao_ai.pipeline.bundle as _pb

    monkeypatch.setattr(_pb, "write_pipeline_bundle", lambda *a, **k: None)

    # Spy on the bundle executor to capture the resolved --target (which carries
    # the cloud) without spawning a real databricks subprocess.
    captured: dict[str, object] = {}

    def _spy_exec(command, *, profile, target, cwd, extra_vars=None, dry_run=False):
        captured["target"] = target
        captured["dry_run"] = dry_run

    monkeypatch.setattr(cli, "_exec_bundle_command", _spy_exec)

    # Should NOT raise SystemExit.
    cli.run_databricks_command(
        ["bundle", "deploy"],
        config=str(parameterised_config_path),
        config_vars={"module_id": "09", "catalog": "nfleming"},
        cloud=None,
        dry_run=True,
        development=False,
    )

    # The auto-generated target falls back to the placeholder cloud
    # (`<app>-<cloud>`) so the dry-run preview still renders.
    assert captured["dry_run"] is True
    assert "<cloud>" in str(captured["target"])


@pytest.mark.unit
def test_appconfig_has_no_deployment_target_field():
    from dao_ai.config import AppModel

    assert "deployment_target" not in AppModel.model_fields


@pytest.mark.unit
def test_appconfig_no_registered_model_no_config_time_error():
    """Removing deployment_target also removes the config-time validator that
    raised when registered_model was absent.  The check now lives at deploy
    time in deploy_model_serving_agent.  Constructing AppConfig with a missing
    registered_model must not raise."""
    import tempfile

    from dao_ai.config import AppConfig

    yaml_text = dedent(
        """\
        schemas:
          s:
            catalog_name: cat
            schema_name: sch

        agents:
          hello:
            name: hello
            description: hello
            model:
              name: databricks-meta-llama-3-1-70b-instruct
            prompt: hi

        app:
          name: no_model_app
          description: "app without registered_model"
          agents:
            - name: hello
              description: hello
              model:
                name: databricks-meta-llama-3-1-70b-instruct
              prompt: hi
        """
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml_text)
        tmp_path = Path(fh.name)

    try:
        # Should not raise — the registered_model check moved to deploy time.
        config = AppConfig.from_file(tmp_path, initialize=False)
        assert config.app is not None
        assert config.app.registered_model is None
    finally:
        tmp_path.unlink(missing_ok=True)
