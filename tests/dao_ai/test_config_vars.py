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
    ConfigVariableError,
    ParameterDeclarationModel,
    find_param_references,
    resolve_parameters,
    substitute_params,
)


@pytest.mark.unit
def test_find_param_references_extracts_distinct_names() -> None:
    text = "a: ${param.foo}\nb: ${param.bar:-default}\nc: ${param.foo}"
    assert find_param_references(text) == {"foo", "bar"}


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
    assert by_name["from_cli"].source == "--var"
    assert by_name["from_cli"].value == "cli-value"
    assert by_name["from_env"].source == "env"
    assert by_name["from_env"].value == "env-value"
    assert by_name["from_default"].source == "default"
    assert by_name["from_default"].value == "d"
    assert by_name["missing"].source == "MISSING"
    assert by_name["missing"].value is None
    assert by_name["missing"].required is True


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
              deployment_target: apps
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

    write_bundle(config, out_dir, force=True, development=False)

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

    for command in ("validate", "graph", "deploy", "monitor", "chat", "vars"):
        with pytest.raises(SystemExit):
            # --help triggers a clean SystemExit(0) after printing.
            parse_args([command, "--help"])
        captured = capsys.readouterr()
        assert "--var" in captured.out, (
            f"--var flag missing from `{command} --help` output"
        )


@pytest.mark.unit
def test_cli_run_databricks_command_forwards_vars_to_databricks_cli(
    parameterised_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--var foo=bar must be appended to the underlying `databricks bundle …`."""
    from dao_ai import cli

    captured: dict[str, list[str]] = {}

    def _fake_run_command(cmd_args: list[str], **_: object) -> None:
        captured["cmd"] = cmd_args

    monkeypatch.setattr(cli, "_apply_profile_context", lambda profile: None)
    monkeypatch.setattr(cli, "detect_cloud_provider", lambda profile: "aws")
    monkeypatch.setattr(
        cli, "generate_bundle_from_template", lambda p, n: Path("databricks.yaml")
    )

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
        config_vars={"module_id": "09", "catalog": "nfleming"},
    )

    cmd = captured["cmd"]
    cmd_str = " ".join(cmd)
    assert '--var="module_id=09"' in cmd_str
    assert '--var="catalog=nfleming"' in cmd_str
