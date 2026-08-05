"""Tests for the ``dao-ai service-principal`` CLI reporting.

Focused on ``_print_grants``, which is what a user actually reads after a
provision/grant run. The regression these guard against: a real run reported

    grants (1 applied, 10 failed):
      [uc] catalog hardware_store -> USE_CATALOG  ✗ FAILED

ten times, while the underlying errors — already captured on each ``Grant`` —
were never printed. The true cause was that the catalog did not exist in that
workspace, but the output (and the log line) pointed at GRANT rights instead.
"""

from __future__ import annotations

import pytest

from dao_ai.cli import _print_grants
from dao_ai.service_principal import (
    GRANT_FAILURE_ABSENT,
    GRANT_FAILURE_DENIED,
    GRANT_FAILURE_ERROR,
    Grant,
    GrantPlan,
)

PRINCIPAL = "11111111-1111-1111-1111-111111111111"


def _plan(*grants: Grant) -> GrantPlan:
    return GrantPlan(principal=PRINCIPAL, grants=list(grants))


def _absent_catalog(name: str = "hardware_store") -> Grant:
    return Grant(
        kind="uc",
        target=name,
        privileges=["USE_CATALOG"],
        securable_type="catalog",
        applied=False,
        error=f"Catalog '{name}' does not exist.",
        failure_kind=GRANT_FAILURE_ABSENT,
    )


@pytest.mark.unit
class TestPrintGrantsFailureReporting:
    def test_prints_the_error_for_a_failed_grant(self, capsys) -> None:
        """The whole bug: ``Grant.error`` was recorded but never rendered."""
        _print_grants(_plan(_absent_catalog()), applied=True)
        out = capsys.readouterr().out
        assert "Catalog 'hardware_store' does not exist." in out

    def test_labels_absent_and_denied_differently(self, capsys) -> None:
        denied = Grant(
            kind="uc",
            target="cat.sch.fn",
            privileges=["EXECUTE"],
            securable_type="function",
            applied=False,
            error="PERMISSION_DENIED: user lacks GRANT",
            failure_kind=GRANT_FAILURE_DENIED,
        )
        _print_grants(_plan(_absent_catalog(), denied), applied=True)
        out = capsys.readouterr().out
        assert "⚠ ABSENT" in out
        assert "✗ DENIED" in out

    def test_generic_failure_still_reads_failed(self, capsys) -> None:
        boom = Grant(
            kind="warehouse",
            target="wh-1",
            privileges=["CAN_USE"],
            applied=False,
            error="kaboom",
            failure_kind=GRANT_FAILURE_ERROR,
        )
        _print_grants(_plan(boom), applied=True)
        out = capsys.readouterr().out
        assert "✗ FAILED" in out
        assert "kaboom" in out

    def test_summary_separates_absent_from_failed(self, capsys) -> None:
        """An absent target is a config/workspace mismatch, not a failure.

        ``_grant_serving_endpoint`` already no-ops silently on an absent
        endpoint, so counting an absent catalog as "failed" was inconsistent.
        """
        ok = Grant(
            kind="uc",
            target="cat.sch",
            privileges=["USE_SCHEMA"],
            securable_type="schema",
            applied=True,
        )
        _print_grants(_plan(ok, _absent_catalog()), applied=True)
        out = capsys.readouterr().out
        assert "1 applied" in out
        assert "1 absent" in out
        assert "failed" not in out.split("\n")[0]

    def test_suggests_a_var_override_when_a_catalog_is_absent(self, capsys) -> None:
        _print_grants(_plan(_absent_catalog()), applied=True)
        out = capsys.readouterr().out
        assert "--var catalog=" in out
        assert "hardware_store" in out

    def test_absent_hint_points_at_config_not_permissions(self, capsys) -> None:
        """The old output blamed GRANT rights for a resource that wasn't there."""
        _print_grants(_plan(_absent_catalog()), applied=True)
        out = capsys.readouterr().out
        assert "-p profile" in out or "-p " in out
        # The absent hint must not send the user auditing ACLs.
        absent_block = out.split("⚠ ABSENT")[1]
        assert "GRANT/MANAGE" not in absent_block.split("\n\n")[0]

    def test_dry_run_prints_no_failure_labels(self, capsys) -> None:
        planned = Grant(
            kind="uc",
            target="cat",
            privileges=["USE_CATALOG"],
            securable_type="catalog",
        )
        _print_grants(_plan(planned), applied=False)
        out = capsys.readouterr().out
        assert "dry-run — nothing applied" in out
        assert "ABSENT" not in out
        assert "FAILED" not in out

    def test_skip_note_is_still_rendered(self, capsys) -> None:
        skipped = Grant(
            kind="lakebase_role",
            target="my-proj",
            privileges=["DATABRICKS_SUPERUSER"],
            note="SKIP: no declared service principal owns this project",
        )
        _print_grants(_plan(skipped), applied=True)
        out = capsys.readouterr().out
        assert "⚠ SKIP" in out
        assert "no declared service principal" in out

    def test_empty_plan_reports_nothing_to_grant(self, capsys) -> None:
        _print_grants(_plan(), applied=True)
        assert "no grantable resources found" in capsys.readouterr().out
