"""Unit tests for `link_experiment_trace_location`.

Covers the idempotent linkage helper introduced to fix silent trace loss on
Databricks Apps redeploys — MLflow rejects re-linking an experiment that
already has traces, so we detect an existing matching linkage via the
experiment's UC tags and skip the API call when it already matches.
"""

from unittest.mock import MagicMock, patch

import pytest


class _FakeExperiment:
    def __init__(self, tags: dict[str, str] | None = None) -> None:
        self.tags = tags or {}


def _cfg(catalog: str, schema: str, prefix: str | None = None) -> MagicMock:
    """Build a mock AppConfig with a trace_location.

    Real AppConfig instantiation drags too much (secrets, resources, etc.);
    the helper only touches `config.app.trace_location.{catalog_name,
    schema_name,resolved_table_prefix}`.
    """
    cfg = MagicMock()
    cfg.app = MagicMock()
    cfg.app.trace_location.catalog_name = catalog
    cfg.app.trace_location.schema_name = schema
    cfg.app.trace_location.resolved_table_prefix = prefix
    return cfg


@pytest.mark.unit
class TestLinkExperimentTraceLocation:
    def test_no_op_when_trace_location_missing(self) -> None:
        from dao_ai.providers.databricks import link_experiment_trace_location

        cfg = MagicMock()
        cfg.app = MagicMock()
        cfg.app.trace_location = None

        with patch(
            "dao_ai.providers.databricks.mlflow.set_experiment"
        ) as set_exp:
            link_experiment_trace_location(cfg, "exp1")
        set_exp.assert_not_called()

    def test_links_when_no_existing_tags(self) -> None:
        from dao_ai.providers.databricks import link_experiment_trace_location

        cfg = _cfg("cat", "sch")
        exp = _FakeExperiment(tags={})
        with (
            patch("dao_ai.providers.databricks.mlflow.set_experiment") as set_exp,
            patch("mlflow.tracking.MlflowClient") as mc,
        ):
            mc.return_value.get_experiment.return_value = exp
            link_experiment_trace_location(cfg, "exp1")
        set_exp.assert_called_once()
        kwargs = set_exp.call_args.kwargs
        assert kwargs["experiment_id"] == "exp1"
        assert kwargs["trace_location"] is not None

    def test_skips_when_already_linked_matching(self) -> None:
        from dao_ai.providers.databricks import link_experiment_trace_location

        cfg = _cfg("cat", "sch", "my_prefix")
        exp = _FakeExperiment(
            tags={
                "mlflow.experiment.databricksTraceDestinationPath": "cat.sch.my_prefix",
            }
        )
        with (
            patch("dao_ai.providers.databricks.mlflow.set_experiment") as set_exp,
            patch("mlflow.tracking.MlflowClient") as mc,
        ):
            mc.return_value.get_experiment.return_value = exp
            link_experiment_trace_location(cfg, "exp1")
        set_exp.assert_not_called()

    def test_links_when_prefix_differs(self) -> None:
        from dao_ai.providers.databricks import link_experiment_trace_location

        cfg = _cfg("cat", "sch", "wanted_prefix")
        exp = _FakeExperiment(
            tags={
                "mlflow.experiment.databricksTraceDestinationPath": "cat.sch.other_prefix",
            }
        )
        with (
            patch("dao_ai.providers.databricks.mlflow.set_experiment") as set_exp,
            patch("mlflow.tracking.MlflowClient") as mc,
        ):
            mc.return_value.get_experiment.return_value = exp
            link_experiment_trace_location(cfg, "exp1")
        set_exp.assert_called_once()

    def test_links_when_schema_differs(self) -> None:
        from dao_ai.providers.databricks import link_experiment_trace_location

        cfg = _cfg("cat", "wanted_schema")
        exp = _FakeExperiment(
            tags={
                "mlflow.experiment.databricksTraceDestinationPath": "cat.old_schema.some_prefix",
            }
        )
        with (
            patch("dao_ai.providers.databricks.mlflow.set_experiment") as set_exp,
            patch("mlflow.tracking.MlflowClient") as mc,
        ):
            mc.return_value.get_experiment.return_value = exp
            link_experiment_trace_location(cfg, "exp1")
        set_exp.assert_called_once()

    def test_prefix_none_matches_any_existing_prefix(self) -> None:
        """Config-side prefix=None (MLflow auto-assigns) → any existing prefix on
        the same catalog/schema is compatible."""
        from dao_ai.providers.databricks import link_experiment_trace_location

        cfg = _cfg("cat", "sch", None)
        exp = _FakeExperiment(
            tags={
                # MLflow substituted the experiment id as the prefix when
                # link was called without an explicit table_prefix.
                "mlflow.experiment.databricksTraceDestinationPath": "cat.sch.2118001471326798",
            }
        )
        with (
            patch("dao_ai.providers.databricks.mlflow.set_experiment") as set_exp,
            patch("mlflow.tracking.MlflowClient") as mc,
        ):
            mc.return_value.get_experiment.return_value = exp
            link_experiment_trace_location(cfg, "exp1")
        set_exp.assert_not_called()

    def test_malformed_destination_path_falls_through_to_link(self) -> None:
        """A destination-path tag with too few segments to parse → attempt link."""
        from dao_ai.providers.databricks import link_experiment_trace_location

        cfg = _cfg("cat", "sch")
        exp = _FakeExperiment(
            tags={
                "mlflow.experiment.databricksTraceDestinationPath": "just_one_part",
            }
        )
        with (
            patch("dao_ai.providers.databricks.mlflow.set_experiment") as set_exp,
            patch("mlflow.tracking.MlflowClient") as mc,
        ):
            mc.return_value.get_experiment.return_value = exp
            link_experiment_trace_location(cfg, "exp1")
        set_exp.assert_called_once()

    def test_get_experiment_error_falls_through_to_link(self) -> None:
        """If we can't read the experiment tags, attempt the link — safest default."""
        from dao_ai.providers.databricks import link_experiment_trace_location

        cfg = _cfg("cat", "sch")
        with (
            patch("dao_ai.providers.databricks.mlflow.set_experiment") as set_exp,
            patch("mlflow.tracking.MlflowClient") as mc,
        ):
            mc.return_value.get_experiment.side_effect = RuntimeError("perm denied")
            link_experiment_trace_location(cfg, "exp1")
        set_exp.assert_called_once()

    def test_underlying_link_error_surfaces(self) -> None:
        """When link genuinely fails (e.g. warehouse permission), raise."""
        from dao_ai.providers.databricks import link_experiment_trace_location

        cfg = _cfg("cat", "sch")
        exp = _FakeExperiment(tags={})
        with (
            patch("dao_ai.providers.databricks.mlflow.set_experiment") as set_exp,
            patch("mlflow.tracking.MlflowClient") as mc,
        ):
            mc.return_value.get_experiment.return_value = exp
            set_exp.side_effect = RuntimeError("warehouse timeout")
            with pytest.raises(RuntimeError, match="warehouse timeout"):
                link_experiment_trace_location(cfg, "exp1")

    def test_already_contains_traces_swallowed_as_fallback(self) -> None:
        """If the tag-check misses but MLflow says 'already contains traces',
        treat as idempotent — the linkage must already be in place."""
        from dao_ai.providers.databricks import link_experiment_trace_location
        import mlflow

        cfg = _cfg("cat", "sch")
        exp = _FakeExperiment(tags={})  # tag lookup says "not linked"
        with (
            patch("dao_ai.providers.databricks.mlflow.set_experiment") as set_exp,
            patch("mlflow.tracking.MlflowClient") as mc,
        ):
            mc.return_value.get_experiment.return_value = exp
            set_exp.side_effect = mlflow.exceptions.RestException(
                {"error_code": "BAD_REQUEST", "message": "already contains traces"}
            )
            # Should NOT raise
            link_experiment_trace_location(cfg, "exp1")
