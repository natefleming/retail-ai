"""Tests for the stale-Delta-checkpoint self-heal in ``create_vector_store``.

A Delta-Sync VS index binds to its source table by a streaming checkpoint
(Delta table GUID + version). When the source table is recreated
(``CREATE OR REPLACE``, overwrite reload, retried ingest) the checkpoint goes
stale and a plain ``index.sync()`` fail-loops forever. The provider now detects
this and drops+recreates the index. Direct-Access indexes (no source table, no
checkpoint) must never be touched by this logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dao_ai.config import (
    IndexModel,
    LLMModel,
    SchemaModel,
    TableModel,
    VectorSearchEndpoint,
    VectorSearchEndpointType,
    VectorStoreModel,
)
from dao_ai.providers import databricks as dbx
from dao_ai.providers.databricks import DatabricksProvider


def _vector_store() -> VectorStoreModel:
    schema = SchemaModel(catalog_name="test_cat", schema_name="test_sch")
    return VectorStoreModel(
        index=IndexModel(schema=schema, name="test_index"),
        source_table=TableModel(schema=schema, name="test_source"),
        embedding_source_column="description",
        embedding_model=LLMModel(name="databricks-gte-large-en"),
        endpoint=VectorSearchEndpoint(
            name="test_endpoint", type=VectorSearchEndpointType.STANDARD
        ),
        primary_key="id",
    )


def _provider() -> tuple[DatabricksProvider, MagicMock]:
    provider = DatabricksProvider()
    mock_vsc = MagicMock()
    provider.vsc = mock_vsc
    return provider, mock_vsc


def _delta_sync_details(detailed_state: str = "ONLINE_NO_PENDING_UPDATE") -> dict:
    return {
        "delta_sync_index_spec": {"source_table": "test_cat.test_sch.test_source"},
        "status": {"detailed_state": detailed_state},
    }


def _direct_access_details(detailed_state: str = "ONLINE_DIRECT_ACCESS") -> dict:
    return {
        "direct_access_index_spec": {},
        "status": {"detailed_state": detailed_state},
    }


@pytest.mark.unit
class TestSelfHeal:
    def test_healthy_index_syncs_not_recreated(self) -> None:
        provider, vsc = _provider()
        idx = MagicMock()
        idx.describe.return_value = _delta_sync_details("ONLINE_NO_PENDING_UPDATE")
        vsc.get_index.return_value = idx
        with (
            patch.object(dbx, "endpoint_exists", return_value=True),
            patch.object(dbx, "index_exists", return_value=True),
            patch.object(dbx, "_source_table_delta_uuid", return_value="uuid-1"),
        ):
            provider.create_vector_store(_vector_store())
        idx.sync.assert_called_once()
        vsc.delete_index.assert_not_called()
        vsc.create_delta_sync_index_and_wait.assert_not_called()

    def test_failed_state_triggers_recreate(self) -> None:
        provider, vsc = _provider()
        idx = MagicMock()
        idx.describe.return_value = _delta_sync_details("OFFLINE_FAILED")
        vsc.get_index.return_value = idx
        with (
            patch.object(dbx, "endpoint_exists", return_value=True),
            patch.object(dbx, "index_exists", return_value=True),
            patch.object(dbx, "_wait_until_index_absent"),
            patch.object(dbx, "_source_table_delta_uuid", return_value="uuid-1"),
        ):
            provider.create_vector_store(_vector_store())
        vsc.delete_index.assert_called_once()
        vsc.create_delta_sync_index_and_wait.assert_called_once()
        idx.sync.assert_not_called()

    def test_uuid_mismatch_triggers_recreate(self) -> None:
        provider, vsc = _provider()
        idx = MagicMock()
        # index recorded uuid-OLD; live source is uuid-NEW → stale.
        details = _delta_sync_details("ONLINE_NO_PENDING_UPDATE")
        details["custom_tags"] = {"dao_ai_source_delta_uuid": "uuid-OLD"}
        idx.describe.return_value = details
        vsc.get_index.return_value = idx
        with (
            patch.object(dbx, "endpoint_exists", return_value=True),
            patch.object(dbx, "index_exists", return_value=True),
            patch.object(dbx, "_wait_until_index_absent"),
            patch.object(dbx, "_source_table_delta_uuid", return_value="uuid-NEW"),
        ):
            provider.create_vector_store(_vector_store())
        vsc.delete_index.assert_called_once()
        vsc.create_delta_sync_index_and_wait.assert_called_once()

    def test_uuid_match_does_not_recreate(self) -> None:
        provider, vsc = _provider()
        idx = MagicMock()
        details = _delta_sync_details("ONLINE_NO_PENDING_UPDATE")
        details["custom_tags"] = {"dao_ai_source_delta_uuid": "uuid-SAME"}
        idx.describe.return_value = details
        vsc.get_index.return_value = idx
        with (
            patch.object(dbx, "endpoint_exists", return_value=True),
            patch.object(dbx, "index_exists", return_value=True),
            patch.object(dbx, "_source_table_delta_uuid", return_value="uuid-SAME"),
        ):
            provider.create_vector_store(_vector_store())
        vsc.delete_index.assert_not_called()
        idx.sync.assert_called_once()

    def test_direct_access_index_never_recreated(self) -> None:
        """Direct-Access index (no source table / checkpoint) even in a FAILED
        state must NOT be dropped by the Delta-Sync self-heal."""
        provider, vsc = _provider()
        idx = MagicMock()
        idx.describe.return_value = _direct_access_details("OFFLINE_FAILED")
        vsc.get_index.return_value = idx
        with (
            patch.object(dbx, "endpoint_exists", return_value=True),
            patch.object(dbx, "index_exists", return_value=True),
        ):
            provider.create_vector_store(_vector_store())
        vsc.delete_index.assert_not_called()
        idx.sync.assert_called_once()

    def test_detection_error_falls_through_to_sync(self) -> None:
        """If describe()/staleness detection raises, we must fall through to a
        normal sync — never block a healthy deploy on a heuristic."""
        provider, vsc = _provider()
        idx = MagicMock()
        idx.describe.side_effect = RuntimeError("transient describe failure")
        vsc.get_index.return_value = idx
        with (
            patch.object(dbx, "endpoint_exists", return_value=True),
            patch.object(dbx, "index_exists", return_value=True),
        ):
            provider.create_vector_store(_vector_store())
        vsc.delete_index.assert_not_called()
        idx.sync.assert_called_once()

    def test_timeout_raises_actionable_error(self) -> None:
        """A still-not-ONLINE index (SDK raises on timeout/OFFLINE) surfaces a
        RuntimeError with actionable remediation (rewrite the source table / raise
        its Delta retention), not a bare hang."""
        provider, vsc = _provider()
        idx = MagicMock()
        idx.describe.return_value = _delta_sync_details("ONLINE_NO_PENDING_UPDATE")
        # First wait_until_ready (pre-sync) ok; final wait raises like the SDK.
        idx.wait_until_ready.side_effect = [None, Exception("did not become online")]
        vsc.get_index.return_value = idx
        with (
            patch.object(dbx, "endpoint_exists", return_value=True),
            patch.object(dbx, "index_exists", return_value=True),
            patch.object(dbx, "_source_table_delta_uuid", return_value="uuid-1"),
        ):
            with pytest.raises(RuntimeError, match="deletedFileRetentionDuration"):
                provider.create_vector_store(_vector_store())


@pytest.mark.unit
class TestSyncPipelineRunningRace:
    """``index.sync()`` 400s ("Pipeline is in state RUNNING") when a Delta-Sync
    pipeline is mid-update. That is benign — a sync is already in flight — so
    provisioning must poll past it, not fail the task."""

    def test_running_pipeline_400_is_retried_then_succeeds(self) -> None:
        provider, vsc = _provider()
        idx = MagicMock()
        idx.describe.return_value = _delta_sync_details("ONLINE_NO_PENDING_UPDATE")
        # First sync() races the pipeline; second succeeds once it's idle.
        idx.sync.side_effect = [
            dbx.AISearchBadRequest(
                "Index is not ready to sync yet. Pipeline is in state RUNNING "
                "and needs to be in one of the following states to sync: "
                "COMPLETED, FAILED, CANCELED."
            ),
            None,
        ]
        vsc.get_index.return_value = idx
        with (
            patch.object(dbx, "endpoint_exists", return_value=True),
            patch.object(dbx, "index_exists", return_value=True),
            patch.object(dbx, "_source_table_delta_uuid", return_value="uuid-1"),
            patch.object(dbx.time, "sleep"),
        ):
            provider.create_vector_store(_vector_store())
        assert idx.sync.call_count == 2
        vsc.delete_index.assert_not_called()

    def test_other_bad_request_is_reraised(self) -> None:
        provider, vsc = _provider()
        idx = MagicMock()
        idx.describe.return_value = _delta_sync_details("ONLINE_NO_PENDING_UPDATE")
        idx.sync.side_effect = dbx.AISearchBadRequest("Some other 400")
        vsc.get_index.return_value = idx
        with (
            patch.object(dbx, "endpoint_exists", return_value=True),
            patch.object(dbx, "index_exists", return_value=True),
            patch.object(dbx, "_source_table_delta_uuid", return_value="uuid-1"),
        ):
            with pytest.raises(dbx.AISearchBadRequest, match="Some other 400"):
                provider.create_vector_store(_vector_store())

    def test_presync_wait_uses_wait_for_updates(self) -> None:
        """The pre-sync gate must wait for the pipeline to be idle
        (wait_for_updates=True), not merely queryable, to avoid the race."""
        provider, vsc = _provider()
        idx = MagicMock()
        idx.describe.return_value = _delta_sync_details("ONLINE_NO_PENDING_UPDATE")
        vsc.get_index.return_value = idx
        with (
            patch.object(dbx, "endpoint_exists", return_value=True),
            patch.object(dbx, "index_exists", return_value=True),
            patch.object(dbx, "_source_table_delta_uuid", return_value="uuid-1"),
        ):
            provider.create_vector_store(_vector_store())
        # Every wait_until_ready in the healthy path waits for updates to settle.
        assert idx.wait_until_ready.call_count >= 2
        assert all(
            call.kwargs.get("wait_for_updates") is True
            for call in idx.wait_until_ready.call_args_list
        )


@pytest.mark.unit
class TestStaleHelpers:
    def test_index_is_delta_sync_true_false(self) -> None:
        assert dbx._index_is_delta_sync(_delta_sync_details()) is True
        assert dbx._index_is_delta_sync(_direct_access_details()) is False
        assert dbx._index_is_delta_sync({}) is False

    def test_read_index_source_uuid_dict_and_list_tags(self) -> None:
        d1 = _delta_sync_details()
        d1["custom_tags"] = {"dao_ai_source_delta_uuid": "abc"}
        assert dbx._read_index_source_uuid(d1) == "abc"
        d2 = _delta_sync_details()
        d2["custom_tags"] = [{"key": "dao_ai_source_delta_uuid", "value": "xyz"}]
        assert dbx._read_index_source_uuid(d2) == "xyz"
        assert dbx._read_index_source_uuid(_delta_sync_details()) is None

    def test_stale_false_when_direct_access(self) -> None:
        idx = MagicMock()
        assert (
            dbx._index_is_stale(idx, _direct_access_details("OFFLINE_FAILED"), None)
            is False
        )
