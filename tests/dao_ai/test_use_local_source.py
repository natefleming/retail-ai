"""Unit tests for the --development / _use_local_source deploy resolution."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dao_ai.providers import databricks as dbx


@pytest.mark.unit
class TestUseLocalSource:
    def test_explicit_true_forces_local(self) -> None:
        # Even when installed from PyPI, --development wins.
        with patch.object(dbx, "is_published", return_value=True):
            assert dbx._use_local_source(True) is True

    def test_explicit_false_forces_pypi(self) -> None:
        # Even from an editable install, --no-development wins.
        with patch.object(dbx, "is_published", return_value=False):
            assert dbx._use_local_source(False) is False

    def test_none_auto_local_when_editable(self) -> None:
        with patch.object(dbx, "is_published", return_value=False):
            assert dbx._use_local_source(None) is True

    def test_none_auto_pypi_when_published(self) -> None:
        with patch.object(dbx, "is_published", return_value=True):
            assert dbx._use_local_source(None) is False
