"""Unit tests for the --development source resolution.

``resolve_use_local_source`` (in ``dao_ai.utils``) is the single source of
truth for the ``--development`` / ``--no-development`` tri-state shared by the
CLI handlers, the deploy notebook, and the Databricks provider. The provider's
``_use_local_source`` is a thin wrapper that must stay in lock-step with it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dao_ai import utils
from dao_ai.providers import databricks as dbx


@pytest.mark.unit
class TestResolveUseLocalSource:
    def test_explicit_true_forces_local(self) -> None:
        # Even when installed from PyPI, --development wins.
        with patch.object(utils, "is_published", return_value=True):
            assert utils.resolve_use_local_source(True) is True

    def test_explicit_false_forces_pypi(self) -> None:
        # Even from an editable install, --no-development wins.
        with patch.object(utils, "is_published", return_value=False):
            assert utils.resolve_use_local_source(False) is False

    def test_none_auto_local_when_editable(self) -> None:
        with patch.object(utils, "is_published", return_value=False):
            assert utils.resolve_use_local_source(None) is True

    def test_none_auto_pypi_when_published(self) -> None:
        with patch.object(utils, "is_published", return_value=True):
            assert utils.resolve_use_local_source(None) is False


@pytest.mark.unit
class TestProviderWrapperDelegates:
    """``_use_local_source`` must return exactly what the shared resolver does."""

    @pytest.mark.parametrize("published", [True, False])
    @pytest.mark.parametrize("development", [True, False, None])
    def test_wrapper_matches_resolver(
        self, development: bool | None, published: bool
    ) -> None:
        with patch.object(utils, "is_published", return_value=published):
            assert dbx._use_local_source(development) is utils.resolve_use_local_source(
                development
            )
