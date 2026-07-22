"""Unit tests for extras threading into deploy-path codegen helpers.

Verifies the generated requirements/pyproject content carries the right
``dao-ai[<extras>]`` suffix across the Apps and MCP bundle helpers, and that
``get_installed_packages`` pins optional packages only when their extra is
requested.
"""

import pytest

from dao_ai.apps.bundle import _make_requirements_txt as apps_reqs
from dao_ai.mcp.generate import _make_requirements_txt as mcp_reqs
from dao_ai.utils import get_installed_packages


# ---------------------------------------------------------------------------
# Apps bundle requirements
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_apps_published_no_extras() -> None:
    assert apps_reqs(development=False) == "dao-ai\n"


@pytest.mark.unit
def test_apps_published_with_extras() -> None:
    assert apps_reqs(development=False, extras="[a2a,rerank]") == "dao-ai[a2a,rerank]\n"


@pytest.mark.unit
def test_apps_dev_with_extras() -> None:
    out = apps_reqs(development=True, wheel_filename="dao_ai-0.2.0.whl", extras="[a2a]")
    assert out == "./dist/dao_ai-0.2.0.whl[a2a]\n"


@pytest.mark.unit
def test_apps_dev_requires_wheel_filename() -> None:
    with pytest.raises(ValueError):
        apps_reqs(development=True)


# ---------------------------------------------------------------------------
# MCP bundle requirements — always carries the mcp extra, merges features
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_mcp_published_default_mcp_extra() -> None:
    assert mcp_reqs(development=False) == "dao-ai[mcp]\n"


@pytest.mark.unit
def test_mcp_published_merged_extras() -> None:
    assert mcp_reqs(development=False, extras="mcp,a2a") == "dao-ai[mcp,a2a]\n"


@pytest.mark.unit
def test_mcp_dev_merged_extras() -> None:
    out = mcp_reqs(development=True, wheel_filename="w.whl", extras="mcp,rerank")
    assert out == "./dist/w.whl[mcp,rerank]\n"


# ---------------------------------------------------------------------------
# get_installed_packages — optional pins gated on the extras set
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_get_installed_packages_core_only_omits_optional() -> None:
    pkgs = get_installed_packages(extras=set())
    joined = " ".join(pkgs)
    for optional in ("flashrank", "langmem", "deepagents", "ddgs", "openpyxl", "a2a-sdk"):
        assert optional not in joined, f"{optional} leaked into core pins"
    # A core package is always pinned.
    assert any(p.startswith("mlflow==") for p in pkgs)


@pytest.mark.unit
def test_get_installed_packages_includes_requested_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub version() so the gating logic is tested regardless of whether the
    # optional packages happen to be installed in the test env.
    monkeypatch.setattr("dao_ai.utils.version", lambda name: "9.9.9")
    pkgs = get_installed_packages(extras={"rerank"})
    assert any(p.startswith("flashrank==") for p in pkgs)
    # An unrequested extra's package stays out.
    assert not any(p.startswith("langmem==") for p in pkgs)


@pytest.mark.unit
def test_get_installed_packages_none_is_core_only() -> None:
    pkgs = get_installed_packages()
    assert not any(p.startswith("flashrank==") for p in pkgs)
