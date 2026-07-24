"""Guard the local ``dao-ai deploy`` dependency set against the protobuf/pyspark
regression tracked in issue #211.

``dao-ai deploy --target model_serving`` logs the MLflow model in-process on the
developer's machine. That path imports spark-connect (via MLflow schema inference
and, for UC-function configs, ``DatabricksFunctionClient`` building a
``DatabricksSession``). A *standalone* ``pyspark`` dependency resolves to 4.x, whose
spark-connect proto gencode requires ``protobuf>=6.33`` — which collides with the
always-installed ``databricks-ai-search``'s ``protobuf<6`` cap and crashes with a
``VersionError``.

``databricks-connect`` (already a core transitive dep via ``databricks-langchain``)
bundles its own protobuf-5-compatible pyspark, so the fix is to never declare a
standalone ``pyspark``. These tests assert that invariant in both ``pyproject.toml``
and the resolved ``uv.lock`` so it can't silently regress.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

_REPO_ROOT: Path = Path(__file__).parents[2]
_PYPROJECT: Path = _REPO_ROOT / "pyproject.toml"
_UV_LOCK: Path = _REPO_ROOT / "uv.lock"


def _pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))


def _requirement_names(specs: list[str]) -> list[str]:
    """Canonical (lowercased) distribution names from requirement strings."""
    return [Requirement(s).name.lower() for s in specs]


class TestPyprojectHasNoStandalonePyspark:
    """Neither the ``databricks`` extra nor group may declare a bare ``pyspark``."""

    def test_databricks_extra_has_no_pyspark(self) -> None:
        extras = _pyproject()["project"]["optional-dependencies"]
        assert "pyspark" not in _requirement_names(extras["databricks"])

    def test_databricks_group_has_no_pyspark(self) -> None:
        groups = _pyproject()["dependency-groups"]
        assert "pyspark" not in _requirement_names(groups["databricks"])

    def test_no_section_declares_standalone_pyspark(self) -> None:
        """Sweep every extra + group: standalone pyspark must not reappear
        anywhere (databricks-connect provides the pyspark namespace)."""
        proj = _pyproject()
        extras = proj["project"].get("optional-dependencies", {})
        groups = proj.get("dependency-groups", {})
        core = proj["project"].get("dependencies", [])

        offenders: list[str] = []
        for name, specs in [("dependencies", core)]:
            if "pyspark" in _requirement_names(specs):
                offenders.append(name)
        for name, specs in extras.items():
            if "pyspark" in _requirement_names(specs):
                offenders.append(f"optional-dependencies.{name}")
        for name, specs in groups.items():
            # groups may contain include-group dicts; keep only str specs.
            str_specs = [s for s in specs if isinstance(s, str)]
            if "pyspark" in _requirement_names(str_specs):
                offenders.append(f"dependency-groups.{name}")

        assert not offenders, (
            "standalone pyspark reintroduced in: "
            + ", ".join(offenders)
            + " — pyspark 4.x drags in protobuf>=6.33 and breaks local "
            "`dao-ai deploy`; rely on databricks-connect's bundled pyspark "
            "(see issue #211)."
        )

    def test_databricks_connect_still_declared(self) -> None:
        """The fix keeps databricks-connect — it supplies the pyspark namespace."""
        extras = _pyproject()["project"]["optional-dependencies"]
        assert "databricks-connect" in _requirement_names(extras["databricks"])


class TestUvLockIsProtobuf5Coherent:
    """The resolved lock must not carry standalone pyspark 4.x or protobuf>=6."""

    def _lock_packages(self) -> list[dict]:
        return tomllib.loads(_UV_LOCK.read_text(encoding="utf-8"))["package"]

    def test_no_standalone_pyspark_package(self) -> None:
        names = {p["name"] for p in self._lock_packages()}
        assert "pyspark" not in names, (
            "uv.lock contains a top-level standalone pyspark package; it must "
            "be provided only by databricks-connect's bundle (issue #211)."
        )

    def test_protobuf_resolves_below_6(self) -> None:
        protobufs = [
            p for p in self._lock_packages() if p["name"] == "protobuf"
        ]
        assert protobufs, "protobuf missing from uv.lock"
        for p in protobufs:
            assert Version(p["version"]) < Version("6"), (
                f"protobuf resolved to {p['version']}; must stay <6 to satisfy "
                "databricks-ai-search and databricks-connect's spark-connect "
                "gencode (issue #211)."
            )

    def test_databricks_connect_present(self) -> None:
        names = {p["name"] for p in self._lock_packages()}
        assert "databricks-connect" in names
