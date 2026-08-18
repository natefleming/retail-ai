"""Every shipped example YAML loads through ``AppConfig``.

The examples are the documented surface of the config schema, and they are
edited whenever a field is renamed or a default moves. Nothing checked that
they still parse: ``test_git_source.py`` walks the same tree but only asserts
the *discovery* heuristic recognizes them as dao-ai configs, which a config
with a bad field passes fine.

``initialize=False`` keeps this offline — no workspace client, no resource
resolution — so it stays a pure schema check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dao_ai.config import AppConfig
from dao_ai.config_vars import WorkspaceVariableError

EXAMPLES: Path = Path(__file__).parents[2] / "examples"

# Files that live under examples/ but are not dao-ai configs. Same set as
# ``test_git_source.py::test_every_shipped_example_config_is_recognized``.
NOT_CONFIGS: frozenset[str] = frozenset(
    {"examples.yaml", "app.yaml", "environment.yaml"}
)

# Known-broken example, failing before this test existed: every agent requires
# a ``model``, and ``general_agent`` declares none. Listed rather than skipped
# silently so fixing it is a one-line deletion here.
KNOWN_INVALID: frozenset[str] = frozenset(
    {"12_middleware/tool_selector_middleware.yaml"}
)


def _example_configs() -> list[Path]:
    if not EXAMPLES.is_dir():
        return []
    return sorted(
        path
        for path in EXAMPLES.rglob("*.yaml")
        if path.name not in NOT_CONFIGS
        and path.relative_to(EXAMPLES).as_posix() not in KNOWN_INVALID
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_path", _example_configs(), ids=lambda p: p.relative_to(EXAMPLES).as_posix()
)
def test_every_shipped_example_validates(config_path: Path) -> None:
    try:
        AppConfig.from_file(config_path, initialize=False)
    except WorkspaceVariableError as exc:
        # A few examples use ``${workspace.*}`` (e.g. current_user), which is
        # resolved at load time and needs a live WorkspaceClient regardless of
        # ``initialize=False``. Skip rather than fail so the suite stays offline
        # — the parse itself got far enough to reach variable resolution.
        pytest.skip(f"needs workspace auth: {exc}")


@pytest.mark.unit
def test_the_example_walk_actually_found_configs() -> None:
    """Guards the guard: an empty parametrize list would make this file green
    while checking nothing."""
    if not EXAMPLES.is_dir():
        pytest.skip("examples/ not present in this checkout")
    assert len(_example_configs()) > 50
