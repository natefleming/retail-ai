"""The CLI opts into argcomplete shell completion.

argcomplete discovers ``dao-ai`` via the ``# PYTHON_ARGCOMPLETE_OK`` marker and
the ``argcomplete.autocomplete(parser)`` hook in ``parse_args``. The hook is
guarded so a normal parse still works whether or not argcomplete is installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dao_ai.cli as cli


@pytest.mark.unit
def test_argcomplete_marker_present() -> None:
    # Must be within argcomplete's first-1KB scan window (it's line 1 here).
    head = Path(cli.__file__).read_text(encoding="utf-8")[:1024]
    assert "# PYTHON_ARGCOMPLETE_OK" in head


@pytest.mark.unit
def test_parse_args_still_works_normally() -> None:
    # The guarded autocomplete() call must not interfere with a real parse.
    opts = cli.parse_args(["version"])
    assert opts.command == "version"
