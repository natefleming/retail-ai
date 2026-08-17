"""Project code declared explicitly via ``app.code_paths``.

Unlike the colocated ``src/`` convention, a ``code_paths`` entry is named in the
config. ``prepend_code_paths_to_sys_path`` puts each resolved entry's *parent* on
``sys.path`` — for the entry ``from_git_lib`` that parent is the config's own
directory, so the package keeps its own name in imports:
``from_git_lib.pricing.create_discount_tool``.
"""

from from_git_lib.pricing import create_discount_tool

__all__ = ["create_discount_tool"]
