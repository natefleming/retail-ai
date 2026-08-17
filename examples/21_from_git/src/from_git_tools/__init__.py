"""Tools shipped inside the repository, under the colocated ``src/`` convention.

``src/`` beside the config needs no declaration: ``AppConfig.from_file`` (and so
``from_git``) discovers it and prepends it to ``sys.path`` at load time, which is
why ``from_git_tools.aisle.find_aisle`` imports prefix-free from a config that was
never checked out by hand.
"""

from from_git_tools.aisle import find_aisle

__all__ = ["find_aisle"]
