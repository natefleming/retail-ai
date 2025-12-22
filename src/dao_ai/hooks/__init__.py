"""
Hook utilities for DAO AI.

For validation hooks, use middleware instead:
- dao_ai.middleware.UserIdValidationMiddleware
- dao_ai.middleware.ThreadIdValidationMiddleware
- dao_ai.middleware.FilterLastHumanMessageMiddleware
"""

from dao_ai.hooks.core import (
    create_hooks,
    null_hook,
    null_initialization_hook,
    null_shutdown_hook,
)

__all__ = [
    "create_hooks",
    "null_hook",
    "null_initialization_hook",
    "null_shutdown_hook",
]
