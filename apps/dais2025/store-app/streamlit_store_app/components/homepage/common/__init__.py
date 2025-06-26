"""Common homepage components shared across roles."""

from .chat_integration import show_persistent_chat, simulate_chat_notification
from .kpi_summary import (
    show_inventory_summary,
    show_kpi_summary,
    show_manager_summary_cards,
)
from .notifications import show_notifications_modal

__all__ = [
    "show_notifications_modal",
    "show_kpi_summary",
    "show_inventory_summary",
    "show_manager_summary_cards",
    "show_persistent_chat",
    "simulate_chat_notification",
]
