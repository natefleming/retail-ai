"""Store manager homepage components."""

from .alerts_tab import show_manager_alerts_tab
from .analytics_tab import show_manager_analytics_tab
from .dashboard_tab import show_manager_dashboard_tab
from .inventory_tab import show_manager_inventory_tab
from .manager_homepage import show_manager_homepage
from .operations_tab import show_manager_operations_tab
from .team_tab import show_manager_team_tab

__all__ = [
    "show_manager_homepage",
    "show_manager_dashboard_tab",
    "show_manager_alerts_tab",
    "show_manager_operations_tab",
    "show_manager_team_tab",
    "show_manager_inventory_tab",
    "show_manager_analytics_tab",
]
