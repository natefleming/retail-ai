"""Main store manager homepage coordinator."""

import streamlit as st

from .alerts_tab import show_manager_alerts_tab
from .analytics_tab import show_manager_analytics_tab
from .dashboard_tab import show_manager_dashboard_tab
from .inventory_tab import show_manager_inventory_tab
from .operations_tab import show_manager_operations_tab
from .team_tab import show_manager_team_tab


def show_manager_homepage(selected_nav: str = "Dashboard"):
    """Display homepage content for store managers based on navigation selection."""
    
    # Store current navigation in session state for VIP notification tracking
    st.session_state.current_manager_tab = selected_nav
    
    # Map navigation selection to content
    if selected_nav == "Dashboard":
        show_manager_dashboard_tab()
    elif selected_nav == "Team Management":
        show_manager_team_tab()
    elif selected_nav == "Inventory":
        show_manager_inventory_tab()
    elif selected_nav == "Sales Analytics":
        show_manager_analytics_tab()
    elif selected_nav == "Operations":
        show_manager_operations_tab()
    elif selected_nav == "Alerts":
        # Alerts tab now directly mapped
        show_manager_alerts_tab()
    else:
        # Default to Dashboard for any other selection
        show_manager_dashboard_tab()
