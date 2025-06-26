"""VP Retail Operations homepage coordinator."""

import streamlit as st

from .ai_insights_tab import show_ai_insights_tab
from .executive_dashboard_tab import show_executive_dashboard_tab
from .geographical_analysis_tab import show_geographical_analysis_tab
from .performance_metrics_tab import show_performance_metrics_tab
from .strategic_insights_tab import show_strategic_insights_tab


def show_vp_homepage(selected_nav: str = "Executive Dashboard"):
    """Display homepage content for VP of Retail Operations based on navigation selection."""
    
    # Map navigation selection to content
    if selected_nav == "Executive Dashboard":
        show_executive_dashboard_tab()
    elif selected_nav == "Regional Analytics":
        show_geographical_analysis_tab()
    elif selected_nav == "Store Performance":
        show_performance_metrics_tab()
    elif selected_nav == "Strategic Planning":
        show_strategic_insights_tab()
    elif selected_nav == "Reports":
        show_ai_insights_tab()
    else:
        # Default to Executive Dashboard for any other selection
        show_executive_dashboard_tab()
