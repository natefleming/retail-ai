"""
TailAdmin components for Streamlit applications.

This module provides TailAdmin-styled components and utilities for Streamlit apps.
"""

from .tailadmin_styles import (
    TAILADMIN_COLORS,
    TAILADMIN_SHADOWS,
    TAILADMIN_TYPOGRAPHY,
    create_tailadmin_button,
    create_tailadmin_card,
    create_tailadmin_metric_card,
    create_tailadmin_progress_bar,
    create_tailadmin_stat_widget,
    get_tailadmin_color,
    inject_tailadmin_css,
)

__all__ = [
    "TAILADMIN_COLORS",
    "TAILADMIN_TYPOGRAPHY",
    "TAILADMIN_SHADOWS",
    "inject_tailadmin_css",
    "get_tailadmin_color",
    "create_tailadmin_button",
    "create_tailadmin_card",
    "create_tailadmin_metric_card",
    "create_tailadmin_progress_bar",
    "create_tailadmin_stat_widget",
]
