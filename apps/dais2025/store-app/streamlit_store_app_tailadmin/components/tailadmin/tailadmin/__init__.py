"""
TailAdmin Components Package

This package contains TailAdmin-styled components and utilities for Streamlit applications.

Components:
- tailadmin_styles: Core CSS styles and color palette
- tailadmin_components: Basic TailAdmin-styled components
- tailadmin_components_enhanced: Advanced interactive components
"""

from .tailadmin_components import (
    create_tailadmin_layout,
    display_tailadmin_alert,
    display_tailadmin_badge,
    display_tailadmin_chart,
    display_tailadmin_info_bar,
    display_tailadmin_metric_card,
    display_tailadmin_metrics_grid,
    display_tailadmin_navigation,
    display_tailadmin_table,
)
from .tailadmin_components_enhanced import (
    create_tailadmin_form,
    create_tailadmin_form_field,
    create_tailadmin_header,
    create_tailadmin_plotly_chart,
    create_tailadmin_sidebar,
    display_enhanced_metrics_grid,
    display_tailadmin_chart_card,
    display_tailadmin_data_table,
    display_tailadmin_notification,
    finalize_tailadmin_app,
    initialize_tailadmin_app,
    show_tailadmin_showcase,
)
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
    # Styles
    "inject_tailadmin_css",
    "get_tailadmin_color",
    "create_tailadmin_button",
    "create_tailadmin_card",
    "create_tailadmin_metric_card",
    "create_tailadmin_progress_bar",
    "create_tailadmin_stat_widget",
    "TAILADMIN_COLORS",
    "TAILADMIN_TYPOGRAPHY",
    "TAILADMIN_SHADOWS",
    # Basic Components
    "display_tailadmin_metrics_grid",
    "display_tailadmin_metric_card",
    "display_tailadmin_chart",
    "display_tailadmin_table",
    "display_tailadmin_alert",
    "display_tailadmin_badge",
    "display_tailadmin_info_bar",
    "display_tailadmin_navigation",
    "create_tailadmin_layout",
    # Enhanced Components
    "initialize_tailadmin_app",
    "create_tailadmin_header",
    "create_tailadmin_sidebar",
    "display_enhanced_metrics_grid",
    "create_tailadmin_plotly_chart",
    "display_tailadmin_chart_card",
    "display_tailadmin_data_table",
    "create_tailadmin_form_field",
    "create_tailadmin_form",
    "display_tailadmin_notification",
    "finalize_tailadmin_app",
    "show_tailadmin_showcase",
]
