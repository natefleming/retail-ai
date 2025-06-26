"""
Centralized CSS styles package for the Streamlit Store App.

This package consolidates all CSS styling to avoid scattered st.markdown() calls
throughout the application. All styles are loaded once at app initialization.

Usage:
    from styles import load_all_styles
    load_all_styles()

Style Organization:
    - base.py: Core app styles, typography, and global overrides
    - components.py: Reusable component styles (cards, metrics, etc.)
    - dashboard.py: Dashboard-specific styles for all role dashboards
    - homepage.py: Homepage and navigation styles
    - theme.py: Dark/light theme variables and utilities

To add new styles:
    1. Add CSS rules to the appropriate module (base, components, dashboard, etc.)
    2. Use theme variables from theme.py for consistent theming
    3. Test in both light and dark modes
    4. Document any new CSS classes or patterns

To modify existing styles:
    1. Locate the style in the appropriate module
    2. Make changes in the centralized location only
    3. Do not add duplicate CSS rules in components
"""

from .base import get_base_styles
from .components import get_component_styles
from .dashboard import get_dashboard_styles
from .homepage import get_homepage_styles
from .theme import get_theme_variables


def load_all_styles():
    """
    Load all centralized CSS styles for the application.
    
    This function should be called once at app initialization to inject
    all CSS rules. It consolidates styles from all modules to ensure
    consistent theming and eliminate redundant CSS.
    """
    import streamlit as st
    
    # Get current theme state - should already be initialized in app.py
    # Using False as fallback to match app.py initialization
    dark_mode = st.session_state.get("dark_mode", False)
    theme_vars = get_theme_variables(dark_mode)
    
    # Consolidate all styles
    all_styles = f"""
    <style>
    {get_base_styles(theme_vars)}
    {get_component_styles(theme_vars)}
    {get_dashboard_styles(theme_vars)}
    {get_homepage_styles(theme_vars)}
    </style>
    """
    
    # Inject all styles at once
    st.markdown(all_styles, unsafe_allow_html=True)


__all__ = [
    "load_all_styles",
    "get_theme_variables",
    "get_base_styles",
    "get_component_styles", 
    "get_dashboard_styles",
    "get_homepage_styles"
]

# Styles package for centralized CSS management 