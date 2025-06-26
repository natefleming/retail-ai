"""
TailAdmin Visual Style Extraction and Application for Streamlit

This module extracts the visual styling elements from TailAdmin dashboard template
and provides them as reusable functions and CSS for Streamlit applications.

Key Visual Elements Extracted:
- Color schemes (brand, gray, success, error, warning palettes)
- Typography scales and font weights
- Spacing and layout patterns
- Shadow and border radius styles
- Interactive states and animations
- Component patterns (cards, buttons, tables, forms)
"""

from typing import Optional, Union

import streamlit as st
import streamlit.components.v1 as components

# ============================================================================
# TAILADMIN COLOR PALETTE EXTRACTION
# ============================================================================

TAILADMIN_COLORS = {
    # Brand Colors (Primary Blue)
    "brand": {
        "25": "#f2f7ff",
        "50": "#ecf3ff",
        "100": "#dde9ff",
        "200": "#c2d6ff",
        "300": "#9cb9ff",
        "400": "#7592ff",
        "500": "#465fff",  # Primary brand color
        "600": "#3641f5",
        "700": "#2a31d8",
        "800": "#252dae",
        "900": "#262e89",
        "950": "#161950",
    },
    # Gray Scale
    "gray": {
        "25": "#fcfcfd",
        "50": "#f9fafb",
        "100": "#f2f4f7",
        "200": "#e4e7ec",
        "300": "#d0d5dd",
        "400": "#98a2b3",
        "500": "#667085",
        "600": "#475467",
        "700": "#344054",
        "800": "#1d2939",
        "900": "#101828",
        "950": "#0c111d",
        "dark": "#1a2231",
    },
    # Success Colors
    "success": {
        "25": "#f6fef9",
        "50": "#ecfdf3",
        "100": "#d1fadf",
        "500": "#12b76a",
        "600": "#039855",
        "700": "#027a48",
    },
    # Error Colors
    "error": {"25": "#fffbfa", "50": "#fef3f2", "100": "#fee4e2", "500": "#f04438", "600": "#d92d20", "700": "#b42318"},
    # Warning Colors
    "warning": {
        "25": "#fffcf5",
        "50": "#fffaeb",
        "100": "#fef0c7",
        "500": "#f79009",
        "600": "#dc6803",
        "700": "#b54708",
    },
    # Special Colors
    "white": "#ffffff",
    "black": "#101828",
}

# ============================================================================
# TAILADMIN TYPOGRAPHY SYSTEM
# ============================================================================

TAILADMIN_TYPOGRAPHY = {
    "font_family": "Outfit, sans-serif",
    "sizes": {
        "title-2xl": {"size": "72px", "line_height": "90px"},
        "title-xl": {"size": "60px", "line_height": "72px"},
        "title-lg": {"size": "48px", "line_height": "60px"},
        "title-md": {"size": "36px", "line_height": "44px"},
        "title-sm": {"size": "30px", "line_height": "38px"},
        "title-xs": {"size": "24px", "line_height": "32px"},
        "theme-xl": {"size": "20px", "line_height": "30px"},
        "theme-sm": {"size": "14px", "line_height": "20px"},
        "theme-xs": {"size": "12px", "line_height": "18px"},
    },
}

# ============================================================================
# TAILADMIN SHADOW SYSTEM
# ============================================================================

TAILADMIN_SHADOWS = {
    "xs": "0px 1px 2px 0px rgba(16, 24, 40, 0.05)",
    "sm": "0px 1px 3px 0px rgba(16, 24, 40, 0.1), 0px 1px 2px 0px rgba(16, 24, 40, 0.06)",
    "md": "0px 4px 8px -2px rgba(16, 24, 40, 0.1), 0px 2px 4px -2px rgba(16, 24, 40, 0.06)",
    "lg": "0px 12px 16px -4px rgba(16, 24, 40, 0.08), 0px 4px 6px -2px rgba(16, 24, 40, 0.03)",
    "xl": "0px 20px 24px -4px rgba(16, 24, 40, 0.08), 0px 8px 8px -4px rgba(16, 24, 40, 0.03)",
    "focus": "0px 0px 0px 4px rgba(70, 95, 255, 0.12)",
}

# ============================================================================
# CORE CSS INJECTION FUNCTION
# ============================================================================


def inject_tailadmin_css():
    """
    Inject TailAdmin's core CSS styles into Streamlit using components.
    This should be called at the start of your Streamlit app.
    """

    css = f"""
    <style>
    /* Import TailAdmin Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap');

    /* Reset Streamlit Default Styles */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }}

    /* TailAdmin Base Styles */
    .tailadmin-app {{
        font-family: '{TAILADMIN_TYPOGRAPHY["font_family"]}';
        color: {TAILADMIN_COLORS["gray"]["700"]};
        background-color: {TAILADMIN_COLORS["gray"]["50"]};
    }}

    /* TailAdmin Card Styles */
    .tailadmin-card {{
        background: {TAILADMIN_COLORS["white"]};
        border: 1px solid {TAILADMIN_COLORS["gray"]["200"]};
            border-radius: 1rem;
            padding: 1.5rem;
        box-shadow: {TAILADMIN_SHADOWS["sm"]};
            transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }}

    .tailadmin-card:hover {{
        box-shadow: {TAILADMIN_SHADOWS["md"]};
            transform: translateY(-2px);
    }}

    /* TailAdmin Button Styles */
    .tailadmin-btn {{
        border-radius: 0.75rem;
        font-weight: 500;
        font-size: {TAILADMIN_TYPOGRAPHY["sizes"]["theme-sm"]["size"]};
        padding: 0.75rem 1.5rem;
        border: none;
        cursor: pointer;
        transition: all 0.2s ease;
        display: inline-flex;
            align-items: center;
        gap: 0.5rem;
    }}

    .tailadmin-btn-primary {{
        background-color: {TAILADMIN_COLORS["brand"]["500"]};
        color: {TAILADMIN_COLORS["white"]};
    }}

    .tailadmin-btn-primary:hover {{
        background-color: {TAILADMIN_COLORS["brand"]["600"]};
        box-shadow: {TAILADMIN_SHADOWS["sm"]};
    }}

    .tailadmin-btn-secondary {{
        background-color: {TAILADMIN_COLORS["gray"]["100"]};
        color: {TAILADMIN_COLORS["gray"]["700"]};
        border: 1px solid {TAILADMIN_COLORS["gray"]["300"]};
    }}

    .tailadmin-btn-secondary:hover {{
        background-color: {TAILADMIN_COLORS["gray"]["200"]};
    }}

    .tailadmin-btn-success {{
        background-color: {TAILADMIN_COLORS["success"]["500"]};
        color: {TAILADMIN_COLORS["white"]};
    }}

    .tailadmin-btn-warning {{
        background-color: {TAILADMIN_COLORS["warning"]["500"]};
        color: {TAILADMIN_COLORS["white"]};
    }}

    .tailadmin-btn-error {{
        background-color: {TAILADMIN_COLORS["error"]["500"]};
        color: {TAILADMIN_COLORS["white"]};
    }}

    /* TailAdmin Badge Styles */
    .tailadmin-badge {{
        display: inline-flex;
            align-items: center;
            gap: 0.25rem;
        padding: 0.25rem 0.75rem;
            border-radius: 9999px;
        font-size: {TAILADMIN_TYPOGRAPHY["sizes"]["theme-xs"]["size"]};
            font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }}

    .tailadmin-badge-success {{
        background-color: {TAILADMIN_COLORS["success"]["100"]};
        color: {TAILADMIN_COLORS["success"]["700"]};
    }}

    .tailadmin-badge-warning {{
        background-color: {TAILADMIN_COLORS["warning"]["100"]};
        color: {TAILADMIN_COLORS["warning"]["700"]};
    }}

    .tailadmin-badge-error {{
        background-color: {TAILADMIN_COLORS["error"]["100"]};
        color: {TAILADMIN_COLORS["error"]["700"]};
    }}

    .tailadmin-badge-info {{
        background-color: {TAILADMIN_COLORS["brand"]["100"]};
        color: {TAILADMIN_COLORS["brand"]["700"]};
    }}

    .tailadmin-badge-secondary {{
        background-color: {TAILADMIN_COLORS["gray"]["100"]};
        color: {TAILADMIN_COLORS["gray"]["700"]};
    }}

    /* TailAdmin Alert Styles */
    .tailadmin-alert {{
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        border-left: 4px solid;
    }}

    .tailadmin-alert-success {{
        background-color: {TAILADMIN_COLORS["success"]["50"]};
        border-left-color: {TAILADMIN_COLORS["success"]["500"]};
        color: {TAILADMIN_COLORS["success"]["700"]};
    }}

    .tailadmin-alert-warning {{
        background-color: {TAILADMIN_COLORS["warning"]["50"]};
        border-left-color: {TAILADMIN_COLORS["warning"]["500"]};
        color: {TAILADMIN_COLORS["warning"]["700"]};
    }}

    .tailadmin-alert-error {{
        background-color: {TAILADMIN_COLORS["error"]["50"]};
        border-left-color: {TAILADMIN_COLORS["error"]["500"]};
        color: {TAILADMIN_COLORS["error"]["700"]};
    }}

    .tailadmin-alert-info {{
        background-color: {TAILADMIN_COLORS["brand"]["50"]};
        border-left-color: {TAILADMIN_COLORS["brand"]["500"]};
        color: {TAILADMIN_COLORS["brand"]["700"]};
    }}

        /* TailAdmin Table Styles */
    .tailadmin-table {{
        width: 100%;
        border-collapse: collapse;
        background: {TAILADMIN_COLORS["white"]};
        border-radius: 0.75rem;
            overflow: hidden;
        box-shadow: {TAILADMIN_SHADOWS["sm"]};
    }}

    .tailadmin-table th {{
        background-color: {TAILADMIN_COLORS["gray"]["50"]};
        color: {TAILADMIN_COLORS["gray"]["700"]};
        font-weight: 600;
        padding: 1rem;
        text-align: left;
        font-size: {TAILADMIN_TYPOGRAPHY["sizes"]["theme-sm"]["size"]};
        border-bottom: 1px solid {TAILADMIN_COLORS["gray"]["200"]};
    }}

    .tailadmin-table td {{
        padding: 1rem;
        border-bottom: 1px solid {TAILADMIN_COLORS["gray"]["100"]};
        color: {TAILADMIN_COLORS["gray"]["600"]};
    }}

    .tailadmin-table tbody tr:hover {{
        background-color: {TAILADMIN_COLORS["gray"]["25"]};
    }}

    /* TailAdmin Form Styles */
    .tailadmin-input {{
        width: 100%;
        padding: 0.75rem 1rem;
        border: 1px solid {TAILADMIN_COLORS["gray"]["300"]};
        border-radius: 0.5rem;
        font-size: {TAILADMIN_TYPOGRAPHY["sizes"]["theme-sm"]["size"]};
        transition: all 0.2s ease;
        background-color: {TAILADMIN_COLORS["white"]};
    }}

    .tailadmin-input:focus {{
        outline: none;
        border-color: {TAILADMIN_COLORS["brand"]["500"]};
        box-shadow: {TAILADMIN_SHADOWS["focus"]};
    }}

    /* TailAdmin Sidebar Styles */
    .tailadmin-sidebar {{
        background: {TAILADMIN_COLORS["white"]};
        border-right: 1px solid {TAILADMIN_COLORS["gray"]["200"]};
        height: 100vh;
        width: 290px;
        padding: 1.25rem;
    }}

    .tailadmin-menu-item {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
            border-radius: 0.5rem;
        color: {TAILADMIN_COLORS["gray"]["700"]};
        text-decoration: none;
            font-weight: 500;
        font-size: {TAILADMIN_TYPOGRAPHY["sizes"]["theme-sm"]["size"]};
        transition: all 0.2s ease;
        margin-bottom: 0.25rem;
    }}

    .tailadmin-menu-item:hover {{
        background-color: {TAILADMIN_COLORS["gray"]["100"]};
    }}

    .tailadmin-menu-item.active {{
        background-color: {TAILADMIN_COLORS["brand"]["50"]};
        color: {TAILADMIN_COLORS["brand"]["500"]};
    }}

    /* TailAdmin Header Styles */
    .tailadmin-header {{
        background: {TAILADMIN_COLORS["white"]};
        border-bottom: 1px solid {TAILADMIN_COLORS["gray"]["200"]};
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    /* Dark Mode Support */
    .dark .tailadmin-card {{
        background-color: {TAILADMIN_COLORS["gray"]["dark"]};
        border-color: {TAILADMIN_COLORS["gray"]["800"]};
        color: {TAILADMIN_COLORS["gray"]["300"]};
    }}

    .dark .tailadmin-table {{
        background-color: {TAILADMIN_COLORS["gray"]["dark"]};
    }}

    .dark .tailadmin-table th {{
        background-color: {TAILADMIN_COLORS["gray"]["800"]};
        color: {TAILADMIN_COLORS["gray"]["300"]};
    }}

    /* Responsive Design */
    @media (max-width: 768px) {{
        .tailadmin-card {{
            padding: 1rem;
        }}

        .tailadmin-btn {{
            padding: 0.5rem 1rem;
            font-size: {TAILADMIN_TYPOGRAPHY["sizes"]["theme-xs"]["size"]};
        }}
    }}

    /* Custom Scrollbar */
    .tailadmin-scrollbar::-webkit-scrollbar {{
        width: 6px;
    }}

    .tailadmin-scrollbar::-webkit-scrollbar-track {{
        background: {TAILADMIN_COLORS["gray"]["100"]};
        border-radius: 3px;
    }}

    .tailadmin-scrollbar::-webkit-scrollbar-thumb {{
        background: {TAILADMIN_COLORS["gray"]["400"]};
        border-radius: 3px;
    }}

    .tailadmin-scrollbar::-webkit-scrollbar-thumb:hover {{
        background: {TAILADMIN_COLORS["gray"]["500"]};
    }}
    </style>
    """

    # Use components.html for better CSS injection
    components.html(css, height=0)


# ============================================================================
# COMPONENT STYLING FUNCTIONS
# ============================================================================


def get_tailadmin_color(category: str, shade: str = "500") -> str:
    """
    Get a TailAdmin color value by category and shade.

    Args:
        category: Color category (brand, gray, success, error, warning)
        shade: Color shade (25, 50, 100, 200, etc.)

    Returns:
        Hex color value
    """
    return TAILADMIN_COLORS.get(category, {}).get(shade, TAILADMIN_COLORS["gray"]["500"])


def create_tailadmin_button(
    text: str, button_type: str = "primary", icon: str | None = None, disabled: bool = False, size: str = "medium"
) -> str:
    """
    Create a TailAdmin-styled button HTML.

    Args:
        text: Button text
        button_type: Button style (primary, secondary, success, warning, error)
        icon: Optional icon (HTML/emoji)
        disabled: Whether button is disabled
        size: Button size (small, medium, large)

    Returns:
        HTML string for the button
    """

    size_classes = {
        "small": "padding: 0.5rem 1rem; font-size: 0.875rem;",
        "medium": "padding: 0.75rem 1.5rem; font-size: 0.875rem;",
        "large": "padding: 1rem 2rem; font-size: 1rem;",
    }

    icon_html = f"<span>{icon}</span>" if icon else ""
    disabled_style = "opacity: 0.6; cursor: not-allowed;" if disabled else ""

    return f"""
    <button class="tailadmin-btn tailadmin-btn-{button_type}"
            style="{size_classes.get(size, size_classes["medium"])} {disabled_style}"
            {"disabled" if disabled else ""}>
        {icon_html}
        <span>{text}</span>
    </button>
    """


def create_tailadmin_card(
    content: str, title: str | None = None, actions: str | None = None, card_class: str = ""
) -> str:
    """
    Create a TailAdmin-styled card component.

    Args:
        content: Card content (HTML)
        title: Optional card title
        actions: Optional action buttons (HTML)
        card_class: Additional CSS classes

    Returns:
        HTML string for the card
    """

    title_html = (
        f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid {TAILADMIN_COLORS["gray"]["200"]};
        ">
            <h3 style="
                font-size: 1.25rem;
                font-weight: 600;
                color: {TAILADMIN_COLORS["gray"]["900"]};
                margin: 0;
            ">{title}</h3>
            {actions or ""}
        </div>
    """
        if title
        else ""
    )

    return f"""
    <div class="tailadmin-card {card_class}">
        {title_html}
        {content}
    </div>
    """


def create_tailadmin_metric_card(
    icon: str,
    value: str,
    label: str,
    change: str | None = None,
    change_type: str = "positive",
    trend_data: list[float] | None = None,
) -> str:
    """
    Create a TailAdmin-styled metric card with optional trend indicator.

    Args:
        icon: Icon (HTML/emoji)
        value: Metric value
        label: Metric label
        change: Change percentage
        change_type: Change direction (positive/negative)
        trend_data: Optional trend data for sparkline

    Returns:
        HTML string for the metric card
    """

    change_html = ""
    if change:
        change_icon = "📈" if change_type == "positive" else "📉"
        change_color = (
            TAILADMIN_COLORS["success"]["500"] if change_type == "positive" else TAILADMIN_COLORS["error"]["500"]
        )
        change_bg = (
            TAILADMIN_COLORS["success"]["100"] if change_type == "positive" else TAILADMIN_COLORS["error"]["100"]
        )

        change_html = f"""
            <div style="
                display: inline-flex;
            align-items: center;
                gap: 0.25rem;
                background-color: {change_bg};
                color: {change_color};
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 500;
                margin-top: 0.75rem;
            ">
                <span>{change_icon}</span>
                <span>{change}</span>
            </div>
        """

    # Simple trend indicator if trend_data provided
    trend_html = ""
    if trend_data and len(trend_data) > 1:
        trend_direction = "↗️" if trend_data[-1] > trend_data[0] else "↘️"
        trend_html = f"""
            <div style="
                position: absolute;
                top: 1rem;
                right: 1rem;
                font-size: 1.5rem;
                opacity: 0.6;
            ">{trend_direction}</div>
        """

    return f"""
    <div class="tailadmin-card" style="position: relative; height: 100%;">
        {trend_html}
        <div style="
            display: flex;
            height: 3.5rem;
            width: 3.5rem;
            align-items: center;
            justify-content: center;
            border-radius: 1rem;
            background-color: {TAILADMIN_COLORS["brand"]["50"]};
            margin-bottom: 1.5rem;
            font-size: 1.75rem;
        ">
            {icon}
        </div>

        <div style="
            font-size: 0.875rem;
            color: {TAILADMIN_COLORS["gray"]["600"]};
            font-weight: 500;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.025em;
        ">
            {label}
        </div>

        <div style="
            font-size: 2.25rem;
            font-weight: 700;
            color: {TAILADMIN_COLORS["gray"]["900"]};
            line-height: 1.2;
            margin-bottom: 0.5rem;
        ">
            {value}
        </div>

            {change_html}
        </div>
    """


def create_tailadmin_progress_bar(
    percentage: float, label: str, color: str = "brand", show_percentage: bool = True
) -> str:
    """
    Create a TailAdmin-styled progress bar.

    Args:
        percentage: Progress percentage (0-100)
        label: Progress label
        color: Color theme (brand, success, warning, error)
        show_percentage: Whether to show percentage text

    Returns:
        HTML string for the progress bar
    """

    percentage_text = f"<span style='font-weight: 600;'>{percentage}%</span>" if show_percentage else ""

    return f"""
    <div style="margin-bottom: 1rem;">
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        ">
            <span style="
                font-size: 0.875rem;
                font-weight: 500;
                color: {TAILADMIN_COLORS["gray"]["700"]};
            ">{label}</span>
            {percentage_text}
        </div>
        <div style="
            width: 100%;
            height: 0.5rem;
            background-color: {TAILADMIN_COLORS["gray"]["200"]};
            border-radius: 9999px;
            overflow: hidden;
        ">
            <div style="
                height: 100%;
                width: {percentage}%;
                background-color: {get_tailadmin_color(color)};
                border-radius: 9999px;
                transition: width 0.3s ease;
            "></div>
        </div>
        </div>
    """


def create_tailadmin_stat_widget(title: str, stats: list[dict], chart_data: str | None = None) -> str:
    """
    Create a TailAdmin-styled statistics widget with multiple metrics.

    Args:
        title: Widget title
        stats: List of stat dictionaries with keys: label, value, change, change_type
        chart_data: Optional chart/graph HTML

    Returns:
        HTML string for the stats widget
    """

    stats_html = ""
    for stat in stats:
        change_indicator = ""
        if stat.get("change"):
            change_color = (
                TAILADMIN_COLORS["success"]["500"]
                if stat.get("change_type") == "positive"
                else TAILADMIN_COLORS["error"]["500"]
            )
            change_icon = "↗" if stat.get("change_type") == "positive" else "↘"
            change_indicator = f"""
                <span style="color: {change_color}; font-size: 0.875rem; font-weight: 500;">
                    {change_icon} {stat["change"]}
                </span>
            """

        stats_html += f"""
            <div style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0.75rem 0;
                border-bottom: 1px solid {TAILADMIN_COLORS["gray"]["100"]};
            ">
                <div>
                    <div style="
                        font-size: 0.875rem;
                        color: {TAILADMIN_COLORS["gray"]["600"]};
                        margin-bottom: 0.25rem;
                    ">{stat["label"]}</div>
                    <div style="
                        font-size: 1.5rem;
                        font-weight: 700;
                        color: {TAILADMIN_COLORS["gray"]["900"]};
                    ">{stat["value"]}</div>
                </div>
                <div>{change_indicator}</div>
            </div>
        """

    chart_section = (
        f"""
        <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid {TAILADMIN_COLORS["gray"]["200"]};">
            {chart_data}
        </div>
    """
        if chart_data
        else ""
    )

    return create_tailadmin_card(content=f"{stats_html}{chart_section}", title=title)
