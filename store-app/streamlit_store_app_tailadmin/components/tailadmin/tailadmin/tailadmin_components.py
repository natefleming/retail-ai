"""TailAdmin-style components for Streamlit integration."""

import plotly.express as px
import streamlit as st


def display_tailadmin_metrics_grid(metrics):
    """
    Display a grid of TailAdmin-style metric cards.

    Args:
        metrics (list): List of metric dictionaries with keys:
            - icon: HTML/emoji icon
            - value: The numeric value
            - label: The metric label
            - change: Optional percentage change
            - change_type: 'positive' or 'negative'
    """
    # Create grid layout
    num_metrics = len(metrics)
    if num_metrics <= 2:
        cols = st.columns(num_metrics)
    elif num_metrics <= 4:
        cols = st.columns(2)
    else:
        cols = st.columns(min(num_metrics, 4))

    for i, metric in enumerate(metrics):
        with cols[i % len(cols)]:
            display_tailadmin_metric_card(**metric)


def display_tailadmin_metric_card(
    icon, value, label, change=None, change_type="positive"
):
    """Display a single TailAdmin-style metric card."""

    # Format change indicator
    change_html = ""
    if change:
        change_icon = "📈" if change_type == "positive" else "📉"
        change_color = "#10b981" if change_type == "positive" else "#ef4444"
        change_bg = (
            "rgba(16, 185, 129, 0.1)"
            if change_type == "positive"
            else "rgba(239, 68, 68, 0.1)"
        )

        change_html = f"""
            <div style="
                display: inline-flex;
                align-items: center;
                gap: 0.25rem;
                background-color: {change_bg};
                color: {change_color};
                padding: 0.25rem 0.5rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 500;
                margin-top: 0.5rem;
            ">
                <span>{change_icon}</span>
                <span>{change}</span>
            </div>
        """

    # Create the metric card
    st.markdown(
        f"""
        <div style="
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            transition: all 0.3s ease;
            margin-bottom: 1rem;
            height: 100%;
        ">
            <div style="
                display: flex;
                height: 3rem;
                width: 3rem;
                align-items: center;
                justify-content: center;
                border-radius: 0.75rem;
                background-color: #f1f5f9;
                margin-bottom: 1.25rem;
                font-size: 1.5rem;
            ">
                {icon}
            </div>
            
            <div style="
                font-size: 0.875rem;
                color: #64748b;
                font-weight: 500;
                margin-bottom: 0.5rem;
            ">
                {label}
            </div>
            
            <div style="
                font-size: 2rem;
                font-weight: 700;
                color: #1e293b;
                line-height: 1.2;
            ">
                {value}
            </div>
            
            {change_html}
        </div>
    """,
        unsafe_allow_html=True,
    )


def display_tailadmin_chart(chart_data, chart_type="line", title="Chart", height=400):
    """
    Display a TailAdmin-style chart with proper styling.

    Args:
        chart_data: Chart data (depends on chart_type)
        chart_type: 'line', 'bar', 'pie', 'area'
        title: Chart title
        height: Chart height in pixels
    """

    # Chart container with TailAdmin styling
    st.markdown(
        f"""
        <div style="
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            margin-bottom: 1.5rem;
        ">
            <div style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 1.5rem;
            ">
                <h3 style="
                    font-size: 1.25rem;
                    font-weight: 600;
                    color: #1e293b;
                    margin: 0;
                ">
                    {title}
                </h3>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Configure chart based on type
    if chart_type == "line":
        fig = px.line(
            chart_data,
            x=chart_data.columns[0],
            y=chart_data.columns[1],
            color_discrete_sequence=["#3b82f6"],
        )
    elif chart_type == "bar":
        fig = px.bar(
            chart_data,
            x=chart_data.columns[0],
            y=chart_data.columns[1],
            color_discrete_sequence=["#3b82f6"],
        )
    elif chart_type == "pie":
        fig = px.pie(
            chart_data,
            values=chart_data.columns[1],
            names=chart_data.columns[0],
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
    elif chart_type == "area":
        fig = px.area(
            chart_data,
            x=chart_data.columns[0],
            y=chart_data.columns[1],
            color_discrete_sequence=["#3b82f6"],
        )

    # Apply TailAdmin styling to chart
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter",
        font_color="#64748b",
        showlegend=False,
    )

    # Style axes
    fig.update_xaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickcolor="#e2e8f0")

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def display_tailadmin_table(data, title="Table", actions=None):
    """
    Display a TailAdmin-style table.

    Args:
        data: DataFrame to display
        title: Table title
        actions: Optional HTML for action buttons
    """

    actions_html = f"<div>{actions}</div>" if actions else ""

    st.markdown(
        f"""
        <div style="
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 1rem;
            overflow: hidden;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            margin-bottom: 1.5rem;
        ">
            <div style="
                background-color: #f8fafc;
                padding: 1.5rem;
                border-bottom: 1px solid #e2e8f0;
                display: flex;
                align-items: center;
                justify-content: space-between;
            ">
                <h3 style="
                    font-size: 1.25rem;
                    font-weight: 600;
                    color: #1e293b;
                    margin: 0;
                ">
                    {title}
                </h3>
                {actions_html}
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Display the dataframe with custom styling
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
        height=min(400, len(data) * 35 + 100),
    )


def display_tailadmin_alert(message, alert_type="info", icon=None, dismissible=False):
    """
    Display a TailAdmin-style alert.

    Args:
        message: Alert message
        alert_type: 'success', 'warning', 'error', 'info'
        icon: Custom icon (optional)
        dismissible: Whether alert can be dismissed
    """

    # Default icons
    icons = {"success": "✅", "warning": "⚠️", "error": "❌", "info": "ℹ️"}

    # Color schemes
    colors = {
        "success": {
            "bg": "rgba(16, 185, 129, 0.1)",
            "border": "rgba(16, 185, 129, 0.2)",
            "text": "#059669",
        },
        "warning": {
            "bg": "rgba(245, 158, 11, 0.1)",
            "border": "rgba(245, 158, 11, 0.2)",
            "text": "#d97706",
        },
        "error": {
            "bg": "rgba(239, 68, 68, 0.1)",
            "border": "rgba(239, 68, 68, 0.2)",
            "text": "#dc2626",
        },
        "info": {
            "bg": "rgba(59, 130, 246, 0.1)",
            "border": "rgba(59, 130, 246, 0.2)",
            "text": "#2563eb",
        },
    }

    alert_icon = icon or icons.get(alert_type, "ℹ️")
    color_scheme = colors.get(alert_type, colors["info"])

    dismiss_btn = (
        """
        <button style="
            background: none;
            border: none;
            color: inherit;
            cursor: pointer;
            font-size: 1rem;
            padding: 0;
            margin-left: auto;
        ">✕</button>
    """
        if dismissible
        else ""
    )

    st.markdown(
        f"""
        <div style="
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            background-color: {color_scheme["bg"]};
            border: 1px solid {color_scheme["border"]};
            color: {color_scheme["text"]};
        ">
            <span style="flex-shrink: 0;">{alert_icon}</span>
            <span style="flex: 1;">{message}</span>
            {dismiss_btn}
        </div>
    """,
        unsafe_allow_html=True,
    )


def display_tailadmin_badge(text, badge_type="secondary"):
    """
    Display a TailAdmin-style badge.

    Args:
        text: Badge text
        badge_type: 'primary', 'secondary', 'success', 'warning', 'error'
    """

    colors = {
        "primary": {"bg": "rgba(59, 130, 246, 0.1)", "text": "#2563eb"},
        "secondary": {"bg": "#f1f5f9", "text": "#475569"},
        "success": {"bg": "rgba(16, 185, 129, 0.1)", "text": "#059669"},
        "warning": {"bg": "rgba(245, 158, 11, 0.1)", "text": "#d97706"},
        "error": {"bg": "rgba(239, 68, 68, 0.1)", "text": "#dc2626"},
    }

    color_scheme = colors.get(badge_type, colors["secondary"])

    return f"""
        <span style="
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
            background-color: {color_scheme["bg"]};
            color: {color_scheme["text"]};
        ">
            {text}
        </span>
    """


def display_tailadmin_info_bar(items):
    """
    Display a TailAdmin-style info bar with multiple items.

    Args:
        items: List of dictionaries with 'icon', 'label', and 'value' keys
    """

    items_html = ""
    for item in items:
        items_html += f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.875rem;
                color: #475569;
            ">
                <span>{item.get("icon", "")}</span>
                <span><strong style='color: #1e293b; font-weight: 600;'>{item.get("label", "")}</strong> {item.get("value", "")}</span>
            </div>
        """

    st.markdown(
        f"""
        <div style="
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 0.75rem;
            padding: 1rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1.5rem;
            flex-wrap: wrap;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        ">
            {items_html}
        </div>
    """,
        unsafe_allow_html=True,
    )


def display_tailadmin_navigation(nav_items, active_item=None):
    """
    Display a TailAdmin-style navigation component.

    Args:
        nav_items: List of dictionaries with 'label', 'icon', and 'key' keys
        active_item: Key of the currently active item
    """

    nav_html = ""
    for item in nav_items:
        is_active = item.get("key") == active_item

        bg_color = "#3b82f6" if is_active else "transparent"
        text_color = "white" if is_active else "#475569"
        hover_bg = "#3b82f6" if is_active else "#f8fafc"

        nav_html += f"""
            <div style="
                padding: 0.75rem 1rem;
                border-radius: 0.5rem;
                color: {text_color};
                background-color: {bg_color};
                display: flex;
                align-items: center;
                gap: 0.75rem;
                transition: all 0.15s ease;
                cursor: pointer;
                margin-bottom: 0.25rem;
            " 
            onmouseover="this.style.backgroundColor='{hover_bg}'"
            onmouseout="this.style.backgroundColor='{bg_color}'">
                <span>{item.get("icon", "")}</span>
                <span>{item.get("label", "")}</span>
            </div>
        """

    st.markdown(
        f"""
        <div style="
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 0.75rem;
            padding: 0.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        ">
            {nav_html}
        </div>
    """,
        unsafe_allow_html=True,
    )


def create_tailadmin_layout():
    """
    Create a TailAdmin-style page layout with proper spacing and structure.
    """
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1400px;
            padding-top: 1.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )
