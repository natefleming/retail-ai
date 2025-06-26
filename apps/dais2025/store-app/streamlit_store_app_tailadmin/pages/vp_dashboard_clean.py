"""VP Retail Operations Dashboard - Clean Streamlit Implementation with TailAdmin Header."""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from components.tailadmin import TAILADMIN_COLORS, get_tailadmin_color


def show_vp_dashboard_clean():
    """Main function for VP dashboard."""

    inject_tailwind_css()
    create_tailadmin_header_v3()
    create_overview_metrics()

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Revenue vs Target")
        st.caption("Monthly performance tracking")
        fig = create_revenue_chart()
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        create_monthly_target_card()

    st.divider()

    create_operational_metrics()

    st.divider()

    create_top_performing_stores()

    st.divider()

    create_geographic_breakdown()

    st.divider()

    create_ai_insights()


def inject_tailwind_css():
    """Inject Tailwind CSS via CDN with dark mode support."""

    # Get dark mode state
    dark_mode = st.session_state.get("dark_mode", True)

    # Dynamic styling based on dark mode
    if dark_mode:
        body_bg = "#111827"
        text_color = "#f9fafb"
        section_bg = "#1f2937"
    else:
        body_bg = "#f9fafb"
        text_color = "#1f2937"
        section_bg = "white"

    st.markdown(
        f"""
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
    /* Apply dark mode class to body */
    .stApp {{
        background-color: {body_bg} !important;
        color: {text_color} !important;
        transition: all 0.3s ease !important;
    }}

    .stApp > header {{
        visibility: hidden;
    }}

    .stDeployButton {{
        display: none;
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    .stDecoration {{
        display: none;
    }}

    .main .block-container {{
        padding-top: 0rem !important;
        padding-left: 0rem;
        padding-right: 0rem;
        max-width: none;
        margin-top: 0rem !important;
        background-color: {body_bg} !important;
    }}

    /* Remove default spacing from first element */
    .main .block-container > div:first-child {{
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }}

    /* Remove spacing from columns */
    div[data-testid="column"] {{
        padding-top: 0rem !important;
    }}

    /* Remove spacing from text inputs and buttons in header */
    div[data-testid="column"] .stTextInput,
    div[data-testid="column"] .stButton {{
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
    }}

    .header-avatar {{
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: #3b82f6;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.875rem;
    }}

    /* Dark mode support for Streamlit components */
    .stTextInput > div > div > input {{
        background-color: {section_bg} !important;
        color: {text_color} !important;
        border-color: {"#374151" if dark_mode else "#e5e7eb"} !important;
    }}

    /* Fix placeholder text visibility in dark mode */
    .stTextInput > div > div > input::placeholder {{
        color: {"#9ca3af" if dark_mode else "#6b7280"} !important;
        opacity: 1 !important;
    }}

    .stButton > button {{
        background-color: {section_bg} !important;
        color: {text_color} !important;
        border-color: {"#374151" if dark_mode else "#e5e7eb"} !important;
    }}

    .stButton > button:hover {{
        background-color: {"#374151" if dark_mode else "#f3f4f6"} !important;
        border-color: {"#4b5563" if dark_mode else "#d1d5db"} !important;
    }}

    /* Dark mode support for dataframes and tables */
    .stDataFrame {{
        background-color: {section_bg} !important;
    }}

    .stDataFrame [data-testid="stTable"] {{
        background-color: {section_bg} !important;
        color: {text_color} !important;
    }}

    /* Dark mode support for plotly charts */
    .js-plotly-plot {{
        background-color: {section_bg} !important;
    }}

    /* Dark mode support for metric containers */
    [data-testid="metric-container"] {{
        background-color: {section_bg} !important;
        color: {text_color} !important;
        border-color: {"#374151" if dark_mode else "#e5e7eb"} !important;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def create_tailadmin_header_v3():
    """Alternative: Sidebar approach for user menu - single row layout with functional dark mode."""

    # Initialize session state for user dropdown and dark mode
    if "user_dropdown_open" not in st.session_state:
        st.session_state.user_dropdown_open = False
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    # Get current dark mode state
    dark_mode = st.session_state.get("dark_mode", True)

    # Add negative margin to pull header up
    st.markdown(
        """
    <style>
    .header-row {
        margin-top: -1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Hide backup Streamlit buttons */
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"] {
        display: none !important;
        visibility: hidden !important;
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        pointer-events: none !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # All controls in one row with custom class
    with st.container():
        st.markdown('<div class="header-row">', unsafe_allow_html=True)
        col1, spacer, col2 = st.columns([3, 0.5, 2])

        with col1:
            st.text_input(
                "Search",
                placeholder="🔍 Search stores, reports, or commands...",
                key="search_v3",
                label_visibility="hidden",
            )

        with spacer:
            # Empty spacer column
            st.write("")

        with col2:
            # Get button styling based on dark mode
            if dark_mode:
                button_style = "background: #1f2937; border: 1px solid #374151; color: #d1d5db;"
            else:
                button_style = "background: white; border: 1px solid #e5e7eb; color: #6b7280;"

            # Group sun, notifications, avatar, and user details together - aligned to the right
            st.markdown(
                f"""
            <div style="display: flex; align-items: center; justify-content: flex-end; gap: 1rem;">
                <button onclick="toggleDarkMode()" style="border: none; cursor: pointer; padding: 0.5rem; width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; {button_style} transition: all 0.2s; font-size: 1.2rem;" title="Toggle dark mode">
                    {"🌙" if not dark_mode else "☀️"}
                </button>
                <button onclick="showNotifications()" style="background: white; border: 1px solid #e5e7eb; cursor: pointer; padding: 0.5rem; width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #6b7280; position: relative; transition: all 0.2s; font-size: 1.2rem; hover:background-color: #f3f4f6;" title="Notifications">
                    <span style="position: absolute; top: 4px; right: 4px; width: 8px; height: 8px; background: #f97316; border-radius: 50%; z-index: 1; animation: pulse 2s infinite;"></span>
                    🔔
                </button>
                <div class="header-avatar">SC</div>
                <div style="padding: 0.25rem 0;">
                    <div style="font-weight: 600; font-size: 0.875rem; color: #1f2937;">Sarah Chen</div>
                    <div style="color: #6b7280; font-size: 0.75rem;">VP Retail Operations</div>
                </div>
            </div>

            <style>
            @keyframes pulse {{
                0%, 100% {{
                    opacity: 1;
                }}
                50% {{
                    opacity: 0.5;
                }}
            }}

            button:hover {{
                background-color: {"#374151" if dark_mode else "#f3f4f6"} !important;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }}
            </style>

            <script>
            function toggleDarkMode() {{
                // Trigger Streamlit component event
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    key: 'dark_mode_toggle',
                    value: true
                }}, '*');
            }}

            function showNotifications() {{
                // Trigger Streamlit component event
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    key: 'notifications_clicked',
                    value: true
                }}, '*');
            }}
            </script>
            """,
                unsafe_allow_html=True,
            )

            # Hidden buttons to handle Streamlit interactions
            col_hidden1, col_hidden2 = st.columns(2)
            with col_hidden1:
                if st.button("Dark Mode", key="dark_mode_btn", help="Toggle dark mode"):
                    st.session_state.dark_mode = not st.session_state.dark_mode
                    st.rerun()

            with col_hidden2:
                if st.button("Notifications", key="notif_btn", help="Show notifications"):
                    st.toast("3 new notifications")


def create_overview_metrics():
    """Create TailAdmin-styled overview metrics using container styling approach with dark mode support."""

    # Get dark mode state
    dark_mode = st.session_state.get("dark_mode", True)

    # Dynamic styling based on dark mode
    if dark_mode:
        # Dark mode styles
        container_bg = "#1f2937"
        container_border = "#374151"
        container_shadow = "0 1px 3px 0 rgba(0, 0, 0, 0.3)"
        hover_shadow = "0 10px 25px 0 rgba(0, 0, 0, 0.4)"
        value_color = "#f9fafb"
        label_color = "#9ca3af"
        # Dark mode icon backgrounds
        icon_bg_blue = "#1e3a8a"
        icon_bg_yellow = "#92400e"
        icon_bg_purple = "#3730a3"
        icon_bg_amber = "#b45309"
    else:
        # Light mode styles
        container_bg = "white"
        container_border = "#e5e7eb"
        container_shadow = "0 1px 3px 0 rgba(0, 0, 0, 0.1)"
        hover_shadow = "0 10px 25px 0 rgba(0, 0, 0, 0.15)"
        value_color = "#1f2937"
        label_color = "#6b7280"
        # Light mode icon backgrounds
        icon_bg_blue = "#dbeafe"
        icon_bg_yellow = "#fef3c7"
        icon_bg_purple = "#e0e7ff"
        icon_bg_amber = "#fef7cd"

    # Custom CSS for containers with dark mode support
    st.markdown(
        f"""
    <style>
    .metric-container {{
        background: {container_bg};
        border: 1px solid {container_border};
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: {container_shadow};
        transition: all 0.3s;
    }}

    .metric-container:hover {{
        transform: translateY(-2px);
        box-shadow: {hover_shadow};
    }}

    .metric-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }}

    .metric-icon {{
        width: 48px;
        height: 48px;
        border-radius: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }}

    .metric-badge {{
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 500;
    }}

    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {value_color};
        margin: 0;
    }}

    .metric-label {{
        color: {label_color};
        font-size: 1.25rem;
        margin: 0.5rem 0 0 0;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-header">
                    <div class="metric-icon" style="background: {icon_bg_blue};">💰</div>
                    <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">↗ 11.5%</div>
                </div>
                <div class="metric-value">$45.2M</div>
                <div class="metric-label">Total Revenue</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-header">
                    <div class="metric-icon" style="background: {icon_bg_yellow};">🏪</div>
                    <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">↗ +3</div>
                </div>
                <div class="metric-value">502</div>
                <div class="metric-label">Active Stores</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col3:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-header">
                    <div class="metric-icon" style="background: {icon_bg_purple};">💳</div>
                    <div class="metric-badge" style="background: #fee2e2; color: #dc2626;">↘ -2.3%</div>
                </div>
                <div class="metric-value">$87.50</div>
                <div class="metric-label">Avg Transaction</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col4:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-header">
                    <div class="metric-icon" style="background: {icon_bg_amber};">⭐</div>
                    <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">↗ 5.2%</div>
                </div>
                <div class="metric-value">4.3/5.0</div>
                <div class="metric-label">Customer Satisfaction</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def create_revenue_chart():
    """Create revenue performance chart."""
    # Get dark mode state for dynamic styling
    dark_mode = st.session_state.get("dark_mode", True)

    # Set colors based on dark mode
    if dark_mode:
        font_color = "#f9fafb"  # Light text for dark mode
        grid_color = "rgba(255,255,255,0.1)"  # Light grid for dark mode
        title_color = "#f9fafb"
    else:
        font_color = "#6b7280"  # Dark text for light mode
        grid_color = "rgba(0,0,0,0.05)"  # Dark grid for light mode
        title_color = "#1f2937"

    months = pd.date_range(start="2024-01-01", periods=12, freq="ME")
    revenue_data = [3.2, 3.8, 4.1, 3.9, 4.3, 4.7, 4.2, 4.5, 4.8, 4.1, 3.9, 4.6]
    target_data = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=months,
            y=revenue_data,
            mode="lines+markers",
            name="Actual Revenue",
            line={"color": "#3b82f6", "width": 3},
            marker={"size": 8, "color": "#3b82f6"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=months,
            y=target_data,
            mode="lines",
            name="Target",
            line={"color": "#ef4444", "width": 2, "dash": "dash"},
            opacity=0.7,
        )
    )

    fig.update_layout(
        height=400,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": True, "gridwidth": 1, "gridcolor": grid_color, "tickfont": {"color": font_color}},
        yaxis={
            "title": {"text": "Revenue ($M)", "font": {"color": font_color}},
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": grid_color,
            "tickfont": {"color": font_color},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": font_color},
        },
        font={"color": font_color},
        hovermode="x unified",
        title={"text": "Revenue vs Target - Monthly Performance", "font": {"color": title_color}},
    )

    return fig


def create_monthly_target_card():
    """Create a monthly target card using Streamlit native components with dark mode support."""
    # Get dark mode state for dynamic styling
    dark_mode = st.session_state.get("dark_mode", True)

    # Set colors based on dark mode
    if dark_mode:
        card_bg = "#1f2937"
        card_border = "#374151"
        text_primary = "#f9fafb"
        text_secondary = "#9ca3af"
    else:
        card_bg = "white"
        card_border = "#e5e7eb"
        text_primary = "#1f2937"
        text_secondary = "#6b7280"

    # Create container with custom styling
    with st.container():
        st.markdown(
            f"""
        <div style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1rem;">
            <h3 style="color: {text_primary}; font-size: 1.25rem; font-weight: 600; margin: 0 0 0.5rem 0;">📋 Monthly Targets</h3>
            <p style="color: {text_secondary}; font-size: 0.875rem; margin: 0 0 1.5rem 0;">Current month performance vs goals</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Use columns for the metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Revenue Target", value="$4.2M", delta="8% above target", delta_color="normal")

        with col2:
            st.metric(label="Store Target", value="495", delta="2% vs goal", delta_color="normal")

        # Progress bar using Streamlit
        st.markdown(
            f"""
        <div style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 1rem; padding: 1rem; margin-top: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: {text_secondary}; font-size: 0.875rem;">Progress to Year-End Goal</span>
                <span style="color: {text_primary}; font-size: 0.875rem; font-weight: 600;">73% Complete</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Use Streamlit's progress bar
        st.progress(0.73)

        # Additional key metrics
        col3, col4 = st.columns(2)

        with col3:
            st.metric(label="Monthly Sales", value="$3.8M", delta="12% vs last month", delta_color="normal")

        with col4:
            st.metric(label="Customer Growth", value="2,847", delta="156 new customers", delta_color="normal")


def create_operational_metrics():
    """Create operational metrics using Streamlit columns with dark mode support."""
    # Get dark mode state for dynamic styling
    dark_mode = st.session_state.get("dark_mode", True)

    # Set colors based on dark mode
    if dark_mode:
        subheader_style = "color: #f9fafb; font-weight: 600; font-size: 1.25rem; margin-bottom: 1rem;"
    else:
        subheader_style = "color: #1f2937; font-weight: 600; font-size: 1.25rem; margin-bottom: 1rem;"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<h3 style="{subheader_style}">📦 Inventory Metrics</h3>', unsafe_allow_html=True)
        st.metric("Inventory Accuracy", "94.2%", "3.1%")
        st.metric("Stock Turnover Rate", "8.3x", "0.4x")
        st.metric("Out of Stock Rate", "2.1%", "-0.5%", delta_color="inverse")
        st.metric("Overstock Items", "156", "-23", delta_color="inverse")

    with col2:
        st.markdown(f'<h3 style="{subheader_style}">⚡ Operational Efficiency</h3>', unsafe_allow_html=True)
        st.metric("Labor Cost Ratio", "18.5%", "-1.2%", delta_color="inverse")
        st.metric("Energy Cost per Store", "$2,840", "-$120", delta_color="inverse")
        st.metric("Avg Transaction Time", "2.3 min", "-0.2 min", delta_color="inverse")
        st.metric("Store Compliance", "97.8%", "1.1%")


def create_top_performing_stores():
    """Create top performing stores table using Streamlit."""
    st.subheader("🏆 Top Performing Stores")

    # Create sample data
    stores_data = {
        "Store": ["#0847", "#0923", "#1205", "#0652", "#1834"],
        "Location": ["Los Angeles, CA", "Houston, TX", "Miami, FL", "Chicago, IL", "Phoenix, AZ"],
        "Revenue": ["$284K", "$267K", "$245K", "$238K", "$225K"],
        "Growth": ["15.2%", "12.8%", "18.3%", "6.4%", "9.1%"],
        "Status": ["🟢 Excellent", "🟢 Excellent", "🟢 Excellent", "🟡 Good", "🟡 Good"],
    }

    df = pd.DataFrame(stores_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def create_geographic_breakdown():
    """Create geographic performance using Streamlit components."""
    st.subheader("🌍 Geographic Performance")
    st.caption("Revenue by region with targets")

    # Region data
    regions_data = {
        "Region": ["Northeast 🔵", "Southeast 🟢", "Midwest 🟡", "West 🟣"],
        "Stores": [145, 132, 98, 127],
        "Revenue": ["$12.5M", "$11.8M", "$9.2M", "$11.7M"],
        "Target": ["$13.0M", "$11.0M", "$9.5M", "$12.0M"],
        "Achievement": ["96%", "107%", "97%", "97.5%"],
    }

    df_regions = pd.DataFrame(regions_data)

    # Display as metrics in a grid
    for i, row in df_regions.iterrows():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            st.write(f"**{row['Region']}**")
            st.caption(f"{row['Stores']} stores")

        with col2:
            st.metric("Revenue", row["Revenue"])

        with col3:
            st.metric("Target", row["Target"])

        with col4:
            achievement = float(row["Achievement"].replace("%", ""))
            delta_color = "normal" if achievement >= 100 else "inverse"
            st.metric("Achievement", row["Achievement"], delta_color=delta_color)

        if i < len(df_regions) - 1:
            st.divider()


def create_ai_insights():
    """Create AI insights section."""
    st.subheader("🤖 AI-Powered Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Revenue Optimization**

        Northeast region is 4% below target. Consider:
        - Promotional campaigns for Q4
        - Inventory rebalancing from Southeast
        - Staff training programs
        """)

        st.success("""
        **Operational Excellence**

        Southeast region exceeding targets by 7%:
        - Replicate successful practices
        - Document best practices
        - Scale to other regions
        """)

    with col2:
        st.warning("""
        **Inventory Management**

        156 overstock items detected:
        - Implement clearance strategies
        - Review demand forecasting
        - Optimize procurement cycles
        """)

        st.error("""
        **Customer Experience**

        Transaction value declining 2.3%:
        - Review product mix strategy
        - Enhance upselling training
        - Analyze competitor pricing
        """)
