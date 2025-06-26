"""Enhanced VP Retail Operations Dashboard - Experimental Version using TailAdmin Components."""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.tailadmin import get_tailadmin_color


def show_vp_dashboard_enhanced():
    """Main function to run the enhanced VP dashboard."""

    load_tailwind_css()

    # Generate mock data
    data = generate_mock_data()

    # Create sidebar controls
    create_sidebar()

    # Main content
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Executive header
    create_executive_header()

    # KPI metrics section
    create_kpi_metrics_section(data)

    # Performance visualizations
    create_performance_visualizations(data)

    # Geographical analysis
    create_geographical_map_section()

    # AI insights
    create_ai_insights_section()

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
    <div style="margin-top: 3rem; padding: 1.5rem; background: #f9fafb; border-radius: 0.75rem; text-align: center; border: 1px solid #e5e7eb;">
        <p style="margin: 0; font-size: 0.875rem; color: #6b7280;">
            🏪 <strong>Retail Operations Dashboard</strong> • Powered by AI & Real-time Analytics •
            <span style="color: #10b981;">●</span> System Status: Operational
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def load_tailwind_css():
    """Load TailwindCSS and custom styling."""
    st.markdown(
        """
    <style>
    @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');

    /* Custom VP Dashboard Styling */
    .main-container {
        padding: 1rem;
        max-width: 1920px;
        margin: 0 auto;
    }

    .metric-card {
        border-radius: 1rem;
        border: 1px solid #e5e7eb;
        background-color: white;
        padding: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }

    .metric-icon {
        display: flex;
        height: 3rem;
        width: 3rem;
        align-items: center;
        justify-content: center;
        border-radius: 0.75rem;
        background-color: #f3f4f6;
    }

    .metric-value {
        margin-top: 1.25rem;
        font-size: 1.25rem;
        font-weight: 700;
        color: #1f2937;
    }

    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
    }

    .metric-change {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        border-radius: 9999px;
        padding: 0.125rem 0.5rem;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .metric-change.positive {
        background-color: #dcfce7;
        color: #16a34a;
    }

    .metric-change.negative {
        background-color: #fee2e2;
        color: #dc2626;
    }

    .chart-container {
        border-radius: 1rem;
        border: 1px solid #e5e7eb;
        background-color: white;
        padding: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    .geography-selector {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }

    .sidebar-section {
        background-color: white;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }

    /* Hide Streamlit default styling */
    .stDeployButton {display:none;}
    .stDecoration {display:none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom grid layout */
    .grid-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    @media (max-width: 768px) {
        .grid-container {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def generate_mock_data():
    """Generate realistic mock data for the dashboard."""
    # Generate data for different geographical levels
    data = {
        "country": {
            "sales_volume": 45200000,
            "avg_transaction": 87.50,
            "inventory_accuracy": 94.2,
            "gross_margin": 32.8,
            "customer_satisfaction": 4.3,
            "change_sales": 11.5,
            "change_transaction": -2.3,
            "change_inventory": 3.1,
            "change_margin": 1.8,
            "change_satisfaction": 5.2,
        },
        "regions": [
            {"name": "Northeast", "sales": 12500000, "stores": 145, "performance": "excellent"},
            {"name": "Southeast", "sales": 11800000, "stores": 132, "performance": "good"},
            {"name": "Midwest", "sales": 9200000, "stores": 98, "performance": "good"},
            {"name": "West", "sales": 11700000, "stores": 127, "performance": "excellent"},
        ],
        "monthly_trends": pd.DataFrame(
            {
                "month": pd.date_range(start="2024-01-01", periods=12, freq="ME"),
                "sales": np.random.normal(3800000, 300000, 12),
                "margin": np.random.normal(32.5, 1.5, 12),
                "satisfaction": np.random.normal(4.2, 0.2, 12),
            }
        ),
    }
    return data


def create_metric_card_html(title, value, change, icon_svg, is_currency=False, is_percentage=False):
    """Create a TailAdmin-style metric card using HTML."""

    # Format value
    if is_currency:
        formatted_value = f"${value:,.0f}" if value >= 1000 else f"${value:.2f}"
    elif is_percentage:
        formatted_value = f"{value:.1f}%"
    else:
        if value >= 1000000:
            formatted_value = f"{value / 1000000:.1f}M"
        elif value >= 1000:
            formatted_value = f"{value / 1000:.0f}K"
        else:
            formatted_value = f"{value:.1f}"

    # Determine change styling
    change_class = "positive" if change >= 0 else "negative"
    change_icon = "↗" if change >= 0 else "↘"
    change_text = f"{abs(change):.1f}%"

    card_html = f"""
    <div class="metric-card">
        <div class="metric-icon">
            {icon_svg}
        </div>
        <div style="display: flex; align-items: end; justify-content: space-between; margin-top: 1.25rem;">
            <div>
                <span class="metric-label">{title}</span>
                <h4 class="metric-value">{formatted_value}</h4>
            </div>
            <span class="metric-change {change_class}">
                {change_icon} {change_text}
            </span>
        </div>
    </div>
    """
    return card_html


def create_sidebar():
    """Create executive dashboard sidebar with navigation and controls."""
    with st.sidebar:
        st.markdown(
            """
        <div class="sidebar-section">
            <h2 style="margin: 0 0 1rem 0; font-size: 1.25rem; font-weight: 600; color: #1f2937;">
                🎯 Executive Controls
            </h2>
            <p style="margin: 0; font-size: 0.875rem; color: #6b7280;">
                Strategic oversight and performance insights
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Geographical Context Selector
        st.markdown(
            """
        <div class="sidebar-section">
            <h3 style="margin: 0 0 0.75rem 0; font-size: 1rem; font-weight: 600; color: #1f2937;">
                📍 Geographic Scope
            </h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        geo_level = st.selectbox("Analysis Level", ["Country", "Region", "Market", "District", "Store"], index=0)

        if geo_level != "Country":
            location = st.selectbox(
                f"Select {geo_level}",
                ["Northeast", "Southeast", "Midwest", "West"]
                if geo_level == "Region"
                else ["Location 1", "Location 2"],
            )

        # Time Period Selector
        st.markdown(
            """
        <div class="sidebar-section">
            <h3 style="margin: 0 0 0.75rem 0; font-size: 1rem; font-weight: 600; color: #1f2937;">
                📅 Time Period
            </h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        time_period = st.selectbox(
            "Analysis Period", ["Last 30 Days", "Last Quarter", "Last 6 Months", "Last Year", "YTD"], index=1
        )

        # Dashboard View Selector
        st.markdown(
            """
        <div class="sidebar-section">
            <h3 style="margin: 0 0 0.75rem 0; font-size: 1rem; font-weight: 600; color: #1f2937;">
                📊 Dashboard View
            </h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        dashboard_view = st.selectbox(
            "Focus Area",
            ["Overall Performance", "Sales Overview", "Inventory Focus", "Customer Experience", "Financial Metrics"],
            index=0,
        )

        # AI Insights Toggle
        st.markdown(
            """
        <div class="sidebar-section">
            <h3 style="margin: 0 0 0.75rem 0; font-size: 1rem; font-weight: 600; color: #1f2937;">
                🤖 AI-Powered Insights
            </h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        enable_ai = st.checkbox("Enable AI Recommendations", value=True)
        auto_refresh = st.checkbox("Auto-refresh Data", value=False)

        return {
            "geo_level": geo_level,
            "location": location if geo_level != "Country" else "United States",
            "time_period": time_period,
            "dashboard_view": dashboard_view,
            "enable_ai": enable_ai,
            "auto_refresh": auto_refresh,
        }


def create_executive_header():
    """Create the executive dashboard header."""
    st.markdown(
        """
    <div style="
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">
                    🏪 Retail Performance Dashboard
                </h1>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.125rem; opacity: 0.9;">
                    VP of Retail Operations - Executive Command Center
                </p>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; opacity: 0.7;">
                    Real-time insights across 502 stores nationwide • Last updated: {datetime.now().strftime('%H:%M:%S')}
                </p>
            </div>
            <div style="text-align: right;">
                <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 0.75rem;">
                    <p style="margin: 0; font-size: 0.75rem; opacity: 0.8;">System Status</p>
                    <p style="margin: 0; font-size: 0.875rem; font-weight: 600;">🟢 All Systems Operational</p>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def create_kpi_metrics_section(data):
    """Create executive-level KPI metric cards using TailAdmin styling."""
    st.markdown(
        """
    <div style="margin-bottom: 2rem;">
        <h2 style="margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 600; color: #1f2937;">
            📊 Executive KPI Overview
        </h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Define icons for each metric

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Sales Volume", f"${data['country']['sales_volume']:,.0f}", f"{data['country']['change_sales']:.1f}%"
        )
        st.metric(
            "Inventory Accuracy",
            f"{data['country']['inventory_accuracy']:.1f}%",
            f"{data['country']['change_inventory']:.1f}%",
        )
        st.metric(
            "Customer Satisfaction",
            f"{data['country']['customer_satisfaction']:.1f}/5.0",
            f"{data['country']['change_satisfaction']:.1f}%",
        )

    with col2:
        st.metric(
            "Avg Transaction Value",
            f"${data['country']['avg_transaction']:.2f}",
            f"{data['country']['change_transaction']:.1f}%",
        )
        st.metric("Gross Margin", f"{data['country']['gross_margin']:.1f}%", f"{data['country']['change_margin']:.1f}%")


def create_performance_visualizations(data):
    """Create executive performance visualizations."""
    st.markdown(
        """
    <div style="margin: 2rem 0 1rem 0;">
        <h2 style="margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 600; color: #1f2937;">
            📈 Performance Trends & Analysis
        </h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Monthly Sales Performance")

        # Create dual-axis chart
        fig = go.Figure()

        # Sales line
        fig.add_trace(
            go.Scatter(
                x=data["monthly_trends"]["month"],
                y=data["monthly_trends"]["sales"],
                mode="lines+markers",
                name="Sales ($)",
                line={"color": get_tailadmin_color("brand"), "width": 3},
                marker={"size": 8},
                yaxis="y",
            )
        )

        # Margin line
        fig.add_trace(
            go.Scatter(
                x=data["monthly_trends"]["month"],
                y=data["monthly_trends"]["margin"],
                mode="lines+markers",
                name="Gross Margin (%)",
                line={"color": get_tailadmin_color("success"), "width": 3},
                marker={"size": 8},
                yaxis="y2",
            )
        )

        fig.update_layout(
            height=400,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis={"showgrid": True, "gridwidth": 1, "gridcolor": "rgba(0,0,0,0.1)", "title": "Month"},
            yaxis={
                "title": "Sales Volume ($)",
                "side": "left",
                "showgrid": True,
                "gridwidth": 1,
                "gridcolor": "rgba(0,0,0,0.1)",
            },
            yaxis2={"title": "Gross Margin (%)", "side": "right", "overlaying": "y", "showgrid": False},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Regional Performance")
        st.caption("Sales volume by region")

        # Create regional performance list
        for region in data["regions"]:
            performance_color = (
                get_tailadmin_color("success")
                if region["performance"] == "excellent"
                else get_tailadmin_color("warning")
            )
            performance_width = "85%" if region["performance"] == "excellent" else "65%"

            st.markdown(
                f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="width: 2rem; height: 2rem; border-radius: 50%; background: linear-gradient(135deg, {performance_color} 0%, {performance_color}aa 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.75rem;">
                        {region["name"][:2]}
                    </div>
                    <div>
                        <p style="margin: 0; font-size: 0.875rem; font-weight: 600; color: #1f2937;">
                            {region["name"]}
                        </p>
                        <span style="font-size: 0.75rem; color: #6b7280;">
                            {region["stores"]} stores
                        </span>
                    </div>
                </div>
                <div style="display: flex; width: 35%; align-items: center; gap: 0.75rem;">
                    <div style="position: relative; height: 0.5rem; width: 100%; border-radius: 0.25rem; background: #e5e7eb;">
                        <div style="position: absolute; left: 0; top: 0; height: 100%; width: {performance_width}; border-radius: 0.25rem; background: {performance_color};"></div>
                    </div>
                    <p style="margin: 0; font-size: 0.875rem; font-weight: 600; color: #1f2937; min-width: fit-content;">
                        ${region["sales"] / 1000000:.1f}M
                    </p>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def create_geographical_map_section():
    """Create geographical analysis section with map visualization."""
    st.markdown(
        """
    <div style="margin: 2rem 0 1rem 0;">
        <h2 style="margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 600; color: #1f2937;">
            🗺️ Geographic Performance Analysis
        </h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Store Performance by State")
        st.caption("Heat map showing sales performance across all locations")

        # Create mock geographic data
        states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]
        sales_data = np.random.uniform(1000000, 5000000, len(states))

        # Create choropleth-style bar chart as placeholder
        fig = px.bar(x=states, y=sales_data, title="", color=sales_data, color_continuous_scale="Viridis")

        fig.update_layout(
            height=400,
            margin={"l": 0, "r": 0, "t": 0, "b": 40},
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis_title="State",
            yaxis_title="Sales Volume ($)",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Performing Markets")
        st.caption("Highest revenue generating markets")

        top_markets = [
            {"name": "Los Angeles, CA", "sales": 4200000, "growth": 12.5},
            {"name": "Houston, TX", "sales": 3800000, "growth": 8.3},
            {"name": "Miami, FL", "sales": 3500000, "growth": 15.2},
            {"name": "Chicago, IL", "sales": 3200000, "growth": 6.1},
            {"name": "Atlanta, GA", "sales": 2900000, "growth": 9.8},
        ]

        for i, market in enumerate(top_markets):
            rank_color = get_tailadmin_color("brand") if i < 2 else get_tailadmin_color("gray", "500")
            st.markdown(
                f"""
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.75rem; border-radius: 0.5rem; background: #f9fafb; margin-bottom: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="width: 1.5rem; height: 1.5rem; border-radius: 50%; background: {rank_color}; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.75rem;">
                        {i + 1}
                    </div>
                    <div>
                        <p style="margin: 0; font-size: 0.875rem; font-weight: 600; color: #1f2937;">
                            {market["name"]}
                        </p>
                        <span style="font-size: 0.75rem; color: #6b7280;">
                            ${market["sales"] / 1000000:.1f}M revenue
                        </span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.75rem; color: {get_tailadmin_color("success")}; font-weight: 600;">
                        +{market["growth"]:.1f}%
                    </span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def create_ai_insights_section():
    """Create AI-powered insights section."""
    st.markdown(
        """
    <div style="margin: 2rem 0 1rem 0;">
        <h2 style="margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 600; color: #1f2937;">
            🤖 AI-Powered Strategic Insights
        </h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    insights = [
        {
            "title": "Inventory Optimization",
            "description": "AI detected 15% overstock in Northeast region. Recommend redistribution to Southeast markets.",
            "action": "Initiate Transfer",
            "impact": "High",
            "color": get_tailadmin_color("error"),
        },
        {
            "title": "Customer Experience",
            "description": "Satisfaction scores declining in 3 major markets. Customer service training recommended.",
            "action": "Schedule Training",
            "impact": "Medium",
            "color": get_tailadmin_color("warning"),
        },
        {
            "title": "Revenue Opportunity",
            "description": "Predictive model shows 8% revenue increase potential through pricing optimization.",
            "action": "Review Pricing",
            "impact": "High",
            "color": get_tailadmin_color("success"),
        },
    ]

    for i, insight in enumerate(insights):
        with [col1, col2, col3][i]:
            st.markdown(
                f"""
            <div style="border: 1px solid #e5e7eb; border-left: 4px solid {insight["color"]}; border-radius: 0.5rem; padding: 1rem; background: white; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <h4 style="margin: 0; font-size: 1rem; font-weight: 600; color: #1f2937;">
                        {insight["title"]}
                    </h4>
                    <span style="padding: 0.125rem 0.5rem; border-radius: 9999px; background: {insight["color"]}20; color: {insight["color"]}; font-size: 0.75rem; font-weight: 500;">
                        {insight["impact"]}
                    </span>
                </div>
                <p style="margin: 0 0 1rem 0; font-size: 0.875rem; color: #6b7280; line-height: 1.4;">
                    {insight["description"]}
                </p>
                <button style="width: 100%; padding: 0.5rem 1rem; background: {insight["color"]}; color: white; border: none; border-radius: 0.375rem; font-size: 0.875rem; font-weight: 500; cursor: pointer;">
                    {insight["action"]}
                </button>
            </div>
            """,
                unsafe_allow_html=True,
            )
