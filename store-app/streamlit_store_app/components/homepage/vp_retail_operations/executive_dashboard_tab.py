"""Executive Dashboard tab for VP of Retail Operations."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_echarts import st_echarts
from streamlit_card import card
from datetime import datetime, timedelta
import time


def show_executive_dashboard_tab():
    """Display the executive dashboard with KPIs, alerts, and regional performance."""
    # Time frame selector
    col1, col2 = st.columns([4, 1])  # Adjust column widths for alignment
    with col1:
        st.markdown("### Executive KPI Dashboard")
    with col2:
        time_frame = st.segmented_control(
            label="",
            options=["Weekly", "Monthly", "Yearly"],
            key="exec_time_frame",
            default="Weekly"  # Set default to 'Weekly'
        )

    # Define new KPIs based on selected time frame
    if time_frame == "Weekly":
        total_sales_volume = 1.5  # Million
        sales_volume_change = 3.0
        gross_margin = 45.0  # %
        gross_margin_change = 1.5
        atv = 50.0  # $
        atv_change = 2.0
        inventory_accuracy = 98.0  # %
        inventory_accuracy_change = 0.5
        csat = 4.5  # out of 5
        csat_change = 0.2
    elif time_frame == "Monthly":
        total_sales_volume = 6.0  # Million
        sales_volume_change = 5.0
        gross_margin = 44.0  # %
        gross_margin_change = 1.0
        atv = 52.0  # $
        atv_change = 1.8
        inventory_accuracy = 97.5  # %
        inventory_accuracy_change = 0.4
        csat = 4.6  # out of 5
        csat_change = 0.3
    else:  # Yearly
        total_sales_volume = 72.0  # Million
        sales_volume_change = 10.0
        gross_margin = 43.0  # %
        gross_margin_change = 0.8
        atv = 51.0  # $
        atv_change = 1.5
        inventory_accuracy = 97.0  # %
        inventory_accuracy_change = 0.3
        csat = 4.7  # out of 5
        csat_change = 0.4

    # Secondary KPI variables based on time frame
    if time_frame == "Weekly":
        sales_per_employee = 125  # K
        employee_productivity = 4.2
        inventory_turns = 8.5
        inventory_improvement = 0.8
        customer_traffic = 2.1  # M
        traffic_growth = 3.5
        omnichannel_mix = 42.0
        digital_growth = 12.3
    elif time_frame == "Monthly":
        sales_per_employee = 130  # K
        employee_productivity = 5.1
        inventory_turns = 8.8
        inventory_improvement = 1.2
        customer_traffic = 8.4  # M
        traffic_growth = 4.2
        omnichannel_mix = 45.0
        digital_growth = 15.8
    else:  # Yearly
        sales_per_employee = 135  # K
        employee_productivity = 6.7
        inventory_turns = 9.2
        inventory_improvement = 1.8
        customer_traffic = 98.2  # M
        traffic_growth = 8.9
        omnichannel_mix = 48.0
        digital_growth = 18.5

    # Display new KPIs using custom HTML styling for consistency
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-header">
                    <!-- Icon removed -->
                </div>
                <div class="metric-value-container">
                    <div class="metric-value">${total_sales_volume:.1f}M</div>
                    <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">↑ {sales_volume_change:.1f}%</div>
                </div>
                <div class="metric-label">Total Sales Volume</div>
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
                    <!-- Icon removed -->
                </div>
                <div class="metric-value-container">
                    <div class="metric-value">{gross_margin:.1f}%</div>
                    <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">↑ {gross_margin_change:.1f}%</div>
                </div>
                <div class="metric-label">Gross Margin</div>
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
                    <!-- Icon removed -->
                </div>
                <div class="metric-value-container">
                    <div class="metric-value">${atv:.1f}</div>
                    <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">↑ {atv_change:.1f}%</div>
                </div>
                <div class="metric-label">Average Transaction Value</div>
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
                    <!-- Icon removed -->
                </div>
                <div class="metric-value-container">
                    <div class="metric-value">{inventory_accuracy:.1f}%</div>
                    <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">↑ {inventory_accuracy_change:.1f}%</div>
                </div>
                <div class="metric-label">Inventory Accuracy</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col5:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-header">
                    <!-- Icon removed -->
                </div>
                <div class="metric-value-container">
                    <div class="metric-value">{csat:.1f}</div>
                    <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">↑ {csat_change:.1f}</div>
                </div>
                <div class="metric-label">Customer Satisfaction Score</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Display revenue target and gauge in the same row

    st.markdown("---")

    # Show metrics and animation directly
    # render_combined_metrics_animation(time_frame)
    performance_data = get_performance_data(time_frame)
    render_header_metrics(performance_data)
    render_progress_bar(performance_data)
    st.markdown("---")
    render_performance_summary_metrics(performance_data)
    st.markdown("---")
    render_performance_summary_cards(performance_data)
    st.markdown("---")
    st.markdown("**Revenue Trajectory**")

    # Ensure 'today' is defined before use
    today = pd.to_datetime("today").normalize()

    # Set a fixed seed for reproducibility
    np.random.seed(42)

    # Update the sales vs target line chart to show future target sales
    if time_frame == "Weekly":
        dates = pd.date_range(start=today - pd.Timedelta(weeks=2), periods=28, freq="D")  # Last two weeks and next two weeks
        current_sales = np.where(dates <= today, np.random.normal(1.5, 0.2, len(dates)) * 1000000, np.nan)  # Mock current sales
        target_sales = np.random.normal(1.5, 0.15, len(dates)) * 1000000  # Mock target sales
    elif time_frame == "Monthly":
        dates = pd.date_range(start=today - pd.DateOffset(months=1), periods=60, freq="D")
        current_sales = np.where(dates <= today, np.random.normal(1.5, 0.2, len(dates)) * 1000000, np.nan)
        target_sales = np.random.normal(1.5, 0.15, len(dates)) * 1000000
    else:  # Yearly
        dates = pd.date_range(start=today - pd.DateOffset(years=1), periods=24, freq="M")
        current_sales = np.where(dates <= today, np.random.normal(1.5, 0.2, len(dates)) * 1000000, np.nan)
        target_sales = np.random.normal(1.5, 0.15, len(dates)) * 1000000

    sales_data = pd.DataFrame({
        "Date": dates,
        "Current Sales": current_sales,
        "Target Sales": target_sales
    })

    # Create line chart
    fig_sales = go.Figure()
    fig_sales.add_trace(
        go.Scatter(
            x=sales_data["Date"],
            y=sales_data["Current Sales"],
            mode="lines+markers",
            name="Current Sales",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=8)
        )
    )
    fig_sales.add_trace(
        go.Scatter(
            x=sales_data["Date"],
            y=sales_data["Target Sales"],
            mode="lines",
            name="Target Sales",
            line=dict(color="#ff7f0e", width=2, dash="dash")
        )
    )

    # Adjust forecasted sales to connect with the end of the current sales
    forecasted_sales = np.where(dates > today, np.random.normal(1.5, 0.2, len(dates)) * 1000000, current_sales)

    # Add forecasted sales to the line chart
    fig_sales.add_trace(
        go.Scatter(
            x=sales_data["Date"],
            y=forecasted_sales,
            mode="lines",
            name="Forecasted Sales",
            line=dict(color="#2ca02c", width=2, dash="dot")
        )
    )

    # Ensure both the line chart and the gauge chart have the same height
    chart_height = 300

    fig_sales.update_layout(
        height=chart_height,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins to match
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1f2937", family="Arial")
    )
    fig_sales.update_yaxes(tickformat="$,.0f")

    st.plotly_chart(fig_sales, use_container_width=True, key="sales_chart")

    # Secondary KPIs with consistent TailAdmin styling
    st.markdown("### Store Operations Excellence")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-header">
                    <!-- Icon removed -->
                </div>
                <div class="metric-value-container">
                    <div class="metric-value">${sales_per_employee:.0f}K</div>
                    <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">↑ {employee_productivity:.1f}% YoY</div>
                </div>
                <div class="metric-label">Sales per Employee (LTM)</div>
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
                    <!-- Icon removed -->
                </div>
                <div class="metric-value-container">
                    <div class="metric-value">{inventory_turns:.1f}x</div>
                    <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">↑ {inventory_improvement:.1f} YoY</div>
                </div>
                <div class="metric-label">Inventory Turns (LTM)</div>
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
                    <!-- Icon removed -->
                </div>
                <div class="metric-value-container">
                    <div class="metric-value">{customer_traffic:.1f}M</div>
                    <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">↑ {traffic_growth:.1f}% QoQ</div>
                </div>
                <div class="metric-label">Customer Traffic (Q4)</div>
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
                    <!-- Icon removed -->
                </div>
                <div class="metric-value-container">
                    <div class="metric-value">{omnichannel_mix:.1f}%</div>
                    <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">↑ {digital_growth:.1f}% YoY</div>
                </div>
                <div class="metric-label">Omnichannel Sales Mix (Q4)</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue and Performance Trends
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Performance vs Target")

        # Performance metrics comparison
        metrics = [
            "Revenue",
            "Customer Satisfaction",
            "Inventory Accuracy",
            "Gross Margin",
        ]
        actual = [103.2, 98.7, 102.1, 106.3]
        target = [100, 100, 100, 100]

        fig_performance = go.Figure()
        fig_performance.add_trace(
            go.Bar(
                x=metrics,
                y=actual,
                name="Actual",
                marker_color=["#28a745" if x >= 100 else "#dc3545" for x in actual],
            )
        )
        fig_performance.add_trace(
            go.Scatter(
                x=metrics,
                y=target,
                mode="lines+markers",
                name="Target",
                line=dict(color="#ffc107", width=3),
                marker=dict(size=8),
            )
        )

        fig_performance.update_layout(
            height=300,
            showlegend=True,
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title="Performance (%)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig_performance, use_container_width=True, key="performance_vs_target")

    with col2:
        st.markdown("#### Regional Revenue Analysis")

        # Regional performance heatmap
        regions_data = pd.DataFrame(
            {
                "Region": [
                    "West Coast",
                    "East Coast",
                    "Central",
                    "Southeast",
                    "Northeast",
                ],
                "Revenue_M": [12.4, 11.8, 9.2, 8.7, 5.7],
                "Growth_Rate": [15.2, 11.8, 8.9, 14.1, 7.3],
                "Store_Count": [87, 82, 61, 68, 44],
                "Avg_Performance": [108.3, 104.7, 98.2, 112.4, 95.8],
            }
        )

        fig_map = px.bar(
            regions_data,
            x="Region",
            y="Revenue_M",
            color="Growth_Rate",
            title="Revenue by Region ($ Millions)",
            color_continuous_scale="Viridis",
        )
        fig_map.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig_map, use_container_width=True, key="regional_performance_1")

    # Geographic Performance Overview
    st.markdown("#### Top Performing Markets")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Regional performance heatmap
        regions_data = pd.DataFrame(
            {
                "Region": [
                    "West Coast",
                    "East Coast",
                    "Central",
                    "Southeast",
                    "Northeast",
                ],
                "Revenue_M": [12.4, 11.8, 9.2, 8.7, 5.7],
                "Growth_Rate": [15.2, 11.8, 8.9, 14.1, 7.3],
                "Store_Count": [87, 82, 61, 68, 44],
                "Avg_Performance": [108.3, 104.7, 98.2, 112.4, 95.8],
            }
        )

        fig_map = px.bar(
            regions_data,
            x="Region",
            y="Revenue_M",
            color="Growth_Rate",
            title="Revenue by Region ($ Millions)",
            color_continuous_scale="Viridis",
        )
        fig_map.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig_map, use_container_width=True, key="regional_performance_2")

    with col2:
        st.markdown("**Top Performing Regions**")

        # Regional performance table with TailAdmin card styling
        top_regions = regions_data.sort_values("Avg_Performance", ascending=False)

        for idx, row in top_regions.iterrows():
            performance_color = (
                "#059669" if row["Avg_Performance"] >= 100
                else "#d97706" if row["Avg_Performance"] >= 95
                else "#dc2626"
            )

            st.markdown(
                f"""
            <div class="region-card" style="border-left: 4px solid {performance_color};">
                <div class="region-name">{row["Region"]}</div>
                <div class="region-details">
                    Performance: <span class="region-metric">{row["Avg_Performance"]:.1f}%</span><br>
                    Revenue: <span class="region-metric">${row["Revenue_M"]:.1f}M</span><br>
                    Growth: <span class="region-metric" style="color: {performance_color};">+{row["Growth_Rate"]:.1f}%</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Strategic Alerts and Insights with TailAdmin styling
    st.markdown("#### Priority Actions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Critical Attention Required**")

        alerts = [
            {
                "type": "warning",
                "title": "Central Region Underperforming",
                "description": "Revenue 8.2% below target. Recommend strategic review.",
                "action": "Schedule leadership review meeting",
            },
            {
                "type": "info",
                "title": "Inventory Optimization Opportunity",
                "description": "AI models suggest 12% efficiency improvement potential.",
                "action": "Deploy advanced inventory AI",
            },
        ]

        for alert in alerts:
            icon = "⚠️" if alert["type"] == "warning" else "💡"
            alert_type_class = "warning" if alert["type"] == "warning" else "info"

            st.markdown(
                f"""
            <div class="alert-card {alert_type_class}">
                <div class="alert-title">{icon} {alert["title"]}</div>
                <div class="alert-description">{alert["description"]}</div>
                <div class="alert-action">Recommended: {alert["action"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("**Growth Opportunities**")

        opportunities = [
            {
                "title": "Southeast Region Expansion",
                "description": "Market analysis shows 23% growth potential in untapped areas.",
                "impact": "Est. +$3.2M annual revenue",
            },
            {
                "title": "Customer Experience Enhancement",
                "description": "AI-driven personalization could improve satisfaction by 0.4 points.",
                "impact": "Est. +8% customer retention",
            },
        ]

        for opp in opportunities:
            st.markdown(
                f"""
            <div class="alert-card success">
                <div class="alert-title">🚀 {opp["title"]}</div>
                <div class="alert-description">{opp["description"]}</div>
                <div class="opportunity-impact">{opp["impact"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Executive Action Center with TailAdmin button styling
    st.markdown("#### Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📋 Generate Executive Report", use_container_width=True):
            st.success("Executive summary report generated and sent to your dashboard!")

    with col2:
        if st.button("📊 Schedule Strategic Review", use_container_width=True):
            st.success("Strategic review meeting scheduled with regional managers!")

    with col3:
        if st.button("🤖 Deploy AI Insights", use_container_width=True):
            st.success(
                "AI optimization models deployed across underperforming regions!"
            )

    with col4:
        if st.button("📈 Launch Growth Initiative", use_container_width=True):
            st.success("Growth initiative approved and resources allocated!")


def get_executive_data():
    # Implementation of get_executive_data function
    pass

# Define the function before using it
def render_combined_metrics_animation(time_frame):
    # Get performance data
    performance_data = get_performance_data(time_frame)
    
    # Render components in sequence
    render_header_metrics(performance_data)
    render_progress_bar(performance_data)
    render_executive_performance_card(performance_data, time_frame)
    render_performance_summary_metrics(performance_data)
    render_enhanced_progress_tracker(performance_data)
    render_performance_summary_cards(performance_data)

def get_performance_data(time_frame):
    """Calculate all performance metrics based on time frame."""
    if time_frame == "Weekly":
        revenue_achieved = 1.8
        revenue_target = 3.5
        days_passed = 2
        total_days = 7
        period_label = "Week"
    elif time_frame == "Monthly":
        revenue_achieved = 7.8
        revenue_target = 15.0
        days_passed = 10
        total_days = 30
        period_label = "Month"
    else:  # Yearly
        revenue_achieved = 85.0
        revenue_target = 150.0
        days_passed = 200
        total_days = 365
        period_label = "Year"

    # Calculate derived metrics
    revenue_cumulative_target = (days_passed / total_days) * revenue_target
    percent_of_target_achieved = (revenue_achieved / revenue_target) * 100
    percent_expected_by_now = (revenue_cumulative_target / revenue_target) * 100
    status_delta = revenue_achieved - revenue_cumulative_target
    days_remaining = total_days - days_passed
    
    # Calculate performance metrics
    variance_amount = status_delta
    variance_percentage = (variance_amount / revenue_cumulative_target * 100) if revenue_cumulative_target > 0 else 0
    performance_ratio = (revenue_achieved / revenue_cumulative_target) if revenue_cumulative_target > 0 else 1
    
    # Determine performance classification
    if performance_ratio >= 1.1:  # 10% above forecast
        performance_class = "EXCEEDING"
        performance_color = "#059669"
        performance_bg = "linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)"
        performance_icon = "🚀"
        performance_trend = "↗️"
    elif performance_ratio >= 1.05:  # 5% above forecast
        performance_class = "OUTPERFORMING" 
        performance_color = "#16a34a"
        performance_bg = "linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)"
        performance_icon = "📈"
        performance_trend = "↗️"
    elif performance_ratio >= 0.95:  # Within 5% of forecast
        performance_class = "ON TARGET"
        performance_color = "#3b82f6"
        performance_bg = "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)"
        performance_icon = "🎯"
        performance_trend = "→"
    elif performance_ratio >= 0.9:  # 5-10% below forecast
        performance_class = "MONITORING"
        performance_color = "#f59e0b"
        performance_bg = "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"
        performance_icon = "⚠️"
        performance_trend = "↘️"
    else:  # >10% below forecast
        performance_class = "UNDERPERFORMING"
        performance_color = "#dc2626"
        performance_bg = "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
        performance_icon = "📉"
        performance_trend = "↘️"
    
    return {
        'revenue_achieved': revenue_achieved,
        'revenue_target': revenue_target,
        'revenue_cumulative_target': revenue_cumulative_target,
        'days_passed': days_passed,
        'total_days': total_days,
        'days_remaining': days_remaining,
        'period_label': period_label,
        'percent_of_target_achieved': percent_of_target_achieved,
        'percent_expected_by_now': percent_expected_by_now,
        'status_delta': status_delta,
        'variance_amount': variance_amount,
        'variance_percentage': variance_percentage,
        'performance_ratio': performance_ratio,
        'performance_class': performance_class,
        'performance_color': performance_color,
        'performance_bg': performance_bg,
        'performance_icon': performance_icon,
        'performance_trend': performance_trend
    }

def render_header_metrics(data):
    """Render the header metrics section."""
    st.markdown(f"**Target Achievement Progress**")
    inner_col1, inner_col2, inner_col3 = st.columns(3)
    
    with inner_col1:
        st.metric(
            label="Current Sales", 
            value=f"${data['revenue_achieved']:.1f}M",
            delta=f"{data['percent_of_target_achieved']:.1f}% of target"
        )
    
    with inner_col2:
        st.metric(
            label=f"{data['period_label']}ly Target",
            value=f"${data['revenue_target']:.1f}M",
            delta=None
        )
    
    with inner_col3:
        st.metric(
            label="Time Remaining",
            value=f"{data['days_remaining']} days",
            delta=f"{(data['days_passed']/data['total_days']*100):.0f}% elapsed"
        )

def render_progress_bar(data):
    """Render the initial progress bar visualization."""
    # Progress bar using Plotly
    fig_progress = go.Figure()
    
    # Background bar (full target)
    fig_progress.add_trace(go.Bar(
        x=[100],
        y=['Progress'],
        orientation='h',
        marker=dict(color='#e2e8f0'),
        name='Target',
        showlegend=False,
        text=f'Target: ${data["revenue_target"]:.1f}M',
        textposition='inside'
    ))
    
    # Actual progress bar
    fig_progress.add_trace(go.Bar(
        x=[data['percent_of_target_achieved']],
        y=['Progress'],
        orientation='h',
        marker=dict(color='#3b82f6'),
        name='Achieved',
        showlegend=False,
        text=f'${data["revenue_achieved"]:.1f}M ({data["percent_of_target_achieved"]:.1f}%)',
        textposition='inside',
        textfont=dict(color='white')
    ))
    
    # Expected progress line
    fig_progress.add_vline(
        x=data['percent_expected_by_now'],
        line_dash="dash",
        line_color="#f59e0b",
        line_width=3,
        annotation_text=f"Expected: {data['percent_expected_by_now']:.1f}%",
        annotation_position="top"
    )
    
    fig_progress.update_layout(
        height=120,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            range=[0, 100],
            title="Percentage of Target",
            showgrid=False
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        barmode='overlay'
    )
    
    st.plotly_chart(fig_progress, use_container_width=True, key="progress_bar")

def render_executive_performance_card(data, time_frame):
    """Render the executive performance card using streamlit-card."""
    # Executive Performance Card using streamlit-card
    performance_card = card(
        title=f"{data['performance_icon']} {data['performance_class']}",
        text=[
            f"**Performance vs Forecast**",
            f"",
            f"**Variance:** {data['performance_trend']} {data['variance_percentage']:+.1f}% (${abs(data['variance_amount']):.1f}M)",
            f"**Performance Ratio:** {data['performance_ratio']:.2f}x",
            f"",
            f"*Current performance vs forecast expectations*"
        ],
        image=None,
        styles={
            "card": {
                "width": "100%",
                "height": "200px",
                "border-radius": "12px",
                "box-shadow": "0 4px 12px rgba(0,0,0,0.1)",
                "border": f"2px solid {data['performance_color']}30",
                "border-left": f"6px solid {data['performance_color']}",
                "background": f"linear-gradient(135deg, {data['performance_color']}15, {data['performance_color']}25)"
            },
            "title": {
                "font-size": "24px",
                "font-weight": "bold",
                "color": data['performance_color'],
                "text-align": "center"
            },
            "text": {
                "font-family": "Arial, sans-serif",
                "text-align": "center",
                "color": "#1f2937"
            }
        },
        key=f"performance_card_{time_frame}"
    )

def render_performance_summary_metrics(data):
    """Render the performance summary metrics section."""

    st.markdown("**Forecast Analysis**")
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    
    # Calculate target probability based on current trajectory
    current_pace = data['revenue_achieved'] / data['days_passed'] if data['days_passed'] > 0 else 0
    projected_total = current_pace * data['total_days']
    target_probability = min(95, max(15, (projected_total / data['revenue_target']) * 100))
    
    with detail_col1:
        st.metric(
            label="Forecast (YTD)",
            value=f"${data['revenue_cumulative_target']:.1f}M",
            delta=f"{data['percent_expected_by_now']:.1f}% expected"
        )
    
    with detail_col2:
        prob_delta = "High likelihood" if target_probability >= 80 else "Moderate likelihood" if target_probability >= 60 else "Action needed"
        prob_delta_color = "normal" if target_probability >= 70 else "inverse" if target_probability < 50 else "off"
        st.metric(
            label="Target Probability",
            value=f"{target_probability:.0f}%",
            delta=prob_delta,
            delta_color=prob_delta_color
        )
    
    with detail_col3:
        forecast_variance = abs(1 - data['performance_ratio'])
        forecast_accuracy = max(75, 100 - (forecast_variance * 100))  # Min 75% accuracy
        accuracy_delta = "High confidence" if forecast_accuracy >= 90 else "Moderate confidence" if forecast_accuracy >= 80 else "Lower confidence"
        accuracy_delta_color = "normal" if forecast_accuracy >= 85 else "inverse" if forecast_accuracy < 80 else "off"
        st.metric(
            label="Forecast Accuracy",
            value=f"{forecast_accuracy:.1f}%",
            delta=accuracy_delta,
            delta_color=accuracy_delta_color
        )

def render_enhanced_progress_tracker(data):
    """Render the enhanced progress tracker with multiple data layers."""
    st.markdown("**Achievement Timeline**")
    
    # Create sophisticated progress bar with multiple data layers
    fig_executive = go.Figure()
    
    # Base target track (0-100%)
    fig_executive.add_trace(go.Bar(
        x=[100],
        y=['Target Achievement'],
        orientation='h',
        marker=dict(
            color='#f1f5f9',
            line=dict(color='#cbd5e1', width=1)
        ),
        name='Total Target',
        showlegend=False,
        hovertemplate='Total Target: 100%<extra></extra>'
    ))
    
    # Expected progress zone (what should be achieved by now)
    fig_executive.add_trace(go.Bar(
        x=[data['percent_expected_by_now']],
        y=['Target Achievement'], 
        orientation='h',
        marker=dict(
            color='#fbbf24',
            opacity=0.6,
            line=dict(color='#f59e0b', width=1)
        ),
        name='Expected Progress',
        showlegend=False,
        hovertemplate=f'Expected by now: {data["percent_expected_by_now"]:.1f}%<extra></extra>'
    ))
    
    # Actual achievement bar with dynamic color based on performance
    achievement_color = data['performance_color']
    fig_executive.add_trace(go.Bar(
        x=[data['percent_of_target_achieved']],
        y=['Target Achievement'],
        orientation='h',
        marker=dict(
            color=achievement_color,
            line=dict(color=achievement_color, width=2)
        ),
        name='Actual Achievement',
        showlegend=False,
        text=f'{data["percent_of_target_achieved"]:.1f}%',
        textposition='inside',
        textfont=dict(color='white', size=16, family='Arial Black'),
        hovertemplate=f'Current Achievement: {data["percent_of_target_achieved"]:.1f}%<br>Revenue: ${data["revenue_achieved"]:.1f}M<extra></extra>'
    ))
    
    # Add milestone markers
    milestones = [25, 50, 75, 100]
    for milestone in milestones:
        fig_executive.add_vline(
            x=milestone,
            line=dict(color='#94a3b8', width=1, dash='dot'),
            opacity=0.5
        )
        fig_executive.add_annotation(
            x=milestone,
            y=0.5,
            text=f"{milestone}%",
            showarrow=False,
            font=dict(size=10, color='#64748b'),
            yshift=20
        )
    
    # Expected progress indicator line
    fig_executive.add_vline(
        x=data['percent_expected_by_now'],
        line=dict(color='#f59e0b', width=4),
        annotation=dict(
            text=f"Expected: {data['percent_expected_by_now']:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor='#f59e0b',
            bgcolor='#fff7ed',
            bordercolor='#f59e0b',
            borderwidth=1,
            font=dict(size=12, color='#92400e')
        )
    )
    
    fig_executive.update_layout(
        height=80,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(
            range=[0, 105],
            showgrid=False,
            showticklabels=True,
            tickmode='array',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['0%', '25%', '50%', '75%', '100%'],
            tickfont=dict(size=12, color='#64748b')
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            fixedrange=True
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        barmode='overlay',
        bargap=0.1
    )
    
    st.plotly_chart(fig_executive, use_container_width=True, key="executive_tracker")

def render_performance_summary_cards(data):
    """Render the performance summary cards section."""
    st.markdown("**Sales Performance**")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    # Calculate required values
    daily_run_rate = data['revenue_achieved'] / data['days_passed'] if data['days_passed'] > 0 else 0
    required_daily_rate = (data['revenue_target'] - data['revenue_achieved']) / data['days_remaining'] if data['days_remaining'] > 0 else 0
    
    # Calculate forecast accuracy (simulated based on performance ratio)
    # Higher accuracy when actual performance is closer to forecast
    forecast_variance = abs(1 - data['performance_ratio'])
    forecast_accuracy = max(75, 100 - (forecast_variance * 100))  # Min 75% accuracy
    
    with summary_col1:
        delta_color = "normal" if data['variance_amount'] >= 0 else "inverse"
        st.metric(
            label="Performance Ratio",
            value=f"{data['performance_ratio']:.2f}x",
            delta=f"{data['variance_percentage']:+.1f}% vs forecast",
            delta_color=delta_color
        )
    
    with summary_col2:
        st.metric(
            label="Current Run Rate",
            value=f"${daily_run_rate:.1f}M",
            delta="per day average"
        )
    
    with summary_col3:
        pace_delta = "On pace" if required_daily_rate <= daily_run_rate else "Needs acceleration"
        pace_delta_color = "normal" if required_daily_rate <= daily_run_rate else "inverse"
        st.metric(
            label="Required Pace",
            value=f"${required_daily_rate:.1f}M",
            delta=pace_delta,
            delta_color=pace_delta_color
        )
