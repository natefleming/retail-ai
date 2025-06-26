"""Associate dashboard tab."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time


def show_associate_dashboard_tab():
    """Display the Dashboard tab with key metrics for associates."""
    
    # Personal achievement notification system
    # Initialize dashboard entry tracking when first entering dashboard
    if "associate_dashboard_entry_time" not in st.session_state:
        st.session_state.associate_dashboard_entry_time = time.time()
        st.session_state.achievement_notification_shown = False
    
    # Check if we need to reset the timer (when navigating back to dashboard)
    current_nav = st.session_state.get("current_associate_tab", "My Dashboard")
    if current_nav == "My Dashboard":
        if st.session_state.get("last_associate_nav") != "My Dashboard":
            # User just navigated to Dashboard - reset timer
            st.session_state.associate_dashboard_entry_time = time.time()
            st.session_state.achievement_notification_shown = False
        st.session_state.last_associate_nav = "My Dashboard"
    
    # Check if 5 seconds have passed since entering the dashboard
    current_time = time.time()
    time_on_dashboard = current_time - st.session_state.associate_dashboard_entry_time
    
    # Show achievement notification after 5 seconds
    if time_on_dashboard >= 5 and not st.session_state.achievement_notification_shown:
        st.toast("🎉 Great job! You've completed 85% of your daily goals ahead of schedule!", icon="⭐")
        st.session_state.achievement_notification_shown = True
    
    # Controlled refresh mechanism
    if not st.session_state.achievement_notification_shown:
        if 2.5 <= time_on_dashboard < 3.5:
            time.sleep(0.5)
            st.rerun()
        elif 4.5 <= time_on_dashboard < 6.5:
            time.sleep(0.5)
            st.rerun()
    
    # Create metrics using custom styling
    st.markdown("### My Performance Today")
    
    # First row of metrics - Daily Performance
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">$3,250</div>
                    <div class="metric-badge positive">+15% vs yesterday</div>
                </div>
                <div class="metric-label">My Sales Today</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">18</div>
                    <div class="metric-badge positive">Target: 15</div>
                </div>
                <div class="metric-label">Customers Served</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col3:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">6/7</div>
                    <div class="metric-badge positive">86% complete</div>
                </div>
                <div class="metric-label">Tasks Completed</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col4:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">4.8/5</div>
                    <div class="metric-badge positive">12 reviews today</div>
                </div>
                <div class="metric-label">Customer Rating</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Second row of metrics - Shift & Goals
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">3h 42m</div>
                    <div class="metric-badge info">Until shift end</div>
                </div>
                <div class="metric-label">Time Remaining</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">$180</div>
                    <div class="metric-badge positive">Above avg: $145</div>
                </div>
                <div class="metric-label">Avg Sale Amount</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col3:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">2</div>
                    <div class="metric-badge info">Follow-up tomorrow</div>
                </div>
                <div class="metric-label">Personal Shopping</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col4:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">96%</div>
                    <div class="metric-badge positive">Great attendance</div>
                </div>
                <div class="metric-label">Weekly Goal Progress</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Sales Performance Chart
    st.markdown("#### My Sales Performance This Week")
    
    # Generate daily sales data for the week
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    current_day = datetime.now().weekday()  # 0=Monday, 6=Sunday
    
    # Mock weekly sales data - use NaN instead of None to avoid conversion issues
    np.random.seed(42)
    my_sales = [2800, 3100, 2950, 3250, np.nan, np.nan, np.nan]  # Current week, only up to today
    my_targets = [2500, 2500, 2500, 2500, 2500, 3000, 3000]  # Daily targets
    
    # Only show actual sales up to current day
    for i in range(current_day + 1, 7):
        my_sales[i] = np.nan
    
    # Create DataFrame
    weekly_data = pd.DataFrame({
        "Day": days,
        "My Sales": my_sales,
        "Daily Target": my_targets
    })
    
    # Create bar chart
    fig_weekly = go.Figure()
    
    # Add actual sales bars with proper NaN handling
    fig_weekly.add_trace(
        go.Bar(
            x=weekly_data["Day"],
            y=weekly_data["My Sales"],
            name="My Sales",
            marker_color="#3b82f6",
            text=[f"${int(x)}" if not pd.isna(x) else "" for x in weekly_data["My Sales"]],
            textposition="outside"
        )
    )
    
    # Add target line
    fig_weekly.add_trace(
        go.Scatter(
            x=weekly_data["Day"],
            y=weekly_data["Daily Target"],
            mode="lines+markers",
            name="Daily Target",
            line=dict(color="#f59e0b", width=3, dash="dash"),
            marker=dict(size=8)
        )
    )
    
    # Get dark mode state for chart styling
    dark_mode = st.session_state.get("dark_mode", False)
    chart_text_color = "#f9fafb" if dark_mode else "#1f2937"
    
    fig_weekly.update_layout(
        height=300,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Day of Week",
        yaxis_title="Sales ($)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=chart_text_color, family="Arial")
    )
    fig_weekly.update_yaxes(tickformat="$,.0f")
    
    st.plotly_chart(fig_weekly, use_container_width=True)

    st.markdown("---")

    # Personal Goals and Recognition
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### My Goals Progress")
        
        # Goal progress with progress bars
        goals = [
            {"name": "Weekly Sales", "current": 11200, "target": 12500, "unit": "$"},
            {"name": "Customer Satisfaction", "current": 4.8, "target": 4.5, "unit": "/5"},
            {"name": "Tasks Completed", "current": 23, "target": 25, "unit": ""},
            {"name": "Product Knowledge", "current": 85, "target": 90, "unit": "%"}
        ]
        
        for goal in goals:
            progress_percent = min(100, (goal["current"] / goal["target"]) * 100)
            
            # Progress bar color based on achievement
            if progress_percent >= 100:
                bar_color = "#10b981"
                status = "🎯 Achieved!"
            elif progress_percent >= 90:
                bar_color = "#3b82f6"
                status = "🔥 Almost there!"
            elif progress_percent >= 75:
                bar_color = "#f59e0b"
                status = "⚡ On track"
            else:
                bar_color = "#ef4444"
                status = "💪 Keep going"
            
            st.markdown(f"**{goal['name']}**")
            st.progress(progress_percent / 100)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption(f"Current: {goal['current']}{goal['unit']}")
            with col_b:
                st.caption(f"Target: {goal['target']}{goal['unit']}")
            
            st.markdown(f"<small style='color: {bar_color}; font-weight: 600;'>{status}</small>", 
                       unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Recent Achievements")
        
        achievements = [
            {
                "icon": "🏆",
                "title": "Top Performer",
                "description": "Exceeded weekly sales goal by 15%",
                "time": "2 days ago",
                "color": "#10b981"
            },
            {
                "icon": "⭐",
                "title": "Customer Favorite",
                "description": "Received 5 perfect ratings this week",
                "time": "1 day ago",
                "color": "#3b82f6"
            },
            {
                "icon": "🎯",
                "title": "Task Master",
                "description": "Completed all assigned tasks on time",
                "time": "Today",
                "color": "#f59e0b"
            },
            {
                "icon": "📚",
                "title": "Product Expert",
                "description": "Completed advanced product training",
                "time": "3 days ago",
                "color": "#8b5cf6"
            }
        ]
        
        for achievement in achievements:
            st.markdown(
                f"""
                <div class="achievement-card" style="
                    border-left: 4px solid {achievement['color']};
                    background: {achievement['color']}10;
                    padding: 12px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                ">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 20px;">{achievement['icon']}</span>
                        <div>
                            <div style="font-weight: 600; color: {achievement['color']};">{achievement['title']}</div>
                            <div style="font-size: 14px; color: #6b7280;">{achievement['description']}</div>
                            <div style="font-size: 12px; color: #9ca3af;">{achievement['time']}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Team Performance and Store Context
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### My Department Performance")
        
        # Department metrics
        dept_metrics = [
            {"label": "Department Sales", "value": "$28,750", "change": "+12%", "positive": True},
            {"label": "Team Members", "value": "8 active", "change": "Full staff", "positive": True},
            {"label": "Customer Wait", "value": "2.1 min", "change": "Target: <3 min", "positive": True},
            {"label": "My Contribution", "value": "11.3%", "change": "Above avg", "positive": True}
        ]
        
        for metric in dept_metrics:
            change_color = "#10b981" if metric["positive"] else "#ef4444"
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 8px 0;
                    border-bottom: 1px solid #e5e7eb;
                ">
                    <div>
                        <div style="font-weight: 600;">{metric['label']}</div>
                        <div style="font-size: 14px; color: #6b7280;">{metric['change']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 18px; font-weight: 700; color: {change_color};">{metric['value']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with col2:
        st.markdown("#### Quick Actions")
        
        # Quick action buttons
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("📋 View My Tasks", use_container_width=True):
                st.switch_page("pages/my_tasks.py")
            
            if st.button("📅 Check Schedule", use_container_width=True):
                st.success("Opening schedule view...")
            
            if st.button("📊 Detailed Performance", use_container_width=True):
                st.success("Loading detailed analytics...")
        
        with col_b:
            if st.button("🛍️ Product Lookup", use_container_width=True):
                st.success("Opening product search...")
            
            if st.button("👥 Team Chat", use_container_width=True):
                st.success("Opening team communication...")
            
            if st.button("🎓 Training Hub", use_container_width=True):
                st.success("Accessing training materials...")

    # Today's Schedule Summary
    st.markdown("---")
    st.markdown("#### Today's Schedule")
    
    # Schedule timeline
    schedule_items = [
        {"time": "9:00 AM", "task": "Shift Start - Women's Fashion", "status": "completed"},
        {"time": "10:30 AM", "task": "BOPIS Order Pickup - Sarah J.", "status": "completed"},
        {"time": "12:00 PM", "task": "Lunch Break", "status": "completed"},
        {"time": "2:00 PM", "task": "Personal Shopping - Emma R.", "status": "current"},
        {"time": "3:30 PM", "task": "Inventory Restock", "status": "upcoming"},
        {"time": "5:00 PM", "task": "End of Shift", "status": "upcoming"}
    ]
    
    cols = st.columns(len(schedule_items))
    
    for i, (col, item) in enumerate(zip(cols, schedule_items)):
        with col:
            if item["status"] == "completed":
                status_color = "#10b981"
                status_icon = "✅"
            elif item["status"] == "current":
                status_color = "#3b82f6"
                status_icon = "🔄"
            else:
                status_color = "#6b7280"
                status_icon = "⏰"
            
            st.markdown(
                f"""
                <div style="text-align: center; padding: 8px;">
                    <div style="color: {status_color}; font-size: 20px;">{status_icon}</div>
                    <div style="font-weight: 600; font-size: 14px;">{item['time']}</div>
                    <div style="font-size: 12px; color: #6b7280; line-height: 1.2;">{item['task']}</div>
                </div>
                """,
                unsafe_allow_html=True
            ) 