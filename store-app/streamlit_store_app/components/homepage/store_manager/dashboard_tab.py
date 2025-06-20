"""Store manager dashboard tab."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time


def show_manager_dashboard_tab():
    """Display the Dashboard tab with key metrics."""
    
    # VIP Customer Notification System
    # Initialize dashboard entry tracking when first entering dashboard
    if "dashboard_entry_time" not in st.session_state:
        st.session_state.dashboard_entry_time = time.time()
        st.session_state.vip_notification_shown = False
    
    # Check if we need to reset the timer (when navigating back to dashboard)
    # This ensures the notification can trigger again if they leave and come back
    current_nav = st.session_state.get("current_manager_tab", "Dashboard")
    if current_nav == "Dashboard":
        if st.session_state.get("last_nav") != "Dashboard":
            # User just navigated to Dashboard - reset timer
            st.session_state.dashboard_entry_time = time.time()
            st.session_state.vip_notification_shown = False
        st.session_state.last_nav = "Dashboard"
    
    # Check if 7 seconds have passed since entering the dashboard
    current_time = time.time()
    time_on_dashboard = current_time - st.session_state.dashboard_entry_time
    
    # Show VIP notification after 7 seconds
    if time_on_dashboard >= 7 and not st.session_state.vip_notification_shown:
        # Enhanced urgent toast notification with custom styling
        st.markdown(
            """
            <style>
            .urgent-toast {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 9999;
                background: linear-gradient(135deg, #dc2626 0%, #b91c1c 50%, #991b1b 100%);
                color: white;
                padding: 20px 24px;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(220, 38, 38, 0.4), 0 0 30px rgba(220, 38, 38, 0.3);
                border: 3px solid #fca5a5;
                animation: urgentPulse 2s infinite, slideInRight 0.5s ease-out;
                max-width: 400px;
                font-weight: 600;
                font-size: 16px;
                line-height: 1.4;
            }
            
            @keyframes urgentPulse {
                0% { box-shadow: 0 20px 60px rgba(220, 38, 38, 0.4), 0 0 30px rgba(220, 38, 38, 0.3); }
                50% { box-shadow: 0 25px 80px rgba(220, 38, 38, 0.6), 0 0 50px rgba(220, 38, 38, 0.5); }
                100% { box-shadow: 0 20px 60px rgba(220, 38, 38, 0.4), 0 0 30px rgba(220, 38, 38, 0.3); }
            }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            .urgent-badge {
                background: #fbbf24;
                color: #92400e;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 8px;
                display: inline-block;
                animation: badgePulse 1.5s infinite;
            }
            
            @keyframes badgePulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Use enhanced toast with dramatic styling
        st.toast("🚨🌟 URGENT PLATINUM MEMBER ALERT 🌟🚨\n\nEmma Rodriguez arriving in 55 minutes for personal styling!\n\n⚡ IMMEDIATE STAFF ASSIGNMENT REQUIRED ⚡", icon="👑")
        
        # Add a brief sound effect simulation with additional emphasis
        st.markdown(
            """
            <div class="urgent-toast">
                <div class="urgent-badge">🔴 URGENT PRIORITY</div>
                <div style="font-size: 20px; margin-bottom: 8px;">👑 PLATINUM VIP ARRIVAL</div>
                <div style="margin-bottom: 12px;">
                    <strong>Emma Rodriguez</strong> arriving in <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 8px; font-weight: 900;">55 MINUTES</span>
                </div>
                <div style="font-size: 14px; opacity: 0.95;">
                    🎯 Personal styling appointment<br>
                    ⚡ Requires immediate staff assignment<br>
                    💼 Professional wardrobe consultation
                </div>
            </div>
            
            <script>
                setTimeout(function() {
                    const toast = document.querySelector('.urgent-toast');
                    if (toast) {
                        toast.style.animation = 'urgentPulse 2s infinite, fadeOut 1s ease-in-out 8s forwards';
                    }
                }, 100);
                
                setTimeout(function() {
                    const toast = document.querySelector('.urgent-toast');
                    if (toast) {
                        toast.remove();
                    }
                }, 10000);
            </script>
            """,
            unsafe_allow_html=True
        )
        
        st.session_state.vip_notification_shown = True
    
    # More controlled refresh mechanism - only refresh twice: once at 3 seconds, once at 7 seconds
    # This is much less aggressive than every second but ensures the notification appears
    if not st.session_state.vip_notification_shown:
        if 2.5 <= time_on_dashboard < 3.5:
            # First check around 3 seconds - minimal delay
            time.sleep(0.5)
            st.rerun()
        elif 6.5 <= time_on_dashboard < 8.5:
            # Second check around 7 seconds to catch the notification
            time.sleep(0.5)
            st.rerun()
    
    # Create metrics using custom styling

    # Executive overview using custom HTML styling for consistency
    st.markdown("### Today's Performance")
    
    # First row of metrics - Core Daily Performance
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">$28,750</div>
                    <div class="metric-badge neutral">96% of target</div>
                </div>
                <div class="metric-label">Daily Sales vs Target</div>
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
                    <div class="metric-value">247</div>
                    <div class="metric-badge positive">+18% vs yesterday</div>
                </div>
                <div class="metric-label">Customer Traffic</div>
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
                    <div class="metric-value">12/15</div>
                    <div class="metric-badge neutral">80% coverage</div>
                </div>
                <div class="metric-label">Staff Coverage</div>
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
                    <div class="metric-value">3.2 min</div>
                    <div class="metric-badge positive">Target: <5 min</div>
                </div>
                <div class="metric-label">Customer Wait Time</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Second row of metrics - Operational Efficiency
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">$171.50</div>
                    <div class="metric-badge positive">+$23 vs target</div>
                </div>
                <div class="metric-label">Average Sale Amount</div>
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
                    <div class="metric-value">$312</div>
                    <div class="metric-badge positive">+8% vs last week</div>
                </div>
                <div class="metric-label">Sales per Hour Worked</div>
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
                    <div class="metric-value">18</div>
                    <div class="metric-badge info">Avg pickup: 1.8 hrs</div>
                </div>
                <div class="metric-label">BOPIS Orders Ready</div>
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
                    <div class="metric-value">12.4%</div>
                    <div class="metric-badge positive">Within target range</div>
                </div>
                <div class="metric-label">Return Rate</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Sales vs Target Chart
    st.markdown("#### Today's Sales vs Target")
    
    # Get current time
    current_hour = datetime.now().hour
    
    # Generate hourly sales data for today
    hours = list(range(6, 23))  # Store hours from 6 AM to 10 PM
    np.random.seed(42)  # For consistent demo data
    
    # Mock hourly sales data
    hourly_sales = []
    hourly_targets = []
    hourly_forecast = []
    
    for hour in hours:
        # Higher sales during peak hours (11-1 PM and 5-7 PM)
        if 11 <= hour <= 13 or 17 <= hour <= 19:
            base_sales = np.random.normal(2800, 400) if hour <= current_hour else None
            forecast_sales = np.random.normal(2600, 300) if hour > current_hour else None  # Slightly lower than actual
            target_sales = 2500
        elif 9 <= hour <= 11 or 14 <= hour <= 17:
            base_sales = np.random.normal(2200, 300) if hour <= current_hour else None
            forecast_sales = np.random.normal(2100, 250) if hour > current_hour else None
            target_sales = 2000
        else:
            base_sales = np.random.normal(1200, 200) if hour <= current_hour else None
            forecast_sales = np.random.normal(1150, 150) if hour > current_hour else None
            target_sales = 1000
            
        # Only add actual sales if the hour has passed
        hourly_sales.append(max(base_sales, 0) if base_sales is not None else None)
        hourly_targets.append(target_sales)
        hourly_forecast.append(max(forecast_sales, 0) if forecast_sales is not None else None)
    
    # Create DataFrame
    sales_data = pd.DataFrame({
        "Hour": [f"{h}:00" for h in hours],
        "Actual Sales": hourly_sales,
        "Target Sales": hourly_targets,
        "Forecasted Sales": hourly_forecast
    })
    
    # Create line chart
    fig_sales = go.Figure()
    
    # Add actual sales line (only for past hours)
    fig_sales.add_trace(
        go.Scatter(
            x=sales_data["Hour"],
            y=sales_data["Actual Sales"],
            mode="lines+markers",
            name="Actual Sales",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=8),
            connectgaps=False  # Don't connect gaps in the data
        )
    )
    
    # Add target sales line (for all hours)
    fig_sales.add_trace(
        go.Scatter(
            x=sales_data["Hour"],
            y=sales_data["Target Sales"],
            mode="lines",
            name="Target Sales",
            line=dict(color="#ff7f0e", width=2, dash="dash")
        )
    )
    
    # Add forecasted sales line (only for future hours)
    fig_sales.add_trace(
        go.Scatter(
            x=sales_data["Hour"],
            y=sales_data["Forecasted Sales"],
            mode="lines",
            name="Forecasted Sales",
            line=dict(color="#2ca02c", width=2, dash="dot"),
            connectgaps=False
        )
    )
    
    # Get dark mode state for chart styling
    dark_mode = st.session_state.get("dark_mode", False)
    chart_text_color = "#f9fafb" if dark_mode else "#1f2937"
    
    # Update layout
    fig_sales.update_layout(
        height=300,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Hour of Day",
        yaxis_title="Sales ($)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=chart_text_color, family="Arial")
    )
    fig_sales.update_yaxes(tickformat="$,.0f")
    
    st.plotly_chart(fig_sales, use_container_width=True)

    st.markdown("---")

    # # Performance trends section
    # st.markdown("#### 📈 Performance Trends")

    # # Performance trends using enhanced cards in a 2x2 grid
    # col1, col2 = st.columns(2)
    
    # # Get theme variables for consistent styling
    # dark_mode = st.session_state.get("dark_mode", False)
    # if dark_mode:
    #     value_color = "#f9fafb"
    #     label_color = "#9ca3af"
    # else:
    #     value_color = "#1f2937"
    #     label_color = "#6b7280"
    
    # with col1:
    #     st.markdown(
    #         f"""
    #         <div class="modern-performance-card">
    #             <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
    #                 <div style="font-weight: 700; color: {value_color};">Weekly Sales</div>
    #                 <div style="color: #10b981; font-weight: 600;">↗ +12%</div>
    #             </div>
    #             <div style="
    #                 background: linear-gradient(90deg, #e2e8f0 0%, #10b981 100%);
    #                 height: 8px;
    #                 border-radius: 4px;
    #                 margin-bottom: 0.5rem;
    #             "></div>
    #             <div style="font-size: 1.1rem; font-weight: 700; color: {value_color}; margin-bottom: 0.25rem;">$142,350</div>
    #             <div style="font-size: 0.85rem; color: {label_color};">vs. last week</div>
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )
        
    #     st.markdown(
    #         f"""
    #         <div class="modern-performance-card">
    #             <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
    #                 <div style="font-weight: 700; color: {value_color};">Customer Satisfaction</div>
    #                 <div style="color: #fbbf24; font-weight: 600;">↗ +0.2</div>
    #             </div>
    #             <div style="
    #                 background: linear-gradient(90deg, #e2e8f0 0%, #fbbf24 94%);
    #                 height: 8px;
    #                 border-radius: 4px;
    #                 margin-bottom: 0.5rem;
    #             "></div>
    #             <div style="font-size: 1.1rem; font-weight: 700; color: {value_color}; margin-bottom: 0.25rem;">4.7/5.0</div>
    #             <div style="font-size: 0.85rem; color: {label_color};">vs. last month</div>
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )

    # with col2:
    #     st.markdown(
    #         f"""
    #         <div class="modern-performance-card">
    #             <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
    #                 <div style="font-weight: 700; color: {value_color};">Monthly Target</div>
    #                 <div style="color: #10b981; font-weight: 600;">On track</div>
    #             </div>
    #             <div style="
    #                 background: linear-gradient(90deg, #e2e8f0 0%, #3b82f6 78%);
    #                 height: 8px;
    #                 border-radius: 4px;
    #                 margin-bottom: 0.5rem;
    #             "></div>
    #             <div style="font-size: 1.1rem; font-weight: 700; color: {value_color}; margin-bottom: 0.25rem;">78%</div>
    #             <div style="font-size: 0.85rem; color: {label_color};">complete</div>
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )
        
    #     st.markdown(
    #         f"""
    #         <div class="modern-performance-card">
    #             <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
    #                 <div style="font-weight: 700; color: {value_color};">Staff Efficiency</div>
    #                 <div style="color: #8b5cf6; font-weight: 600;">↗ +3%</div>
    #             </div>
    #             <div style="
    #                 background: linear-gradient(90deg, #e2e8f0 0%, #8b5cf6 94%);
    #                 height: 8px;
    #                 border-radius: 4px;
    #                 margin-bottom: 0.5rem;
    #             "></div>
    #             <div style="font-size: 1.1rem; font-weight: 700; color: {value_color}; margin-bottom: 0.25rem;">94%</div>
    #             <div style="font-size: 0.85rem; color: {label_color};">vs. average</div>
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )
