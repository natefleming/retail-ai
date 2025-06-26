"""Homepage for the TailAdmin store management app."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from components.tailadmin import TAILADMIN_COLORS, create_tailadmin_metric_card, get_tailadmin_color


def show_homepage():
    """Main homepage with TailAdmin styling and components."""

    # Initialize session state defaults if not already set
    if "config" not in st.session_state:
        st.session_state.config = {
            "employees": {
                "store_associate": {"name": "Sarah Johnson"},
                "store_manager": {"name": "Mike Rodriguez"},
                "vp_retail_operations": {"name": "Jennifer Chen"},
            }
        }

    if "store_name" not in st.session_state:
        st.session_state.store_name = "BrickMart San Francisco"

    # Get user role and store context
    user_role = st.session_state.get("user_role", "store_associate")
    employee_name = st.session_state.config.get("employees", {}).get(user_role, {}).get("name", "Employee")
    store_name = st.session_state.get("store_name", "BrickMart San Francisco")

    # Extract location from store name
    if store_name.startswith("BrickMart "):
        location = store_name.replace("BrickMart ", "").strip()
    else:
        location = store_name

    # TailAdmin Page Header
    col1, col2 = st.columns([8, 2])

    with col1:
        st.markdown(
            f"""
            <div style="margin-bottom: 1.5rem;">
                <h1 style="
                    font-size: 2.25rem;
                    font-weight: 800;
                    color: #1e293b;
                    margin: 0;
                    line-height: 1.2;
                ">BrickMart - {location}</h1>
                <p style="
                    color: #64748b;
                    margin-top: 0.5rem;
                    font-size: 1.1rem;
                    font-weight: 500;
                ">Welcome back, {employee_name}! Here's your store overview for today.</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🤖 AI Assistant", key="header_chat_btn", type="primary", use_container_width=True):
            st.info("AI Assistant functionality would be implemented here")

    # Store Info Bar
    current_time = datetime.now().strftime("%I:%M %p")
    current_date = datetime.now().strftime("%A, %B %d")

    info_items = [
        {"icon": "🕐", "label": current_time, "value": f"• {current_date}"},
        {"icon": "🌤️", "label": "Weather", "value": "72°F ☀️"},
        {"icon": "📍", "label": "Address", "value": "789 Market St, San Francisco, CA 94102"},
        {"icon": "⏰", "label": "Hours", "value": "8:00 AM - 9:00 PM"},
        {"icon": "📞", "label": "Phone", "value": "(415) 555-9876"},
    ]

    # Display info bar using TailAdmin styling
    info_html = f"""
    <div style="
        background: {TAILADMIN_COLORS["white"]};
        border: 1px solid {TAILADMIN_COLORS["gray"]["200"]};
        border-radius: 1rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 2rem;
        flex-wrap: wrap;
        justify-content: space-between;
    ">
    """

    for item in info_items:
        info_html += f"""
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.2rem;">{item["icon"]}</span>
            <div>
                <div style="font-weight: 600; color: {TAILADMIN_COLORS["gray"]["900"]}; font-size: 0.875rem;">
                    {item["label"]}
                </div>
                <div style="color: {TAILADMIN_COLORS["gray"]["600"]}; font-size: 0.75rem;">
                    {item["value"]}
                </div>
            </div>
        </div>
        """

    info_html += "</div>"
    components.html(info_html, height=100)

    # Role-based alerts
    if user_role == "store_manager":
        st.warning("📊 Monthly sales target: 85% complete (15 days remaining)")
        st.info("👥 Staff meeting scheduled for tomorrow at 10 AM")
    elif user_role == "vp_retail_operations":
        st.success("📈 Q4 performance review: 12 stores above target")
    else:
        st.info("✅ Remember to complete your daily inventory check by 6 PM")

    # Key Metrics based on role
    if user_role == "store_manager":
        metrics_data = [
            {"icon": "💰", "value": "$24,589", "label": "Today's Sales", "change": "12.5%", "change_type": "positive"},
            {"icon": "👥", "value": "186", "label": "Customers Today", "change": "8.3%", "change_type": "positive"},
            {"icon": "📦", "value": "45", "label": "Orders Processed", "change": "5.2%", "change_type": "negative"},
            {"icon": "⭐", "value": "4.8", "label": "Customer Rating", "change": "0.2", "change_type": "positive"},
        ]
    elif user_role == "vp_retail_operations":
        metrics_data = [
            {"icon": "🏪", "value": "24", "label": "Active Stores", "change": "2 new", "change_type": "positive"},
            {"icon": "💰", "value": "$1.2M", "label": "Total Revenue", "change": "15.8%", "change_type": "positive"},
            {"icon": "📊", "value": "87%", "label": "Performance Score", "change": "3.2%", "change_type": "positive"},
            {"icon": "👥", "value": "2,847", "label": "Total Customers", "change": "11.5%", "change_type": "positive"},
        ]
    else:  # store_associate
        metrics_data = [
            {"icon": "✅", "value": "23", "label": "Tasks Completed", "change": "4 pending", "change_type": "positive"},
            {"icon": "📦", "value": "156", "label": "Items Processed", "change": "12.3%", "change_type": "positive"},
            {"icon": "🕐", "value": "6.5h", "label": "Hours Worked", "change": "1.5h left", "change_type": "positive"},
            {"icon": "⭐", "value": "95%", "label": "Accuracy Rate", "change": "2%", "change_type": "positive"},
        ]

    # Display metrics using TailAdmin components
    cols = st.columns(4)
    for i, metric in enumerate(metrics_data):
        with cols[i]:
            metric_html = create_tailadmin_metric_card(
                icon=metric["icon"],
                value=metric["value"],
                label=metric["label"],
                change=metric["change"],
                change_type=metric["change_type"],
            )
            components.html(metric_html, height=200)

    # Charts and Tables section
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Sales Trend (Last 30 Days)")

        # Sample sales data
        dates = pd.date_range(start=datetime.now() - timedelta(days=29), end=datetime.now(), freq="D")
        sales_data = pd.DataFrame(
            {
                "Date": dates,
                "Sales": np.random.randint(15000, 35000, len(dates)) + np.cumsum(np.random.randn(len(dates)) * 500),
            }
        )

        fig = px.line(sales_data, x="Date", y="Sales", color_discrete_sequence=[get_tailadmin_color("brand")])
        fig.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=get_tailadmin_color("gray", "700"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🛍️ Category Performance")

        # Sample category performance
        category_data = pd.DataFrame(
            {
                "Category": ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"],
                "Revenue": [45000, 32000, 28000, 22000, 15000],
            }
        )

        fig = px.bar(category_data, x="Category", y="Revenue", color_discrete_sequence=[get_tailadmin_color("brand")])
        fig.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=get_tailadmin_color("gray", "700"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recent Activity Table
    if user_role == "store_manager":
        st.subheader("📋 Recent Store Activity")
        recent_data = pd.DataFrame(
            {
                "Time": ["2:45 PM", "2:30 PM", "2:15 PM", "2:00 PM", "1:45 PM"],
                "Action": [
                    "Sale Completed",
                    "Inventory Updated",
                    "Customer Return",
                    "Staff Check-in",
                    "Product Restocked",
                ],
                "Employee": ["Sarah M.", "Mike C.", "Emma L.", "John D.", "Sarah M."],
                "Amount": ["$156.99", "-", "$89.50", "-", "-"],
                "Status": ["Completed", "Completed", "Processed", "Active", "Completed"],
            }
        )
    else:
        st.subheader("📝 Your Tasks & Activities")
        recent_data = pd.DataFrame(
            {
                "Task": [
                    "Inventory Count - Electronics",
                    "Customer Service - Refund",
                    "Stock Replenishment",
                    "Price Update - Home Goods",
                    "Customer Assistance",
                ],
                "Priority": ["High", "Medium", "High", "Low", "Medium"],
                "Status": ["In Progress", "Completed", "Pending", "Completed", "Completed"],
                "Due Time": ["3:00 PM", "2:30 PM", "4:00 PM", "2:00 PM", "2:15 PM"],
                "Assigned To": ["You", "Sarah M.", "You", "Mike C.", "You"],
            }
        )

    st.dataframe(recent_data, use_container_width=True, hide_index=True)

    # Role-specific sections
    if user_role == "store_manager":
        st.markdown("### 🎛️ Store Management")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Sales Reports", use_container_width=True):
                st.info("Sales reports functionality")

        with col2:
            if st.button("👥 Staff Management", use_container_width=True):
                st.info("Staff management functionality")

        with col3:
            if st.button("📦 Inventory Control", use_container_width=True):
                st.info("Inventory control functionality")

    elif user_role == "vp_retail_operations":
        st.markdown("### 📊 Operations Overview")

        # Multi-store performance chart
        store_performance = pd.DataFrame(
            {"Store": ["Downtown", "Mall Plaza", "Uptown", "Suburban", "Airport"], "Performance": [95, 88, 92, 85, 90]}
        )

        fig = px.bar(
            store_performance, x="Store", y="Performance", color_discrete_sequence=[get_tailadmin_color("success")]
        )
        fig.update_layout(
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=get_tailadmin_color("gray", "700"),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # store_associate
        st.markdown("### ⚡ Quick Actions")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("📋 Check Tasks", use_container_width=True):
                st.info("Task management")

        with col2:
            if st.button("📦 Inventory", use_container_width=True):
                st.info("Inventory tasks")

        with col3:
            if st.button("🛍️ Sales", use_container_width=True):
                st.info("Sales assistance")

        with col4:
            if st.button("❓ Help", use_container_width=True):
                st.info("Get help")
