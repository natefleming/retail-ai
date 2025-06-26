"""TailAdmin-styled homepage for the store management app."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import streamlit_modal as modal

from components.chat import show_chat_container
from components.tailadmin import (
    display_tailadmin_alert,
    display_tailadmin_badge,
    display_tailadmin_chart,
    display_tailadmin_info_bar,
    display_tailadmin_metrics_grid,
    display_tailadmin_table,
)
from utils.database import get_stores
from utils.store_context import check_permission


def show_tailadmin_homepage():
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

    # Chat modal setup
    chat_notifications = st.session_state.get("chat_notifications", 0)
    chat_modal = modal.Modal(title="AI Assistant", key="homepage_chat_modal", max_width=700, padding=20)

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
        button_text = f"AI Assistant ({chat_notifications})" if chat_notifications > 0 else "AI Assistant"
        if st.button(button_text, key="header_chat_btn", type="primary", use_container_width=True):
            st.session_state.chat_notifications = 0
            chat_modal.open()

    # Store Info Bar
    current_time = datetime.now().strftime("%I:%M %p")
    current_date = datetime.now().strftime("%A, %B %d")

    # Get store data
    stores_df = get_stores()
    current_store_data = None
    if not stores_df.empty:
        matching_stores = stores_df[stores_df["name"] == store_name]
        if not matching_stores.empty:
            current_store_data = matching_stores.iloc[0]

    # Build store info
    if current_store_data is not None:
        full_address = f"{current_store_data['address']}, {current_store_data['city']}, {current_store_data['state']} {current_store_data['zip_code']}"
        store_phone = current_store_data["phone"]
        store_hours = "24/7" if current_store_data.get("is_24_hours", False) else "8:00 AM - 9:00 PM"
    else:
        full_address = "789 Market St, San Francisco, CA 94102"
        store_phone = "(415) 555-9876"
        store_hours = "8:00 AM - 9:00 PM"

    info_items = [
        {"icon": "🕐", "label": current_time, "value": f"• {current_date}"},
        {"icon": "🌤️", "label": "Weather", "value": "72°F ☀️"},
        {"icon": "📍", "label": "Address", "value": full_address},
        {"icon": "⏰", "label": "Hours", "value": store_hours},
        {"icon": "📞", "label": "Phone", "value": store_phone},
    ]

    display_tailadmin_info_bar(info_items)

    # Chat Modal
    if chat_modal.is_open():
        with chat_modal.container():
            chat_config = st.session_state.get("config", {}).get(
                "chat", {"placeholder": "How can I help you today?", "max_tokens": 1000, "temperature": 0.7}
            )
            show_chat_container(chat_config)

    # Role-based alerts
    if user_role == "store_manager":
        display_tailadmin_alert("Monthly sales target: 85% complete (15 days remaining)", "warning")
        display_tailadmin_alert("Staff meeting scheduled for tomorrow at 10 AM", "info")
    elif user_role == "vp_retail_operations":
        display_tailadmin_alert("Q4 performance review: 12 stores above target", "success")
    else:
        display_tailadmin_alert("Remember to complete your daily inventory check by 6 PM", "info")

    # Key Metrics based on role
    if user_role == "store_manager":
        metrics = [
            {"icon": "💰", "value": "$24,589", "label": "Today's Sales", "change": "12.5%", "change_type": "positive"},
            {"icon": "👥", "value": "186", "label": "Customers Today", "change": "8.3%", "change_type": "positive"},
            {"icon": "📦", "value": "45", "label": "Orders Processed", "change": "5.2%", "change_type": "negative"},
            {"icon": "⭐", "value": "4.8", "label": "Customer Rating", "change": "0.2", "change_type": "positive"},
        ]
    elif user_role == "vp_retail_operations":
        metrics = [
            {"icon": "🏪", "value": "24", "label": "Active Stores", "change": "2 new", "change_type": "positive"},
            {"icon": "💰", "value": "$1.2M", "label": "Total Revenue", "change": "15.8%", "change_type": "positive"},
            {"icon": "📊", "value": "87%", "label": "Performance Score", "change": "3.2%", "change_type": "positive"},
            {"icon": "👥", "value": "2,847", "label": "Total Customers", "change": "11.5%", "change_type": "positive"},
        ]
    else:  # store_associate
        metrics = [
            {"icon": "✅", "value": "23", "label": "Tasks Completed", "change": "4 pending", "change_type": "positive"},
            {"icon": "📦", "value": "156", "label": "Items Processed", "change": "12.3%", "change_type": "positive"},
            {"icon": "🕐", "value": "6.5h", "label": "Hours Worked", "change": "1.5h left", "change_type": "positive"},
            {"icon": "⭐", "value": "95%", "label": "Accuracy Rate", "change": "2%", "change_type": "positive"},
        ]

    display_tailadmin_metrics_grid(metrics)

    # Charts and Tables section
    col1, col2 = st.columns(2)

    with col1:
        # Sample sales data
        dates = pd.date_range(start=datetime.now() - timedelta(days=29), end=datetime.now(), freq="D")
        sales_data = pd.DataFrame(
            {
                "Date": dates,
                "Sales": np.random.randint(15000, 35000, len(dates)) + np.cumsum(np.random.randn(len(dates)) * 500),
            }
        )

        display_tailadmin_chart(sales_data, chart_type="line", title="Sales Trend (Last 30 Days)", height=350)

    with col2:
        # Sample category performance
        category_data = pd.DataFrame(
            {
                "Category": ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"],
                "Revenue": [45000, 32000, 28000, 22000, 15000],
            }
        )

        display_tailadmin_chart(category_data, chart_type="bar", title="Category Performance", height=350)

    # Recent Activity Table
    if user_role == "store_manager":
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
        table_title = "Recent Store Activity"
    else:
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
        table_title = "Your Tasks & Activities"

    display_tailadmin_table(
        recent_data,
        title=table_title,
        actions='<button style="background: #3b82f6; color: white; border: none; border-radius: 0.5rem; padding: 0.5rem 1rem; font-size: 0.875rem; cursor: pointer;">View All</button>',
    )

    # Role-specific sections
    if user_role == "store_manager":
        st.markdown("### Store Management")

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
        st.markdown("### Operations Overview")

        # Multi-store performance chart
        store_performance = pd.DataFrame(
            {"Store": ["Downtown", "Mall Plaza", "Uptown", "Suburban", "Airport"], "Performance": [95, 88, 92, 85, 90]}
        )

        display_tailadmin_chart(store_performance, chart_type="bar", title="Store Performance Scores", height=300)

    else:  # store_associate
        st.markdown("### Quick Actions")

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


if __name__ == "__main__":
    show_tailadmin_homepage()
