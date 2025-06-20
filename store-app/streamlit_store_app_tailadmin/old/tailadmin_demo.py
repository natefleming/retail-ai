"""TailAdmin Demo Page - Showcases TailAdmin components integrated into Streamlit."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from components.tailadmin import (
    display_tailadmin_alert,
    display_tailadmin_badge,
    display_tailadmin_chart,
    display_tailadmin_info_bar,
    display_tailadmin_metrics_grid,
    display_tailadmin_navigation,
    display_tailadmin_table,
)


def show_tailadmin_demo():
    """Display a comprehensive demo of TailAdmin components in Streamlit."""

    # Page Header - TailAdmin Style
    st.markdown(
        """
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
        ">
            <div>
                <h1 style="
                    font-size: 2rem;
                    font-weight: 700;
                    color: #1e293b;
                    margin: 0;
                ">TailAdmin Integration Demo</h1>
                <p style="color: #64748b; margin-top: 0.5rem; font-size: 1rem;">
                    Showcasing TailAdmin's beautiful components integrated into Streamlit
                </p>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Info Bar
    info_items = [
        {"icon": "🕐", "label": "Last Updated", "value": datetime.now().strftime("%I:%M %p")},
        {"icon": "🌤️", "label": "Weather", "value": "72°F ☀️"},
        {"icon": "📍", "label": "Location", "value": "San Francisco, CA"},
        {"icon": "⏰", "label": "Store Hours", "value": "8:00 AM - 9:00 PM"},
    ]
    display_tailadmin_info_bar(info_items)

    # Alerts Demo
    st.markdown("### Alerts")
    col1, col2 = st.columns(2)

    with col1:
        display_tailadmin_alert("Sales target achieved for this month! Great work team.", "success")
        display_tailadmin_alert("Low inventory alert: Restocking needed for 3 items.", "warning")

    with col2:
        display_tailadmin_alert("System maintenance scheduled for tonight at 11 PM.", "info")
        display_tailadmin_alert("Failed to sync data with external system.", "error")

    # Metrics Grid - TailAdmin Style
    st.markdown("### Key Metrics")

    metrics = [
        {"icon": "👥", "value": "3,782", "label": "Customers", "change": "11.01%", "change_type": "positive"},
        {"icon": "📦", "value": "5,359", "label": "Orders", "change": "9.05%", "change_type": "negative"},
        {"icon": "💰", "value": "$89,532", "label": "Revenue", "change": "15.3%", "change_type": "positive"},
        {"icon": "📊", "value": "2.4%", "label": "Conversion Rate", "change": "0.8%", "change_type": "positive"},
    ]

    display_tailadmin_metrics_grid(metrics)

    # Charts Section
    st.markdown("### Analytics Charts")

    # Create sample data for charts
    dates = pd.date_range(start="2024-01-01", end="2024-01-30", freq="D")
    sales_data = pd.DataFrame(
        {
            "Date": dates,
            "Sales": np.random.randint(1000, 5000, len(dates)) + np.cumsum(np.random.randn(len(dates)) * 50),
        }
    )

    inventory_data = pd.DataFrame(
        {"Category": ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"], "Stock": [450, 820, 320, 180, 95]}
    )

    col1, col2 = st.columns(2)

    with col1:
        display_tailadmin_chart(sales_data, chart_type="line", title="Sales Trend (Last 30 Days)", height=350)

    with col2:
        display_tailadmin_chart(inventory_data, chart_type="bar", title="Inventory by Category", height=350)

    # Table Demo
    st.markdown("### Data Tables")

    # Create sample table data
    table_data = pd.DataFrame(
        {
            "Product": ["iPhone 15 Pro", 'Samsung TV 55"', "Nike Air Max", "Coffee Maker", "Wireless Headphones"],
            "Category": ["Electronics", "Electronics", "Sports", "Home", "Electronics"],
            "Price": ["$999", "$899", "$120", "$89", "$199"],
            "Stock": [45, 23, 78, 156, 89],
            "Status": ["In Stock", "Low Stock", "In Stock", "In Stock", "In Stock"],
        }
    )

    # Add status badges to the table
    def format_status(status):
        if status == "In Stock":
            return display_tailadmin_badge(status, "success")
        elif status == "Low Stock":
            return display_tailadmin_badge(status, "warning")
        else:
            return display_tailadmin_badge(status, "error")

    # Create action buttons for table
    table_actions = """
        <button style="
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-size: 0.875rem;
            cursor: pointer;
            margin-right: 0.5rem;
        ">Export</button>
        <button style="
            background: #10b981;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-size: 0.875rem;
            cursor: pointer;
        ">Add Product</button>
    """

    display_tailadmin_table(table_data, title="Product Inventory", actions=table_actions)

    # Navigation Demo
    st.markdown("### Navigation Components")

    col1, col2 = st.columns([1, 2])

    with col1:
        nav_items = [
            {"label": "Dashboard", "icon": "📊", "key": "dashboard"},
            {"label": "Products", "icon": "📦", "key": "products"},
            {"label": "Orders", "icon": "🛒", "key": "orders"},
            {"label": "Customers", "icon": "👥", "key": "customers"},
            {"label": "Analytics", "icon": "📈", "key": "analytics"},
            {"label": "Settings", "icon": "⚙️", "key": "settings"},
        ]

        display_tailadmin_navigation(nav_items, active_item="dashboard")

    with col2:
        st.markdown("""
            #### Navigation Features

            The navigation component includes:
            - **Active state highlighting** with TailAdmin's primary blue color
            - **Hover effects** for better user interaction
            - **Icon support** with consistent spacing
            - **Responsive design** that works on all screen sizes
            - **Customizable styling** that matches TailAdmin's design system

            This component can be easily integrated into your Streamlit sidebar or main content area.
        """)

    # Badge Demo
    st.markdown("### Badges & Status Indicators")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("**Primary**")
        st.markdown(display_tailadmin_badge("New Feature", "primary"), unsafe_allow_html=True)

    with col2:
        st.markdown("**Success**")
        st.markdown(display_tailadmin_badge("Completed", "success"), unsafe_allow_html=True)

    with col3:
        st.markdown("**Warning**")
        st.markdown(display_tailadmin_badge("Pending", "warning"), unsafe_allow_html=True)

    with col4:
        st.markdown("**Error**")
        st.markdown(display_tailadmin_badge("Failed", "error"), unsafe_allow_html=True)

    with col5:
        st.markdown("**Secondary**")
        st.markdown(display_tailadmin_badge("Draft", "secondary"), unsafe_allow_html=True)

    # Integration Instructions
    st.markdown("---")
    st.markdown("### Integration Guide")

    st.markdown("""
    #### How to Use TailAdmin Components in Your Streamlit App

    1. **Import the components**:
    ```python
    from components.tailadmin_styles import load_tailadmin_css
    from components.tailadmin import (
        display_tailadmin_metrics_grid,
        display_tailadmin_chart,
        display_tailadmin_table
    )
    ```

    2. **Load the CSS** in your app initialization:
    ```python
    load_tailadmin_css()
    ```

    3. **Use the components** in your pages:
    ```python
    # Display metrics
    metrics = [
        {"icon": "👥", "value": "1,234", "label": "Users", "change": "12%", "change_type": "positive"}
    ]
    display_tailadmin_metrics_grid(metrics)

    # Display charts
    display_tailadmin_chart(data, chart_type="line", title="Sales Trend")

    # Display tables
    display_tailadmin_table(df, title="User Data")
    ```

    #### Benefits of TailAdmin Integration

    - **Consistent Design**: Matches TailAdmin's professional design system
    - **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
    - **Easy Customization**: Modify colors, spacing, and styling easily
    - **Professional Appearance**: Gives your Streamlit app a polished, dashboard-like look
    - **Component Reusability**: Use the same components across different pages
    """)


if __name__ == "__main__":
    show_tailadmin_demo()
