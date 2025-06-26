"""Inventory page for the Streamlit Store App."""

import streamlit as st
import streamlit_modal as modal

from components.chat import show_chat_container
from components.navigation import show_nav
from components.styles import load_css


def main():
    """Main inventory page."""
    # Load CSS
    load_css()

    # Show navigation
    show_nav()

    # Page header
    col1, col2 = st.columns([8, 2])
    with col1:
        st.title("📦 Inventory Management")
        st.markdown("**Monitor stock levels and manage inventory**")

    with col2:
        if st.button("🤖 AI Assistant", use_container_width=True):
            st.session_state.show_chat = True

    # Create the chat modal
    chat_modal = modal.Modal("AI Assistant", key="inventory_chat_modal", max_width=800)

    # Handle chat modal
    if st.session_state.get("show_chat", False):
        chat_modal.open()
        st.session_state.show_chat = False

    # Modal content
    if chat_modal.is_open():
        with chat_modal.container():
            # Get chat config with fallback
            chat_config = st.session_state.get("config", {}).get(
                "chat",
                {
                    "placeholder": "How can I help you with inventory management?",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            )

            # Show the chat container
            show_chat_container(chat_config)

    # Add custom CSS for better tab styling (same as daily operations)
    st.markdown(
        """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px !important;
        padding: 12px 24px !important;
        background-color: #f8f9fa !important;
        border-radius: 8px 8px 0px 0px !important;
        border: 1px solid #dee2e6 !important;
        border-bottom: none !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #495057 !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 22px !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef !important;
        color: #212529 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
    }
    .stTabs [aria-selected="true"] p {
        color: white !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Main content in tabs - fully tab-based experience
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Current Stock", "Low Stock Alerts", "Stock Trends", "Reorder Management"]
    )

    with tab1:
        show_current_inventory()

    with tab2:
        show_low_stock_alerts()

    with tab3:
        show_stock_trends()

    with tab4:
        show_reorder_management()


def show_current_inventory():
    """Display current inventory with enhanced styling."""
    # Inventory Overview at top of tab
    st.markdown("#### 📊 Inventory Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="inventory-summary-card critical">
                <div class="inventory-icon">⚠️</div>
                <div class="inventory-value">12</div>
                <div class="inventory-label">Low Stock Items</div>
                <div class="inventory-detail">Requires attention</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="inventory-summary-card low">
                <div class="inventory-icon">❌</div>
                <div class="inventory-value">3</div>
                <div class="inventory-label">Out of Stock</div>
                <div class="inventory-detail">Immediate reorder</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="inventory-summary-card good">
                <div class="inventory-icon">💰</div>
                <div class="inventory-value">$847K</div>
                <div class="inventory-label">Total Value</div>
                <div class="inventory-detail">Current inventory</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="inventory-summary-card new">
                <div class="inventory-icon">📦</div>
                <div class="inventory-value">28</div>
                <div class="inventory-label">New Arrivals</div>
                <div class="inventory-detail">This week</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("### 📋 Current Inventory")

    # Mock inventory data focused on retail/fashion
    inventory_data = [
        {
            "category": "Women's Fashion",
            "items": [
                {
                    "name": "Designer Handbags",
                    "stock": 45,
                    "reorder": 20,
                    "price": 299.99,
                    "status": "good",
                },
                {
                    "name": "Silk Scarves",
                    "stock": 8,
                    "reorder": 15,
                    "price": 89.99,
                    "status": "low",
                },
                {
                    "name": "Evening Dresses",
                    "stock": 23,
                    "reorder": 10,
                    "price": 199.99,
                    "status": "good",
                },
                {
                    "name": "Fall Jackets",
                    "stock": 0,
                    "reorder": 12,
                    "price": 149.99,
                    "status": "out",
                },
            ],
        },
        {
            "category": "Electronics",
            "items": [
                {
                    "name": "iPhone 15 Cases",
                    "stock": 67,
                    "reorder": 25,
                    "price": 49.99,
                    "status": "good",
                },
                {
                    "name": "Wireless Headphones",
                    "stock": 12,
                    "reorder": 15,
                    "price": 199.99,
                    "status": "low",
                },
                {
                    "name": "Smart Watches",
                    "stock": 34,
                    "reorder": 20,
                    "price": 299.99,
                    "status": "good",
                },
                {
                    "name": "Wireless Chargers",
                    "stock": 5,
                    "reorder": 20,
                    "price": 39.99,
                    "status": "low",
                },
            ],
        },
        {
            "category": "Men's Fashion",
            "items": [
                {
                    "name": "Dress Shirts",
                    "stock": 28,
                    "reorder": 15,
                    "price": 79.99,
                    "status": "good",
                },
                {
                    "name": "Leather Belts",
                    "stock": 15,
                    "reorder": 10,
                    "price": 59.99,
                    "status": "good",
                },
                {
                    "name": "Casual Sneakers",
                    "stock": 7,
                    "reorder": 12,
                    "price": 129.99,
                    "status": "low",
                },
                {
                    "name": "Winter Coats",
                    "stock": 19,
                    "reorder": 8,
                    "price": 249.99,
                    "status": "good",
                },
            ],
        },
    ]

    for category in inventory_data:
        st.markdown(f"#### {category['category']}")

        for item in category["items"]:
            show_inventory_item_card(item)


def show_inventory_item_card(item):
    """Display an enhanced inventory item card."""
    status_colors = {"good": "#10b981", "low": "#f59e0b", "out": "#ef4444"}
    status_icons = {"good": "✅", "low": "⚠️", "out": "❌"}

    col1, col2, col3 = st.columns([6, 2, 2])

    with col1:
        html_content = f"""
            <div class="inventory-item-card">
                <div class="item-header">
                    <span class="item-name">{item["name"]}</span>
                    <span class="item-status" style="color: {status_colors[item["status"]]}">
                        {status_icons[item["status"]]} {item["status"].upper()}
                    </span>
                </div>
                <div class="item-details">
                    <div><strong>Current Stock:</strong> {item["stock"]} units</div>
                    <div><strong>Reorder Point:</strong> {item["reorder"]} units</div>
                    <div><strong>Unit Price:</strong> ${item["price"]}</div>
                </div>
            </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)

    with col2:
        if item["status"] == "low":
            if st.button(
                "Reorder", key=f"reorder_{item['name']}", use_container_width=True
            ):
                st.success(f"Reorder request submitted for {item['name']}")
        elif item["status"] == "out":
            if st.button(
                "Urgent Order",
                key=f"urgent_{item['name']}",
                use_container_width=True,
                type="primary",
            ):
                st.success(f"Urgent order placed for {item['name']}")

    with col3:
        if st.button(
            "View Details", key=f"details_{item['name']}", use_container_width=True
        ):
            st.info(f"Detailed view for {item['name']} would open here")

    st.markdown("---")


def show_low_stock_alerts():
    """Display low stock alerts."""
    st.markdown("### ⚠️ Low Stock Alerts")

    alerts = [
        {
            "item": "Silk Scarves",
            "current": 8,
            "reorder": 15,
            "category": "Women's Fashion",
            "severity": "medium",
        },
        {
            "item": "Fall Jackets",
            "current": 0,
            "reorder": 12,
            "category": "Women's Fashion",
            "severity": "high",
        },
        {
            "item": "Wireless Headphones",
            "current": 12,
            "reorder": 15,
            "category": "Electronics",
            "severity": "low",
        },
        {
            "item": "Wireless Chargers",
            "current": 5,
            "reorder": 20,
            "category": "Electronics",
            "severity": "medium",
        },
        {
            "item": "Casual Sneakers",
            "current": 7,
            "reorder": 12,
            "category": "Men's Fashion",
            "severity": "low",
        },
    ]

    for alert in alerts:
        show_alert_card(alert)


def show_alert_card(alert):
    """Display an alert card."""

    html_content = f"""
        <div class="alert-card">
            <div class="alert-type">{alert["item"]} - {alert["category"]}</div>
            <div class="alert-details">
                Current Stock: {alert["current"]} units | Reorder Point: {alert["reorder"]} units
            </div>
        </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def show_stock_trends():
    """Display stock trends."""
    st.markdown("### 📈 Stock Trends")
    st.info("Stock trend analytics would be displayed here with charts and graphs.")


def show_reorder_management():
    """Display reorder management."""
    st.markdown("### 🔄 Reorder Management")
    st.info("Reorder management interface would be displayed here.")


# Add custom CSS for inventory components
st.markdown(
    """
    <style>
    /* Global font improvements */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Modern inventory summary card styling - Clean styling without colored borders */
    .inventory-summary-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .inventory-summary-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .inventory-summary-card .inventory-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .inventory-summary-card .inventory-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        display: block;
        line-height: 1.2;
    }
    
    .inventory-summary-card .inventory-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
        display: block;
    }
    
    .inventory-summary-card .inventory-detail {
        font-size: 0.875rem;
        color: #94a3b8;
        font-weight: 400;
    }
    
    /* Enhanced inventory item cards */
    .inventory-item-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .inventory-item-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    }
    
    .item-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .item-name {
        font-weight: 700;
        font-size: 1.25rem;
        color: #1e293b;
        line-height: 1.3;
    }
    
    .item-status {
        font-weight: 600;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.9);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .item-details {
        color: #475569;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .item-details div {
        margin-bottom: 0.5rem;
        padding: 0.25rem 0;
    }
    
    .item-details strong {
        color: #334155;
        font-weight: 600;
    }
    
    /* Enhanced alert cards - Clean without colored borders */
    .alert-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
    }
    
    .alert-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    .alert-type {
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-size: 1.125rem;
    }
    
    .alert-details {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Enhanced selectbox styling */
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Page title styling */
    h1 {
        color: #1e293b;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h3 {
        color: #334155;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h4 {
        color: #475569;
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
    }
    
    /* Enhanced markdown text */
    .stMarkdown p {
        font-size: 1rem;
        line-height: 1.6;
        color: #64748b;
    }
    
    /* Success/info message styling */
    .stSuccess, .stInfo {
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 500;
    }
    </style>
""",
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    main()
