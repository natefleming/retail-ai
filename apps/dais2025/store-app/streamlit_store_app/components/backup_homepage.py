"""Homepage components for the retail store employee iPad app."""

from datetime import datetime

import streamlit as st
import streamlit_modal as modal
from streamlit_card import card

from components.chat import show_chat_container
from utils.database import get_stores

# Import VP dashboard functions from the components directory
try:
    from components.vp_executive import show_vp_homepage
except ImportError:
    # Fallback if the import structure is different
    def show_vp_homepage():
        st.error("VP Dashboard not available. Please check the installation.")


def show_notifications_modal():
    """Display notifications in an expandable modal."""
    # Initialize notification state
    if "show_notifications" not in st.session_state:
        st.session_state.show_notifications = False

    # Notification button with count badge
    notification_count = 4  # Mock count

    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button(
            f"🔔 {notification_count}",
            key="notifications_toggle",
            help="View notifications",
        ):
            st.session_state.show_notifications = (
                not st.session_state.show_notifications
            )

    # Show notifications modal if toggled
    if st.session_state.show_notifications:
        with st.expander("📢 Notifications", expanded=True):
            # Categorized notifications
            st.markdown("#### 🚨 Urgent")
            urgent_notifications = [
                {
                    "message": "Security system maintenance in 30 minutes - Electronics section",
                    "time": "5 min ago",
                },
                {
                    "message": "Platinum Member arriving at 2 PM - Personal styling appointment requires immediate assignment",
                    "time": "15 min ago",
                },
            ]

            for notif in urgent_notifications:
                st.markdown(
                    f"""
                    <div class="notification-item urgent">
                        <div class="notification-message">{notif["message"]}</div>
                        <div class="notification-time">{notif["time"]}</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("#### ⚠️ Important")
            important_notifications = [
                {
                    "message": "New designer collection arriving tomorrow - Prepare display area",
                    "time": "1 hour ago",
                },
                {
                    "message": "Staff meeting moved to 3 PM in conference room",
                    "time": "2 hours ago",
                },
            ]

            for notif in important_notifications:
                st.markdown(
                    f"""
                    <div class="notification-item important">
                        <div class="notification-message">{notif["message"]}</div>
                        <div class="notification-time">{notif["time"]}</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )


def show_kpi_summary():
    """Display condensed KPI dashboard for store managers."""
    st.markdown("### Store Performance")

    # Mock retail data
    today_sales = 28750.00
    yesterday_sales = 24320.00
    sales_change = ((today_sales - yesterday_sales) / yesterday_sales) * 100

    pending_orders = 15
    completed_orders = 89

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-summary-card sales">
                <div class="kpi-icon">💰</div>
                <div class="kpi-value">${today_sales:,.0f}</div>
                <div class="kpi-label">Today's Sales</div>
                <div class="kpi-change positive">+{sales_change:.1f}%</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-summary-card orders">
                <div class="kpi-icon">📦</div>
                <div class="kpi-value">{completed_orders}</div>
                <div class="kpi-label">Orders Complete</div>
                <div class="kpi-change">{pending_orders} pending</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="kpi-summary-card traffic">
                <div class="kpi-icon">👥</div>
                <div class="kpi-value">247</div>
                <div class="kpi-label">Customers Today</div>
                <div class="kpi-change">Peak: 2-4 PM</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="kpi-summary-card conversion">
                <div class="kpi-icon">📈</div>
                <div class="kpi-value">68%</div>
                <div class="kpi-label">Conversion Rate</div>
                <div class="kpi-change positive">+5% vs avg</div>
            </div>
        """,
            unsafe_allow_html=True,
        )


def show_inventory_summary():
    """Display condensed inventory status for all employees."""
    st.markdown("### Inventory Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="inventory-summary-card critical">
                <div class="inventory-icon">🚨</div>
                <div class="inventory-value">3</div>
                <div class="inventory-label">Critical Stock</div>
                <div class="inventory-detail">Designer Jeans, iPhone Cases</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="inventory-summary-card low">
                <div class="inventory-icon">⚠️</div>
                <div class="inventory-value">12</div>
                <div class="inventory-label">Low Stock</div>
                <div class="inventory-detail">Seasonal items</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="inventory-summary-card good">
                <div class="inventory-icon">✅</div>
                <div class="inventory-value">892</div>
                <div class="inventory-label">Well Stocked</div>
                <div class="inventory-detail">Core inventory</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="inventory-summary-card new">
                <div class="inventory-icon">🆕</div>
                <div class="inventory-value">24</div>
                <div class="inventory-label">New Arrivals</div>
                <div class="inventory-detail">Fall collection</div>
            </div>
        """,
            unsafe_allow_html=True,
        )


def show_manager_summary_cards():
    """Display summary cards for store managers with navigation."""
    st.markdown("### Quick Access")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 Daily Operations", key="daily_ops", use_container_width=True):
            st.switch_page("pages/daily_operations.py")

        st.markdown(
            """
            <div class="summary-card operations">
                <div class="summary-stats">
                    <div class="stat-item">
                        <span class="stat-value">5/8</span>
                        <span class="stat-label">Tasks Complete</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">2</span>
                        <span class="stat-label">Urgent Items</span>
                    </div>
                </div>
                <div class="summary-preview">
                    • Morning inventory ✅<br>
                    • Vendor delivery 🔄<br>
                    • Staff meeting ⏳
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        if st.button("Team Insights", key="team_insights", use_container_width=True):
            st.switch_page("pages/team_insights.py")

        st.markdown(
            """
            <div class="summary-card team">
                <div class="summary-stats">
                    <div class="stat-item">
                        <span class="stat-value">12/15</span>
                        <span class="stat-label">Staff Present</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">94%</span>
                        <span class="stat-label">Avg Performance</span>
                    </div>
                </div>
                <div class="summary-preview">
                    🏆 Top: Sarah Chen (98%)<br>
                    ⚠️ Coverage gap: 3-4 PM<br>
                    3 shift changes today
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        if st.button(
            "Detailed Inventory", key="detailed_inventory", use_container_width=True
        ):
            st.switch_page("pages/inventory.py")

        st.markdown(
            """
            <div class="summary-card inventory">
                <div class="summary-stats">
                    <div class="stat-item">
                        <span class="stat-value">$2.1M</span>
                        <span class="stat-label">Total Value</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">15</span>
                        <span class="stat-label">Reorder Needed</span>
                    </div>
                </div>
                <div class="summary-preview">
                    📱 Electronics: 95% stocked<br>
                    👗 Apparel: 87% stocked<br>
                    👟 Footwear: 92% stocked
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )


def show_associate_homepage_with_chat(chat_modal, chat_notifications):
    """Display homepage content for store associates with integrated chat button."""
    # Create tabs with chat button on the same line
    col1, col2 = st.columns([8, 2])

    with col1:
        # Main content in tabs - fully tab-based experience
        tab1, tab2, tab3, tab4 = st.tabs(
            ["My Work", "Schedule", "Products", "Performance"]
        )

    with col2:
        # Chat button aligned with tabs
        if chat_notifications > 0:
            button_text = f"AI Assistant ({chat_notifications})"
        else:
            button_text = "AI Assistant"

        if st.button(
            button_text,
            key="associate_chat_btn",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.chat_notifications = 0
            chat_modal.open()

    # Tab content
    with tab1:
        show_my_work_tab()

    with tab2:
        show_schedule_tab()

    with tab3:
        show_products_tab()

    with tab4:
        show_performance_tab()


def show_associate_homepage():
    """Display homepage content for store associates with improved tab-based layout."""
    # Add custom CSS for better tab styling with stronger selectors
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

    # Main content in tabs - fully tab-based experience with clean styling
    tab1, tab2, tab3, tab4 = st.tabs(
        ["My Tasks", "Schedule", "Products", "Performance"]
    )

    with tab1:
        show_my_work_tab()

    with tab2:
        show_schedule_tab()

    with tab3:
        show_products_tab()

    with tab4:
        show_performance_tab()


def show_my_work_tab():
    """Display the My Tasks tab with tasks and immediate priorities."""
    # Initialize personal shopping modal state
    if "show_personal_shopping_modal" not in st.session_state:
        st.session_state.show_personal_shopping_modal = False

    # Add modern CSS for associate overview cards
    st.markdown(
        """
    <style>
    .associate-overview-card {
        width: 100%;
        height: 140px;
        border-radius: 16px;
        margin: 0;
        padding: 1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .associate-overview-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.2) !important;
    }
    
    .associate-card-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        display: block;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
    }
    
    .associate-card-value {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        display: block;
        line-height: 1.2;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .associate-card-label {
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.9;
    }
    
    .associate-card-label:hover {
        opacity: 1;
        letter-spacing: 1.5px;
    }
    
    .modern-work-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
    }
    
    .modern-work-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
            opacity: 1;
        }
        50% {
            transform: scale(1.05);
            opacity: 0.8;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Quick status overview at top - modern cards matching manager style
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="associate-overview-card" style="
                box-shadow: 0 4px 20px rgba(34, 197, 94, 0.4), 0 1px 3px rgba(0, 0, 0, 0.1);
                background: linear-gradient(135deg, #22c55e 0%, #16a34a 50%, #15803d 100%);
                border: 1px solid rgba(34, 197, 94, 0.3);
                color: white;
            ">
                <div class="associate-card-icon">🟢</div>
                <div class="associate-card-value">On Shift</div>
                <div class="associate-card-label">3h 37m remaining</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="associate-overview-card" style="
                box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4), 0 1px 3px rgba(0, 0, 0, 0.1);
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
                border: 1px solid rgba(59, 130, 246, 0.3);
                color: white;
            ">
                <div class="associate-card-icon">📋</div>
                <div class="associate-card-value">7 Tasks</div>
                <div class="associate-card-label">3 high priority</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="associate-overview-card" style="
                box-shadow: 0 4px 20px rgba(245, 158, 11, 0.4), 0 1px 3px rgba(0, 0, 0, 0.1);
                background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #d97706 100%);
                border: 1px solid rgba(245, 158, 11, 0.3);
                color: white;
            ">
                <div class="associate-card-icon">📦</div>
                <div class="associate-card-value">Inventory</div>
                <div class="associate-card-label">3 critical items</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="associate-overview-card" style="
                box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4), 0 1px 3px rgba(0, 0, 0, 0.1);
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #b91c1c 100%);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: white;
            ">
                <div class="associate-card-icon">🔔</div>
                <div class="associate-card-value">4 Alerts</div>
                <div class="associate-card-label">2 urgent</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Today's Priorities")

        # Fixed height container for scrollable priorities
        with st.container(height=400):
            # Personal shopping appointment - clickable card with Details button
            col1_task, col2_task = st.columns([4, 1])

            with col1_task:
                st.markdown(
                    """
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1rem;
                        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                        border-radius: 12px;
                        color: white;
                        margin-bottom: 0.5rem;
                        position: relative;
                    ">
                        <div>
                            <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.25rem;">Personal shopping appointment</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Due: 2:00 PM • Emma Rodriguez</div>
                        </div>
                        <div style="
                            background: rgba(255,255,255,0.2);
                            padding: 0.5rem 1rem;
                            border-radius: 8px;
                            font-weight: 600;
                            font-size: 0.875rem;
                        ">Service</div>
                        <div style="
                            position: absolute;
                            top: -8px;
                            right: -8px;
                            background: #ef4444;
                            color: white;
                            padding: 4px 8px;
                            border-radius: 12px;
                            font-size: 0.75rem;
                            font-weight: 700;
                            animation: pulse 2s infinite;
                        ">NEW • 1 min ago</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2_task:
                # Details button for personal shopping appointment
                if st.button(
                    "Details", key="personal_shopping_details", use_container_width=True
                ):
                    st.session_state.show_personal_shopping_modal = True
                    st.rerun()

            # BOPIS Order task with Details button
            col1_bopis, col2_bopis = st.columns([4, 1])

            with col1_bopis:
                st.markdown(
                    """
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1rem;
                        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                        border-radius: 12px;
                        color: white;
                        margin-bottom: 0.5rem;
                    ">
                        <div>
                            <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.25rem;">BOPIS Order #B2024-0156</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Due: 10:30 AM • Sarah Johnson</div>
                        </div>
                        <div style="
                            background: rgba(255,255,255,0.2);
                            padding: 0.5rem 1rem;
                            border-radius: 8px;
                            font-weight: 600;
                            font-size: 0.875rem;
                        ">BOPIS</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2_bopis:
                if st.button("Details", key="bopis_details", use_container_width=True):
                    st.info(
                        "📦 BOPIS Order #B2024-0156 - Sarah Johnson\n\nItems: 2x Women's Blazer (Navy), 1x Dress Pants (Black)\nPickup Time: 10:30 AM\nStatus: Ready for pickup"
                    )

            # Restock task with Details button
            col1_restock, col2_restock = st.columns([4, 1])

            with col1_restock:
                st.markdown(
                    """
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1rem;
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        border-radius: 12px;
                        color: white;
                        margin-bottom: 0.5rem;
                    ">
                        <div>
                            <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.25rem;">Restock designer section</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Due: 12:00 PM • Floor 2</div>
                        </div>
                        <div style="
                            background: rgba(255,255,255,0.2);
                            padding: 0.5rem 1rem;
                            border-radius: 8px;
                            font-weight: 600;
                            font-size: 0.875rem;
                        ">Restock</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2_restock:
                if st.button(
                    "Details", key="restock_details", use_container_width=True
                ):
                    st.info(
                        "📦 Restock Designer Section - Floor 2\n\nItems needed: 15x Designer Handbags, 8x Luxury Scarves\nLocation: Aisle D2-D4\nPriority: Medium"
                    )

        if st.button(
            "📋 View All Tasks", key="view_all_tasks", use_container_width=True
        ):
            st.switch_page("pages/my_tasks.py")

    with col2:
        st.markdown("#### Current Assignment")

        st.markdown(
            """
            <div class="modern-work-card">
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <div style="
                        text-align: center;
                        padding: 1.5rem;
                        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                        border-radius: 12px;
                        color: white;
                        margin-bottom: 1rem;
                    ">
                        <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">Women's Fashion</div>
                        <div style="font-size: 1rem; opacity: 0.9;">Designer Area</div>
                    </div>
                    <div style="
                        display: flex;
                        flex-direction: column;
                        gap: 0.75rem;
                        padding: 1rem;
                        background: #f8fafc;
                        border-radius: 12px;
                        border: 1px solid #e2e8f0;
                    ">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: 600; color: #64748b;">Coverage:</span>
                            <span style="color: #1e293b; font-weight: 600;">Solo until 2 PM</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: 600; color: #64748b;">Break Due:</span>
                            <span style="color: #ef4444; font-weight: 600;">Now!</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: 600; color: #64748b;">Performance:</span>
                            <span style="color: #10b981; font-weight: 600;">94%</span>
                        </div>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🚨 Quick Actions")

        # Quick action buttons with modern styling
        if st.button("🛒 Check BOPIS Orders", use_container_width=True):
            st.switch_page("pages/my_tasks.py")

        if st.button("📦 Report Low Stock", use_container_width=True):
            st.info("Stock reporting form would open")

        if st.button("🤝 Request Help", use_container_width=True):
            st.info("Help request sent to manager")

        if st.button("☕ Take Break", use_container_width=True):
            st.success("Break started - timer activated")

    # Personal Shopping Appointment Modal
    if st.session_state.show_personal_shopping_modal:

        @st.dialog("Personal Shopping Appointment", width="large")
        def show_personal_shopping_details():
            st.markdown(
                """
            <style>
            /* Modal styling */
            div[data-testid="stDialog"] {
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                right: 0 !important;
                bottom: 0 !important;
                display: flex !important;
                align-items: flex-start !important;
                justify-content: center !important;
                z-index: 1000 !important;
                background: rgba(0, 0, 0, 0.5) !important;
                width: 100vw !important;
                height: 100vh !important;
                padding-top: 2rem !important;
                overflow-y: auto !important;
            }
            div[data-testid="stDialog"] > div {
                max-width: 900px !important;
                width: 85vw !important;
                position: relative !important;
                background: white !important;
                border-radius: 16px !important;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
                margin: 0 auto !important;
                max-height: calc(100vh - 4rem) !important;
                overflow-y: auto !important;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            # Header with urgent styling
            st.markdown(
                """
            <div style="
                background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 50%, #6d28d9 100%);
                color: white;
                padding: 32px;
                border-radius: 20px;
                margin-bottom: 32px;
                position: relative;
                overflow: hidden;
                border: 3px solid #c4b5fd;
                box-shadow: 0 0 30px rgba(139, 92, 246, 0.4);
            ">
                <div style="
                    position: absolute;
                    top: -50%;
                    right: -20%;
                    width: 200px;
                    height: 200px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 50%;
                    filter: blur(40px);
                "></div>
                <div style="position: relative; z-index: 2;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                        <div style="display: flex; align-items: center; gap: 16px;">
                            <div style="
                                background: rgba(255, 255, 255, 0.2);
                                padding: 16px;
                                border-radius: 20px;
                                backdrop-filter: blur(10px);
                            ">
                                <span style="font-size: 32px;">🛍️</span>
                            </div>
                            <div>
                                <div style="
                                    background: #fef3c7;
                                    color: #92400e;
                                    padding: 8px 16px;
                                    border-radius: 25px;
                                    font-size: 14px;
                                    font-weight: 900;
                                    text-transform: uppercase;
                                    letter-spacing: 1px;
                                    margin-bottom: 8px;
                                    border: 2px solid #fbbf24;
                                ">URGENT - PERSONAL STYLING APPOINTMENT</div>
                                <h1 style="
                                    margin: 0;
                                    font-size: 28px;
                                    font-weight: 900;
                                    letter-spacing: -0.02em;
                                ">ASSIGNMENT NOTIFICATION</h1>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="
                                background: rgba(255, 255, 255, 0.2);
                                padding: 12px 20px;
                                border-radius: 15px;
                                backdrop-filter: blur(10px);
                                border: 1px solid rgba(255, 255, 255, 0.3);
                            ">
                                <div style="
                                    font-size: 24px;
                                    font-weight: 900;
                                    margin-bottom: 4px;
                                ">⏰ 55 MIN</div>
                                <div style="
                                    font-size: 12px;
                                    opacity: 0.9;
                                    text-transform: uppercase;
                                    letter-spacing: 1px;
                                ">Until Appointment</div>
                            </div>
                        </div>
                    </div>
                    <div style="
                        background: rgba(255, 255, 255, 0.15);
                        padding: 20px;
                        border-radius: 15px;
                        backdrop-filter: blur(10px);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                    ">
                        <div style="
                            font-size: 18px;
                            font-weight: 600;
                            line-height: 1.6;
                            margin-bottom: 12px;
                        ">
                            <span style="
                                background: #fbbf24;
                                color: #92400e;
                                padding: 4px 8px;
                                border-radius: 8px;
                                font-weight: 900;
                                margin-right: 8px;
                            ">PLATINUM MEMBER</span>
                            Emma Rodriguez arriving in 55 minutes
                        </div>
                        <div style="
                            font-size: 16px;
                            opacity: 0.95;
                            line-height: 1.5;
                        ">
                            • <strong>Personal styling appointment assigned to you</strong><br>
                            • <strong>Customer intelligence and preparation recommendations included</strong>
                        </div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Customer Information Section
            st.markdown("### 👤 Customer Information")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                <div style="
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 4px solid #8b5cf6;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #64748b;">Customer</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">Victoria Chen</div>
                    <div style="font-size: 14px; color: #64748b;">Platinum Member (5+ years)</div>
                </div>
                
                <div style="
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 4px solid #10b981;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #64748b;">Appointment</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">11:00 AM (55 minutes remaining)</div>
                    <div style="font-size: 14px; color: #64748b;">Personal Shopping - Women's Professional Wear</div>
                </div>
                
                <div style="
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 4px solid #f59e0b;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #64748b;">Average Spend</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">$850 per visit</div>
                    <div style="font-size: 14px; color: #64748b;">High-value customer</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    """
                <div style="
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 4px solid #3b82f6;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #64748b;">Last Purchase</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">$1,200 business wardrobe</div>
                    <div style="font-size: 14px; color: #64748b;">3 weeks ago</div>
                </div>
                
                <div style="
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 4px solid #ef4444;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #64748b;">Style Profile</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">Classic Professional</div>
                    <div style="font-size: 14px; color: #64748b;">Size 8, prefers neutral colors</div>
                </div>
                
                <div style="
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 4px solid #8b5cf6;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #64748b;">Purchase Triggers</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">Career Milestones</div>
                    <div style="font-size: 14px; color: #64748b;">Buys for promotions, presentations, networking events</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Action Buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    "✅ Accept Assignment", type="primary", use_container_width=True
                ):
                    st.success(
                        "✅ Assignment accepted! Customer preparation initiated."
                    )
                    st.balloons()
                    st.session_state.show_personal_shopping_modal = False
                    st.rerun()

            with col2:
                if st.button("📱 View Customer Profile", use_container_width=True):
                    st.info("📱 Full customer profile would open in new window")

            with col3:
                if st.button("❌ Close", use_container_width=True):
                    st.session_state.show_personal_shopping_modal = False
                    st.rerun()

        # Show the modal
        show_personal_shopping_details()


def show_schedule_tab():
    """Display the Schedule tab with shift info and time tracking."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⏰ Current Shift")

        st.markdown(
            """
            <div class="modern-work-card">
                <div style="display: flex; flex-direction: column; gap: 1.5rem;">
                    <div style="
                        text-align: center;
                        padding: 1.5rem;
                        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; line-height: 1.2;">12:23 PM</div>
                        <div style="font-size: 1rem; opacity: 0.9;">Current Time</div>
                    </div>
                    <div style="
                        padding: 1rem;
                        background: #f8fafc;
                        border-radius: 12px;
                        border: 1px solid #e2e8f0;
                    ">
                        <div style="
                            background: #e2e8f0;
                            border-radius: 12px;
                            height: 12px;
                            margin-bottom: 0.75rem;
                            overflow: hidden;
                            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
                        ">
                            <div style="
                                background: linear-gradient(90deg, #3b82f6, #10b981);
                                height: 100%;
                                width: 55%;
                                border-radius: 12px;
                                transition: width 0.3s ease;
                                box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
                            "></div>
                        </div>
                        <div style="
                            text-align: center;
                            color: #64748b;
                            font-size: 1rem;
                            font-weight: 500;
                            margin-bottom: 1rem;
                        ">4h 23m worked • 3h 37m remaining</div>
                        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: 600; color: #64748b;">Shift:</span>
                                <span style="color: #1e293b; font-weight: 600;">8:00 AM - 4:00 PM</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: 600; color: #64748b;">Break:</span>
                                <span style="color: #ef4444; font-weight: 600;">12:00 - 12:30 PM (Due now!)</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: 600; color: #64748b;">Department:</span>
                                <span style="color: #1e293b; font-weight: 600;">Women's Fashion</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("View Full Schedule", use_container_width=True):
            st.switch_page("pages/my_schedule.py")

    with col2:
        st.markdown("#### This Week")

        st.markdown(
            """
            <div class="modern-work-card">
                <div style="display: flex; flex-direction: column; gap: 1.5rem;">
                    <div style="
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 1rem;
                        padding: 1rem;
                        background: #f8fafc;
                        border-radius: 12px;
                        border: 1px solid #e2e8f0;
                    ">
                        <div style="text-align: center; padding: 1rem;">
                            <div style="font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem;">32/40</div>
                            <div style="font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Hours</div>
                        </div>
                        <div style="text-align: center; padding: 1rem;">
                            <div style="font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem;">4/5</div>
                            <div style="font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Days</div>
                        </div>
                        <div style="text-align: center; padding: 1rem;">
                            <div style="font-size: 2rem; font-weight: 700; color: #10b981; margin-bottom: 0.25rem;">94%</div>
                            <div style="font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Performance</div>
                        </div>
                        <div style="text-align: center; padding: 1rem;">
                            <div style="font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem;">$2,847</div>
                            <div style="font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Sales</div>
                        </div>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📝 Upcoming")

        st.markdown(
            """
            <div class="modern-work-card">
                <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1rem;
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div>
                            <div style="font-size: 1.1rem; font-weight: 700;">Tomorrow</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">8 AM - 4 PM - Electronics</div>
                        </div>
                        <div style="font-size: 1.5rem;">📱</div>
                    </div>
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1rem;
                        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div>
                            <div style="font-size: 1.1rem; font-weight: 700;">Friday</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">9 AM - 5 PM - Women's Fashion</div>
                        </div>
                        <div style="font-size: 1.5rem;">👗</div>
                    </div>
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1rem;
                        background: linear-gradient(135deg, #64748b 0%, #475569 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div>
                            <div style="font-size: 1.1rem; font-weight: 700;">Saturday</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Day Off</div>
                        </div>
                        <div style="font-size: 1.5rem;">🏖️</div>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )


def show_products_tab():
    """Display the Products tab with promotions and product info."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔥 Active Promotions")

        st.markdown(
            """
            <div class="modern-work-card">
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1.25rem;
                        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div>
                            <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.25rem;">Fall Fashion Sale</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Ends: End of week</div>
                        </div>
                        <div style="
                            background: rgba(255,255,255,0.2);
                            padding: 0.75rem 1.25rem;
                            border-radius: 8px;
                            font-weight: 700;
                            font-size: 1.1rem;
                        ">40% off</div>
                    </div>
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1.25rem;
                        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div>
                            <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.25rem;">Designer Handbags</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Ends: Tomorrow</div>
                        </div>
                        <div style="
                            background: rgba(255,255,255,0.2);
                            padding: 0.75rem 1.25rem;
                            border-radius: 8px;
                            font-weight: 700;
                            font-size: 1.1rem;
                        ">25% off</div>
                    </div>
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1.25rem;
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div>
                            <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.25rem;">Tech Accessories</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Ends: 3 days</div>
                        </div>
                        <div style="
                            background: rgba(255,255,255,0.2);
                            padding: 0.75rem 1.25rem;
                            border-radius: 8px;
                            font-weight: 700;
                            font-size: 1.1rem;
                        ">Buy 2 Get 1</div>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("View All Promotions", use_container_width=True):
            st.switch_page("pages/products_promotions.py")

    with col2:
        st.markdown("#### 🆕 New This Week")

        st.markdown(
            """
            <div class="modern-work-card">
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <div style="
                        padding: 1.25rem;
                        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                        border-radius: 12px;
                        border: 1px solid #cbd5e1;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                            <div style="font-size: 1.25rem; font-weight: 700; color: #1e293b;">iPhone 15 Pro Cases</div>
                            <div style="font-size: 1.5rem;">📱</div>
                        </div>
                        <div style="color: #64748b; font-size: 0.9rem;">Electronics • E3</div>
                    </div>
                    <div style="
                        padding: 1.25rem;
                        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                        border-radius: 12px;
                        border: 1px solid #cbd5e1;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                            <div style="font-size: 1.25rem; font-weight: 700; color: #1e293b;">Winter Coats</div>
                            <div style="font-size: 1.5rem;">🧥</div>
                        </div>
                        <div style="color: #64748b; font-size: 0.9rem;">Women's Apparel • W2</div>
                    </div>
                    <div style="
                        padding: 1.25rem;
                        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                        border-radius: 12px;
                        border: 1px solid #cbd5e1;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                            <div style="font-size: 1.25rem; font-weight: 700; color: #1e293b;">Designer Sneakers</div>
                            <div style="font-size: 1.5rem;">👟</div>
                        </div>
                        <div style="color: #64748b; font-size: 0.9rem;">Footwear • F4</div>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🔥 Trending")

        st.markdown(
            """
            <div class="modern-work-card">
                <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1rem;
                        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div>
                            <div style="font-size: 1.1rem; font-weight: 700;">🎧 Wireless Earbuds Pro</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Growth: +45%</div>
                        </div>
                        <div style="font-size: 1.5rem;">📈</div>
                    </div>
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1rem;
                        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div>
                            <div style="font-size: 1.1rem; font-weight: 700;">👗 Oversized Blazers</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Growth: +60%</div>
                        </div>
                        <div style="font-size: 1.5rem;">🔥</div>
                    </div>
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1rem;
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div>
                            <div style="font-size: 1.1rem; font-weight: 700;">⌚ Minimalist Watches</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Growth: +35%</div>
                        </div>
                        <div style="font-size: 1.5rem;">⭐</div>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )


def show_performance_tab():
    """Display the Performance tab with personal metrics and achievements."""

    # Add CSS for performance grid cards
    st.markdown(
        """
    <style>
    .performance-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        height: 100%;
    }
    
    .performance-metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .performance-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
    }
    
    .trend-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    
    .trend-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Performance Metrics")

        # Fixed height container for performance metrics in 2x2 grid
        with st.container(height=350):
            st.markdown(
                """
                <div class="performance-grid">
                    <div class="performance-metric-card" style="border-left: 4px solid #10b981;">
                        <div class="metric-icon">🛒</div>
                        <div class="metric-value" style="color: #10b981;">12</div>
                        <div class="metric-label">BOPIS Orders</div>
                    </div>
                    <div class="performance-metric-card" style="border-left: 4px solid #3b82f6;">
                        <div class="metric-icon">🤝</div>
                        <div class="metric-value" style="color: #3b82f6;">8</div>
                        <div class="metric-label">Customer Assists</div>
                    </div>
                    <div class="performance-metric-card" style="border-left: 4px solid #fbbf24;">
                        <div class="metric-icon">💰</div>
                        <div class="metric-value" style="color: #fbbf24;">$2,450</div>
                        <div class="metric-label">Sales Today</div>
                    </div>
                    <div class="performance-metric-card" style="border-left: 4px solid #8b5cf6;">
                        <div class="metric-icon">⭐</div>
                        <div class="metric-value" style="color: #8b5cf6;">4.8/5</div>
                        <div class="metric-label">Customer Rating</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 🏆 Overall Performance")

        st.markdown(
            """
            <div class="modern-work-card">
                <div style="
                    text-align: center;
                    padding: 2rem;
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    border-radius: 12px;
                    color: white;
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        position: absolute;
                        top: -50%;
                        right: -20%;
                        width: 200px;
                        height: 200px;
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 50%;
                        filter: blur(40px);
                    "></div>
                    <div style="position: relative; z-index: 2;">
                        <div style="font-size: 4rem; font-weight: 900; margin-bottom: 0.5rem; line-height: 1; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">94%</div>
                        <div style="font-size: 1.2rem; font-weight: 600; opacity: 0.9;">Overall Score</div>
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">Excellent Performance</div>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### 📈 Performance Trends")

        # Fixed height container for performance trends in 2x2 grid
        with st.container(height=350):
            st.markdown(
                """
                <div class="performance-grid">
                    <div class="trend-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div style="font-weight: 700; color: #1e293b;">Sales Trend</div>
                            <div style="color: #10b981; font-weight: 600;">↗ +18%</div>
                        </div>
                        <div style="
                            background: linear-gradient(90deg, #e2e8f0 0%, #10b981 100%);
                            height: 8px;
                            border-radius: 4px;
                            margin-bottom: 0.5rem;
                        "></div>
                        <div style="font-size: 0.85rem; color: #64748b;">vs. last week</div>
                    </div>
                    <div class="trend-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div style="font-weight: 700; color: #1e293b;">Customer Rating</div>
                            <div style="color: #fbbf24; font-weight: 600;">↗ +0.3</div>
                        </div>
                        <div style="
                            background: linear-gradient(90deg, #e2e8f0 0%, #fbbf24 96%);
                            height: 8px;
                            border-radius: 4px;
                            margin-bottom: 0.5rem;
                        "></div>
                        <div style="font-size: 0.85rem; color: #64748b;">4.8/5 average</div>
                    </div>
                    <div class="trend-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div style="font-weight: 700; color: #1e293b;">Task Completion</div>
                            <div style="color: #3b82f6; font-weight: 600;">↗ +12%</div>
                        </div>
                        <div style="
                            background: linear-gradient(90deg, #e2e8f0 0%, #3b82f6 87%);
                            height: 8px;
                            border-radius: 4px;
                            margin-bottom: 0.5rem;
                        "></div>
                        <div style="font-size: 0.85rem; color: #64748b;">87% completion rate</div>
                    </div>
                    <div class="trend-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div style="font-weight: 700; color: #1e293b;">Speed Score</div>
                            <div style="color: #8b5cf6; font-weight: 600;">↗ +8%</div>
                        </div>
                        <div style="
                            background: linear-gradient(90deg, #e2e8f0 0%, #8b5cf6 92%);
                            height: 8px;
                            border-radius: 4px;
                            margin-bottom: 0.5rem;
                        "></div>
                        <div style="font-size: 0.85rem; color: #64748b;">92% efficiency</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 🏅 Recent Achievements")

        st.markdown(
            """
            <div class="modern-work-card">
                <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <div style="
                        display: flex;
                        align-items: center;
                        gap: 1rem;
                        padding: 1rem;
                        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div style="font-size: 1.5rem;">🌟</div>
                        <div>
                            <div style="font-size: 1.1rem; font-weight: 700;">Customer Service Excellence</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Recently earned</div>
                        </div>
                    </div>
                    <div style="
                        display: flex;
                        align-items: center;
                        gap: 1rem;
                        padding: 1rem;
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div style="font-size: 1.5rem;">💰</div>
                        <div>
                            <div style="font-size: 1.1rem; font-weight: 700;">Sales Target Exceeded</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Recently earned</div>
                        </div>
                    </div>
                    <div style="
                        display: flex;
                        align-items: center;
                        gap: 1rem;
                        padding: 1rem;
                        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                        border-radius: 12px;
                        color: white;
                    ">
                        <div style="font-size: 1.5rem;">⚡</div>
                        <div>
                            <div style="font-size: 1.1rem; font-weight: 700;">Speed Champion (BOPIS)</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">Recently earned</div>
                        </div>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )


def simulate_chat_notification():
    """Simulate receiving a chat notification (for demo purposes)."""
    if "chat_notifications" not in st.session_state:
        st.session_state.chat_notifications = 0

    # Only add notifications if chat is closed
    if not st.session_state.get("chat_window_open", False):
        st.session_state.chat_notifications += 1


def show_persistent_chat():
    """Display a floating chat icon in the lower right corner that opens a chat modal."""
    # Initialize chat state
    if "chat_notifications" not in st.session_state:
        st.session_state.chat_notifications = 0

    # Get current chat status
    chat_status = st.session_state.get("chat_status", "available")

    # Create notification badge HTML
    notification_badge = ""
    if st.session_state.chat_notifications > 0:
        notification_badge = f"""
        <span style="
            position: absolute;
            top: -5px;
            right: -5px;
            background: #ff4757;
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        ">{st.session_state.chat_notifications}</span>
        """

    # Create status indicator
    status_indicators = {
        "available": {"color": "#28a745", "pulse": ""},
        "typing": {"color": "#007bff", "pulse": "animation: pulse 1.5s infinite;"},
        "processing": {"color": "#ffc107", "pulse": "animation: pulse 1s infinite;"},
        "error": {"color": "#dc3545", "pulse": "animation: pulse 2s infinite;"},
    }

    status_info = status_indicators.get(chat_status, status_indicators["available"])

    # Status indicator dot HTML
    status_dot = f"""
    <span style="
        position: absolute;
        bottom: -2px;
        left: -2px;
        background: {status_info["color"]};
        border: 2px solid white;
        border-radius: 50%;
        width: 16px;
        height: 16px;
        {status_info["pulse"]}
    "></span>
    """

    # Floating chat button HTML
    chat_button_html = f"""
    <style>
    @keyframes pulse {{
        0% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.1); opacity: 0.7; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    </style>
    
    <div id="floating-chat-container" style="
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        z-index: 1000;
    ">
        <div style="
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
            position: relative;
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(0,0,0,0.2)'"
           onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)'"
           onclick="document.getElementById('hidden-chat-btn').click()">
            💬
            {notification_badge}
            {status_dot}
        </div>
    </div>
    """

    # Display the floating chat icon
    st.markdown(chat_button_html, unsafe_allow_html=True)

    # Hidden button for modal trigger
    st.markdown(
        """
    <style>
    #hidden-chat-btn {
        display: none !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create the modal
    chat_modal = modal.Modal(
        title="AI Assistant", key="chat_modal", max_width=600, padding=20
    )

    # Hidden button to trigger modal
    if st.button("Open Chat", key="hidden_chat_btn", help="Open AI Assistant"):
        # Clear notifications when chat is opened
        st.session_state.chat_notifications = 0
        chat_modal.open()

    # Modal content
    if chat_modal.is_open():
        with chat_modal.container():
            # Get chat config with fallback
            chat_config = st.session_state.get("config", {}).get(
                "chat",
                {
                    "placeholder": "How can I help you today?",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            )

            # Show the chat container
            show_chat_container(chat_config)

    # Show appropriate homepage content based on user role (back to original functions)
    if user_role == "store_manager":
        # Show new tab-based manager homepage
        show_manager_homepage()
    else:
        # Show new tab-based associate homepage
        show_associate_homepage()


def show_manager_homepage_with_chat(chat_modal, chat_notifications):
    """Display tab-based homepage content for store managers with integrated chat button."""
    # Create tabs with chat button on the same line
    col1, col2 = st.columns([8, 2])

    with col1:
        # Main content in tabs - fully tab-based experience
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "Store Overview",
                "Operations",
                "Team",
                "📦 Inventory",
                "Analytics",
                "💡 Alerts",
            ]
        )

    with col2:
        # Chat button aligned with tabs
        if chat_notifications > 0:
            button_text = f"AI Assistant ({chat_notifications})"
        else:
            button_text = "AI Assistant"

        if st.button(
            button_text,
            key="manager_chat_btn",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.chat_notifications = 0
            chat_modal.open()

    # Tab content
    with tab1:
        show_manager_dashboard_tab()

    with tab2:
        show_manager_alerts_tab()

    with tab3:
        show_manager_operations_tab()

    with tab4:
        show_manager_team_tab()

    with tab5:
        show_manager_inventory_tab()

    with tab6:
        show_manager_analytics_tab()


def show_manager_homepage():
    """Display tab-based homepage content for store managers."""
    # Add custom CSS for better tab styling with stronger selectors
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

    # Main content in tabs - fully tab-based experience with clean styling
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Dashboard", "Alerts", "Operations", "Team", "Inventory", "Analytics"]
    )

    with tab1:
        show_manager_dashboard_tab()

    with tab2:
        show_manager_alerts_tab()

    with tab3:
        show_manager_operations_tab()

    with tab4:
        show_manager_team_tab()

    with tab5:
        show_manager_inventory_tab()

    with tab6:
        show_manager_analytics_tab()


def show_manager_dashboard_tab():
    """Display the Dashboard tab with key metrics."""
    # Add custom CSS for modern card hover effects
    st.markdown(
        """
    <style>
    .modern-overview-card {
        width: 100%;
        height: 140px;
        border-radius: 16px;
        margin: 0;
        padding: 1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .modern-overview-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.2) !important;
    }
    
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 0.25rem;
        transition: transform 0.3s ease;
    }
    
    .modern-overview-card:hover .card-icon {
        transform: scale(1.1);
    }
    
    .card-value {
        font-size: 2.25rem;
        color: white;
        font-weight: 900;
        margin-bottom: 0.125rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        line-height: 1;
        transition: all 0.3s ease;
    }
    
    .card-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .modern-overview-card:hover .card-label {
        color: rgba(255, 255, 255, 1);
        letter-spacing: 1.5px;
    }
    
    .modern-performance-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
    }
    
    .modern-performance-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Quick executive dashboard at top
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="modern-overview-card" style="
                box-shadow: 0 4px 20px rgba(34, 197, 94, 0.4), 0 1px 3px rgba(0, 0, 0, 0.1);
                background: linear-gradient(135deg, #22c55e 0%, #16a34a 50%, #15803d 100%);
                border: 1px solid rgba(34, 197, 94, 0.3);
            ">
                <div class="card-icon">💰</div>
                <div class="card-value">$28,750</div>
                <div class="card-label">Today's Sales (+18%)</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="modern-overview-card" style="
                box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4), 0 1px 3px rgba(0, 0, 0, 0.1);
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
                border: 1px solid rgba(59, 130, 246, 0.3);
            ">
                <div class="card-icon">👥</div>
                <div class="card-value">12/15</div>
                <div class="card-label">Staff Present (94% avg)</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="modern-overview-card" style="
                box-shadow: 0 4px 20px rgba(245, 158, 11, 0.4), 0 1px 3px rgba(0, 0, 0, 0.1);
                background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #d97706 100%);
                border: 1px solid rgba(245, 158, 11, 0.3);
            ">
                <div class="card-icon">📋</div>
                <div class="card-value">5/8</div>
                <div class="card-label">Tasks Complete (2 urgent)</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="modern-overview-card" style="
                box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4), 0 1px 3px rgba(0, 0, 0, 0.1);
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #b91c1c 100%);
                border: 1px solid rgba(239, 68, 68, 0.3);
            ">
                <div class="card-icon">🔔</div>
                <div class="card-value">4</div>
                <div class="card-label">Active Alerts</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Add CSS for manager performance grid cards
    st.markdown(
        """
    <style>
    .manager-performance-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        height: 100%;
    }
    
    .manager-metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .manager-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .manager-metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    .manager-metric-value {
        font-size: 2rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        line-height: 1;
    }
    
    .manager-metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
    }
    
    .manager-metric-change {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    
    .manager-trend-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    
    .manager-trend-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Today's Performance")

        # Fixed height container for today's performance metrics in 2x2 grid
        with st.container(height=350):
            st.markdown(
                """
                <div class="manager-performance-grid">
                    <div class="manager-metric-card" style="border-left: 4px solid #10b981;">
                        <div class="manager-metric-icon">💰</div>
                        <div class="manager-metric-value" style="color: #10b981;">$28,750</div>
                        <div class="manager-metric-label">Sales Target</div>
                        <div class="manager-metric-change" style="color: #10b981;">96% (+18%)</div>
                    </div>
                    <div class="manager-metric-card" style="border-left: 4px solid #3b82f6;">
                        <div class="manager-metric-icon">👥</div>
                        <div class="manager-metric-value" style="color: #3b82f6;">247</div>
                        <div class="manager-metric-label">Customer Traffic</div>
                        <div class="manager-metric-change" style="color: #64748b;">Peak: 2-4 PM</div>
                    </div>
                    <div class="manager-metric-card" style="border-left: 4px solid #fbbf24;">
                        <div class="manager-metric-icon">📈</div>
                        <div class="manager-metric-value" style="color: #fbbf24;">68%</div>
                        <div class="manager-metric-label">Conversion Rate</div>
                        <div class="manager-metric-change" style="color: #10b981;">+5% vs avg</div>
                    </div>
                    <div class="manager-metric-card" style="border-left: 4px solid #8b5cf6;">
                        <div class="manager-metric-icon">💳</div>
                        <div class="manager-metric-value" style="color: #8b5cf6;">$171.50</div>
                        <div class="manager-metric-label">Avg Transaction</div>
                        <div class="manager-metric-change" style="color: #10b981;">+12% vs avg</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### 📈 Performance Trends")

        # Fixed height container for performance trends in 2x2 grid
        with st.container(height=350):
            st.markdown(
                """
                <div class="manager-performance-grid">
                    <div class="manager-trend-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div style="font-weight: 700; color: #1e293b;">Weekly Sales</div>
                            <div style="color: #10b981; font-weight: 600;">↗ +12%</div>
                        </div>
                        <div style="
                            background: linear-gradient(90deg, #e2e8f0 0%, #10b981 100%);
                            height: 8px;
                            border-radius: 4px;
                            margin-bottom: 0.5rem;
                        "></div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem;">$142,350</div>
                        <div style="font-size: 0.85rem; color: #64748b;">vs. last week</div>
                    </div>
                    <div class="manager-trend-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div style="font-weight: 700; color: #1e293b;">Monthly Target</div>
                            <div style="color: #10b981; font-weight: 600;">On track</div>
                        </div>
                        <div style="
                            background: linear-gradient(90deg, #e2e8f0 0%, #3b82f6 78%);
                            height: 8px;
                            border-radius: 4px;
                            margin-bottom: 0.5rem;
                        "></div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem;">78%</div>
                        <div style="font-size: 0.85rem; color: #64748b;">complete</div>
                    </div>
                    <div class="manager-trend-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div style="font-weight: 700; color: #1e293b;">Customer Satisfaction</div>
                            <div style="color: #fbbf24; font-weight: 600;">↗ +0.2</div>
                        </div>
                        <div style="
                            background: linear-gradient(90deg, #e2e8f0 0%, #fbbf24 94%);
                            height: 8px;
                            border-radius: 4px;
                            margin-bottom: 0.5rem;
                        "></div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem;">4.7/5.0</div>
                        <div style="font-size: 0.85rem; color: #64748b;">vs. last month</div>
                    </div>
                    <div class="manager-trend-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div style="font-weight: 700; color: #1e293b;">Staff Efficiency</div>
                            <div style="color: #8b5cf6; font-weight: 600;">↗ +3%</div>
                        </div>
                        <div style="
                            background: linear-gradient(90deg, #e2e8f0 0%, #8b5cf6 94%);
                            height: 8px;
                            border-radius: 4px;
                            margin-bottom: 0.5rem;
                        "></div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem;">94%</div>
                        <div style="font-size: 0.85rem; color: #64748b;">vs. average</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )


def show_manager_alerts_tab():
    """Display the Alerts tab with interactive counters and scrollable alert containers."""
    # Add custom CSS for scrollable alert containers and interactive elements
    st.markdown(
        """
    <style>
    .alerts-container {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        background: #f8f9fa;
        margin-bottom: 1rem;
    }
    
    .alert-item {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
        cursor: pointer;
        position: relative;
    }
    
    .alert-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .alert-item.urgent {
        border-left-color: #dc3545;
        background: linear-gradient(90deg, #fff5f5 0%, white 10%);
    }
    
    .alert-item.important {
        border-left-color: #ffc107;
        background: linear-gradient(90deg, #fffbf0 0%, white 10%);
    }
    
    .alert-item.resolved {
        opacity: 0.6;
        background: #f8f9fa;
    }
    
    .alert-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .alert-type {
        font-weight: 700;
        color: #495057;
        font-size: 0.9rem;
    }
    
    .alert-severity {
        background: #dc3545;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 700;
    }
    
    .alert-severity.important {
        background: #ffc107;
        color: #212529;
    }
    
    .alert-time {
        font-size: 0.75rem;
        color: #6c757d;
    }
    
    .alert-message {
        color: #495057;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    
    .alert-actions {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }
    
    .click-hint {
        font-size: 0.75rem;
        color: #007bff;
        font-style: italic;
        margin-top: 0.25rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize alert state
    if "resolved_alerts" not in st.session_state:
        st.session_state.resolved_alerts = set()
    if "show_alert_modal" not in st.session_state:
        st.session_state.show_alert_modal = False
    if "modal_alert_type" not in st.session_state:
        st.session_state.modal_alert_type = ""
    if "selected_alert_id" not in st.session_state:
        st.session_state.selected_alert_id = None

    # All alerts data with detailed information
    all_alerts = [
        {
            "id": 1,
            "type": "Personal Styling",
            "message": "Platinum Member Victoria Chen arriving in 1 hour - Personal stylist still unassigned",
            "severity": "urgent",
            "action": "Assign personal stylist",
            "time": "15 min ago",
            "details": {
                "customer_name": "Victoria Chen",
                "membership_tier": "Platinum Member (5+ years)",
                "appointment_time": "11:00 AM (58 minutes from now)",
                "service_type": "Personal Shopping - Women's Professional Wear",
                "avg_purchase": "$850 per visit",
                "last_visit": "3 weeks ago, purchased $1,200 business wardrobe",
                "original_stylist": "Jessica Martinez (called in sick)",
                "backup_failed": "Auto-reassignment system failed",
                "available_stylists": [
                    {
                        "name": "Maria Santos",
                        "rating": "4.9/5",
                        "specialty": "Women's Fashion",
                        "status": "Available",
                    },
                    {
                        "name": "David Kim",
                        "rating": "4.7/5",
                        "specialty": "Cross-trained",
                        "status": "Available",
                    },
                    {
                        "name": "Lisa Park",
                        "rating": "4.8/5",
                        "specialty": "Women's Fashion",
                        "status": "Busy until 11:30 AM",
                    },
                ],
            },
        },
        {
            "id": 2,
            "type": "Critical Stock",
            "message": "Designer Jeans - only 2 left",
            "severity": "urgent",
            "action": "Reorder now",
            "time": "20 min ago",
        },
        {
            "id": 3,
            "type": "Staff Coverage",
            "message": "Electronics understaffed 3-4 PM",
            "severity": "urgent",
            "action": "Find coverage",
            "time": "30 min ago",
        },
        {
            "id": 4,
            "type": "Delivery Update",
            "message": "New designer collection arriving tomorrow - Prepare display area",
            "severity": "important",
            "action": "Prep display area",
            "time": "1 hour ago",
        },
        {
            "id": 5,
            "type": "Schedule Change",
            "message": "Staff meeting moved to 3 PM in conference room",
            "severity": "important",
            "action": "Update team",
            "time": "2 hours ago",
        },
        {
            "id": 6,
            "type": "Personal Styling",
            "message": "Preferred Client Sarah Johnson arriving at 2 PM for wardrobe consultation",
            "severity": "important",
            "action": "Prep personal shopper",
            "time": "2 hours ago",
        },
        {
            "id": 7,
            "type": "Delivery Delay",
            "message": "Designer collection delayed to 4:30 PM",
            "severity": "important",
            "action": "Update team",
            "time": "3 hours ago",
        },
    ]

    # Calculate real-time counters
    urgent_alerts = [
        a
        for a in all_alerts
        if a["severity"] == "urgent" and a["id"] not in st.session_state.resolved_alerts
    ]
    important_alerts = [
        a
        for a in all_alerts
        if a["severity"] == "important"
        and a["id"] not in st.session_state.resolved_alerts
    ]
    resolved_alerts = [
        a for a in all_alerts if a["id"] in st.session_state.resolved_alerts
    ]
    total_active = len(urgent_alerts) + len(important_alerts)

    urgent_count = len(urgent_alerts)
    important_count = len(important_alerts)
    resolved_count = len(resolved_alerts)

    # Real-time counters with color-coded statistics using streamlit-card
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Urgent alerts card - Bright Red theme
        urgent_clicked = card(
            title="Urgent",
            text=f"{urgent_count} alerts",
            styles={
                "card": {
                    "width": "100%",
                    "height": "120px",
                    "border-radius": "16px",
                    "box-shadow": "0 4px 20px rgba(239, 68, 68, 0.4), 0 1px 3px rgba(0, 0, 0, 0.1)",
                    "background": "linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #b91c1c 100%)",
                    "border": "1px solid rgba(239, 68, 68, 0.3)",
                    "margin": "0",
                    "padding": "1rem",
                    "text-align": "center",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease",
                    "position": "relative",
                    "overflow": "hidden",
                },
                "title": {
                    "font-size": "2.5rem",
                    "color": "white",
                    "font-weight": "900",
                    "margin-bottom": "0.1rem",
                    "text-shadow": "0 2px 4px rgba(0, 0, 0, 0.3)",
                    "line-height": "1",
                },
                "text": {
                    "font-size": "1.1rem",
                    "color": "rgba(255, 255, 255, 0.9)",
                    "font-weight": "700",
                    "text-transform": "uppercase",
                    "letter-spacing": "1px",
                    "text-shadow": "0 1px 2px rgba(0, 0, 0, 0.2)",
                },
            },
            key="urgent_card_v5",
        )

        if urgent_clicked:
            st.session_state.modal_alert_type = "urgent"

    with col2:
        # Important alerts card - Bright Yellow-Orange theme
        important_clicked = card(
            title="Important",
            text=f"{important_count} alerts",
            styles={
                "card": {
                    "width": "100%",
                    "height": "120px",
                    "border-radius": "16px",
                    "box-shadow": "0 4px 20px rgba(245, 158, 11, 0.4), 0 1px 3px rgba(0, 0, 0, 0.1)",
                    "background": "linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #d97706 100%)",
                    "border": "1px solid rgba(245, 158, 11, 0.3)",
                    "margin": "0",
                    "padding": "1rem",
                    "text-align": "center",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease",
                    "position": "relative",
                    "overflow": "hidden",
                },
                "title": {
                    "font-size": "2.5rem",
                    "color": "white",
                    "font-weight": "900",
                    "margin-bottom": "0.1rem",
                    "text-shadow": "0 2px 4px rgba(0, 0, 0, 0.3)",
                    "line-height": "1",
                },
                "text": {
                    "font-size": "1.1rem",
                    "color": "rgba(255, 255, 255, 0.9)",
                    "font-weight": "700",
                    "text-transform": "uppercase",
                    "letter-spacing": "1px",
                    "text-shadow": "0 1px 2px rgba(0, 0, 0, 0.2)",
                },
            },
            key="important_card_v5",
        )

        if important_clicked:
            st.session_state.modal_alert_type = "important"

    with col3:
        # Resolved alerts card - Bright Emerald Green theme
        resolved_clicked = card(
            title="Resolved",
            text=f"{resolved_count} alerts",
            styles={
                "card": {
                    "width": "100%",
                    "height": "120px",
                    "border-radius": "16px",
                    "box-shadow": "0 4px 20px rgba(34, 197, 94, 0.4), 0 1px 3px rgba(0, 0, 0, 0.1)",
                    "background": "linear-gradient(135deg, #22c55e 0%, #16a34a 50%, #15803d 100%)",
                    "border": "1px solid rgba(34, 197, 94, 0.3)",
                    "margin": "0",
                    "padding": "1rem",
                    "text-align": "center",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease",
                    "position": "relative",
                    "overflow": "hidden",
                },
                "title": {
                    "font-size": "2.5rem",
                    "color": "white",
                    "font-weight": "900",
                    "margin-bottom": "0.1rem",
                    "text-shadow": "0 2px 4px rgba(0, 0, 0, 0.3)",
                    "line-height": "1",
                },
                "text": {
                    "font-size": "1.1rem",
                    "color": "rgba(255, 255, 255, 0.9)",
                    "font-weight": "700",
                    "text-transform": "uppercase",
                    "letter-spacing": "1px",
                    "text-shadow": "0 1px 2px rgba(0, 0, 0, 0.2)",
                },
            },
            key="resolved_card_v5",
        )

        if resolved_clicked:
            st.session_state.modal_alert_type = "resolved"

    with col4:
        # Total active alerts card - Bright Blue theme
        total_clicked = card(
            title="Total Active",
            text=f"{total_active} alerts",
            styles={
                "card": {
                    "width": "100%",
                    "height": "120px",
                    "border-radius": "16px",
                    "box-shadow": "0 4px 20px rgba(59, 130, 246, 0.4), 0 1px 3px rgba(0, 0, 0, 0.1)",
                    "background": "linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%)",
                    "border": "1px solid rgba(59, 130, 246, 0.3)",
                    "margin": "0",
                    "padding": "1rem",
                    "text-align": "center",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease",
                    "position": "relative",
                    "overflow": "hidden",
                },
                "title": {
                    "font-size": "2.5rem",
                    "color": "white",
                    "font-weight": "900",
                    "margin-bottom": "0.1rem",
                    "text-shadow": "0 2px 4px rgba(0, 0, 0, 0.3)",
                    "line-height": "1",
                },
                "text": {
                    "font-size": "1.1rem",
                    "color": "rgba(255, 255, 255, 0.9)",
                    "font-weight": "700",
                    "text-transform": "uppercase",
                    "letter-spacing": "1px",
                    "text-shadow": "0 1px 2px rgba(0, 0, 0, 0.2)",
                },
            },
            key="total_card_v5",
        )

        if total_clicked:
            st.session_state.modal_alert_type = "all"

    # Set default alert type if none selected
    if not st.session_state.modal_alert_type:
        st.session_state.modal_alert_type = "all"

    # Always show the alert display area below the cards
    # Filter alerts based on current modal type
    if st.session_state.modal_alert_type == "urgent":
        display_alerts = urgent_alerts
        display_title = f"Urgent Alerts ({len(display_alerts)})"
    elif st.session_state.modal_alert_type == "important":
        display_alerts = important_alerts
        display_title = f"Important Alerts ({len(display_alerts)})"
    elif st.session_state.modal_alert_type == "resolved":
        display_alerts = resolved_alerts
        display_title = f"Resolved Alerts ({len(display_alerts)})"
    else:  # all
        display_alerts = urgent_alerts + important_alerts
        display_title = f"All Active Alerts ({len(display_alerts)})"

    # Display the selected alert type
    st.markdown(f"### {display_title}")

    # Fixed height container with alert cards
    with st.container(height=350):
        if len(display_alerts) > 0:
            for alert in display_alerts:
                is_resolved = alert["id"] in st.session_state.resolved_alerts
                "resolved" if is_resolved else alert["severity"]
                severity_label = (
                    "RESOLVED" if is_resolved else alert["severity"].upper()
                )

                # Color coding for alert cards
                if alert["severity"] == "urgent":
                    border_color = "#dc3545"
                    bg_color = "#fff5f5" if not is_resolved else "#f8f9fa"
                    severity_bg = "#dc3545"
                    severity_text = "white"
                elif alert["severity"] == "important":
                    border_color = "#ffc107"
                    bg_color = "#fffbf0" if not is_resolved else "#f8f9fa"
                    severity_bg = "#ffc107"
                    severity_text = "#212529"
                else:
                    border_color = "#6c757d"
                    bg_color = "#f8f9fa"
                    severity_bg = "#6c757d"
                    severity_text = "white"

                # Create columns for alert content and button
                col1, col2 = st.columns([4, 1])

                with col1:
                    # Alert card display
                    st.markdown(
                        f"""
                    <div style="
                        background: {bg_color};
                        border-left: 4px solid {border_color};
                        border-radius: 8px;
                        padding: 1rem;
                        margin-bottom: 0.75rem;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        opacity: {"0.6" if is_resolved else "1"};
                    ">
                        <div style="
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            margin-bottom: 0.5rem;
                        ">
                            <span style="
                                font-weight: 700;
                                color: #495057;
                                font-size: 0.9rem;
                            ">{alert["type"]}</span>
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span style="
                                    background: {severity_bg};
                                    color: {severity_text};
                                    padding: 0.2rem 0.5rem;
                                    border-radius: 12px;
                                    font-size: 0.85rem;
                                    font-weight: 700;
                                ">{severity_label}</span>
                                <span style="
                                    font-size: 0.75rem;
                                    color: #6c757d;
                                ">{alert["time"]}</span>
                            </div>
                        </div>
                        <div style="
                            color: #495057;
                            margin-bottom: 0.5rem;
                            line-height: 1.4;
                        ">{alert["message"]}</div>
                        <div style="
                            color: #007bff;
                            font-size: 0.8rem;
                        ">→ {alert["action"]}</div>
                        <div class="click-hint">Click "Details" for more information</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    # Details button for each alert
                    if st.button(
                        "Details",
                        key=f"alert_details_{alert['id']}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_alert_id = alert["id"]
                        st.rerun()
        else:
            st.info("No alerts to display.")

    # Alert details modal using st.dialog - ONLY triggered by Details buttons
    if st.session_state.selected_alert_id:
        selected_alert = next(
            (a for a in all_alerts if a["id"] == st.session_state.selected_alert_id),
            None,
        )

        if selected_alert:
            try:

                @st.dialog("Alert Details", width="large")
                def show_alert_details():
                    # Add modern CSS styling
                    st.markdown(
                        """
                    <style>
                    .modern-card {
                        background: white;
                        border-radius: 16px;
                        padding: 24px;
                        margin: 16px 0;
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
                        border: 1px solid rgba(0, 0, 0, 0.04);
                        transition: all 0.3s ease;
                    }
                    .modern-card:hover {
                        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.12);
                    }
                    .section-header {
                        font-size: 18px;
                        font-weight: 700;
                        color: #1a1a1a;
                        margin-bottom: 20px;
                        display: flex;
                        align-items: center;
                        gap: 12px;
                        letter-spacing: -0.02em;
                    }
                    .info-grid {
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 16px;
                        margin-bottom: 24px;
                    }
                    .info-item {
                        background: #f8fafc;
                        border-radius: 12px;
                        padding: 16px;
                        border-left: 4px solid;
                        transition: all 0.2s ease;
                    }
                    .info-item:hover {
                        background: #f1f5f9;
                        transform: translateY(-2px);
                    }
                    .info-label {
                        font-size: 12px;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                        margin-bottom: 8px;
                        opacity: 0.7;
                    }
                    .info-value {
                        font-size: 16px;
                        font-weight: 600;
                        color: #1a1a1a;
                        line-height: 1.4;
                    }
                    .status-badge {
                        display: inline-flex;
                        align-items: center;
                        gap: 6px;
                        padding: 6px 12px;
                        border-radius: 20px;
                        font-size: 13px;
                        font-weight: 600;
                        letter-spacing: 0.02em;
                    }
                    .status-available {
                        background: #dcfce7;
                        color: #166534;
                        border: 1px solid #bbf7d0;
                    }
                    .status-busy {
                        background: #fef3c7;
                        color: #92400e;
                        border: 1px solid #fde68a;
                    }
                    .recommended-badge {
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        color: white;
                        padding: 4px 10px;
                        border-radius: 12px;
                        font-size: 11px;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                        margin-left: 8px;
                    }
                    /* Additional CSS for wider dialog */
                    div[data-testid="stDialog"] {
                        position: fixed !important;
                        top: 0 !important;
                        left: 0 !important;
                        right: 0 !important;
                        bottom: 0 !important;
                        display: flex !important;
                        align-items: flex-start !important;
                        justify-content: center !important;
                        z-index: 1000 !important;
                        background: rgba(0, 0, 0, 0.5) !important;
                        width: 100vw !important;
                        height: 100vh !important;
                        padding-top: 2rem !important;
                        overflow-y: auto !important;
                    }
                    div[data-testid="stDialog"] > div {
                        max-width: 1000px !important;
                        width: 85vw !important;
                        position: relative !important;
                        background: white !important;
                        border-radius: 16px !important;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
                        margin: 0 auto !important;
                        max-height: calc(100vh - 4rem) !important;
                        overflow-y: auto !important;
                    }
                    </style>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Create a scrollable container for the modal content
                    with st.container(height=650):
                        # Modern header for Personal Styling alerts
                        if selected_alert["type"] == "Personal Styling":
                            st.markdown(
                                """
                            <div style="
                                background: linear-gradient(135deg, #dc2626 0%, #b91c1c 50%, #991b1b 100%);
                                color: white;
                                padding: 32px;
                                border-radius: 20px;
                                margin-bottom: 32px;
                                position: relative;
                                overflow: hidden;
                                border: 3px solid #fca5a5;
                                box-shadow: 0 0 30px rgba(220, 38, 38, 0.4);
                                animation: pulse-urgent 2s infinite;
                            ">
                                <style>
                                @keyframes pulse-urgent {
                                    0% { box-shadow: 0 0 30px rgba(220, 38, 38, 0.4); }
                                    50% { box-shadow: 0 0 50px rgba(220, 38, 38, 0.8); }
                                    100% { box-shadow: 0 0 30px rgba(220, 38, 38, 0.4); }
                                }
                                </style>
                                <div style="
                                    position: absolute;
                                    top: -50%;
                                    right: -20%;
                                    width: 200px;
                                    height: 200px;
                                    background: rgba(255, 255, 255, 0.1);
                                    border-radius: 50%;
                                    filter: blur(40px);
                                "></div>
                                <div style="position: relative; z-index: 2;">
                                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                                        <div style="display: flex; align-items: center; gap: 16px;">
                                            <div style="
                                                background: rgba(255, 255, 255, 0.2);
                                                padding: 16px;
                                                border-radius: 20px;
                                                backdrop-filter: blur(10px);
                                            ">
                                                <span style="font-size: 32px;">⚠️</span>
                                            </div>
                                            <div>
                                                <div style="
                                                    background: #fef2f2;
                                                    color: #dc2626;
                                                    padding: 8px 16px;
                                                    border-radius: 25px;
                                                    font-size: 14px;
                                                    font-weight: 900;
                                                    text-transform: uppercase;
                                                    letter-spacing: 1px;
                                                    margin-bottom: 8px;
                                                    border: 2px solid #fca5a5;
                                                ">URGENT - IMMEDIATE ACTION REQUIRED</div>
                                                <h1 style="
                                                    margin: 0;
                                                    font-size: 28px;
                                                    font-weight: 900;
                                                    letter-spacing: -0.02em;
                                                ">STYLIST UNASSIGNED</h1>
                                            </div>
                                        </div>
                                        <div style="text-align: right;">
                                            <div style="
                                                background: rgba(255, 255, 255, 0.2);
                                                padding: 12px 20px;
                                                border-radius: 15px;
                                                backdrop-filter: blur(10px);
                                                border: 1px solid rgba(255, 255, 255, 0.3);
                                            ">
                                                <div style="
                                                    font-size: 24px;
                                                    font-weight: 900;
                                                    margin-bottom: 4px;
                                                ">⏰ 58 MIN</div>
                                                <div style="
                                                    font-size: 12px;
                                                    opacity: 0.9;
                                                    text-transform: uppercase;
                                                    letter-spacing: 1px;
                                                ">Until Appointment</div>
                                            </div>
                                        </div>
                                    </div>
                                    <div style="
                                        background: rgba(255, 255, 255, 0.15);
                                        padding: 20px;
                                        border-radius: 15px;
                                        backdrop-filter: blur(10px);
                                        border: 1px solid rgba(255, 255, 255, 0.2);
                                    ">
                                        <div style="
                                            font-size: 18px;
                                            font-weight: 600;
                                            line-height: 1.6;
                                            margin-bottom: 12px;
                                        ">
                                            <span style="
                                                background: #fbbf24;
                                                color: #92400e;
                                                padding: 4px 8px;
                                                border-radius: 8px;
                                                font-weight: 900;
                                                margin-right: 8px;
                                            ">PLATINUM MEMBER</span>
                                            Victoria Chen arriving in 1 hour
                                        </div>
                                        <div style="
                                            font-size: 16px;
                                            opacity: 0.95;
                                            line-height: 1.5;
                                            • <strong>Risk of service disruption and customer dissatisfaction</strong>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        else:
                            # Standard header for other alert types
                            st.markdown(f"### {selected_alert['message']}")
                            st.markdown(
                                f"**Severity:** {selected_alert['severity'].title()}"
                            )
                            st.markdown(f"**Time:** {selected_alert['time']}")
                        # Show additional details if available
                        if "details" in selected_alert:
                            details = selected_alert["details"]

                            # Issue Details Section - MOVED TO TOP
                            st.markdown(
                                f"""
                            <div class="modern-card" style="border-left: 4px solid #ef4444;">
                                <div class="section-header">
                                    <span style="color: #ef4444; font-size: 20px;">⚠️</span>
                                    Issue Analysis
                                </div>
                                <div style="
                                    background: #fef2f2;
                                    border-radius: 12px;
                                    padding: 20px;
                                    border: 1px solid #fecaca;
                                ">
                                    <div style="margin-bottom: 12px;">
                                        <span style="color: #991b1b; font-weight: 600; font-size: 14px;">Original Stylist:</span>
                                        <span style="color: #1f2937; margin-left: 8px;">{details["original_stylist"]}</span>
                                    </div>
                                    <div>
                                        <span style="color: #991b1b; font-weight: 600; font-size: 14px;">Root Cause:</span>
                                        <span style="color: #1f2937; margin-left: 8px;">{details["backup_failed"]}</span>
                                    </div>
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Customer Information Section - CLEANED UP
                            st.markdown(
                                f"""
                            <div class="modern-card">
                                <div class="section-header">
                                    <span style="
                                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                        -webkit-background-clip: text;
                                        -webkit-text-fill-color: transparent;
                                        background-clip: text;
                                        font-size: 20px;
                                    ">👤</span>
                                    Customer Profile
                                </div>
                                <div class="info-grid">
                                    <div class="info-item" style="border-left-color: #667eea;">
                                        <div class="info-label" style="color: #667eea;">Customer</div>
                                        <div class="info-value">{details["customer_name"]}</div>
                                    </div>
                                    <div class="info-item" style="border-left-color: #f59e0b;">
                                        <div class="info-label" style="color: #f59e0b;">Membership</div>
                                        <div class="info-value">{details["membership_tier"]}</div>
                                    </div>
                                    <div class="info-item" style="border-left-color: #ef4444;">
                                        <div class="info-label" style="color: #ef4444;">Appointment</div>
                                        <div class="info-value">{details["appointment_time"]}</div>
                                    </div>
                                    <div class="info-item" style="border-left-color: #10b981;">
                                        <div class="info-label" style="color: #10b981;">Average Purchase</div>
                                        <div class="info-value">{details["avg_purchase"]}</div>
                                    </div>
                                </div>
                                <div style="
                                    background: #f8fafc;
                                    border-radius: 12px;
                                    padding: 16px;
                                    border: 1px solid #e2e8f0;
                                ">
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 14px;">
                                        <div>
                                            <span style="color: #64748b; font-weight: 600;">Last Visit:</span><br>
                                            <span style="color: #1e293b;">{details["last_visit"]}</span>
                                        </div>
                                        <div>
                                            <span style="color: #64748b; font-weight: 600;">Service Type:</span><br>
                                            <span style="color: #1e293b;">{details["service_type"]}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # CDP Intelligence Section using native Streamlit components - FIXED LISTS
                            st.markdown("---")
                            st.markdown("### 🧠 AI Customer Intelligence")

                            col1, col2 = st.columns(2)

                            with col1:
                                st.info("""**Purchase Pattern**

• Shops quarterly  
• Prefers premium brands  
• Avg basket: 3-4 items  
• Low return rate (1.2%)""")

                                st.info("""**Brand Affinity**

• Theory, Ann Taylor, Kate Spade  
• Color pref: Navy, black, cream""")

                            with col2:
                                st.info("""**Style Profile**

• Professional wardrobe focus  
• Sizes: 8 (dress), M (tops), 8.5 (shoes)""")

                                st.info("""**Current Need**

• Executive wardrobe upgrade  
• Budget range: $1,500-2,500""")

                            st.success("""**🎯 Stylist Recommendations**

• Focus on versatile pieces for business travel  
• Suggest coordinating sets for efficiency  
• Emphasize quality fabrics and classic cuts  
• Show care instructions (she always asks)""")
                            # Available Stylists Section
                            st.markdown("---")
                            st.markdown("### 👥 Available Stylists")

                            # Display each stylist card individually with properly integrated buttons
                            for i, stylist in enumerate(details["available_stylists"]):
                                if stylist["status"] == "Available":
                                    status_bg = "#dcfce7"
                                    status_color = "#166534"
                                    status_border = "#bbf7d0"
                                    status_icon = "✓"
                                else:
                                    status_bg = "#fef3c7"
                                    status_color = "#92400e"
                                    status_border = "#fde68a"
                                    status_icon = "⏳"

                                # Special styling for recommended stylist
                                if stylist["name"] == "Maria Santos":
                                    card_style = "border: 2px solid #10b981; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);"
                                    recommended_badge = '<span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-left: 8px;">RECOMMENDED</span>'
                                else:
                                    card_style = "border: 1px solid #e2e8f0;"
                                    recommended_badge = ""

                                # Create a container for each stylist with integrated button using columns
                                col_info, col_button = st.columns([4, 1])

                                with col_info:
                                    st.markdown(
                                        f"""
                                    <div style="
                                        background: white;
                                        border-radius: 16px;
                                        padding: 24px;
                                        margin: 12px 0px;
                                        {card_style}
                                        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
                                        transition: all 0.3s ease;
                                        height: 120px;
                                        display: flex;
                                        align-items: center;
                                    ">
                                        <div style="display: flex; align-items: center; gap: 12px; width: 100%;">
                                            <div style="
                                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                                color: white;
                                                width: 40px;
                                                height: 40px;
                                                border-radius: 12px;
                                                display: flex;
                                                align-items: center;
                                                justify-content: center;
                                                font-size: 18px;
                                            ">👤</div>
                                            <div style="flex: 1;">
                                                <div style="
                                                    font-size: 18px;
                                                    font-weight: 700;
                                                    color: #1a1a1a;
                                                    letter-spacing: -0.01em;
                                                    margin-bottom: 4px;
                                                ">{stylist["name"]}{recommended_badge}</div>
                                                <div style="
                                                    font-size: 14px;
                                                    color: #64748b;
                                                    font-weight: 500;
                                                    margin-bottom: 8px;
                                                ">{stylist["specialty"]}</div>
                                                <div style="display: flex; align-items: center; gap: 12px;">
                                                    <div style="
                                                        display: inline-flex;
                                                        align-items: center;
                                                        gap: 6px;
                                                        padding: 6px 12px;
                                                        border-radius: 20px;
                                                        font-size: 13px;
                                                        font-weight: 600;
                                                        background: {status_bg};
                                                        color: {status_color};
                                                        border: 1px solid {status_border};
                                                    ">
                                                        {status_icon} {stylist["status"]}
                                                    </div>
                                                    <div style="
                                                        font-size: 14px;
                                                        color: #f59e0b;
                                                        font-weight: 600;
                                                    ">⭐ {stylist["rating"]}</div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    """,
                                        unsafe_allow_html=True,
                                    )

                                with col_button:
                                    # Add some top margin to align with the card
                                    st.markdown(
                                        "<div style='margin-top: 50px;'></div>",
                                        unsafe_allow_html=True,
                                    )

                                    if stylist["status"] == "Available":
                                        button_type = (
                                            "primary"
                                            if stylist["name"] == "Maria Santos"
                                            else "secondary"
                                        )
                                        button_text = (
                                            "Assign"
                                            if stylist["name"] == "Maria Santos"
                                            else "Assign"
                                        )

                                        if st.button(
                                            button_text,
                                            key=f"assign_{stylist['name'].replace(' ', '_')}_individual",
                                            type=button_type,
                                            use_container_width=True,
                                        ):
                                            # Enhanced orchestrated response demonstration
                                            st.success(
                                                f"✅ {stylist['name']} assigned to Victoria Chen's appointment!"
                                            )

                                            # Show orchestrated workflow details
                                            st.info("""
                                            **🔄 Automated Workflow Initiated:**
                                            
                                            **📱 Staff Notification:**
                                            • Maria Santos notified via mobile app
                                            • Customer profile and preferences sent
                                            • Appointment details and preparation time provided
                                            
                                            **🧠 AI-Powered Preparation:**
                                            • Victoria's style profile and purchase history loaded
                                            • Recommended items pre-selected based on preferences
                                            • Inventory availability confirmed for suggested pieces
                                            
                                            **📋 System Updates:**
                                            • Appointment status updated across all systems
                                            • Customer service team notified of assignment
                                            • Performance tracking initiated for service quality
                                            
                                            **⏰ Timeline:** All actions completed in <5 seconds
                                            """)

                                            st.balloons()
                                            st.session_state.selected_alert_id = None
                                            st.rerun()
                                    else:
                                        st.button(
                                            "N/A",
                                            key=f"unavailable_{stylist['name'].replace(' ', '_')}",
                                            disabled=True,
                                            use_container_width=True,
                                            help=f"{stylist['name']} is {stylist['status']}",
                                        )
                            # For alerts without detailed information
                            st.markdown("---")
                            if st.button("Close", key="close_simple_dialog"):
                                st.session_state.selected_alert_id = None
                                st.rerun()

                # Show the dialog
                show_alert_details()

            except Exception as e:
                st.error(f"Dialog error: {e}")
                # Fallback: Show details in an expander
                with st.expander(
                    f"📋 {selected_alert['type']} - Alert Details", expanded=True
                ):
                    st.markdown(f"### {selected_alert['message']}")
                    st.markdown(f"**Severity:** {selected_alert['severity'].title()}")
                    st.markdown(f"**Time:** {selected_alert['time']}")
                    st.markdown(f"**Recommended Action:** {selected_alert['action']}")

                    if "details" in selected_alert:
                        details = selected_alert["details"]
                        st.markdown("---")
                        st.markdown("### Customer Information")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Customer:** {details['customer_name']}")
                            st.markdown(f"**Membership:** {details['membership_tier']}")
                            st.markdown(
                                f"**Appointment:** {details['appointment_time']}"
                            )
                            st.markdown(f"**Service:** {details['service_type']}")

                        with col2:
                            st.markdown(f"**Avg Purchase:** {details['avg_purchase']}")
                            st.markdown(f"**Last Visit:** {details['last_visit']}")
                            st.markdown(
                                f"**Original Stylist:** {details['original_stylist']}"
                            )
                            st.markdown(f"**Issue:** {details['backup_failed']}")

                        st.markdown("### Available Stylists")
                        for stylist in details["available_stylists"]:
                            status_color = (
                                "#28a745"
                                if stylist["status"] == "Available"
                                else "#ffc107"
                            )
                            st.markdown(
                                f"""
                            <div style="
                                background: #f8f9fa;
                                border-radius: 8px;
                                padding: 0.75rem;
                                margin-bottom: 0.5rem;
                                border-left: 3px solid {status_color};
                            ">
                                <strong>{stylist["name"]}</strong> - {stylist["specialty"]}<br>
                                Rating: {stylist["rating"]} | Status: <span style="color: {status_color}; font-weight: bold;">{stylist["status"]}</span>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Add individual assign buttons for fallback section too
                            if stylist["status"] == "Available":
                                if st.button(
                                    f"Assign {stylist['name']}",
                                    key=f"assign_{stylist['name'].replace(' ', '_')}_fallback",
                                ):
                                    # Enhanced orchestrated response demonstration
                                    st.success(
                                        f"✅ {stylist['name']} assigned to Victoria Chen's appointment!"
                                    )

                                    # Show orchestrated workflow details
                                    st.info("""
                                    **🔄 Automated Workflow Initiated:**
                                    
                                    **📱 Staff Notification:**
                                    • Maria Santos notified via mobile app
                                    • Customer profile and preferences sent
                                    • Appointment details and preparation time provided
                                    
                                    **🧠 AI-Powered Preparation:**
                                    • Victoria's style profile and purchase history loaded
                                    • Recommended items pre-selected based on preferences
                                    • Inventory availability confirmed for suggested pieces
                                    
                                    **📋 System Updates:**
                                    • Appointment status updated across all systems
                                    • Customer service team notified of assignment
                                    • Performance tracking initiated for service quality
                                    
                                    **⏰ Timeline:** All actions completed in <5 seconds
                                    """)

                                    st.session_state.selected_alert_id = None
                                    st.rerun()
                    else:
                        # For alerts without detailed information
                        st.markdown("---")
                        if st.button("Close", key="close_simple_dialog"):
                            st.session_state.selected_alert_id = None
                            st.rerun()


def show_manager_operations_tab():
    """Display the Operations tab with daily priorities and tasks."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Today's Priorities")

        operations = [
            {
                "task": "Morning inventory check",
                "status": "completed",
                "time": "08:00",
                "owner": "Sarah Chen",
            },
            {
                "task": "Staff meeting - Holiday prep",
                "status": "in_progress",
                "time": "09:30",
                "owner": "All Staff",
            },
            {
                "task": "Vendor delivery - Designer Collection",
                "status": "pending",
                "time": "11:00",
                "owner": "Mike Rodriguez",
            },
            {
                "task": "Weekly sales report review",
                "status": "pending",
                "time": "15:00",
                "owner": "Manager",
            },
            {
                "task": "Evening shift handover",
                "status": "pending",
                "time": "18:00",
                "owner": "Emma Wilson",
            },
        ]

        for op in operations:
            status_colors = {
                "completed": "#28a745",
                "in_progress": "#007bff",
                "pending": "#6c757d",
            }
            status_icons = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}

            st.markdown(
                f"""
                <div class="operation-preview-card">
                    <div class="operation-header">
                        <span class="operation-task">{op["task"]}</span>
                        <span class="operation-time">{op["time"]}</span>
                    </div>
                    <div class="operation-details">
                        <span style="color: {status_colors[op["status"]]}">
                            {status_icons[op["status"]]} {op["status"].replace("_", " ").title()}
                        </span>
                        • Assigned to: {op["owner"]}
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### Quick Actions")

        if st.button("➕ Add Task", use_container_width=True):
            st.info("Task creation form would open")

        if st.button("📞 Call Staff", use_container_width=True):
            st.info("Staff contact list would open")

        if st.button("🚚 Track Deliveries", use_container_width=True):
            st.info("Delivery tracking would open")

        if st.button("Generate Report", use_container_width=True):
            st.info("Report generator would open")

        st.markdown("#### 🕐 Store Hours")
        st.markdown(
            """
            <div class="store-hours-card">
                <div class="hours-today">
                    <div class="hours-label">Today:</div>
                    <div class="hours-time">8:00 AM - 9:00 PM</div>
                </div>
                <div class="hours-status">
                    <div class="status-open">🟢 Open</div>
                    <div class="hours-remaining">5h 37m remaining</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )


def show_manager_team_tab():
    """Display the Team tab with staff overview and performance."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Team Status")

        team_members = [
            {
                "name": "Sarah Chen",
                "role": "Store Associate",
                "status": "active",
                "performance": 98,
                "location": "Women's Fashion",
            },
            {
                "name": "Mike Rodriguez",
                "role": "Store Associate",
                "status": "active",
                "performance": 95,
                "location": "Electronics",
            },
            {
                "name": "Emma Wilson",
                "role": "Store Associate",
                "status": "break",
                "performance": 92,
                "location": "Customer Service",
            },
            {
                "name": "James Park",
                "role": "Visual Merchandiser",
                "status": "active",
                "performance": 88,
                "location": "All Floors",
            },
            {
                "name": "Lisa Wong",
                "role": "Store Associate",
                "status": "off",
                "performance": 75,
                "location": "Men's Fashion",
            },
        ]

        for member in team_members:
            status_colors = {"active": "#28a745", "break": "#ffc107", "off": "#6c757d"}
            status_icons = {"active": "🟢", "break": "☕", "off": "🔴"}

            st.markdown(
                f"""
                <div class="team-member-card">
                    <div class="member-header">
                        <span class="member-name">{member["name"]}</span>
                        <span class="member-status" style="color: {status_colors[member["status"]]}">
                            {status_icons[member["status"]]} {member["status"].title()}
                        </span>
                    </div>
                    <div class="member-details">
                        <div>{member["role"]} • {member["location"]}</div>
                        <div>Performance: {member["performance"]}%</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        if st.button("View Team Insights", use_container_width=True):
            st.switch_page("pages/team_insights.py")

    with col2:
        st.markdown("#### Team Metrics")

        st.markdown(
            """
            <div class="team-metrics-card">
                <div class="team-metric">
                    <span class="metric-label">Average Performance:</span>
                    <span class="metric-value">94%</span>
                </div>
                <div class="team-metric">
                    <span class="metric-label">Customer Satisfaction:</span>
                    <span class="metric-value">4.6/5</span>
                </div>
                <div class="team-metric">
                    <span class="metric-label">Tasks Completed:</span>
                    <span class="metric-value">47/52</span>
                </div>
                <div class="team-metric">
                    <span class="metric-label">Schedule Adherence:</span>
                    <span class="metric-value">98%</span>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("#### ⚠️ Team Alerts")
        team_alerts = [
            {"alert": "Coverage gap: 3-4 PM Electronics", "icon": "⚠️"},
            {"alert": "Sarah Chen: 42 hours this week", "icon": "⏰"},
            {"alert": "3 employees need safety training", "icon": "📚"},
        ]

        for alert in team_alerts:
            st.markdown(
                f"""
                <div class="new-item-preview">
                    <div class="new-item-name">{alert["icon"]} Alert</div>
                    <div class="new-item-details">{alert["alert"]}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )


def show_manager_inventory_tab():
    """Display the Inventory tab with stock levels and alerts."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📦 Inventory Overview")

        # Use the existing inventory summary but in a more compact format
        inventory_categories = [
            {
                "name": "Critical Stock",
                "count": 3,
                "items": ["Designer Jeans", "iPhone Cases", "Silk Scarves"],
                "color": "#dc3545",
            },
            {
                "name": "Low Stock",
                "count": 12,
                "items": ["Fall Jackets", "Wireless Headphones", "Boots"],
                "color": "#ffc107",
            },
            {
                "name": "Well Stocked",
                "count": 892,
                "items": ["Core inventory items"],
                "color": "#28a745",
            },
            {
                "name": "New Arrivals",
                "count": 24,
                "items": ["Winter Collection", "Holiday Items"],
                "color": "#6f42c1",
            },
        ]

        for category in inventory_categories:
            st.markdown(
                f"""
                <div class="inventory-category-card" style="border-left-color: {category["color"]}">
                    <div class="category-header">
                        <span class="category-name">{category["name"]}</span>
                        <span class="category-count">{category["count"]}</span>
                    </div>
                    <div class="category-items">{", ".join(category["items"][:3])}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        if st.button("Detailed Inventory", use_container_width=True):
            st.switch_page("pages/inventory.py")

    with col2:
        st.markdown("#### 💰 Inventory Value")

        st.markdown(
            """
            <div class="inventory-value-card">
                <div class="value-metric">
                    <span class="value-label">Total Inventory Value:</span>
                    <span class="value-amount">$2.1M</span>
                </div>
                <div class="value-metric">
                    <span class="value-label">Turnover Rate:</span>
                    <span class="value-amount">4.2x/year</span>
                </div>
                <div class="value-metric">
                    <span class="value-label">Reorder Needed:</span>
                    <span class="value-amount">15 items</span>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Department Stock Levels")
        departments = [
            {"name": "Electronics", "level": 95, "color": "#28a745"},
            {"name": "Women's Fashion", "level": 87, "color": "#ffc107"},
            {"name": "Men's Fashion", "level": 92, "color": "#28a745"},
            {"name": "Footwear", "level": 78, "color": "#ffc107"},
        ]

        for dept in departments:
            st.markdown(
                f"""
                <div class="department-stock-bar">
                    <div class="dept-name">{dept["name"]}</div>
                    <div class="stock-bar">
                        <div class="stock-fill" style="width: {dept["level"]}%; background-color: {dept["color"]}"></div>
                    </div>
                    <div class="stock-percentage">{dept["level"]}%</div>
                </div>
            """,
                unsafe_allow_html=True,
            )


def show_manager_analytics_tab():
    """Display the Analytics tab with trends and insights."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sales Trends")

        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-metric">
                    <span class="metric-label">Week-over-Week Growth:</span>
                    <span class="metric-value positive">+12.5%</span>
                </div>
                <div class="analytics-metric">
                    <span class="metric-label">Best Performing Category:</span>
                    <span class="metric-value">Electronics (+23%)</span>
                </div>
                <div class="analytics-metric">
                    <span class="metric-label">Peak Sales Hour:</span>
                    <span class="metric-value">2:00 - 4:00 PM</span>
                </div>
                <div class="analytics-metric">
                    <span class="metric-label">Return Rate:</span>
                    <span class="metric-value">2.8% (below target)</span>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Goals Progress")
        goals = [
            {"name": "Monthly Sales Target", "progress": 78, "target": "$450K"},
            {"name": "Customer Satisfaction", "progress": 92, "target": "4.5/5"},
            {"name": "Inventory Turnover", "progress": 85, "target": "4.5x/year"},
        ]

        for goal in goals:
            color = (
                "#28a745"
                if goal["progress"] >= 90
                else "#ffc107"
                if goal["progress"] >= 70
                else "#dc3545"
            )
            st.markdown(
                f"""
                <div class="goal-progress-card">
                    <div class="goal-name">{goal["name"]}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {goal["progress"]}%; background-color: {color}"></div>
                    </div>
                    <div class="goal-details">{goal["progress"]}% to {goal["target"]}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### 💡 Insights & Recommendations")

        insights = [
            {
                "title": "Optimize Staffing",
                "insight": "Add 1 associate during 2-4 PM peak hours",
                "impact": "High",
                "icon": "👥",
            },
            {
                "title": "Inventory Alert",
                "insight": "Reorder designer jeans before weekend rush",
                "impact": "High",
                "icon": "📦",
            },
            {
                "title": "Promotion Opportunity",
                "insight": "Electronics trending +45% - extend promotion",
                "impact": "Medium",
                "icon": "📈",
            },
            {
                "title": "Training Need",
                "insight": "Customer service scores dipped in Men's Fashion",
                "impact": "Medium",
                "icon": "📚",
            },
        ]

        for insight in insights:
            impact_colors = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-header">
                        <span class="insight-icon">{insight["icon"]}</span>
                        <span class="insight-title">{insight["title"]}</span>
                        <span class="insight-impact" style="background-color: {impact_colors[insight["impact"]]}">
                            {insight["impact"]}
                        </span>
                    </div>
                    <div class="insight-text">{insight["insight"]}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )


def show_homepage():
    """Main homepage function that routes to appropriate view based on user role."""
    # Get user role from session state
    user_role = st.session_state.get("user_role", "store_associate")

    # Get employee name and store name
    employee_name = st.session_state.config["employees"][st.session_state.user_role][
        "name"
    ]
    store_name = st.session_state.store_name

    # Add chat modal setup
    chat_notifications = st.session_state.get("chat_notifications", 0)

    # Create the modal first
    chat_modal = modal.Modal(
        title="AI Assistant", key="homepage_chat_modal", max_width=700, padding=20
    )

    # Page header with integrated store info and chat button
    col1, col2 = st.columns([8, 2])

    with col1:
        # Enhanced title with company and location
        # Extract location from store name (remove "BrickMart" prefix if present)
        if store_name.startswith("BrickMart "):
            location = store_name.replace("BrickMart ", "").strip()
        else:
            location = store_name

        st.title(f"BrickMart - {location}")

        # Integrated store info bar - blends with header
        current_time = datetime.now().strftime("%I:%M %p")
        current_date = datetime.now().strftime("%A, %B %d")

        # Get actual store data from database
        stores_df = get_stores()

        # Find current store data
        current_store_data = None
        if not stores_df.empty:
            # Find the store that matches the current store name
            matching_stores = stores_df[stores_df["name"] == store_name]
            if not matching_stores.empty:
                current_store_data = matching_stores.iloc[0]

        # Build store info from database or use fallback
        if current_store_data is not None:
            # Build complete address from database fields
            full_address = f"{current_store_data['address']}, {current_store_data['city']}, {current_store_data['state']} {current_store_data['zip_code']}"
            store_phone = current_store_data["phone"]

            # Determine hours based on is_24_hours flag
            if current_store_data.get("is_24_hours", False):
                store_hours = "24/7"
            else:
                store_hours = "8:00 AM - 9:00 PM"  # Default hours
        else:
            # Fallback data if store not found in database
            full_address = "789 Market St, San Francisco, CA 94102"
            store_phone = "(415) 555-9876"
            store_hours = "8:00 AM - 9:00 PM"

        store_info = {
            "address": full_address,
            "phone": store_phone,
            "hours": store_hours,
            "weather": "72°F ☀️",  # Weather remains mock for now
        }

        # Create a seamless info bar under the title
        st.markdown(
            f"""
            <div style="
                margin-top: -10px;
                margin-bottom: 15px;
                padding: 8px 0px;
                border-bottom: 1px solid #e9ecef;
                color: #6c757d;
                font-size: 14px;
            ">
                <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
                    <div style="font-weight: 700; color: #495057;">
                        <strong>Welcome back, {employee_name}!</strong>
                    </div>
                    <div style="display: flex; align-items: center; gap: 15px; font-size: 13px;">
                        <span><strong>🕐 {current_time}</strong> • {current_date}</span>
                        <span>🌤️ {store_info["weather"]}</span>
                        <span>📍 {store_info["address"]}</span>
                        <span>⏰ {store_info["hours"]}</span>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        # Add some spacing to align with title
        st.markdown("<br>", unsafe_allow_html=True)
        # Chat button with notification badge
        if chat_notifications > 0:
            button_text = f"AI Assistant ({chat_notifications})"
        else:
            button_text = "AI Assistant"

        if st.button(
            button_text, key="header_chat_btn", type="primary", use_container_width=True
        ):
            st.session_state.chat_notifications = 0
            chat_modal.open()

    # Modal content
    if chat_modal.is_open():
        with chat_modal.container():
            # Get chat config with fallback
            chat_config = st.session_state.get("config", {}).get(
                "chat",
                {
                    "placeholder": "How can I help you today?",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            )

            # Show the chat container
            show_chat_container(chat_config)

    # Show appropriate homepage content based on user role
    if user_role == "store_manager":
        show_manager_homepage()
    elif user_role == "vp_retail_operations":
        show_vp_homepage()
    else:
        show_associate_homepage()
