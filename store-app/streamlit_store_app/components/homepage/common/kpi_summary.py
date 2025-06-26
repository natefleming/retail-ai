"""KPI and inventory summary components."""

import streamlit as st


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
