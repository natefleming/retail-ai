"""Store manager inventory tab."""

import streamlit as st


def show_manager_inventory_tab():
    """Display the Inventory tab with stock levels and alerts."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📦 Inventory Overview")

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
                <div style="
                    background: white;
                    border-left: 4px solid {category["color"]};
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 0.75rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 0.5rem;
                    ">
                        <span style="font-weight: 700; color: #495057;">{category["name"]}</span>
                        <span style="color: {category["color"]}; font-weight: 600;">{category["count"]}</span>
                    </div>
                    <div style="color: #6c757d; font-size: 0.9rem;">
                        {", ".join(category["items"][:3])}
                    </div>
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
            <div style="
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                border: 1px solid rgba(226, 232, 240, 0.6);
                margin-bottom: 1.5rem;
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 1rem;
                ">
                    <span style="font-weight: 600; color: #64748b;">Total Inventory Value:</span>
                    <span style="color: #1e293b; font-weight: 600;">$2.1M</span>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 1rem;
                ">
                    <span style="font-weight: 600; color: #64748b;">Turnover Rate:</span>
                    <span style="color: #1e293b; font-weight: 600;">4.2x/year</span>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                ">
                    <span style="font-weight: 600; color: #64748b;">Reorder Needed:</span>
                    <span style="color: #1e293b; font-weight: 600;">15 items</span>
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
                <div style="
                    background: white;
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 0.5rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 0.5rem;
                    ">
                        <span style="font-weight: 600; color: #495057;">{dept["name"]}</span>
                        <span style="color: {dept["color"]}; font-weight: 600;">{dept["level"]}%</span>
                    </div>
                    <div style="
                        background: #e2e8f0;
                        border-radius: 8px;
                        height: 8px;
                        overflow: hidden;
                    ">
                        <div style="
                            background: {dept["color"]};
                            height: 100%;
                            width: {dept["level"]}%;
                            border-radius: 8px;
                        "></div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )
