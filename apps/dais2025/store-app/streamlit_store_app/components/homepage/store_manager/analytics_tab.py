"""Store manager analytics tab."""

import streamlit as st


def show_manager_analytics_tab():
    """Display the Analytics tab with trends and insights."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sales Trends")

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
                    <span style="font-weight: 600; color: #64748b;">Week-over-Week Growth:</span>
                    <span style="color: #10b981; font-weight: 600;">+12.5%</span>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 1rem;
                ">
                    <span style="font-weight: 600; color: #64748b;">Best Performing Category:</span>
                    <span style="color: #1e293b; font-weight: 600;">Electronics (+23%)</span>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 1rem;
                ">
                    <span style="font-weight: 600; color: #64748b;">Peak Sales Hour:</span>
                    <span style="color: #1e293b; font-weight: 600;">2:00 - 4:00 PM</span>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                ">
                    <span style="font-weight: 600; color: #64748b;">Return Rate:</span>
                    <span style="color: #10b981; font-weight: 600;">2.8% (below target)</span>
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
                <div style="
                    background: white;
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 0.75rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="
                        font-weight: 600;
                        color: #495057;
                        margin-bottom: 0.5rem;
                    ">{goal["name"]}</div>
                    <div style="
                        background: #e2e8f0;
                        border-radius: 8px;
                        height: 8px;
                        overflow: hidden;
                        margin-bottom: 0.5rem;
                    ">
                        <div style="
                            background: {color};
                            height: 100%;
                            width: {goal["progress"]}%;
                            border-radius: 8px;
                        "></div>
                    </div>
                    <div style="
                        color: #6c757d;
                        font-size: 0.9rem;
                    ">{goal["progress"]}% to {goal["target"]}</div>
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
                <div style="
                    background: white;
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 0.75rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-left: 4px solid {impact_colors[insight["impact"]]};
                ">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 0.5rem;
                    ">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="font-size: 1.2rem;">{insight["icon"]}</span>
                            <span style="font-weight: 700; color: #495057;">{insight["title"]}</span>
                        </div>
                        <span style="
                            background: {impact_colors[insight["impact"]]};
                            color: white;
                            padding: 0.2rem 0.5rem;
                            border-radius: 12px;
                            font-size: 0.8rem;
                            font-weight: 600;
                        ">{insight["impact"]}</span>
                    </div>
                    <div style="color: #6c757d; font-size: 0.9rem;">
                        {insight["insight"]}
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )
