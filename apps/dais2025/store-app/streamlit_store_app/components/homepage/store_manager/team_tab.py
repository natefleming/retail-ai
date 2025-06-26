"""Store manager team tab."""

import streamlit as st


def show_manager_team_tab():
    """Display the Team tab with staff overview and performance."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Team Status")

        team_members = [
            {
                "name": "Victoria Chen",
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
                <div style="
                    background: white;
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 0.75rem;
                    border-left: 4px solid {status_colors[member["status"]]};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 0.5rem;
                    ">
                        <span style="font-weight: 700; color: #495057;">{member["name"]}</span>
                        <span style="color: {status_colors[member["status"]]}">
                            {status_icons[member["status"]]} {member["status"].title()}
                        </span>
                    </div>
                    <div style="color: #6c757d; font-size: 0.9rem;">
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
            <div style="
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                border: 1px solid rgba(226, 232, 240, 0.6);
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 1rem;
                ">
                    <span style="font-weight: 600; color: #64748b;">Average Performance:</span>
                    <span style="color: #1e293b; font-weight: 600;">94%</span>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 1rem;
                ">
                    <span style="font-weight: 600; color: #64748b;">Customer Satisfaction:</span>
                    <span style="color: #1e293b; font-weight: 600;">4.6/5</span>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 1rem;
                ">
                    <span style="font-weight: 600; color: #64748b;">Tasks Completed:</span>
                    <span style="color: #1e293b; font-weight: 600;">47/52</span>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                ">
                    <span style="font-weight: 600; color: #64748b;">Schedule Adherence:</span>
                    <span style="color: #1e293b; font-weight: 600;">98%</span>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("#### ⚠️ Team Alerts")
        team_alerts = [
            {"alert": "Coverage gap: 2-6 PM Electronics (Mike Rodriguez sick)", "icon": "⚠️"},
            {"alert": "Victoria Chen: 42 hours this week", "icon": "⏰"},
            {"alert": "3 employees need safety training", "icon": "📚"},
        ]

        for alert in team_alerts:
            st.markdown(
                f"""
                <div style="
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                    border-radius: 8px;
                    padding: 0.75rem;
                    margin-bottom: 0.5rem;
                ">
                    <div style="font-weight: 600; color: #856404;">{alert["icon"]} Alert</div>
                    <div style="color: #856404; font-size: 0.9rem;">{alert["alert"]}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )
