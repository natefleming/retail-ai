"""Store manager operations tab."""

import streamlit as st


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
                <div style="
                    background: white;
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 0.75rem;
                    border-left: 4px solid {status_colors[op["status"]]};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 0.5rem;
                    ">
                        <span style="font-weight: 700; color: #495057;">{op["task"]}</span>
                        <span style="color: #6c757d; font-size: 0.9rem;">{op["time"]}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <span style="color: {status_colors[op["status"]]}">
                            {status_icons[op["status"]]} {op["status"].replace("_", " ").title()}
                        </span>
                        <span style="color: #6c757d;">• Assigned to: {op["owner"]}</span>
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
                    align-items: center;
                    margin-bottom: 1rem;
                ">
                    <span style="font-weight: 600; color: #64748b;">Today:</span>
                    <span style="color: #1e293b; font-weight: 600;">8:00 AM - 9:00 PM</span>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <span style="color: #10b981; font-weight: 600;">🟢 Open</span>
                    <span style="color: #64748b;">5h 37m remaining</span>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )
