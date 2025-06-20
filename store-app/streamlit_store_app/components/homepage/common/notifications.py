"""Notifications modal component."""

import streamlit as st


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
