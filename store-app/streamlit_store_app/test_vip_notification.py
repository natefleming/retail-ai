"""Test script for VIP customer notification feature."""

import streamlit as st
import time

# Configure the page
st.set_page_config(page_title="VIP Notification Test", layout="wide")

st.title("🧪 VIP Customer Notification Test")
st.markdown("This page tests the VIP customer notification that appears after 7 seconds on the manager dashboard.")

# Add instructions
st.markdown("""
### Instructions:
1. Select **Store Manager** role from the sidebar
2. Navigate to the **Dashboard** tab
3. Wait 7 seconds - a Platinum Member notification should appear as a toast
4. Navigate away and back to Dashboard to test the reset functionality

### Expected Behavior:
- Toast notification appears after exactly 7 seconds: "🌟 Platinum Member Alert: Emma Rodriguez arriving soon and needs personal styling assistance!"
- Notification only appears once per dashboard visit
- Timer resets when navigating away and back to Dashboard
""")

# Show current session state for debugging
with st.expander("🔍 Debug Information"):
    current_time = time.time()
    dashboard_entry_time = st.session_state.get("dashboard_entry_time", 0)
    time_on_dashboard = current_time - dashboard_entry_time if dashboard_entry_time else 0
    
    st.write(f"**Current Time:** {current_time:.2f}")
    st.write(f"**Dashboard Entry Time:** {dashboard_entry_time:.2f}")
    st.write(f"**Time on Dashboard:** {time_on_dashboard:.2f} seconds")
    st.write(f"**VIP Notification Shown:** {st.session_state.get('vip_notification_shown', False)}")
    st.write(f"**Current Manager Tab:** {st.session_state.get('current_manager_tab', 'None')}")
    st.write(f"**Last Nav:** {st.session_state.get('last_nav', 'None')}")

# Add reset button for testing
if st.button("🔄 Reset VIP Notification Test"):
    if "dashboard_entry_time" in st.session_state:
        del st.session_state.dashboard_entry_time
    if "vip_notification_shown" in st.session_state:
        del st.session_state.vip_notification_shown
    if "last_nav" in st.session_state:
        del st.session_state.last_nav
    st.success("VIP notification test state reset!")
    st.rerun()

st.markdown("---")
st.markdown("**Next Steps:** Navigate to the main app and test the Store Manager Dashboard!") 