"""Navigation components for the Streamlit Store App."""

import streamlit as st


def show_nav():
    """Display the navigation sidebar."""
    st.sidebar.page_link("app.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/orders.py", label="Orders", icon="🛍️")
    st.sidebar.page_link("pages/inventory.py", label="Inventory", icon="📦")
    st.sidebar.page_link("pages/staff.py", label="Staff", icon="👥")

    # Handle navigation state changes
    if "nav_change" not in st.session_state:
        st.session_state.nav_change = True

    # Listen for navigation changes
    if st.session_state.nav_change:
        st.session_state.page = st.session_state.get("page", "home")


def show_bottom_nav():
    """Display the bottom navigation bar."""
    st.markdown(
        """
        <div class="bottom-nav">
            <div>🏠 Home</div>
            <div>📦 Orders</div>
            <div>📊 Inventory</div>
            <div>⚙️ Settings</div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def show_header(user_name: str = "Sarah", role: str = "Store Associate"):
    """Display the application header with user info."""
    st.markdown(
        f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h1 style="margin: 0;">Store Companion</h1>
                <div style="text-align: right;">
                    <p style="margin: 0;">Welcome, {user_name}</p>
                    <p style="margin: 0; color: #666;">{role}</p>
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )
