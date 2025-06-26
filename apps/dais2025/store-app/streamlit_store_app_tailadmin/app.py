"""
TailAdmin Store Operations Dashboard - Multi-Page Application

This is the main application file that combines all TailAdmin-styled pages
into a single Streamlit application with navigation.
"""

import streamlit as st
import streamlit.components.v1 as components

from components.tailadmin import inject_tailadmin_css


def main():
    """Main application with page navigation."""

    # Configure page
    st.set_page_config(
        page_title="TailAdmin Store Dashboard", page_icon="🏪", layout="wide", initial_sidebar_state="expanded"
    )

    # Initialize session state for navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Homepage"

    # Inject TailAdmin CSS
    inject_tailadmin_css()

    # Sidebar navigation
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
            <h1 style="color: #465fff; font-size: 1.5rem; font-weight: 700; margin: 0;">
                🏪 TailAdmin Dashboard
            </h1>
            <p style="color: #6b7280; font-size: 0.875rem; margin: 0.5rem 0 0 0;">
                Store Operations & Analytics
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Navigation menu
        page_options = [
            "🏠 Homepage",
            "👔 VP Dashboard (Clean)",
            "📊 VP Dashboard (Enhanced)",
            "🎨 Implementation Guide",
            "📋 Components Demo",
        ]

        # Get current index based on session state
        current_index = 0
        if st.session_state.current_page in page_options:
            current_index = page_options.index(st.session_state.current_page)

        page = st.selectbox("Navigate to:", page_options, index=current_index, key="page_selector")

        # Update session state when page changes
        if page != st.session_state.current_page:
            st.session_state.current_page = page
            st.rerun()

        st.markdown("---")

        # Info about current selection
        page_info = {
            "🏠 Homepage": "Store homepage with TailAdmin styling and role-based views",
            "👔 VP Dashboard (Clean)": "Clean VP dashboard with TailAdmin components and dark mode",
            "📊 VP Dashboard (Enhanced)": "Enhanced VP dashboard with advanced TailAdmin styling",
            "🎨 Implementation Guide": "Complete guide to implementing TailAdmin styles in Streamlit",
            "📋 Components Demo": "Demonstration of all available TailAdmin components",
        }

        st.info(page_info.get(page, "Select a page to view"))

        # Additional controls
        st.markdown("### Settings")

        # User role selection for homepage
        if page == "🏠 Homepage":
            user_role = st.selectbox(
                "User Role:", ["store_associate", "store_manager", "vp_retail_operations"], index=1
            )
            st.session_state.user_role = user_role

        # Dark mode toggle for VP dashboards
        if "VP Dashboard" in page:
            dark_mode = st.checkbox("Dark Mode", value=True)
            st.session_state.dark_mode = dark_mode

    # Main content area with error handling
    try:
        if page == "🏠 Homepage":
            from pages.homepage import show_homepage

            show_homepage()

        elif page == "👔 VP Dashboard (Clean)":
            from pages.vp_dashboard_clean import show_vp_dashboard_clean

            show_vp_dashboard_clean()

        elif page == "📊 VP Dashboard (Enhanced)":
            from pages.vp_dashboard_enhanced import show_vp_dashboard_enhanced

            show_vp_dashboard_enhanced()

        elif page == "🎨 Implementation Guide":
            from pages.implementation_guide import show_implementation_guide

            show_implementation_guide()

        elif page == "📋 Components Demo":
            from pages.components_demo import show_components_demo

            show_components_demo()

    except Exception as e:
        st.error(f"""
        **Error loading page: {page}**

        {str(e)}

        Please try refreshing the page or selecting a different page from the sidebar.
        """)

        # Add debug info in development
        if st.secrets.get("DEBUG", False):
            st.code(f"Error details: {e}")


if __name__ == "__main__":
    main()
