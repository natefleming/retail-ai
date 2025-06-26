"""
Enhanced Navigation Components for Store Operations App

This module provides enhanced navigation using popular third-party Streamlit components
to improve the user experience and provide more intuitive navigation patterns.
"""

import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
from streamlit_pills import pills

# Dark mode initialization is now handled centrally in app.py
# Removed conflicting initialization to prevent mode switching on refresh

def create_role_based_navigation(user_role: str = "vp_retail_operations"):
    """
    Create a beautiful navigation menu based on user role using streamlit-option-menu.
    
    Args:
        user_role: The role of the current user (store_associate, store_manager, vp_retail_operations)
    
    Returns:
        str: The selected menu option
    """
    
    # Get dark mode state - consistent with app.py initialization
    dark_mode = st.session_state.get("dark_mode", False)
    
    # Define navigation options based on role
    if user_role == "vp_retail_operations":
        menu_options = ["Executive Dashboard", "Regional Analytics", "Store Performance", "Strategic Planning", "Reports"]
        menu_icons = ["speedometer2", "bar-chart", "shop", "diagram-3", "file-earmark-text"]
        menu_key = "vp_nav"
    elif user_role == "store_manager":
        menu_options = ["Dashboard", "Team Management", "Inventory", "Sales Analytics", "Operations", "Alerts"]
        menu_icons = ["house", "people", "box", "graph-up", "gear", "exclamation-triangle"]
        menu_key = "manager_nav"
    else:  # store_associate
        menu_options = ["My Dashboard", "My Tasks", "My Schedule", "Product Lookup", "Customer Service"]
        menu_icons = ["house", "check-square", "calendar", "search", "headset"]
        menu_key = "associate_nav"
    
    # Consolidated CSS with reduced spacing and responsive sidebar handling
    nav_css = f"""
    <style>
    /* Compact navigation spacing */
    .streamlit-option-menu {{
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }}
    
    .streamlit-option-menu .nav-link-selected {{
        color: #ffffff !important;
    }}
    
    /* Responsive navigation container */
    [data-testid="stHorizontalBlock"] [data-baseweb="tab-list"] {{
        background-color: {"#1f2937" if dark_mode else "#f9fafb"} !important;
        border-radius: 12px !important;
        padding: 4px 8px !important;
        margin: 0 !important;
        box-shadow: {"0 4px 6px rgba(0, 0, 0, 0.3)" if dark_mode else "0 2px 4px rgba(0, 0, 0, 0.1)"} !important;
        width: 100% !important;
        display: flex !important;
        flex-wrap: wrap !important;
        justify-content: space-between !important;
        transition: all 0.3s ease !important;
    }}
    
    /* Responsive navigation items */
    .streamlit-option-menu .nav-item {{
        flex: 1 !important;
        min-width: fit-content !important;
        transition: all 0.3s ease !important;
    }}
    
    /* Remove extra spacing from option menu container */
    div[data-testid="stHorizontalBlock"]:has(.streamlit-option-menu) {{
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
        padding: 0 !important;
        width: 100% !important;
    }}
    
    /* Responsive adjustments for different screen sizes */
    @media (max-width: 768px) {{
        [data-testid="stHorizontalBlock"] [data-baseweb="tab-list"] {{
            flex-direction: column !important;
        }}
        
        .streamlit-option-menu .nav-item {{
            margin-bottom: 2px !important;
        }}
    }}
    
    /* Sidebar state responsive adjustments */
    .main .block-container {{
        transition: max-width 0.3s ease, padding 0.3s ease !important;
    }}
    
    /* Navigation menu responsive to main content width */
    .streamlit-option-menu {{
        max-width: 100% !important;
        overflow-x: auto !important;
    }}
    
    /* Ensure navigation scales with content area */
    div[data-testid="stHorizontalBlock"] {{
        width: 100% !important;
        max-width: 100% !important;
    }}
    </style>
    """

    st.markdown(nav_css, unsafe_allow_html=True)
    
    # Streamlined styles with consistent spacing and responsive design
    base_styles = {
        "container": {
            "padding": "4px 8px",  # Reduced from 8px 16px
            "background-color": "#1f2937" if dark_mode else "#f9fafb",
            "border": f"1px solid {'#374151' if dark_mode else '#e5e7eb'}",
            "border-radius": "12px",
            "margin": "0px",  # Reduced from 8px 0px
            "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.3)" if dark_mode else "0 2px 4px rgba(0, 0, 0, 0.1)",
            "width": "100%",
            "max-width": "100%",
            "display": "flex",
            "flex-wrap": "wrap",
            "justify-content": "space-between"
        },
        "icon": {
            "color": "#9ca3af" if dark_mode else "#6b7280",
            "font-size": "16px"
        }, 
        "nav-link": {
            "font-size": "18px",
            "text-align": "center",
            "margin": "0px 2px",  # Reduced from 0px 4px
            "color": "#e5e7eb" if dark_mode else "#374151",
            "background-color": "rgba(55, 65, 81, 0.6)" if dark_mode else "rgba(255, 255, 255, 0.8)",
            "border-radius": "8px",
            "padding": "8px 12px",  # Adjusted for better responsive fit
            "border": f"1px solid {'rgba(75, 85, 99, 0.3)' if dark_mode else 'rgba(229, 231, 235, 0.5)'}",
            "font-weight": "500",
            "transition": "all 0.2s ease",
            "box-shadow": "0 1px 2px rgba(0, 0, 0, 0.05)" if not dark_mode else "none",
            "--hover-color": "rgba(75, 85, 99, 0.8)" if dark_mode else "rgba(243, 244, 246, 0.9)",
            "flex": "1 1 auto",
            "min-width": "fit-content",
            "white-space": "nowrap"
        },
        "nav-link-selected": {
            "background-color": "#3b82f6",
            "color": "white",
            "font-weight": "600",
            "border": "1px solid #3b82f6",
            "box-shadow": "0 4px 12px rgba(59, 130, 246, 0.3)" if dark_mode else "0 4px 12px rgba(59, 130, 246, 0.25)"
        },
    }
    
    # Create the navigation menu with compact styling
    selected = option_menu(
        menu_title=None,
        options=menu_options,
        icons=menu_icons,
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles=base_styles,
        key=menu_key
    )
    
    return selected


def create_quick_actions_pills(user_role: str = "store_associate"):
    """
    Create quick action pills for common tasks using streamlit-pills.
    
    Args:
        user_role: The role of the current user
        
    Returns:
        str: The selected action
    """
    
    # Get dark mode state
    dark_mode = st.session_state.get("dark_mode", False)
    
    if user_role == "vp_retail_operations":
        actions = ["📊 KPI Overview", "🏪 Store Comparison", "📈 Trends", "⚠️ Alerts", "�� Reports"]
    elif user_role == "store_manager":
        actions = ["👥 Staff Status", "📦 Inventory Check", "💰 Sales Today", "⚠️ Alerts", "📞 Support"]
    else:  # store_associate
        actions = ["✅ My Tasks", "🕐 Clock In/Out", "❓ Help", "📞 Manager", "🔔 Notifications"]
    
    # Compact CSS for pills with reduced spacing
    pill_css = f"""
    <style>
    /* Compact pills container */
    .stPills {{
        margin: 0 !important;
        padding: 0 !important;
    }}
    
    .stPills [data-baseweb="tab-list"] {{
        background-color: {"#1f2937" if dark_mode else "#f9fafb"} !important;
        border: 1px solid {"#374151" if dark_mode else "#e5e7eb"} !important;
        border-radius: 8px !important;
        padding: 2px !important;
        margin: 0 !important;
    }}
    
    .stPills [data-baseweb="tab"] {{
        background-color: {"#374151" if dark_mode else "#ffffff"} !important;
        color: {"#f9fafb" if dark_mode else "#1f2937"} !important;
        border: 1px solid {"#4b5563" if dark_mode else "#e5e7eb"} !important;
        margin: 1px !important;
        padding: 4px 8px !important;
        font-size: 13px !important;
    }}
    
    .stPills [data-baseweb="tab"]:hover {{
        background-color: {"#4b5563" if dark_mode else "#f3f4f6"} !important;
        color: {"#ffffff" if dark_mode else "#1f2937"} !important;
    }}
    
    .stPills [data-baseweb="tab"][aria-selected="true"] {{
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
    }}
    
    /* Remove extra spacing around pills component */
    div:has(.stPills) {{
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
    }}
    </style>
    """
    
    st.markdown(pill_css, unsafe_allow_html=True)
    
    selected_action = pills(
        "Quick Actions:",
        actions,
        clearable=True,
        index=None,
        key=f"quick_actions_{user_role}"
    )
    
    return selected_action


def create_sidebar_navigation(user_role: str = "store_associate"):
    """
    Create an enhanced sidebar navigation with collapsible sections - compact spacing.
    
    Args:
        user_role: The role of the current user
    """
    
    # Get dark mode state
    dark_mode = st.session_state.get("dark_mode", False)
    
    # Compact sidebar styling with reduced spacing
    sidebar_css = f"""
    <style>
    .stSidebar {{
        background-color: {"#111827" if dark_mode else "#ffffff"} !important;
    }}
    
    .stSidebar .stMarkdown {{
        color: {"#f9fafb" if dark_mode else "#1f2937"} !important;
        margin-bottom: 0.5rem !important;
    }}
    
    .stSidebar .stButton > button {{
        background-color: {"#374151" if dark_mode else "#f9fafb"} !important;
        color: {"#f9fafb" if dark_mode else "#1f2937"} !important;
        border: 1px solid {"#4b5563" if dark_mode else "#e5e7eb"} !important;
        transition: all 0.2s ease !important;
        margin-bottom: 0.25rem !important;
        padding: 0.25rem 0.5rem !important;
        font-size: 0.875rem !important;
    }}
    
    .stSidebar .stButton > button:hover {{
        background-color: {"#4b5563" if dark_mode else "#f3f4f6"} !important;
        border-color: {"#6b7280" if dark_mode else "#d1d5db"} !important;
        transform: translateY(-1px) !important;
    }}
    
    .stSidebar .stButton > button:active {{
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
    }}
    
    .stSidebar .stExpander {{
        background-color: {"#1f2937" if dark_mode else "#ffffff"} !important;
        border: 1px solid {"#374151" if dark_mode else "#e5e7eb"} !important;
        margin-bottom: 0.5rem !important;
    }}
    
    .stSidebar .stExpander .streamlit-expanderHeader {{
        background-color: {"#1f2937" if dark_mode else "#ffffff"} !important;
        color: {"#f9fafb" if dark_mode else "#1f2937"} !important;
        padding: 0.5rem !important;
    }}
    
    .stSidebar .stExpander .streamlit-expanderContent {{
        padding: 0.25rem !important;
    }}
    
    .stSidebar .stMetric {{
        background-color: {"#1f2937" if dark_mode else "#f9fafb"} !important;
        border: 1px solid {"#374151" if dark_mode else "#e5e7eb"} !important;
        padding: 0.5rem !important;
        border-radius: 6px !important;
        margin-bottom: 0.25rem !important;
    }}
    
    .stSidebar .stMetric [data-testid="metric-container"] {{
        background-color: transparent !important;
        color: {"#f9fafb" if dark_mode else "#1f2937"} !important;
    }}
    
    /* Compact dividers */
    .stSidebar hr {{
        margin: 0.5rem 0 !important;
    }}
    
    /* Compact title spacing */
    .stSidebar h3 {{
        margin-bottom: 0.5rem !important;
        margin-top: 0 !important;
    }}
    </style>
    """
    
    st.markdown(sidebar_css, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🏪 BrickMart Operations")
        
        # Compact user info section
        role_info = {
            "vp_retail_operations": ("👔 VP Retail Operations", "#1e40af" if dark_mode else None),
            "store_manager": ("👨‍💼 Store Manager", "#059669" if dark_mode else None),
            "store_associate": ("👤 Store Associate", "#7c3aed" if dark_mode else None)
        }
        
        role_text, bg_color = role_info.get(user_role, ("👤 Store Associate", "#7c3aed" if dark_mode else None))
        
        if dark_mode and bg_color:
            st.markdown(f'<div style="background-color: {bg_color}; color: #ffffff; padding: 6px; border-radius: 6px; margin: 4px 0; font-size: 0.875rem;">{role_text}</div>', unsafe_allow_html=True)
        else:
            st.info(role_text)
        
        st.markdown("---")
        
        # Compact navigation sections with reduced content
        if user_role == "vp_retail_operations":
            # VP Navigation - compact
            with st.expander("📊 Executive Overview", expanded=True):
                for nav_item, key in [
                    ("🎯 KPI Dashboard", "kpi_dashboard"),
                    ("🏪 Store Performance", "store_performance"),
                    ("📈 Regional Trends", "regional_trends")
                ]:
                    if st.button(nav_item, use_container_width=True, key=f"sidebar_{key}"):
                        st.session_state.nav_selection = key
            
            with st.expander("🔍 Analytics & Reports"):
                for nav_item, key in [
                    ("📊 Sales Analytics", "sales_analytics"),
                    ("👥 Workforce Analytics", "workforce_analytics"),
                    ("📋 Custom Reports", "custom_reports")
                ]:
                    if st.button(nav_item, use_container_width=True, key=f"sidebar_{key}"):
                        st.session_state.nav_selection = key
        
        elif user_role == "store_manager":
            # Manager Navigation - compact
            with st.expander("🏪 Store Operations", expanded=True):
                for nav_item, key in [
                    ("📊 Store Dashboard", "store_dashboard"),
                    ("📦 Inventory Management", "inventory"),
                    ("💰 Sales Overview", "sales")
                ]:
                    if st.button(nav_item, use_container_width=True, key=f"sidebar_{key}"):
                        st.session_state.nav_selection = key
            
            with st.expander("👥 Team Management"):
                for nav_item, key in [
                    ("📅 Staff Scheduling", "scheduling"),
                    ("📈 Performance Reviews", "performance"),
                    ("🎓 Training", "training")
                ]:
                    if st.button(nav_item, use_container_width=True, key=f"sidebar_{key}"):
                        st.session_state.nav_selection = key
            
            # Add Demo Alert Controls for Store Managers
            with st.expander("🎬 Demo Controls"):
                from .homepage.store_manager.demo_alerts import DemoAlertSystem
                
                demo_system = DemoAlertSystem()
                demo_system.initialize_demo_state()
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("▶️ Start", use_container_width=True, key="sidebar_start_demo"):
                        demo_system.start_demo()
                        st.rerun()
                
                with col2:
                    if st.button("⏹️ Stop", use_container_width=True, key="sidebar_stop_demo"):
                        demo_system.stop_demo()
                        st.rerun()
                
                # Demo status (without auto-refresh)
                if st.session_state.get("demo_start_time") is not None:
                    status, progress = demo_system.get_demo_status()
                    st.write(f"**Status:** {status}")
                    if progress > 0:
                        st.progress(progress / 100)
                    
                    # Show upcoming alerts count
                    import time
                    elapsed = time.time() - st.session_state.demo_start_time
                    
                    upcoming_count = 0
                    for alert in demo_system.demo_alerts:
                        time_until = alert["trigger_after"] - elapsed
                        if time_until > 0 and alert["id"] not in [a["id"] for a in st.session_state.get("demo_active_alerts", [])]:
                            upcoming_count += 1
                    
                    if upcoming_count > 0:
                        st.caption(f"⏰ {upcoming_count} alerts pending")
                    
                    # Manual refresh button instead of auto-refresh
                    if st.button("🔄 Refresh", use_container_width=True, key="sidebar_refresh_demo"):
                        st.rerun()
        
        else:
            # Associate Navigation - compact
            with st.expander("📋 My Work", expanded=True):
                for nav_item, key in [
                    ("✅ My Tasks", "my_tasks"),
                    ("📅 My Schedule", "my_schedule"),
                    ("🕐 Time Clock", "time_clock")
                ]:
                    if st.button(nav_item, use_container_width=True, key=f"sidebar_{key}"):
                        st.session_state.nav_selection = key
            
            with st.expander("🛍️ Customer Service"):
                for nav_item, key in [
                    ("🏷️ Product Lookup", "product_lookup"),
                    ("💳 Process Returns", "returns"),
                    ("❓ Customer Help", "customer_help")
                ]:
                    if st.button(nav_item, use_container_width=True, key=f"sidebar_{key}"):
                        st.session_state.nav_selection = key
        
        st.markdown("---")
        
        # Compact quick stats
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Today's Sales", "$12.5K", "8.2%")
        with col2:
            st.metric("Active Staff", "23", "2")


def create_breadcrumb_navigation(current_page: str, user_role: str = "store_associate"):
    """
    Create breadcrumb navigation to show current location.
    
    Args:
        current_page: The current page/section
        user_role: The role of the current user
    """
    
    # Define breadcrumb paths based on role and page
    breadcrumb_paths = {
        "vp_retail_operations": {
            "dashboard": ["🏢 Executive", "📊 Dashboard"],
            "analytics": ["🏢 Executive", "📈 Analytics"],
            "reports": ["🏢 Executive", "📋 Reports"],
        },
        "store_manager": {
            "dashboard": ["🏪 Store", "📊 Dashboard"],
            "inventory": ["🏪 Store", "📦 Inventory"],
            "team": ["🏪 Store", "👥 Team"],
        },
        "store_associate": {
            "dashboard": ["👤 My Work", "📊 Dashboard"],
            "tasks": ["👤 My Work", "✅ Tasks"],
            "schedule": ["👤 My Work", "📅 Schedule"],
        }
    }
    
    # Get the breadcrumb for current page
    breadcrumb = breadcrumb_paths.get(user_role, {}).get(current_page, ["🏪 Store", "📊 Dashboard"])
    
    # Display breadcrumb
    breadcrumb_text = " > ".join(breadcrumb)
    st.markdown(f"**{breadcrumb_text}**")


def create_tab_navigation(tabs: list, key_prefix: str = "tab"):
    """
    Create enhanced tab navigation using extra-streamlit-components.
    
    Args:
        tabs: List of tab names
        key_prefix: Prefix for the component key
        
    Returns:
        int: Index of selected tab
    """
    
    # Get dark mode state
    dark_mode = st.session_state.get("dark_mode", False)
    
    # Dark mode friendly styling for tabs
    if dark_mode:
        tab_css = """
        <style>
        .stx-tab-bar {
            background-color: #1f2937 !important;
            border: 1px solid #374151 !important;
            border-radius: 8px !important;
        }
        .stx-tab-bar .stx-tab {
            background-color: #374151 !important;
            color: #f9fafb !important;
            border: 1px solid #4b5563 !important;
        }
        .stx-tab-bar .stx-tab:hover {
            background-color: #4b5563 !important;
        }
        .stx-tab-bar .stx-tab.active {
            background-color: #3b82f6 !important;
            color: #ffffff !important;
            border-color: #3b82f6 !important;
        }
        </style>
        """
    else:
        tab_css = """
        <style>
        .stx-tab-bar {
            background-color: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 8px !important;
        }
        .stx-tab-bar .stx-tab {
            background-color: #f9fafb !important;
            color: #1f2937 !important;
            border: 1px solid #e5e7eb !important;
        }
        .stx-tab-bar .stx-tab:hover {
            background-color: #f3f4f6 !important;
        }
        .stx-tab-bar .stx-tab.active {
            background-color: #3b82f6 !important;
            color: #ffffff !important;
            border-color: #3b82f6 !important;
        }
        </style>
        """
    
    st.markdown(tab_css, unsafe_allow_html=True)
    
    # Convert tabs to TabBarItemData
    tab_data = [
        stx.TabBarItemData(id=i, title=tab, description=f"{tab} section")
        for i, tab in enumerate(tabs)
    ]
    
    chosen_id = stx.tab_bar(
        data=tab_data,
        default=0,
        key=f"{key_prefix}_navigation"
    )
    
    return chosen_id


def show_navigation_demo():
    """Demonstrate all navigation components."""
    
    st.title("🧭 Enhanced Navigation Demo")
    
    # Dark mode toggle at the top
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        dark_mode = st.session_state.get("dark_mode", False)
        if st.button("🌙 Toggle Dark Mode" if not dark_mode else "☀️ Toggle Light Mode", 
                     use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    # Show current mode
    current_mode = "🌙 Dark Mode" if st.session_state.get("dark_mode", False) else "☀️ Light Mode"
    st.info(f"Current theme: {current_mode}")
    
    # Role selector for demo
    user_role = st.selectbox(
        "Select User Role:",
        ["store_associate", "store_manager", "vp_retail_operations"],
        key="nav_demo_role"
    )
    
    st.markdown("---")
    
    # Main navigation
    st.subheader("🎯 Main Navigation")
    selected_main = create_role_based_navigation(user_role)
    st.success(f"Selected: {selected_main}")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    selected_action = create_quick_actions_pills(user_role)
    if selected_action:
        st.info(f"Quick action: {selected_action}")
    
    st.markdown("---")
    
    # Breadcrumb demo
    st.subheader("🍞 Breadcrumb Navigation")
    create_breadcrumb_navigation("dashboard", user_role)
    
    st.markdown("---")
    
    # Tab navigation demo
    st.subheader("📑 Tab Navigation")
    demo_tabs = ["Overview", "Details", "Analytics", "Settings"]
    selected_tab_id = create_tab_navigation(demo_tabs, "demo")
    st.write(f"Selected tab: {demo_tabs[selected_tab_id]}")
    
    # Sidebar navigation is automatically shown
    
    # Show current navigation state
    if hasattr(st.session_state, 'nav_selection'):
        st.sidebar.success(f"Last selected: {st.session_state.nav_selection}")
    
    # Show dark mode benefits
    st.markdown("---")
    st.subheader("✨ Dark Mode Benefits")
    
    benefits_col1, benefits_col2 = st.columns(2)
    
    with benefits_col1:
        st.markdown("""
        **🌙 Dark Mode Features:**
        - Reduced eye strain in low light
        - Better battery life on OLED displays  
        - Modern, professional appearance
        - Consistent color scheme
        - Improved accessibility
        """)
    
    with benefits_col2:
        st.markdown("""
        **☀️ Light Mode Features:**
        - High contrast for detailed work
        - Better readability in bright environments
        - Familiar traditional interface
        - Clear visual hierarchy
        - Optimal for print materials
        """)


if __name__ == "__main__":
    show_navigation_demo() 