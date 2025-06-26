"""Main homepage coordinator module."""

from datetime import datetime

import streamlit as st
import streamlit_modal as modal
import streamlit_shadcn_ui as ui

from components.chat import show_chat_container, show_chat_widget
from utils.database import get_stores

# Import enhanced components
from components.enhanced_navigation import create_role_based_navigation, create_quick_actions_pills, create_sidebar_navigation
from components.enhanced_charts import create_sales_performance_chart, create_inventory_status_gauge

from .associate import show_associate_homepage

# Import from the modular structure
from .store_manager import show_manager_homepage
from .vp_retail_operations import show_vp_homepage


# User and Role Management Functions
def get_role_info(user_role, employee_name):
    """Get role-specific display information."""
    role_info = {
        "vp_retail_operations": {
            "title": "VP Retail Operations",
            "initials": "".join([name[0] for name in employee_name.split()]),
        },
        "store_manager": {
            "title": "Store Manager", 
            "initials": "".join([name[0] for name in employee_name.split()]),
        },
        "store_associate": {
            "title": "Store Associate",
            "initials": "".join([name[0] for name in employee_name.split()]),
        }
    }
    return role_info.get(user_role, role_info["store_associate"])


def get_store_data(user_role, store_name):
    """Fetch and process store data based on user role."""
    if user_role != "vp_retail_operations":
        stores_df = get_stores()

        # Find current store data
        current_store_data = None
        if not stores_df.empty:
            # Find the store that matches the current store name
            matching_stores = stores_df[stores_df["name"] == store_name]
            if not matching_stores.empty:
                current_store_data = matching_stores.iloc[0]

        # Build store info from database or use fallback
        if current_store_data is not None:
            # Build complete address from database fields
            full_address = f"{current_store_data['address']}, {current_store_data['city']}, {current_store_data['state']} {current_store_data['zip_code']}"
            store_phone = current_store_data["phone"]

            # Determine hours based on is_24_hours flag
            if current_store_data.get("is_24_hours", False):
                store_hours = "24/7"
            else:
                store_hours = "8:00 AM - 9:00 PM"  # Default hours
        else:
            # Fallback data if store not found in database
            full_address = "789 Market St, San Francisco, CA 94102"
            store_phone = "(415) 555-9876"
            store_hours = "8:00 AM - 9:00 PM"

        return {
            "address": full_address,
            "phone": store_phone,
            "hours": store_hours,
            "weather": "72°F ☀️",  # Weather remains mock for now
        }
    else:
        # VP sees regional info instead of specific store details
        return {
            "region": "United States",  
            "total_stores": "847 locations",
            "weather": "Multi-regional coverage",
            "coverage": "All time zones"
        }


def get_page_context(user_role, store_name):
    """Get page title and context information based on user role."""
    if user_role == "vp_retail_operations":
        # VP sees regional overview
        title = "🏢 BrickMart Executive Dashboard"
        subtitle = "United States Regional Operations"
        context_info = {
            "scope": "United States",
            "role": "Vice President, Retail Operations", 
            "coverage": "All BrickMart locations nationwide"
        }
    elif user_role == "store_manager":
        # Manager sees store-specific title with management context
        location = store_name.replace("BrickMart ", "").strip() if store_name.startswith("BrickMart ") else store_name
        title = f"🏪 BrickMart {location} - Manager Dashboard"
        subtitle = "Store Management Operations"
        context_info = {
            "scope": f"{location} Store",
            "role": "Store Manager",
            "coverage": f"Managing {store_name} operations"
        }
    else:
        # Associate sees store-specific title with associate context
        location = store_name.replace("BrickMart ", "").strip() if store_name.startswith("BrickMart ") else store_name
        title = f"🏪 BrickMart {location} - Associate Portal"
        subtitle = "Store Operations & Customer Service"
        context_info = {
            "scope": f"{location} Store",
            "role": "Store Associate",
            "coverage": f"Working at {store_name}"
        }
    
    return title, subtitle, context_info


# Header and UI Component Functions
def create_streamlined_header(chat_modal, chat_notifications=0):
    """Create streamlined header with search bar and AI Assistant button - no extra padding."""
    
    # Header with search and AI Assistant button - single container, no extra margins
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.text_input(
            "Search",
            placeholder="🔍 Search stores, products, or commands...",
            key="homepage_search",
            label_visibility="hidden",
        )

    with col3:
        # Chat button with notification badge
        if chat_notifications > 0:
            button_text = f"AI Assistant ({chat_notifications})"
        else:
            button_text = "AI Assistant"

        if st.button(
            button_text, key="header_chat_btn", type="primary", use_container_width=True
        ):
            st.session_state.chat_notifications = 0
            chat_modal.open()


def create_compact_page_header(user_role, employee_name, store_name):
    """Create compact page header with title and user controls - minimal padding."""
    title, subtitle, context_info = get_page_context(user_role, store_name)
    current_role = get_role_info(user_role, employee_name)
    
    # Single row with proper alignment - no extra spacing
    title_col, controls_col = st.columns([3, 2], vertical_alignment="center")
    
    with title_col:
        st.title(title)
    
    with controls_col:
        # All controls in a single HTML block to avoid column spacing issues
        dark_mode = st.session_state.get("dark_mode", False)
        dark_mode_icon = "🌙" if not dark_mode else "☀️"
        
        # Create controls row - simplified to reduce alignment issues
        switch_col, user_profile_col = st.columns([1, 2], vertical_alignment="center")
        
        with switch_col:
            # Dark mode switch with shadcn UI - commented out
            # dark_mode_switch = ui.switch(
            #     default_checked=dark_mode, 
            #     label="🌙" if not dark_mode else "☀️",
            #     key="dark_mode_switch"
            # )
            
            # # Handle state change
            # if dark_mode_switch != dark_mode:
            #     st.session_state.dark_mode = dark_mode_switch
            #     st.rerun()
            pass
        
        with user_profile_col:
            # Popover button styling - modern, subtle appearance
            st.markdown(
                f"""
                <style>
                /* Modern popover button styling - completely remove all outlines */
                .stPopover > button,
                .stPopover button,
                div[data-testid="stPopover"] > button,
                [class*="stPopover"] button {{
                    height: 40px !important;
                    min-height: 40px !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: flex-start !important;
                    padding: 8px 16px !important;
                    font-size: 0.75rem !important;
                    font-weight: 500 !important;
                    border: none !important;
                    border-radius: 20px !important;
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
                    box-shadow: none !important;
                    outline: none !important;
                    white-space: pre-line !important;
                    line-height: 1.2 !important;
                    text-align: left !important;
                    width: 100% !important;
                    color: #374151 !important;
                    transition: all 0.2s ease !important;
                }}
                
                .stPopover > button:hover,
                .stPopover button:hover,
                div[data-testid="stPopover"] > button:hover,
                [class*="stPopover"] button:hover {{
                    background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%) !important;
                    transform: translateY(-1px) !important;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
                    color: #1f2937 !important;
                    border: none !important;
                    outline: none !important;
                }}
                
                .stPopover > button:focus,
                .stPopover button:focus,
                div[data-testid="stPopover"] > button:focus,
                [class*="stPopover"] button:focus {{
                    outline: none !important;
                    border: none !important;
                    background: linear-gradient(135deg, #ddd6fe 0%, #c4b5fd 100%) !important;
                    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
                    color: #1f2937 !important;
                }}
                
                .stPopover > button:active,
                .stPopover button:active,
                div[data-testid="stPopover"] > button:active,
                [class*="stPopover"] button:active {{
                    transform: translateY(0px) !important;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
                    border: none !important;
                    outline: none !important;
                }}
                
                /* Force remove any potential button outlines */
                .stPopover > button:focus-visible,
                .stPopover button:focus-visible,
                div[data-testid="stPopover"] > button:focus-visible,
                [class*="stPopover"] button:focus-visible {{
                    outline: none !important;
                    border: none !important;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # Popover with modern styling
            with st.popover(
                label=(
                    f"{employee_name}\n"
                    f"{current_role['title']}"
                ),
                help="User profile and settings",
                use_container_width=True
            ):
                create_compact_popover_content(user_role, employee_name, store_name, current_role)


def create_compact_popover_content(user_role, employee_name, store_name, current_role):
    """Create compact popover content without excessive spacing."""
    store_info = get_store_data(user_role, store_name)
    
    # Avatar at the top, centered
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 1.5rem;">
            <div style="width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 2rem; margin-bottom: 12px; box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4); border: 3px solid rgba(255, 255, 255, 0.3);">
                {current_role['initials']}
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 600; font-size: 1.1rem; margin: 0; color: #1f2937;">{employee_name}</div>
                <div style="font-style: italic; margin: 4px 0 0 0; color: #6b7280; font-size: 0.9rem;">{current_role['title']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Role badge - compact
    role_badges = {
        "vp_retail_operations": "🏢 **Executive Level**",
        "store_manager": "🏪 **Management Level**"
    }
    st.markdown(role_badges.get(user_role, "👤 **Associate Level**"))
    
    st.divider()
    
    # Context info - compact
    st.markdown("**Current Context:**")
    if user_role == "vp_retail_operations":
        st.info(f"🌍 **Region:** United States\n🏪 **Stores:** 847 locations\n🕐 **Time:** {datetime.now().strftime('%I:%M %p EST')}")
    else:
        location = store_name.replace("BrickMart ", "").strip() if store_name.startswith("BrickMart ") else store_name
        st.info(f"📍 **Location:** {location}\n📞 **Phone:** {store_info.get('phone', 'N/A')}\n🕐 **Time:** {datetime.now().strftime('%I:%M %p')}")
    
    st.divider()
    
    # Quick actions - compact grid
    st.markdown("**Quick Actions:**")
    actions = [
        ("⚙️ Settings", "popover_settings", "⚙️ Settings opened"),
        ("📊 Performance", "popover_stats", "📊 Performance stats displayed"),
        ("🔔 Notifications", "popover_notifications", "📧 3 new notifications"),
        ("🔐 Sign Out", "popover_signout", "🔐 Signed out successfully")
    ]
    
    for action_text, key, toast_msg in actions:
        if st.button(action_text, use_container_width=True, key=key):
            st.toast(toast_msg)
    
    st.divider()
    
    # Theme status
    dark_mode = st.session_state.get("dark_mode", False)
    theme_status = "🌙 Dark Mode" if dark_mode else "☀️ Light Mode"
    st.caption(f"🎨 **Theme:** {theme_status}")


def setup_chat_modal(dark_mode):
    """Setup and configure the chat modal with proper styling."""
    if dark_mode:
        st.markdown(
            """
            <style>
            /* Target all possible modal containers with high specificity */
            body > div:last-child,
            body > div:last-child > div,
            body > div:last-child > div > div,
            div[data-stale="false"] > div,
            div[data-stale="false"] > div > div,
            iframe + div,
            [data-testid="stVerticalBlock"],
            [data-testid="stHorizontalBlock"] {
                background-color: #111827 !important;
                color: #f9fafb !important;
            }
            
            /* Force all child elements */
            body > div:last-child *,
            [data-testid="stVerticalBlock"] *,
            [data-testid="stHorizontalBlock"] * {
                background-color: inherit !important;
                color: inherit !important;
            }
            
            /* Override any hardcoded white backgrounds */
            div[style*="background-color: white"],
            div[style*="background-color: #fff"],
            div[style*="background-color: #ffffff"],
            div[style*="background: white"],
            div[style*="background: #fff"],
            div[style*="background: #ffffff"] {
                background-color: #111827 !important;
                color: #f9fafb !important;
            }
            </style>
            
            <script>
            // Force modal styling with JavaScript after DOM is ready
            setTimeout(function() {
                // Find and style the modal container
                const targetSelectors = [
                    'body > div:last-child',
                    '[data-testid="stVerticalBlock"]', 
                    '[data-testid="stHorizontalBlock"]',
                    'div[data-stale="false"]'
                ];
                
                targetSelectors.forEach(selector => {
                    const elements = document.querySelectorAll(selector);
                    elements.forEach(el => {
                        el.style.setProperty('background-color', '#111827', 'important');
                        el.style.setProperty('color', '#f9fafb', 'important');
                        
                        // Also style all children
                        const children = el.querySelectorAll('*');
                        children.forEach(child => {
                            const computedStyle = window.getComputedStyle(child);
                            if (computedStyle.backgroundColor === 'rgb(255, 255, 255)' || 
                                computedStyle.backgroundColor === 'white' ||
                                computedStyle.backgroundColor === '#ffffff') {
                                child.style.setProperty('background-color', '#111827', 'important');
                                child.style.setProperty('color', '#f9fafb', 'important');
                            }
                        });
                    });
                });
            }, 50);
            </script>
            """,
            unsafe_allow_html=True,
        )
    
    # Get chat config with fallback
    chat_config = st.session_state.get("config", {}).get(
        "chat",
        {
            "placeholder": "How can I help you today?",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
    )

    # Show chat container with role-specific context
    show_chat_container(config=chat_config)


def route_to_homepage(user_role, selected_nav):
    """Route to appropriate homepage based on user role and selected navigation."""
    if selected_nav:
        # Handle navigation selection
        if user_role == "vp_retail_operations":
            show_vp_homepage(selected_nav)
        elif user_role == "store_manager":
            show_manager_homepage(selected_nav)
        else:  # store_associate
            show_associate_homepage(selected_nav)
    else:
        # Default behavior - show role-specific homepage
        if user_role == "vp_retail_operations":
            show_vp_homepage()
        elif user_role == "store_manager":
            show_manager_homepage()
        else:
            show_associate_homepage()


# Main Homepage Function
def show_homepage():
    """Main homepage function that routes to appropriate view based on user role."""
    # Setup chat modal and notifications
    chat_notifications = st.session_state.get("chat_notifications", 0)
    chat_modal = modal.Modal(
        title="AI Assistant", key="homepage_chat_modal", max_width=700, padding=20
    )
    
    # Create the TailAdmin header with chat button
    #create_streamlined_header(chat_modal, chat_notifications)

    # Get user information
    user_role = st.session_state.get("user_role", "store_associate")
    employee_name = st.session_state.config["employees"][user_role]["name"]
    store_name = st.session_state.store_name

    # Create enhanced sidebar navigation
    create_sidebar_navigation(user_role)

    # Render page header with user controls
    create_compact_page_header(user_role, employee_name, store_name)
    
    # Create columns for the chat button section
    col0, col1, col2, col3, col4 = st.columns([1, 2, 2, 2, 2])
    
    with col4:
        # Enhanced chat button styling for prominence
        st.markdown(
            """
            <style>
            /* Make AI Assistant button stand out - Orange theme */
            div[data-testid="column"]:nth-child(5) .stButton > button {
                background: linear-gradient(135deg, #ff8a00 0%, #e55100 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 25px !important;
                font-weight: 700 !important;
                font-size: 0.9rem !important;
                padding: 12px 24px !important;
                height: 50px !important;
                min-height: 50px !important;
                box-shadow: 0 8px 25px rgba(255, 138, 0, 0.4) !important;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                text-transform: uppercase !important;
                letter-spacing: 0.5px !important;
                position: relative !important;
                overflow: hidden !important;
            }
            
            /* Pulsing animation for notifications */
            div[data-testid="column"]:nth-child(5) .stButton > button[aria-label*="💬 AI Assistant ("] {
                animation: pulse-glow 2s infinite !important;
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
                box-shadow: 0 8px 25px rgba(255, 107, 107, 0.5) !important;
            }
            
            /* Hover effects */
            div[data-testid="column"]:nth-child(5) .stButton > button:hover {
                transform: translateY(-3px) scale(1.05) !important;
                box-shadow: 0 12px 35px rgba(255, 138, 0, 0.6) !important;
                background: linear-gradient(135deg, #e55100 0%, #ff8a00 100%) !important;
            }
            
            /* Active state */
            div[data-testid="column"]:nth-child(5) .stButton > button:active {
                transform: translateY(-1px) scale(1.02) !important;
                box-shadow: 0 6px 20px rgba(255, 138, 0, 0.4) !important;
            }
            
            /* Shimmer effect */
            div[data-testid="column"]:nth-child(5) .stButton > button::before {
                content: '' !important;
                position: absolute !important;
                top: 0 !important;
                left: -100% !important;
                width: 100% !important;
                height: 100% !important;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent) !important;
                animation: shimmer 3s infinite !important;
            }
            
            /* Pulsing glow animation */
            @keyframes pulse-glow {
                0%, 100% {
                    box-shadow: 0 8px 25px rgba(255, 107, 107, 0.5) !important;
                }
                50% {
                    box-shadow: 0 8px 25px rgba(255, 107, 107, 0.8), 0 0 30px rgba(255, 107, 107, 0.3) !important;
                }
            }
            
            /* Shimmer animation */
            @keyframes shimmer {
                0% {
                    left: -100%;
                }
                100% {
                    left: 100%;
                }
            }
            
            /* Focus state for accessibility */
            div[data-testid="column"]:nth-child(5) .stButton > button:focus {
                outline: 3px solid rgba(255, 138, 0, 0.5) !important;
                outline-offset: 2px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Chat button with notifications
        if chat_notifications > 0:
            button_text = f"💬 AI Assistant ({chat_notifications})"
            button_type = "primary"
        else:
            button_text = "💬 AI Assistant"
            button_type = "secondary"
            
        if st.button(
            button_text,
            key="main_chat_button",
            type=button_type,
            use_container_width=True,
            help="Click to open AI Assistant chat"
        ):
            # Clear notifications and open modal
            st.session_state.chat_notifications = 0
            chat_modal.open()

    # Enhanced Navigation Menu
    selected_nav = create_role_based_navigation(user_role)

    # Create store info for chat context (without displaying info bar)
    store_info = get_store_data(user_role, store_name)

    # Handle chat modal
    if chat_modal.is_open():
        with chat_modal.container():
            dark_mode = st.session_state.get("dark_mode", False)
            setup_chat_modal(dark_mode)

    # Route to appropriate homepage based on user role and selected navigation
    route_to_homepage(user_role, selected_nav)
