"""Store context management utilities."""

from datetime import datetime

import streamlit as st

from utils.database import get_stores

# Fallback store data for Downtown Market
DOWNTOWN_MARKET_FALLBACK = {
    "id": "081",
    "name": "BrickMart Downtown Market",
    "address": "789 Market St",
    "city": "San Francisco",
    "state": "CA",
    "zip_code": "94102",
    "phone": "(415) 555-9876",
    "email": "downtown-market@brickmart.com",
    "size_sqft": 4500,
    "rating": 4.8,
    "type": "flagship",
    "is_24_hours": False,
    "latitude": 37.7849,
    "longitude": -122.4094,
    "region_id": "6",
    "hours": {
        "monday": {"open": "08:00", "close": "22:00"},
        "tuesday": {"open": "08:00", "close": "22:00"},
        "wednesday": {"open": "08:00", "close": "22:00"},
        "thursday": {"open": "08:00", "close": "22:00"},
        "friday": {"open": "08:00", "close": "23:00"},
        "saturday": {"open": "09:00", "close": "23:00"},
        "sunday": {"open": "09:00", "close": "21:00"},
    },
}


def _initialize_session_state():
    """Initialize session state variables with defaults."""
    defaults = {
        "store_id": None,
        "store_name": None,
        "user_role": None,
        "show_context_switcher": True,
        "store_details": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def _set_fallback_store():
    """Set the fallback store context."""
    st.session_state.store_id = DOWNTOWN_MARKET_FALLBACK["id"]
    st.session_state.store_name = DOWNTOWN_MARKET_FALLBACK["name"]
    st.session_state.store_details = DOWNTOWN_MARKET_FALLBACK.copy()


def init_store_context():
    """Initialize store context in session state."""
    # Initialize session state variables
    _initialize_session_state()

    # Auto-set store to Downtown Market in San Francisco if not already set
    if not st.session_state.store_id:
        try:
            stores_df = get_stores()
            if not stores_df.empty:
                # Try to find the Downtown Market store in database
                downtown_market = stores_df[
                    stores_df["name"] == "BrickMart Downtown Market"
                ]
                if not downtown_market.empty:
                    store = downtown_market.iloc[0]
                    st.session_state.store_id = store["id"]
                    st.session_state.store_name = store["name"]
                    st.session_state.store_details = store.to_dict()
                else:
                    # Use fallback if not found in database
                    _set_fallback_store()
            else:
                # Use fallback if no stores in database
                _set_fallback_store()
        except Exception:
            # Use fallback if database fails
            _set_fallback_store()

    # Auto-set default role to vp_retail_operations if not already set
    if not st.session_state.user_role:
        st.session_state.user_role = "vp_retail_operations"


def toggle_context_switcher():
    """Toggle the visibility of the context switcher."""
    st.session_state.show_context_switcher = not st.session_state.show_context_switcher


def check_permission(permission: str) -> bool:
    """Check if current user role has a specific permission."""
    if not st.session_state.user_role:
        return False
    return st.session_state.config["roles"][st.session_state.user_role].get(
        permission, False
    )


def format_store_details(store_details: dict) -> str:
    """Format store details for display."""
    # Get employee name from config
    employee_name = st.session_state.config["employees"][st.session_state.user_role][
        "name"
    ]
    role_title = st.session_state.user_role.replace("_", " ").title()

    # Handle 24/7 stores
    if store_details.get("is_24_hours", False):
        hours_display = "24/7"
    else:
        current_day = datetime.now().strftime("%A").lower()
        hours = store_details["hours"][current_day]
        hours_display = f"{hours['open']} - {hours['close']}"

    # Remove "BrickMart" from store name for display
    clean_store_name = store_details["name"].replace("BrickMart ", "").strip()

    return (
        f"**Working as:** {employee_name} ({role_title})\n\n"
        f"**Store:** {clean_store_name}\n"
        f"**Type:** {store_details['type'].title()}\n"
        f"**Address:** {store_details['address']}\n"
        f"**Location:** {store_details['city']}, {store_details['state']} {store_details['zip_code']}\n"
        f"**Hours Today:** {hours_display}\n"
        f"**Size:** {store_details['size_sqft']:,} sq ft\n"
        f"**Rating:** {'⭐' * int(store_details['rating'])}"
    )


def show_context_selector():
    """Display the store context selector."""
    if not st.session_state.show_context_switcher:
        return

    with st.sidebar:
        st.markdown("### Store Context")

        # Fixed store context - Downtown Market, San Francisco
        st.info("🏪 **Fixed Store Context**\nBrickMart Downtown Market, San Francisco")

        # Role selection
        roles = list(st.session_state.config["roles"].keys())

        # Find current role index
        current_role_index = 0  # Default to first role (store_manager)
        if st.session_state.get("user_role") and st.session_state.user_role in roles:
            current_role_index = roles.index(st.session_state.user_role)

        def on_role_change():
            """Callback function when role changes."""
            selected_role = st.session_state.role_selector
            if selected_role != st.session_state.user_role:
                st.session_state.user_role = selected_role

        st.selectbox(
            "Select Role:",
            options=roles,
            index=current_role_index,
            format_func=lambda x: x.replace("_", " ").title(),
            key="role_selector",
            on_change=on_role_change,
        )

        # Show current role status
        if st.session_state.user_role:
            role_display = st.session_state.user_role.replace("_", " ").title()
            st.success(f"✅ Current Role: {role_display}")

        # Add demo controls for store managers
        if st.session_state.user_role == "store_manager":
            st.markdown("---")
            st.markdown("### 🎬 Demo Controls")
            
            # Import demo controls here to avoid circular imports
            try:
                from components.homepage.store_manager.demo_alerts import show_demo_alert_controls
                show_demo_alert_controls()
            except ImportError:
                st.error("Demo system not available")

    # Show current context
    if st.session_state.store_name and st.session_state.user_role:
        st.sidebar.success(format_store_details(st.session_state.store_details))
    else:
        st.sidebar.warning("Please select a role to continue")


# COMMENTED OUT: Store selection functionality (preserved for future use)
"""
def show_store_selector():
    # Get stores from database
    try:
        stores_df = get_stores()
        if stores_df.empty:
            st.error("No stores available. Please check your database connection.")
            return
        
        # Convert DataFrame to list of dicts for easier handling
        stores = stores_df.to_dict('records')
        
        # Create display names with format "City, State - Store Name" and remove "BrickMart"
        store_display_options = []
        store_display_to_store = {}
        
        for store in stores:
            # Remove "BrickMart" from store name
            clean_name = store["name"].replace("BrickMart ", "").strip()
            display_name = f"{store['city']}, {store['state']} - {clean_name}"
            store_display_options.append(display_name)
            store_display_to_store[display_name] = store
        
        # Find current selection index
        current_selection_index = None
        if st.session_state.get("store_name"):
            current_store_name = st.session_state.store_name
            for i, (display_name, store) in enumerate(store_display_to_store.items()):
                if store["name"] == current_store_name:
                    current_selection_index = i
                    break
        
        # Store selection
        selected_store_display = st.selectbox(
            "Select Store:",
            options=store_display_options,
            index=current_selection_index,
            placeholder="Choose a store..."
        )
    except Exception as e:
        st.error(f"Error loading stores: {str(e)}")
        st.info("Please check your database configuration.")
"""
