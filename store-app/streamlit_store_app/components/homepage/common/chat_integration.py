"""Chat integration components."""

import streamlit as st
import streamlit_modal as modal

from components.chat import show_chat_container


def simulate_chat_notification():
    """Simulate receiving a chat notification (for demo purposes)."""
    if "chat_notifications" not in st.session_state:
        st.session_state.chat_notifications = 0

    # Only add notifications if chat is closed
    if not st.session_state.get("chat_window_open", False):
        st.session_state.chat_notifications += 1


def show_persistent_chat():
    """Display a floating chat icon in the lower right corner that opens a chat modal."""
    # Initialize chat state
    if "chat_notifications" not in st.session_state:
        st.session_state.chat_notifications = 0

    # Get current chat status
    chat_status = st.session_state.get("chat_status", "available")

    # Create notification badge HTML
    notification_badge = ""
    if st.session_state.chat_notifications > 0:
        notification_badge = f"""
        <span style="
            position: absolute;
            top: -5px;
            right: -5px;
            background: #ff4757;
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        ">{st.session_state.chat_notifications}</span>
        """

    # Create status indicator
    status_indicators = {
        "available": {"color": "#28a745", "pulse": ""},
        "typing": {"color": "#007bff", "pulse": "animation: pulse 1.5s infinite;"},
        "processing": {"color": "#ffc107", "pulse": "animation: pulse 1s infinite;"},
        "error": {"color": "#dc3545", "pulse": "animation: pulse 2s infinite;"},
    }

    status_info = status_indicators.get(chat_status, status_indicators["available"])

    # Ensure color and pulse values are properly retrieved
    status_color = status_info.get("color", "#28a745")
    status_pulse = status_info.get("pulse", "")

    # Status indicator dot HTML with proper escaping
    status_dot = f"""
    <span style="
        position: absolute;
        bottom: -2px;
        left: -2px;
        background: {status_color};
        border: 2px solid white;
        border-radius: 50%;
        width: 16px;
        height: 16px;
        {status_pulse}
    "></span>
    """

    # Floating chat button HTML
    chat_button_html = f"""
    <style>
    @keyframes pulse {{
        0% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.1); opacity: 0.7; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    </style>
    
    <div id="floating-chat-container" style="
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        z-index: 1000;
    ">
        <div style="
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
            position: relative;
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(0,0,0,0.2)'"
           onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)'"
           onclick="document.getElementById('hidden-chat-btn').click()">
            💬
            {notification_badge}
            {status_dot}
        </div>
    </div>
    """

    # Display the floating chat icon
    st.markdown(chat_button_html, unsafe_allow_html=True)

    # Hidden button for modal trigger
    st.markdown(
        """
    <style>
    #hidden-chat-btn {
        display: none !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create the modal
    chat_modal = modal.Modal(
        title="AI Assistant", key="chat_modal", max_width=600, padding=20
    )

    # Hidden button to trigger modal
    if st.button("Open Chat", key="hidden_chat_btn", help="Open AI Assistant"):
        # Clear notifications when chat is opened
        st.session_state.chat_notifications = 0
        chat_modal.open()

    # Modal content
    if chat_modal.is_open():
        with chat_modal.container():
            # Get chat config with fallback
            chat_config = st.session_state.get("config", {}).get(
                "chat",
                {
                    "placeholder": "How can I help you today?",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            )

            # Show the chat container
            show_chat_container(chat_config)
