"""Development Playground - Create and test visual components in isolation."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import streamlit as st

# Remove the global sys.path modification to prevent interference with main app imports


def show_component():
    """Display a round button that toggles a modal window."""
    
    # Initialize session state for modal visibility and user dropdown
    if "modal_open" not in st.session_state:
        st.session_state.modal_open = False
    if "user_dropdown_open" not in st.session_state:
        st.session_state.user_dropdown_open = False
    
    # TailAdmin-style header with Streamlit-compatible components
    st.markdown("""
    <style>
    .tailadmin-header-container {
        background: white;
        border-bottom: 1px solid #e5e7eb;
        padding: 1rem 0;
        margin: -1rem -1rem 1.5rem -1rem;
    }
    
    .user-profile-btn {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 0.75rem !important;
        padding: 0.5rem 1rem !important;
        color: #1f2937 !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .user-profile-btn:hover {
        background: #f3f4f6 !important;
        border-color: #d1d5db !important;
    }
    
    .notification-btn-style {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        font-size: 1.1rem !important;
        padding: 0 !important;
    }
    
    .dropdown-container {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        padding: 0.5rem 0;
        margin-top: 0.5rem;
    }
    
    .dropdown-item-btn {
        width: 100% !important;
        text-align: left !important;
        background: transparent !important;
        border: none !important;
        padding: 0.75rem 1rem !important;
        color: #374151 !important;
        font-weight: 400 !important;
        border-radius: 0 !important;
    }
    
    .dropdown-item-btn:hover {
        background: #f9fafb !important;
    }
    
    .danger-btn:hover {
        background: #fef2f2 !important;
        color: #dc2626 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header container
    st.markdown('<div class="tailadmin-header-container">', unsafe_allow_html=True)
    
    # Header layout with columns
    header_col1, header_col2 = st.columns([2, 1])
    
    with header_col1:
        st.markdown("""
        <div>
            <h2 style="margin: 0; color: #1f2937; font-size: 1.25rem; font-weight: 600;">
                🧪 TailAdmin Playground
            </h2>
            <p style="margin: 0; color: #6b7280; font-size: 0.875rem;">
                Development environment for UI components
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with header_col2:
        # Right side header controls
        notification_col, user_col = st.columns([1, 3])
        
        with notification_col:
            if st.button("🔔 3", key="notifications_btn", help="You have 3 notifications"):
                st.toast("🔔 You have 3 new notifications!")
        
        with user_col:
            # User profile button
            user_display = "👤 Sarah Chen (Developer)"
            if st.button(user_display, key="user_profile_btn", help="User menu"):
                st.session_state.user_dropdown_open = not st.session_state.user_dropdown_open
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # User dropdown menu (appears below header when open)
    if st.session_state.user_dropdown_open:
        st.markdown("---")
        st.markdown("### 👤 User Menu")
        
        # Create dropdown with columns for better layout
        dropdown_col1, dropdown_col2, dropdown_col3 = st.columns([1, 2, 1])
        
        with dropdown_col2:
            st.markdown('<div class="dropdown-container">', unsafe_allow_html=True)
            
            # Profile option
            if st.button("👤 My Profile", key="profile_option", use_container_width=True):
                st.success("📱 Opening user profile...")
                st.session_state.user_dropdown_open = False
                st.rerun()
            
            # Settings option
            if st.button("⚙️ Account Settings", key="settings_option", use_container_width=True):
                st.success("⚙️ Opening account settings...")
                st.session_state.user_dropdown_open = False
                st.rerun()
            
            # Preferences option
            if st.button("🎨 Preferences", key="preferences_option", use_container_width=True):
                st.success("🎨 Opening preferences...")
                st.session_state.user_dropdown_open = False
                st.rerun()
            
            # Help option
            if st.button("❓ Help & Support", key="help_option", use_container_width=True):
                st.success("❓ Opening help center...")
                st.session_state.user_dropdown_open = False
                st.rerun()
            
            # Logout option (with different styling)
            if st.button("🚪 Sign Out", key="logout_option", use_container_width=True, type="secondary"):
                if st.session_state.get("confirm_logout", False):
                    st.error("🚪 Signing out...")
                    st.session_state.user_dropdown_open = False
                    st.session_state.confirm_logout = False
                    st.rerun()
                else:
                    st.session_state.confirm_logout = True
                    st.warning("Click 'Sign Out' again to confirm logout")
                    st.rerun()
            
            # Close dropdown option
            if st.button("✕ Close Menu", key="close_dropdown", use_container_width=True):
                st.session_state.user_dropdown_open = False
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Reset confirmation if dropdown is closed
    if not st.session_state.user_dropdown_open:
        st.session_state.confirm_logout = False
    
    # Separator
    st.markdown("---")
    
    # Rest of the existing modal functionality
    # Custom CSS for round button and modal styling
    st.markdown("""
    <style>
    .round-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    
    .round-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .modal-content {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        max-width: 500px;
        width: 90%;
        position: relative;
        animation: modalFadeIn 0.3s ease-out;
    }
    
    .modal-close {
        position: absolute;
        top: 10px;
        right: 15px;
        background: none;
        border: none;
        font-size: 24px;
        cursor: pointer;
        color: #666;
        padding: 5px;
    }
    
    .modal-close:hover {
        color: #333;
    }
    
    @keyframes modalFadeIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create columns for centering the button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("### Round Button Demo")
        st.markdown("Click the round button below to open the modal!")
        
        # Round button
        if st.button("🚀", key="round_btn", help="Click to open modal"):
            st.session_state.modal_open = not st.session_state.modal_open
    
    # Display modal if open
    if st.session_state.modal_open:
        # Create modal overlay that appears on top of everything
        st.markdown(f"""
        <div id="modal-overlay" style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(0, 0, 0, 0.6);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(3px);
        ">
            <div style="
                background: white;
                padding: 2.5rem;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
                max-width: 500px;
                width: 90%;
                max-height: 80vh;
                overflow-y: auto;
                position: relative;
                animation: modalSlideIn 0.3s ease-out;
                border: 1px solid #e1e5e9;
            ">
                <div style="
                    position: absolute;
                    top: 15px;
                    right: 20px;
                    font-size: 28px;
                    color: #666;
                    cursor: pointer;
                    user-select: none;
                    line-height: 1;
                    width: 30px;
                    height: 30px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 15px;
                    transition: all 0.2s ease;
                " 
                onmouseover="this.style.backgroundColor='#f5f5f5'; this.style.color='#333';"
                onmouseout="this.style.backgroundColor='transparent'; this.style.color='#666';"
                onclick="document.getElementById('modal-overlay').style.display='none';">
                    ×
                </div>
                
                <h2 style="
                    color: #667eea; 
                    margin-top: 0; 
                    margin-bottom: 1.5rem;
                    font-size: 1.8rem;
                    border-bottom: 2px solid #f0f2f6;
                    padding-bottom: 0.5rem;
                ">🎉 Modal Window</h2>
                
                <p style="margin-bottom: 1rem; color: #444; line-height: 1.6;">
                    This is a beautiful modal window that appears on top of the existing screen!
                </p>
                
                <p style="margin-bottom: 1rem; color: #444; font-weight: 500;">
                    You can add any content here:
                </p>
                
                <ul style="margin-bottom: 1.5rem; color: #555; line-height: 1.8;">
                    <li>📝 Forms and inputs</li>
                    <li>📊 Charts and visualizations</li>
                    <li>⚙️ Additional functionality</li>
                    <li>🎨 Custom components</li>
                    <li>💡 Whatever you need!</li>
                </ul>
                
                <div style="
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 1rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                    border-left: 4px solid #667eea;
                ">
                    <p style="margin: 0; color: #555; font-style: italic;">
                        💡 <strong>Tip:</strong> Click the × button above, click outside the modal, 
                        or use the Streamlit close button below to dismiss this modal.
                    </p>
                </div>
                
                <div style="text-align: center; margin-top: 1.5rem;">
                    <div style="
                        display: inline-block;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 0.7rem 1.5rem;
                        border-radius: 25px;
                        font-weight: 500;
                        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                    ">
                        Modal is fully functional! 🚀
                    </div>
                </div>
            </div>
        </div>
        
        <style>
        @keyframes modalSlideIn {{
            from {{
                opacity: 0;
                transform: scale(0.9) translateY(-20px);
            }}
            to {{
                opacity: 1;
                transform: scale(1) translateY(0);
            }}
        }}
        
        /* Prevent body scroll when modal is open */
        body {{
            overflow: hidden;
        }}
        </style>
        
        <script>
        // Click outside to close modal
        document.getElementById('modal-overlay').addEventListener('click', function(e) {{
            if (e.target === this) {{
                this.style.display = 'none';
            }}
        }});
        
        // Prevent clicks inside modal from closing it
        document.querySelector('#modal-overlay > div').addEventListener('click', function(e) {{
            e.stopPropagation();
        }});
        </script>
        """, unsafe_allow_html=True)
        
        # Streamlit-based close button (appears below the modal for backup)
        st.markdown("---")
        st.markdown("**🔹 Modal Control Panel** (Modal is currently open above)")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("✕ Close Modal (Streamlit)", key="close_modal", type="primary"):
                st.session_state.modal_open = False
                st.rerun()
    
    # Additional demo content
    st.markdown("---")
    st.markdown("### Features:")
    st.markdown("""
    - **🎯 Round Button**: Beautifully styled circular button with gradient background
    - **✨ Hover Effects**: Smooth animations and shadow effects
    - **🔄 Toggle Functionality**: Click to open/close the modal
    - **📱 Modal Window**: Overlay modal with fade-in animation
    - **🎨 Modern Design**: Clean, professional styling
    """)
    
    # Show current state for debugging
    with st.expander("Debug Info"):
        st.write(f"Modal State: {'Open' if st.session_state.modal_open else 'Closed'}")
        if st.button("Reset State"):
            st.session_state.modal_open = False
            st.rerun()


def main():
    """Main development playground page."""
    st.set_page_config(
        page_title="Playground",
        page_icon="🧪",
        layout="wide",
    )

    st.title("🧪 TailAdmin Header Component")
    st.markdown("A pixel-perfect recreation of the TailAdmin dashboard header")
    
    show_component()


if __name__ == "__main__":
    main() 