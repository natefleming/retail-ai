"""
CSS styles for the Streamlit Store App.

This module has been refactored to use a centralized CSS management system.
All styles are now organized in the styles/ package for better maintainability.

DEPRECATED: The old scattered CSS approach has been replaced.
Use the new centralized system in styles/ package instead.
"""

import streamlit as st


def load_css():
    """
    Load custom CSS styles for the application.
    
    This function now redirects to the centralized CSS management system
    in the styles/ package to eliminate scattered CSS throughout the app.
    """
    # Import the new centralized CSS system
    try:
        from styles import load_all_styles
        load_all_styles()
    except ImportError:
        # Fallback to old system if new styles package not available
        st.warning("New centralized CSS system not available. Using fallback styles.")
        _load_fallback_css()


def _load_fallback_css():
    """
    Fallback CSS loader for compatibility.
    
    This provides minimal styling if the new centralized system fails.
    Only use this as a last resort - prefer the centralized system.
    """
    st.markdown(
        """
        <style>
        /* Minimal Fallback Styles */
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background-color: #f8fafc;
        }
        
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        h1, h2, h3, h4 {
            color: #1e293b;
            font-weight: 700;
        }
        
        .stMarkdown p {
            color: #64748b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
