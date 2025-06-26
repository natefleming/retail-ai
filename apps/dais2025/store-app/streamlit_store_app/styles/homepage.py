"""
Homepage and navigation CSS styles.

This module contains styles specific to the homepage layout, navigation,
and header components.
"""


def get_homepage_styles(theme_vars):
    """
    Generate homepage CSS styles using theme variables.
    
    Args:
        theme_vars (dict): Theme variables from theme.py
        
    Returns:
        str: CSS styles string
    """
    return f"""
    /* Homepage Specific Overrides */
    .homepage .main .block-container {{
        padding-top: 0rem !important;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: none;
        margin-top: 0rem !important;
        background-color: {theme_vars['body_bg']} !important;
    }}

    /* Header Row Styling */
    .header-row {{
        margin-top: -1rem !important;
        margin-bottom: 1.5rem !important;
    }}

    /* TailwindCSS Integration */
    .tailwind-loaded {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }}

    /* Role Selection and Context */
    .role-selector {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: {theme_vars['container_shadow']};
    }}

    .role-selector h4 {{
        color: {theme_vars['value_color']};
        margin-bottom: 0.75rem;
        font-weight: 600;
    }}

    .role-selector .stSelectbox {{
        margin-bottom: 0.5rem;
    }}

    /* Store Context */
    .store-context {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-bottom: 1rem;
        font-size: 0.875rem;
        color: {theme_vars['label_color']};
    }}

    .store-context strong {{
        color: {theme_vars['value_color']};
    }}

    /* Navigation Tab Styling */
    .nav-tabs {{
        border-bottom: 1px solid {theme_vars['container_border']};
        margin-bottom: 1.5rem;
    }}

    .nav-tab {{
        padding: 0.75rem 1rem;
        border-bottom: 2px solid transparent;
        color: {theme_vars['label_color']};
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }}

    .nav-tab:hover {{
        color: {theme_vars['value_color']};
        border-bottom-color: {theme_vars['container_border']};
    }}

    .nav-tab.active {{
        color: #3b82f6;
        border-bottom-color: #3b82f6;
        font-weight: 600;
    }}

    /* Chat Widget Styling */
    .chat-widget {{
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        z-index: 1000;
    }}

    .chat-button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        position: relative;
        border: 3px solid rgba(255, 255, 255, 0.2);
    }}

    .chat-button:before {{
        content: "💬";
        font-size: 1.8rem;
        display: block;
    }}

    .chat-button:hover {{
        transform: scale(1.1) translateY(-2px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }}

    .chat-button:active {{
        transform: scale(1.05) translateY(-1px);
    }}

    /* Alternative text-based chat button for better visibility */
    .chat-button.text-style {{
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 30px;
        width: auto;
        min-width: 60px;
        padding: 0 1rem;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }}

    .chat-button.text-style:before {{
        content: "💬 Chat";
        font-size: 0.9rem;
    }}

    /* Pulsing animation for attention */
    @keyframes chat-pulse {{
        0% {{
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }}
        50% {{
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.8), 0 0 0 10px rgba(102, 126, 234, 0.1);
        }}
        100% {{
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }}
    }}

    .chat-button.pulse {{
        animation: chat-pulse 2s infinite;
    }}

    /* Dark mode specific adjustments */
    .dark-mode .chat-button {{
        background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
        border: 3px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5);
    }}

    .dark-mode .chat-button:hover {{
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        box-shadow: 0 12px 35px rgba(139, 92, 246, 0.7);
    }}

    .chat-notification {{
        position: absolute;
        top: -5px;
        right: -5px;
        background: #ef4444;
        color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
    }}

    .chat-notification.bounce {{
        animation: bounce 1s infinite;
    }}

    /* Modal Styling */
    .modal-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 2000;
    }}

    .modal-content {{
        background: {theme_vars['container_bg']};
        border-radius: 12px;
        padding: 2rem;
        max-width: 90%;
        max-height: 90%;
        overflow-y: auto;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        border: 1px solid {theme_vars['container_border']};
    }}

    .modal-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid {theme_vars['container_border']};
    }}

    .modal-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {theme_vars['value_color']};
        margin: 0;
    }}

    .modal-close {{
        background: none;
        border: none;
        font-size: 1.5rem;
        color: {theme_vars['label_color']};
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 0.25rem;
        transition: all 0.3s ease;
    }}

    .modal-close:hover {{
        background: {theme_vars['hover_bg']};
        color: {theme_vars['value_color']};
    }}

    /* Responsive Layout */
    @media (max-width: 768px) {{
        .main .block-container {{
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }}

        .header-row {{
            margin-left: -0.5rem;
            margin-right: -0.5rem;
        }}

        .summary-stats {{
            flex-direction: column;
            gap: 0.5rem;
        }}

        .nav-card {{
            padding: 1rem;
        }}

        .nav-card-icon {{
            font-size: 2rem;
        }}

        .chat-widget {{
            bottom: 1rem;
            right: 1rem;
        }}

        .chat-button {{
            width: 50px;
            height: 50px;
            font-size: 1.25rem;
        }}
    }}

    /* Dark Mode Specific Adjustments */
    .dark-mode .stTextInput > div > div > input {{
        background-color: {theme_vars['section_bg']} !important;
        color: {theme_vars['text_color']} !important;
        border-color: {theme_vars['container_border']} !important;
    }}

    .dark-mode .stSelectbox > div > div {{
        background-color: {theme_vars['section_bg']} !important;
        color: {theme_vars['text_color']} !important;
        border-color: {theme_vars['container_border']} !important;
    }}

    /* Loading States */
    .loading-spinner {{
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid {theme_vars['container_border']};
        border-radius: 50%;
        border-top-color: #3b82f6;
        animation: spin 1s ease-in-out infinite;
    }}

    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}

    /* Accessibility Improvements */
    .focus-visible {{
        outline: 2px solid #3b82f6;
        outline-offset: 2px;
    }}

    .sr-only {{
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }}

    /* Avatar and Header Styles */
    .avatar-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0;
        padding: 0;
    }}

    .header-avatar {{
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        margin: 0;
    }}

    .header-avatar:hover {{
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }}

    /* Header Button Styling */
    div[data-testid="column"] .stButton > button {{
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        padding: 8px 16px !important;
        margin: 0 !important;
        font-size: 16px !important;
        line-height: 1 !important;
        border-radius: 8px !important;
        border: 1px solid {theme_vars['container_border']} !important;
        background-color: {theme_vars['container_bg']} !important;
        color: {theme_vars['value_color']} !important;
        box-shadow: {theme_vars['container_shadow']} !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}

    div[data-testid="column"] .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: {theme_vars['hover_shadow']} !important;
        border-color: #3b82f6 !important;
    }}

    /* Smaller Popover Font Styles */
    [data-testid="stPopover"] {{
        font-size: 12px !important;
    }}
    
    [data-testid="stPopover"] > div {{
        font-size: 12px !important;
        background: {theme_vars['container_bg']} !important;
        border: 1px solid {theme_vars['container_border']} !important;
        color: {theme_vars['label_color']} !important;
    }}
    
    [data-testid="stPopover"] p {{
        font-size: 12px !important;
        line-height: 1.4 !important;
        margin: 0.25rem 0 !important;
        color: {theme_vars['label_color']} !important;
        white-space: pre-line;
    }}
    
    [data-testid="stPopover"] strong {{
        color: {theme_vars['value_color']} !important;
        font-weight: 600 !important;
    }}

    /* Base Typography and Layout */
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
        background-color: {theme_vars['body_bg']} !important;
        color: {theme_vars['text_color']} !important;
        line-height: 1.6;
        letter-spacing: -0.01em;
    }}

    /* Streamlit Component Styling */
    .stApp {{
        background-color: {theme_vars['body_bg']} !important;
    }}

    .stSidebar {{
        background-color: {theme_vars['section_bg']} !important;
    }}

    .stSidebar .stSelectbox > div > div {{
        background-color: {theme_vars['section_bg']} !important;
        color: {theme_vars['text_color']} !important;
        border: 1px solid {theme_vars['container_border']} !important;
    }}

    .stSidebar h1, .stSidebar h2, .stSidebar h3 {{
        color: {theme_vars['text_color']} !important;
    }}

    .stSidebar .stMarkdown {{
        color: {theme_vars['text_color']} !important;
    }}

    .stTextInput > div > div > input {{
        background-color: {theme_vars['section_bg']} !important;
        color: {theme_vars['text_color']} !important;
        border: 1px solid {theme_vars['container_border']} !important;
        border-radius: 8px !important;
    }}

    .stTextInput > div > div > input:focus {{
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }}

    .stButton > button[data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    }}

    .stButton > button[data-testid="baseButton-primary"]:hover {{
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6) !important;
    }}

    .stSelectbox > div > div {{
        background-color: {theme_vars['section_bg']} !important;
        color: {theme_vars['text_color']} !important;
        border: 1px solid {theme_vars['container_border']} !important;
        border-radius: 8px !important;
    }}

    .stSelectbox > div > div > div {{
        color: {theme_vars['text_color']} !important;
    }}
    """ 