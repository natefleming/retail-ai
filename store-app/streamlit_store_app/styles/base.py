"""
Base CSS styles for the Streamlit Store App.

This module contains core app styles, typography, and global Streamlit overrides
that apply across the entire application.
"""


def get_base_styles(theme_vars):
    """
    Generate base CSS styles using theme variables.
    
    Args:
        theme_vars (dict): Theme variables from theme.py
        
    Returns:
        str: CSS styles string
    """
    return f"""
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Loading Test - Remove this after confirming fonts work */
    .stApp {{
        /* Temporary debug - remove after testing */
        border-top: 3px solid #3b82f6 !important;
    }}
    
    /* Global App Styles with Modern Typography */
    .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        font-feature-settings: 'liga' 1, 'kern' 1;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        font-size: 16px;
        line-height: 1.5;
        background-color: {theme_vars['body_bg']} !important;
        color: {theme_vars['text_color']} !important;
        transition: all 0.3s ease !important;
    }}
    
    /* Aggressive font family enforcement */
    *, *::before, *::after {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    }}
    
    /* Streamlit specific elements */
    .stApp, .stApp *, 
    [data-testid], [class*="st"], 
    .stMarkdown, .stMarkdown *,
    .stButton, .stButton *,
    .stMetric, .stMetric *,
    .stSelectbox, .stSelectbox *,
    .stTextInput, .stTextInput * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    }}
    
    .stApp > header {{
        visibility: hidden;
    }}

    .stDeployButton {{
        display: none;
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    .stDecoration {{
        display: none;
    }}
    
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 1200px;
        background-color: {theme_vars['body_bg']} !important;
        margin-top: 0rem !important;
    }}

    /* Remove default spacing from first element */
    .main .block-container > div:first-child {{
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }}

    /* Remove spacing from columns */
    div[data-testid="column"] {{
        padding-top: 0rem !important;
    }}

    /* Modern Minimal Typography Scale */
    h1 {{
        color: {theme_vars['text_color']};
        font-weight: 700;
        font-size: 2.25rem;
        line-height: 1.25;
        margin-bottom: 1rem;
        letter-spacing: -0.025em;
        font-feature-settings: 'liga' 1, 'kern' 1;
    }}
    
    h2 {{
        color: {theme_vars['text_color']};
        font-weight: 600;
        font-size: 1.875rem;
        line-height: 1.3;
        margin-bottom: 0.875rem;
        letter-spacing: -0.015em;
    }}
    
    h3 {{
        color: {theme_vars['text_color']};
        font-weight: 600;
        font-size: 1.5rem;
        line-height: 1.35;
        margin-bottom: 0.75rem;
        letter-spacing: -0.01em;
    }}
    
    h4 {{
        color: {theme_vars['text_color']};
        font-weight: 500;
        font-size: 1.25rem;
        line-height: 1.4;
        margin-bottom: 0.625rem;
        letter-spacing: 0;
    }}
    
    h5 {{
        color: {theme_vars['text_color']};
        font-weight: 500;
        font-size: 1.125rem;
        line-height: 1.45;
        margin-bottom: 0.5rem;
    }}
    
    h6 {{
        color: {theme_vars['text_color']};
        font-weight: 500;
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.875rem;
    }}
    
    .stMarkdown p {{
        font-size: 1rem;
        line-height: 1.6;
        color: {theme_vars['label_color']};
        margin-bottom: 1rem;
        font-weight: 400;
    }}
    
    .stMarkdown .small {{
        font-size: 0.875rem;
        line-height: 1.5;
    }}
    
    .stMarkdown .large {{
        font-size: 1.125rem;
        line-height: 1.6;
    }}

    /* Text Utility Classes */
    .text-xs {{
        font-size: 0.75rem;
        line-height: 1.4;
    }}
    
    .text-sm {{
        font-size: 0.875rem;
        line-height: 1.5;
    }}
    
    .text-base {{
        font-size: 1rem;
        line-height: 1.6;
    }}
    
    .text-lg {{
        font-size: 1.125rem;
        line-height: 1.6;
    }}
    
    .text-xl {{
        font-size: 1.25rem;
        line-height: 1.5;
    }}
    
    .font-light {{
        font-weight: 300;
    }}
    
    .font-normal {{
        font-weight: 400;
    }}
    
    .font-medium {{
        font-weight: 500;
    }}
    
    .font-semibold {{
        font-weight: 600;
    }}
    
    .font-bold {{
        font-weight: 700;
    }}

    /* Streamlit Component Typography Overrides */
    .stTextInput > div > div > input {{
        background-color: {theme_vars['section_bg']} !important;
        color: {theme_vars['text_color']} !important;
        border-color: {theme_vars['container_border']} !important;
        font-size: 0.875rem;
        font-weight: 400;
        line-height: 1.5;
    }}

    .stTextInput > div > div > input::placeholder {{
        color: {theme_vars['label_color']} !important;
        opacity: 0.7 !important;
        font-weight: 400;
    }}

    .stButton > button {{
        background-color: {theme_vars['container_bg']} !important;
        color: {theme_vars['text_color']} !important;
        border: 1px solid {theme_vars['container_border']} !important;
        font-weight: 500;
        font-size: 0.875rem;
        letter-spacing: 0.025em;
        padding: 0.625rem 1.25rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: {theme_vars['container_shadow']};
    }}
    
    .stButton > button:hover {{
        background-color: {theme_vars['hover_bg']} !important;
        border-color: {theme_vars['container_border']} !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: {theme_vars['container_shadow']};
    }}

    /* Metric Typography */
    [data-testid="metric-container"] [data-testid="metric-value"] {{
        font-size: 1.875rem !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
        letter-spacing: -0.025em !important;
        color: {theme_vars['value_color']} !important;
        margin: 0 !important;
    }}

    [data-testid="metric-container"] [data-testid="metric-label"] {{
        color: {theme_vars['label_color']} !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.025em !important;
        text-transform: uppercase !important;
        margin: 0 !important;
    }}

    [data-testid="metric-container"] [data-testid="metric-delta"] {{
        font-size: 0.75rem !important;
        font-weight: 500 !important;
    }}

    /* Selectbox Typography */
    .stSelectbox > div > div > div {{
        font-size: 0.875rem;
        font-weight: 400;
    }}

    /* Sidebar Typography */
    .stSidebar {{
        font-size: 0.875rem;
        line-height: 1.5;
    }}
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {{
        font-weight: 600;
        line-height: 1.3;
    }}

    /* Remove spacing from text inputs and buttons in header */
    div[data-testid="column"] .stTextInput,
    div[data-testid="column"] .stButton {{
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
    }}

    /* Dark mode support for dataframes and tables */
    .stDataFrame {{
        background-color: {theme_vars['section_bg']} !important;
        font-size: 0.875rem;
    }}

    .stDataFrame [data-testid="stTable"] {{
        background-color: {theme_vars['section_bg']} !important;
        color: {theme_vars['text_color']} !important;
    }}

    /* Dark mode support for plotly charts */
    .js-plotly-plot {{
        background-color: {theme_vars['section_bg']} !important;
    }}

    /* Utility Classes */
    .text-center {{
        text-align: center;
    }}

    .flex {{
        display: flex;
    }}

    .justify-between {{
        justify-content: space-between;
    }}

    .align-center {{
        align-items: center;
    }}

    .gap-4 {{
        gap: 1rem;
    }}

    /* Add custom CSS to minimize vertical gaps between components */
    [data-testid="stVerticalBlock"] > div {{
        margin-bottom: 0.5rem;
    }}

    /* Ensure no extra space is added by Streamlit's default layout */
    .block-container {{
        padding-top: 0rem;
        padding-bottom: 0rem;
    }}

    /* Avoid affecting the sidebar */
    .sidebar .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    """ 