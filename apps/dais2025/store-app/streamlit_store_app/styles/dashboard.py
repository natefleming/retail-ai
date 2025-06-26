"""
Dashboard CSS styles for role-specific dashboards.

This module contains styles specific to dashboard views including metrics,
performance cards, charts, and executive dashboard components.
"""

from .theme import BRAND_COLORS, STATUS_COLORS


def get_dashboard_styles(theme_vars):
    """
    Generate dashboard CSS styles using theme variables.
    
    Args:
        theme_vars (dict): Theme variables from theme.py
        
    Returns:
        str: CSS styles string
    """
    return f"""
    /* Modern Metric Containers */
    .metric-container {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: {theme_vars['container_shadow']};
        transition: all 0.3s ease-in-out;
    }}

    .metric-container:hover {{
        transform: translateY(-4px);
        box-shadow: {theme_vars['hover_shadow']};
    }}

    .metric-header {{
        display: flex;
        align-items: center;
        justify-content: flex-start;
        margin-bottom: 0.75rem;
    }}

    .metric-icon {{
        width: 40px;
        height: 40px;
        border-radius: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.125rem;
    }}

    .metric-value-container {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.25rem;
    }}

    .metric-badge {{
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: 700;
    }}

    .metric-badge.positive {{
        background-color: #d1fae5;
        color: #065f46;
    }}

    .metric-badge.negative {{
        background-color: #fee2e2;
        color: #991b1b;
    }}

    .metric-badge.neutral {{
        background-color: #fef3c7;
        color: #92400e;
    }}

    .metric-badge.info {{
        background-color: #dbeafe;
        color: #1e40af;
    }}

    .metric-value {{
        font-size: 1.75rem;
        font-weight: 600;
        color: {theme_vars['value_color']};
        margin: 0;
    }}

    .metric-label {{
        color: {theme_vars['label_color']};
        font-size: 1rem;
        margin: 0;
    }}

    /* Update CSS for st.metric components with increased specificity */
    [data-testid="metric-container"] {{
        background: {theme_vars['container_bg']} !important;
        color: {theme_vars['value_color']} !important;
        border: 1px solid {theme_vars['container_border']} !important;
        border-radius: 0.75rem !important;
        padding: 1.25rem !important;
        margin-bottom: 1rem !important;
        box-shadow: {theme_vars['container_shadow']} !important;
        transition: all 0.3s ease-in-out !important;
    }}

    [data-testid="metric-container"]:hover {{
        transform: translateY(-4px) !important;
        box-shadow: {theme_vars['hover_shadow']} !important;
    }}

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
        text-transform: uppercase !important;
        letter-spacing: 0.025em !important;
        margin: 0 !important;
        line-height: 1.4 !important;
    }}

    /* Modern Performance Cards */
    .modern-performance-card {{
        background: {theme_vars['container_bg']};
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: {theme_vars['container_shadow']};
        border: 1px solid {theme_vars['container_border']};
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }}
    
    .modern-performance-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_shadow']};
    }}

    /* Executive Dashboard Specific Styles */
    .executive-overview-card {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: {theme_vars['container_shadow']};
        transition: all 0.3s ease-in-out;
        text-align: center;
    }}

    .executive-overview-card:hover {{
        transform: translateY(-4px);
        box-shadow: {theme_vars['hover_shadow']};
    }}

    /* Segmented Control Styling */
    .st-segmented-control > div {{
        background-color: {theme_vars['hover_bg']};
        color: {theme_vars['text_color']};
        border-radius: 0.5rem;
        padding: 0.5rem;
        transition: background-color 0.3s ease;
        font-size: 0.875rem;
        font-weight: 500;
    }}
    
    .st-segmented-control > div:hover {{
        background-color: {theme_vars['active_bg']};
    }}
    
    .st-segmented-control > div:active {{
        background-color: {theme_vars['container_bg']};
    }}
    
    .st-segmented-control > div > div {{
        background-color: {theme_vars['hover_bg']};
        color: {theme_vars['label_color']};
        border: 1px solid {theme_vars['container_border']};
        font-size: 0.75rem;
        font-weight: 500;
    }}
    
    .st-segmented-control > div > div[aria-selected="true"] {{
        background-color: {BRAND_COLORS['primary']};
        color: #ffffff;
        border: 1px solid {BRAND_COLORS['primary']};
        font-weight: 600;
    }}

    /* Performance Trends Section */
    .performance-trends-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}

    .trend-card {{
        background: {theme_vars['container_bg']};
        border-radius: 12px;
        padding: 1rem;
        box-shadow: {theme_vars['container_shadow']};
        border: 1px solid {theme_vars['container_border']};
        transition: all 0.3s ease;
    }}

    .trend-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_shadow']};
    }}

    .trend-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }}

    .trend-title {{
        font-weight: 600;
        font-size: 0.875rem;
        line-height: 1.4;
        color: {theme_vars['value_color']};
    }}

    .trend-change {{
        font-weight: 600;
        font-size: 0.75rem;
        line-height: 1.4;
    }}

    .trend-change.positive {{
        color: {STATUS_COLORS['positive']};
    }}

    .trend-change.negative {{
        color: {STATUS_COLORS['negative']};
    }}

    .trend-change.neutral {{
        color: {STATUS_COLORS['neutral']};
    }}

    .trend-progress {{
        height: 8px;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #e2e8f0 0%, {STATUS_COLORS['positive']} 100%);
    }}

    .trend-value {{
        font-size: 1.125rem;
        font-weight: 600;
        line-height: 1.3;
        letter-spacing: -0.015em;
        color: {theme_vars['value_color']};
        margin-bottom: 0.25rem;
    }}

    .trend-subtitle {{
        font-size: 0.75rem;
        color: {theme_vars['label_color']};
        font-weight: 400;
        line-height: 1.5;
    }}

    /* Alert and Opportunity Cards */
    .strategic-alert-card {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 16px;
        box-shadow: {theme_vars['container_shadow']};
        transition: all 0.3s;
    }}

    .strategic-alert-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_shadow']};
    }}

    .strategic-alert-card.warning {{
        border-left: 4px solid {STATUS_COLORS['warning']};
    }}

    .strategic-alert-card.info {{
        border-left: 4px solid {BRAND_COLORS['info']};
    }}

    .strategic-alert-card.success {{
        border-left: 4px solid {STATUS_COLORS['positive']};
    }}

    .alert-title {{
        font-weight: 600;
        font-size: 0.875rem;
        line-height: 1.4;
        margin-bottom: 8px;
        color: {theme_vars['value_color']};
    }}

    .alert-description {{
        font-size: 0.75rem;
        margin-bottom: 12px;
        color: {theme_vars['label_color']};
        line-height: 1.5;
        font-weight: 400;
    }}

    .alert-action {{
        font-size: 0.625rem;
        font-style: italic;
        color: {theme_vars['label_color']};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .opportunity-impact {{
        font-size: 0.625rem;
        color: {STATUS_COLORS['positive']};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Regional Performance Cards */
    .regional-performance-card {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: {theme_vars['container_shadow']};
        transition: all 0.3s;
    }}

    .regional-performance-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_shadow']};
    }}

    .regional-title {{
        font-weight: 600;
        font-size: 0.875rem;
        line-height: 1.4;
        color: {theme_vars['value_color']};
        margin-bottom: 8px;
    }}

    .regional-metrics {{
        font-size: 0.75rem;
        color: {theme_vars['label_color']};
        line-height: 1.5;
        font-weight: 400;
    }}

    .regional-performance {{
        font-weight: 600;
        color: {theme_vars['value_color']};
    }}

    .regional-revenue {{
        font-weight: 600;
        color: {theme_vars['value_color']};
    }}

    .regional-growth {{
        font-weight: 600;
    }}

    /* Chart Container Styles */
    .chart-container {{
        background: {theme_vars['container_bg']};
        border-radius: 12px;
        padding: 1rem;
        box-shadow: {theme_vars['container_shadow']};
        border: 1px solid {theme_vars['container_border']};
        margin-bottom: 1rem;
    }}

    .chart-title {{
        font-size: 1rem;
        font-weight: 600;
        line-height: 1.4;
        color: {theme_vars['value_color']};
        margin-bottom: 1rem;
        text-align: center;
    }}

    /* Action Center Buttons */
    .action-center {{
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid {theme_vars['container_border']};
    }}

    .action-button {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        color: {theme_vars['text_color']};
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        letter-spacing: 0.025em;
        transition: all 0.3s ease;
        cursor: pointer;
        text-align: center;
        width: 100%;
        line-height: 1.4;
    }}

    .action-button:hover {{
        background: {theme_vars['hover_bg']};
        transform: translateY(-1px);
        box-shadow: {theme_vars['container_shadow']};
    }}

    /* Icon Background Colors for Executive Dashboard */
    .metric-icon.blue {{ 
        background: {"#1e3a8a" if theme_vars.get('dark_mode') else "#dbeafe"}; 
    }}
    .metric-icon.yellow {{ 
        background: {"#92400e" if theme_vars.get('dark_mode') else "#fef3c7"}; 
    }}
    .metric-icon.purple {{ 
        background: {"#3730a3" if theme_vars.get('dark_mode') else "#e0e7ff"}; 
    }}
    .metric-icon.amber {{ 
        background: {"#b45309" if theme_vars.get('dark_mode') else "#fef7cd"}; 
    }}
    .metric-icon.green {{ 
        background: {"#065f46" if theme_vars.get('dark_mode') else "#d1fae5"}; 
    }}
    .metric-icon.pink {{ 
        background: {"#9d174d" if theme_vars.get('dark_mode') else "#fce7f3"}; 
    }}
    .metric-icon.indigo {{ 
        background: {"#312e81" if theme_vars.get('dark_mode') else "#e0e7ff"}; 
    }}
    .metric-icon.red {{ 
        background: {"#991b1b" if theme_vars.get('dark_mode') else "#fee2e2"}; 
    }}

    /* Alert and Opportunity Card Styles */
    .alert-card {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 16px;
        box-shadow: {theme_vars['container_shadow']};
        transition: all 0.3s;
    }}
    
    .alert-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_shadow']};
    }}
    
    .alert-card.warning {{
        border-left: 4px solid #f59e0b;
        background: {"#fffbeb" if not theme_vars.get('dark_mode') else theme_vars['container_bg']};
    }}
    
    .alert-card.info {{
        border-left: 4px solid #3b82f6;
        background: {"#eff6ff" if not theme_vars.get('dark_mode') else theme_vars['container_bg']};
    }}
    
    .alert-card.success {{
        border-left: 4px solid #10b981;
        background: {"#ecfdf5" if not theme_vars.get('dark_mode') else theme_vars['container_bg']};
    }}
    
    .alert-title {{
        font-weight: 600;
        margin-bottom: 8px;
        color: {theme_vars['value_color']};
    }}
    
    .alert-description {{
        font-size: 14px;
        margin-bottom: 12px;
        color: {theme_vars['label_color']};
        line-height: 1.5;
    }}
    
    .alert-action {{
        font-size: 12px;
        font-style: italic;
        color: {theme_vars['label_color']};
        font-weight: 500;
    }}
    
    .opportunity-impact {{
        font-size: 12px;
        color: #059669;
        font-weight: 600;
    }}
    
    /* Region Performance Cards */
    .region-card {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: {theme_vars['container_shadow']};
        transition: all 0.3s;
    }}
    
    .region-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_shadow']};
    }}
    
    .region-name {{
        font-weight: 600;
        color: {theme_vars['value_color']};
        margin-bottom: 8px;
    }}
    
    .region-details {{
        font-size: 14px;
        color: {theme_vars['label_color']};
        line-height: 1.5;
    }}
    
    .region-metric {{
        font-weight: 600;
        color: {theme_vars['value_color']};
    }}

    /* Aggressive Vertical Spacing Removal for Executive Dashboard */
    [data-testid="stVerticalBlock"] > div {{
        margin-bottom: 0rem !important;
        margin-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-top: 0rem !important;
    }}
    
    [data-testid="stHorizontalBlock"] > div {{
        margin-bottom: 0rem !important;
        margin-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-top: 0rem !important;
    }}
    
    div[data-testid="column"] {{
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
    }}
    
    .stMarkdown {{
        margin-bottom: 0.5rem !important;
        margin-top: 0rem !important;
    }}
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
        margin-bottom: 0.5rem !important;
        margin-top: 0.5rem !important;
        padding: 0rem !important;
    }}
    
    .stPlotlyChart {{
        margin-bottom: 0rem !important;
        margin-top: 0rem !important;
    }}
    
    .stButton {{
        margin-bottom: 0rem !important;
        margin-top: 0rem !important;
    }}
    
    .stSegmentedControl {{
        margin-bottom: 0rem !important;
        margin-top: 0rem !important;
    }}
    
    .element-container {{
        margin-bottom: 0rem !important;
        margin-top: 0rem !important;
    }}
    
    .streamlit-expanderHeader, .streamlit-expanderContent {{
        margin-bottom: 0rem !important;
        margin-top: 0rem !important;
    }}
    
    .stMarkdown p {{
        line-height: 1.2 !important;
        margin-bottom: 0.25rem !important;
        margin-top: 0rem !important;
    }}
    
    .main .block-container > div:first-child {{
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }}
    
    div[data-testid="column"] > div {{
        margin-bottom: 0rem !important;
        margin-top: 0rem !important;
    }}
    """ 