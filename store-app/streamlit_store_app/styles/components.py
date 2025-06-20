"""
Component CSS styles for reusable UI elements.

This module contains styles for cards, metrics, alerts, and other reusable
components used throughout the application.
"""

from .theme import BRAND_COLORS, STATUS_COLORS, GRADIENTS


def get_component_styles(theme_vars):
    """
    Generate component CSS styles using theme variables.
    
    Args:
        theme_vars (dict): Theme variables from theme.py
        
    Returns:
        str: CSS styles string
    """
    return f"""
    /* Enhanced KPI Summary Cards */
    .kpi-summary-card {{
        background: {theme_vars['card_bg']};
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: {theme_vars['card_shadow']};
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        text-align: center;
        border: 1px solid {theme_vars['container_border']};
    }}
    
    .kpi-summary-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_card_shadow']};
    }}
    
    .kpi-icon {{
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
    }}
    
    .kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
        letter-spacing: -0.025em;
        color: {theme_vars['value_color']};
        margin-bottom: 0.5rem;
    }}
    
    .kpi-label {{
        font-size: 0.875rem;
        color: {theme_vars['label_color']};
        margin-bottom: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        line-height: 1.4;
    }}
    
    .kpi-change {{
        font-size: 0.75rem;
        color: {theme_vars['label_color']};
        font-weight: 500;
        line-height: 1.4;
    }}
    
    .kpi-change.positive {{
        color: {STATUS_COLORS['positive']};
        font-weight: 600;
    }}

    /* Enhanced Inventory Summary Cards */
    .inventory-summary-card {{
        background: {theme_vars['card_bg']};
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: {theme_vars['card_shadow']};
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        text-align: center;
        border: 1px solid {theme_vars['container_border']};
    }}
    
    .inventory-summary-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_card_shadow']};
    }}
    
    .inventory-icon {{
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
    }}
    
    .inventory-value {{
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
        letter-spacing: -0.025em;
        color: {theme_vars['value_color']};
        margin-bottom: 0.5rem;
    }}
    
    .inventory-label {{
        font-size: 0.875rem;
        color: {theme_vars['label_color']};
        margin-bottom: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        line-height: 1.4;
    }}
    
    .inventory-detail {{
        font-size: 0.75rem;
        color: {theme_vars['label_color']};
        font-weight: 400;
        line-height: 1.5;
    }}

    /* Enhanced Summary Cards for Navigation */
    .summary-card {{
        background: {theme_vars['card_bg']};
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: {theme_vars['card_shadow']};
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        border: 1px solid {theme_vars['container_border']};
    }}
    
    .summary-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_card_shadow']};
    }}
    
    .summary-card.team {{
        border-left-color: {STATUS_COLORS['positive']};
    }}
    
    .summary-card.inventory {{
        border-left-color: {BRAND_COLORS['purple']};
    }}
    
    .summary-card.tasks {{
        border-left-color: {STATUS_COLORS['negative']};
    }}
    
    .summary-card.schedule {{
        border-left-color: {STATUS_COLORS['warning']};
    }}
    
    .summary-card.products {{
        border-left-color: {BRAND_COLORS['info']};
    }}
    
    .summary-stats {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 1.25rem;
        gap: 1rem;
    }}
    
    .stat-item {{
        text-align: center;
        flex: 1;
    }}
    
    .stat-value {{
        font-size: 1.25rem;
        font-weight: 600;
        line-height: 1.3;
        letter-spacing: -0.015em;
        color: {theme_vars['value_color']};
        margin-bottom: 0.25rem;
    }}
    
    .stat-label {{
        font-size: 0.75rem;
        color: {theme_vars['label_color']};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        line-height: 1.4;
    }}

    /* Enhanced Alert Cards */
    .alert-card {{
        background: {theme_vars['card_bg']};
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid;
        box-shadow: {theme_vars['card_shadow']};
        transition: all 0.3s ease;
    }}
    
    .alert-card:hover {{
        transform: translateY(-1px);
        box-shadow: {theme_vars['hover_card_shadow']};
    }}
    
    .alert-card.high {{
        border-left-color: {STATUS_COLORS['negative']};
    }}
    
    .alert-card.medium {{
        border-left-color: {STATUS_COLORS['warning']};
    }}
    
    .alert-card.low {{
        border-left-color: {STATUS_COLORS['info']};
    }}
    
    .alert-title {{
        font-weight: 600;
        font-size: 0.875rem;
        line-height: 1.4;
        margin-bottom: 0.5rem;
        color: {theme_vars['value_color']};
    }}
    
    .alert-description {{
        font-size: 0.75rem;
        color: {theme_vars['label_color']};
        margin-bottom: 0.5rem;
        line-height: 1.5;
        font-weight: 400;
    }}
    
    .alert-meta {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.625rem;
        color: {theme_vars['muted_color']};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .alert-card.warning {{
        border-left-color: {STATUS_COLORS['warning']};
    }}
    
    .alert-card.error {{
        border-left-color: {STATUS_COLORS['negative']};
    }}
    
    .alert-card.success {{
        border-left-color: {STATUS_COLORS['positive']};
    }}
    
    .alert-card.info {{
        border-left-color: {BRAND_COLORS['info']};
    }}

    /* Enhanced Task Cards */
    .task-card {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: {theme_vars['container_shadow']};
        transition: all 0.3s ease-in-out;
        position: relative;
    }}
    
    .task-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_shadow']};
    }}

    .task-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }}

    .task-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {theme_vars['value_color']};
        margin: 0;
    }}

    .task-due {{
        font-size: 0.875rem;
        color: {theme_vars['label_color']};
        margin: 0;
    }}

    .task-type-badge {{
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .task-type-service {{
        background: #8b5cf6;
        color: white;
    }}

    .task-type-bopis {{
        background: #3b82f6;
        color: white;
    }}

    .task-type-restock {{
        background: #10b981;
        color: white;
    }}

    .task-badge {{
        position: absolute;
        top: -8px;
        right: 1rem;
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
    }}

    /* Navigation Cards */
    .nav-card {{
        background: {theme_vars['card_bg']};
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: {theme_vars['card_shadow']};
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        border: 1px solid {theme_vars['container_border']};
        cursor: pointer;
        text-decoration: none;
        color: inherit;
    }}
    
    .nav-card:hover {{
        transform: translateY(-4px);
        box-shadow: {theme_vars['hover_card_shadow']};
        text-decoration: none;
        color: inherit;
    }}
    
    .nav-card-icon {{
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-align: center;
        display: block;
    }}
    
    .nav-card-title {{
        font-size: 1.125rem;
        font-weight: 600;
        line-height: 1.4;
        letter-spacing: -0.01em;
        color: {theme_vars['value_color']};
        margin-bottom: 0.5rem;
        text-align: center;
    }}
    
    .nav-card-description {{
        font-size: 0.75rem;
        color: {theme_vars['label_color']};
        text-align: center;
        line-height: 1.5;
        font-weight: 400;
    }}

    /* Header Avatar */
    .header-avatar {{
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: {BRAND_COLORS['primary']};
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.025em;
    }}

    /* Progress Bars */
    .progress-bar {{
        width: 100%;
        height: 8px;
        background-color: {theme_vars['container_border']};
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }}
    
    .progress-fill {{
        height: 100%;
        background: linear-gradient(90deg, {theme_vars['container_border']} 0%, {STATUS_COLORS['positive']} 100%);
        transition: width 0.3s ease;
    }}

    /* Badge Components */
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.625rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        line-height: 1.2;
    }}
    
    .badge.success {{
        background-color: rgba(16, 185, 129, 0.1);
        color: {STATUS_COLORS['positive']};
    }}
    
    .badge.warning {{
        background-color: rgba(245, 158, 11, 0.1);
        color: {STATUS_COLORS['warning']};
    }}
    
    .badge.danger {{
        background-color: rgba(239, 68, 68, 0.1);
        color: {STATUS_COLORS['negative']};
    }}
    
    .badge.info {{
        background-color: rgba(59, 130, 246, 0.1);
        color: {STATUS_COLORS['info']};
    }}

    /* Schedule Card Styles */
    .schedule-card {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: {theme_vars['container_shadow']};
        transition: all 0.3s ease-in-out;
    }}

    .schedule-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_shadow']};
    }}

    .schedule-day {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {theme_vars['value_color']};
        margin-bottom: 0.5rem;
    }}

    .schedule-time {{
        font-size: 0.95rem;
        color: {theme_vars['label_color']};
        margin-bottom: 0.25rem;
    }}

    .schedule-status {{
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }}

    .status-today {{
        background: #3b82f6;
        color: white;
    }}

    .status-upcoming {{
        background: #10b981;
        color: white;
    }}

    .status-off {{
        background: #6b7280;
        color: white;
    }}

    /* Product Card Styles */
    .product-card {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: {theme_vars['container_shadow']};
        transition: all 0.3s ease-in-out;
    }}

    .product-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_shadow']};
    }}

    .product-name {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {theme_vars['value_color']};
        margin-bottom: 0.5rem;
    }}

    .product-details {{
        font-size: 0.875rem;
        color: {theme_vars['label_color']};
        margin-bottom: 0.25rem;
    }}

    .stock-badge {{
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }}

    .stock-low {{
        background: #fee2e2;
        color: #991b1b;
    }}

    .stock-critical {{
        background: #ef4444;
        color: white;
    }}

    .stock-good {{
        background: #d1fae5;
        color: #065f46;
    }}

    .promotion-badge {{
        background: #fbbf24;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }}

    /* Performance Card Styles */
    .performance-card {{
        background: {theme_vars['container_bg']};
        border: 1px solid {theme_vars['container_border']};
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: {theme_vars['container_shadow']};
        transition: all 0.3s ease-in-out;
        text-align: center;
    }}

    .performance-card:hover {{
        transform: translateY(-2px);
        box-shadow: {theme_vars['hover_shadow']};
    }}

    .performance-icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
        display: block;
    }}

    .performance-value {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        display: block;
        line-height: 1;
        color: {theme_vars['value_color']};
    }}

    .performance-label {{
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: {theme_vars['label_color']};
        margin-bottom: 0.5rem;
    }}

    .performance-change {{
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
    }}

    .performance-change.positive {{
        background-color: #d1fae5;
        color: #065f46;
    }}

    .performance-change.negative {{
        background-color: #fee2e2;
        color: #991b1b;
    }}
    """ 