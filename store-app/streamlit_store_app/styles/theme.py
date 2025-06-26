"""
Theme variables and color definitions for dark/light mode support.

This module centralizes all theme-related variables to ensure consistent
styling across the entire application.
"""


def get_theme_variables(dark_mode=False):
    """
    Get theme variables for the specified mode.
    
    Args:
        dark_mode (bool): Whether to use dark mode colors
        
    Returns:
        dict: Dictionary containing all theme variables
    """
    if dark_mode:
        # Dark mode color palette
        return {
            # Background colors
            "body_bg": "#111827",
            "container_bg": "#1f2937", 
            "section_bg": "#1f2937",
            "card_bg": "#1f2937",
            
            # Border colors
            "container_border": "#374151",
            "border_color": "#374151",
            
            # Text colors
            "text_color": "#f9fafb",
            "value_color": "#f9fafb",
            "label_color": "#9ca3af",
            "muted_color": "#6b7280",
            
            # Shadow effects
            "container_shadow": "0 1px 3px 0 rgba(0, 0, 0, 0.3)",
            "hover_shadow": "0 10px 25px 0 rgba(0, 0, 0, 0.4)",
            "card_shadow": "0 4px 20px rgba(0,0,0,0.3)",
            "hover_card_shadow": "0 8px 30px rgba(0,0,0,0.4)",
            
            # Interactive states
            "hover_bg": "#374151",
            "active_bg": "#1f2937",
            
            # Icon backgrounds (dark mode)
            "icon_bg_blue": "#1e3a8a",
            "icon_bg_yellow": "#92400e", 
            "icon_bg_purple": "#3730a3",
            "icon_bg_amber": "#b45309",
            "icon_bg_green": "#065f46",
            "icon_bg_pink": "#9d174d",
            "icon_bg_indigo": "#312e81",
            "icon_bg_red": "#991b1b"
        }
    else:
        # Light mode color palette
        return {
            # Background colors
            "body_bg": "#f8fafc",
            "container_bg": "white",
            "section_bg": "white", 
            "card_bg": "white",
            
            # Border colors
            "container_border": "#e5e7eb",
            "border_color": "#e5e7eb",
            
            # Text colors
            "text_color": "#1f2937",
            "value_color": "#1f2937",
            "label_color": "#6b7280",
            "muted_color": "#9ca3af",
            
            # Shadow effects
            "container_shadow": "0 1px 3px 0 rgba(0, 0, 0, 0.1)",
            "hover_shadow": "0 10px 25px 0 rgba(0, 0, 0, 0.15)",
            "card_shadow": "0 4px 20px rgba(0,0,0,0.08)",
            "hover_card_shadow": "0 8px 30px rgba(0,0,0,0.12)",
            
            # Interactive states
            "hover_bg": "#f3f4f6",
            "active_bg": "#e5e7eb",
            
            # Icon backgrounds (light mode)
            "icon_bg_blue": "#dbeafe",
            "icon_bg_yellow": "#fef3c7",
            "icon_bg_purple": "#e0e7ff", 
            "icon_bg_amber": "#fef7cd",
            "icon_bg_green": "#d1fae5",
            "icon_bg_pink": "#fce7f3",
            "icon_bg_indigo": "#e0e7ff",
            "icon_bg_red": "#fee2e2"
        }


# Brand colors (consistent across themes)
BRAND_COLORS = {
    "primary": "#3b82f6",
    "success": "#10b981", 
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#06b6d4",
    "purple": "#8b5cf6"
}


# Status colors for metrics and alerts
STATUS_COLORS = {
    "positive": "#10b981",
    "negative": "#ef4444", 
    "neutral": "#6b7280",
    "warning": "#f59e0b",
    "info": "#3b82f6"
}


# Gradient definitions
GRADIENTS = {
    "card_light": "linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)",
    "card_dark": "linear-gradient(135deg, #1f2937 0%, #111827 100%)",
    "blue": "linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%)",
    "green": "linear-gradient(135deg, #22c55e 0%, #16a34a 50%, #15803d 100%)",
    "amber": "linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #d97706 100%)",
    "red": "linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #b91c1c 100%)",
    "purple": "linear-gradient(135deg, #8b5cf6 0%, #7c3aed 50%, #6d28d9 100%)"
} 