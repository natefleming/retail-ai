# Centralized CSS Management System

This directory contains all CSS styling for the Streamlit Store App in a centralized, maintainable structure. All scattered `st.markdown()` CSS injections have been consolidated here.

## 🎯 Overview

**Before:** CSS was scattered across component files with duplicate code and inconsistent styling
**After:** All CSS is centralized, theme-aware, and loaded once at app startup

## 📁 File Structure

```
styles/
├── __init__.py          # Main entry point, exports load_all_styles()
├── theme.py             # Dark/light theme variables and color definitions
├── base.py              # Core app styles, typography, global overrides
├── components.py        # Reusable UI components (cards, metrics, alerts)
├── dashboard.py         # Dashboard-specific styles for all roles
├── homepage.py          # Homepage, navigation, and header styles
└── README.md           # This documentation file
```

## 🚀 Quick Start

### Loading Styles (Already Setup)
```python
# In app.py - styles are loaded once at startup
from styles import load_all_styles

def init_app():
    # ... other initialization
    load_all_styles()  # Loads all centralized styles
```

### Using Styles in Components
```python
# ✅ CORRECT - No CSS injection needed, just use class names
st.markdown(
    f"""
    <div class="metric-container">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ❌ WRONG - Don't inject CSS in components anymore
st.markdown(
    """
    <style>
    .metric-container { ... }  <!-- This is now centralized -->
    </style>
    """,
    unsafe_allow_html=True,
)
```

## 🎨 Available CSS Classes

### Metric Components
```css
.metric-container        # Main metric card container
.metric-value           # Large metric value text
.metric-label           # Metric label text
.metric-badge           # Status badges (positive, negative, neutral, info)
.metric-header          # Metric header with icon
.metric-icon            # Icon container with color variants
```

### Card Components
```css
.task-card              # Task cards with hover effects
.schedule-card          # Schedule/calendar cards
.product-card           # Product inventory cards
.alert-card             # Alert and notification cards
.performance-card       # Performance metric cards
.nav-card               # Navigation cards
```

### Dashboard Specific
```css
.executive-overview-card     # Executive dashboard cards
.regional-performance-card   # Regional performance displays
.alert-title                # Alert card titles
.alert-description          # Alert card descriptions
.opportunity-impact         # Opportunity impact badges
```

### Status Classes
```css
.positive               # Green/success styling
.negative              # Red/error styling
.neutral               # Yellow/warning styling
.info                  # Blue/info styling
```

## 🌗 Theme System

### Theme Variables
All styles use centralized theme variables that automatically switch between light/dark modes:

```python
# theme.py provides these variables
theme_vars = {
    'container_bg': '#ffffff',      # Card backgrounds
    'container_border': '#e5e7eb',  # Border colors
    'value_color': '#1f2937',       # Primary text
    'label_color': '#6b7280',       # Secondary text
    'container_shadow': '...',      # Box shadows
    'hover_shadow': '...',          # Hover effects
    # ... and more
}
```

### Dark Mode Support
All styles automatically adapt to dark mode via `st.session_state.dark_mode`:

```python
# Automatic theme switching
dark_mode = st.session_state.get("dark_mode", False)
theme_vars = get_theme_variables(dark_mode)
```

## 🔧 Making Style Changes

### Adding New Styles
1. **Choose the right module:**
   - `base.py` - Core app styles, typography
   - `components.py` - Reusable UI components  
   - `dashboard.py` - Dashboard-specific styles
   - `homepage.py` - Navigation and header styles

2. **Add your CSS using theme variables:**
```python
def get_component_styles(theme_vars):
    return f"""
    .my-new-component {{
        background: {theme_vars['container_bg']};
        color: {theme_vars['value_color']};
        border: 1px solid {theme_vars['container_border']};
    }}
    """
```

3. **Test in both light and dark modes**

### Modifying Existing Styles
1. **Find the style in the appropriate module**
2. **Make changes in the centralized location ONLY**
3. **Never add duplicate CSS in component files**

## 📝 Migration Notes

### Removed Functions
These CSS injection functions have been removed from component files:
- `inject_executive_dashboard_css()`
- `inject_manager_dashboard_css()`
- `inject_associate_css()`
- `inject_performance_css()`
- `inject_schedule_css()`
- `inject_products_css()`
- `inject_base_css()`
- `inject_avatar_css()`
- `inject_component_css()`
- `inject_tailwind_css()`

### Updated Component Files
All these files now use centralized styles only:
- `executive_dashboard_tab.py`
- `dashboard_tab.py` (store manager)
- `my_tasks_tab.py`
- `performance_tab.py`
- `schedule_tab.py`
- `products_tab.py`
- `homepage.py`

## 🎯 Benefits

### Before (Scattered CSS)
```python
# In executive_dashboard_tab.py
def inject_executive_dashboard_css():
    st.markdown("""<style>.metric-container {...}</style>""")

# In dashboard_tab.py  
def inject_manager_dashboard_css():
    st.markdown("""<style>.metric-container {...}</style>""")

# In my_tasks_tab.py
def inject_associate_css():
    st.markdown("""<style>.metric-container {...}</style>""")
```

### After (Centralized CSS)
```python
# In app.py - loads once for entire app
from styles import load_all_styles
load_all_styles()

# All components just use the classes
st.markdown('<div class="metric-container">...</div>')
```

### Advantages
- **No Duplication:** CSS defined once, used everywhere
- **Consistent Theming:** All styles use same theme variables
- **Better Performance:** CSS loaded once instead of per-component
- **Easier Maintenance:** Single location for style changes
- **Dark Mode Support:** Automatic theme switching
- **Type Safety:** Centralized imports prevent missing functions

## 🐛 Troubleshooting

### Styles Not Applying
1. **Check that `load_all_styles()` is called in `app.py`**
2. **Verify CSS class names match the centralized definitions**
3. **Check browser developer tools for CSS conflicts**

### Missing Styles
1. **Look in the appropriate module** (`base.py`, `components.py`, etc.)
2. **Check if the style uses theme variables correctly**
3. **Verify the style is included in the module's return string**

### Theme Issues
1. **Ensure `dark_mode` is set in `st.session_state`**
2. **Check that theme variables are being passed correctly**
3. **Test style in both light and dark modes**

## 🎉 Success!

The centralized CSS system is now active! All styling is:
- ✅ Centralized in the `styles/` package
- ✅ Theme-aware with dark/light mode support  
- ✅ Performance optimized (loaded once)
- ✅ Maintainable with clear organization
- ✅ Consistent across all components

No more scattered `st.markdown()` CSS injections! 🎊 