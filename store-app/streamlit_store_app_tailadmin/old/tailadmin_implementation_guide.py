"""
TailAdmin Style Implementation Guide for Streamlit

This script demonstrates how to extract and apply TailAdmin styles to Streamlit apps with examples and best practices.

Key Implementation Methods:
1. Direct CSS injection with st.components.v1.html()
2. Custom HTML components with TailAdmin classes using components
3. Interactive components with JavaScript functionality
4. Complete layout patterns mimicking TailAdmin dashboard

Run this script to see a full demonstration.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Import our TailAdmin styling modules
from components.tailadmin import (
    TAILADMIN_COLORS,
    create_tailadmin_header,
    create_tailadmin_plotly_chart,
    display_enhanced_metrics_grid,
    display_tailadmin_chart_card,
    display_tailadmin_data_table,
    display_tailadmin_notification,
    finalize_tailadmin_app,
    initialize_tailadmin_app,
    inject_tailadmin_css,
)


def main():
    """Main implementation guide and demonstration."""

    # ========================================================================
    # STEP 1: INITIALIZE TAILADMIN APP
    # ========================================================================

    initialize_tailadmin_app()

    # Add JavaScript message listener for component interactions
    components.html(
        """
    <script>
        // Listen for messages from components
        window.addEventListener('message', function(event) {
            if (event.data.type === 'metricClicked') {
                console.log('Streamlit received metric click:', event.data.metric);
            } else if (event.data.type === 'actionClicked') {
                console.log('Streamlit received action click:', event.data.action);
            } else if (event.data.type === 'menuSelected') {
                console.log('Streamlit received menu selection:', event.data.index);
            }
        });
    </script>
    """,
        height=0,
    )

    st.markdown("""
    # 🎨 TailAdmin Style Implementation Guide (Enhanced)

    This guide demonstrates how to extract and apply TailAdmin's visual styles to Streamlit applications using `st.components.v1.html` for enhanced interactivity.

    ## Implementation Methods Covered:

    1. **CSS Extraction & Injection** - Direct CSS styling with components
    2. **Interactive Components** - JavaScript-enabled TailAdmin components
    3. **Chart Styling** - TailAdmin-themed Plotly charts
    4. **Enhanced Tables** - Interactive tables with search, sort, pagination
    5. **Layout Patterns** - Dashboard layout structures with functionality

    ---
    """)

    # ========================================================================
    # STEP 2: DEMONSTRATE INTERACTIVE STYLING CONCEPTS
    # ========================================================================

    st.markdown("## 🎯 Interactive Styling Concepts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Color Palette with Theme Configuration
        TailAdmin colors are now integrated into Streamlit's theme via `.streamlit/config.toml`:
        """)

        # Show color palette with interactive demo
        color_demo = f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0;">
            <div onclick="colorClicked('brand')" style="
                padding: 1rem;
                background: {TAILADMIN_COLORS["brand"]["500"]};
                color: white;
                border-radius: 0.5rem;
                text-align: center;
                cursor: pointer;
                transition: transform 0.2s ease;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <strong>Brand Primary</strong><br>#{TAILADMIN_COLORS["brand"]["500"][1:]}
            </div>
            <div onclick="colorClicked('success')" style="
                padding: 1rem;
                background: {TAILADMIN_COLORS["success"]["500"]};
                color: white;
                border-radius: 0.5rem;
                text-align: center;
                cursor: pointer;
                transition: transform 0.2s ease;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <strong>Success</strong><br>#{TAILADMIN_COLORS["success"]["500"][1:]}
            </div>
            <div onclick="colorClicked('warning')" style="
                padding: 1rem;
                background: {TAILADMIN_COLORS["warning"]["500"]};
                color: white;
                border-radius: 0.5rem;
                text-align: center;
                cursor: pointer;
                transition: transform 0.2s ease;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <strong>Warning</strong><br>#{TAILADMIN_COLORS["warning"]["500"][1:]}
            </div>
            <div onclick="colorClicked('error')" style="
                padding: 1rem;
                background: {TAILADMIN_COLORS["error"]["500"]};
                color: white;
                border-radius: 0.5rem;
                text-align: center;
                cursor: pointer;
                transition: transform 0.2s ease;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <strong>Error</strong><br>#{TAILADMIN_COLORS["error"]["500"][1:]}
            </div>
            <div onclick="colorClicked('gray-light')" style="
                padding: 1rem;
                background: {TAILADMIN_COLORS["gray"]["100"]};
                color: {TAILADMIN_COLORS["gray"]["700"]};
                border-radius: 0.5rem;
                text-align: center;
                cursor: pointer;
                transition: transform 0.2s ease;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <strong>Gray Light</strong><br>#{TAILADMIN_COLORS["gray"]["100"][1:]}
            </div>
            <div onclick="colorClicked('gray-dark')" style="
                padding: 1rem;
                background: {TAILADMIN_COLORS["gray"]["700"]};
                color: white;
                border-radius: 0.5rem;
                text-align: center;
                cursor: pointer;
                transition: transform 0.2s ease;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <strong>Gray Dark</strong><br>#{TAILADMIN_COLORS["gray"]["700"][1:]}
            </div>
        </div>

        <script>
            function colorClicked(colorName) {{
                alert('Color clicked: ' + colorName);
                window.parent.postMessage({{type: 'colorClicked', color: colorName}}, '*');
            }}
        </script>
        """
        components.html(color_demo, height=220)

    with col2:
        st.markdown("""
        ### Interactive Typography System
        TailAdmin typography with hover effects and interactions:
        """)

        typography_demo = f"""
        <div style="font-family: 'Outfit', sans-serif; padding: 1rem; background: {TAILADMIN_COLORS["gray"]["50"]}; border-radius: 0.75rem;">
            <h1 onclick="fontClicked('title-xl')" style="
                font-size: 2.25rem;
                font-weight: 800;
                color: {TAILADMIN_COLORS["gray"]["900"]};
                margin: 0.5rem 0;
                cursor: pointer;
                transition: color 0.2s ease;
            " onmouseover="this.style.color='{TAILADMIN_COLORS["brand"]["500"]}'"
               onmouseout="this.style.color='{TAILADMIN_COLORS["gray"]["900"]}'">
                Title XL (Interactive)
            </h1>

            <h2 onclick="fontClicked('title-md')" style="
                font-size: 1.5rem;
                font-weight: 600;
                color: {TAILADMIN_COLORS["gray"]["700"]};
                margin: 0.5rem 0;
                cursor: pointer;
                transition: color 0.2s ease;
            " onmouseover="this.style.color='{TAILADMIN_COLORS["brand"]["500"]}'"
               onmouseout="this.style.color='{TAILADMIN_COLORS["gray"]["700"]}'">
                Title Medium (Click me)
            </h2>

            <p onclick="fontClicked('body')" style="
                font-size: 1rem;
                color: {TAILADMIN_COLORS["gray"]["600"]};
                margin: 0.5rem 0;
                cursor: pointer;
                transition: color 0.2s ease;
            " onmouseover="this.style.color='{TAILADMIN_COLORS["brand"]["500"]}'"
               onmouseout="this.style.color='{TAILADMIN_COLORS["gray"]["600"]}'">
                Body text with hover effects and interaction
            </p>

            <small onclick="fontClicked('small')" style="
                font-size: 0.875rem;
                color: {TAILADMIN_COLORS["gray"]["500"]};
                cursor: pointer;
                transition: color 0.2s ease;
            " onmouseover="this.style.color='{TAILADMIN_COLORS["brand"]["500"]}'"
               onmouseout="this.style.color='{TAILADMIN_COLORS["gray"]["500"]}'">
                Small text (Interactive)
            </small>
        </div>

        <script>
            function fontClicked(fontType) {{
                console.log('Font clicked:', fontType);
                window.parent.postMessage({{type: 'fontClicked', font: fontType}}, '*');
            }}
        </script>
        """
        components.html(typography_demo, height=220)

    # ========================================================================
    # STEP 3: DEMONSTRATE ENHANCED COMPONENT IMPLEMENTATIONS
    # ========================================================================

    st.markdown("## 📊 Enhanced Interactive Components")

    # Enhanced Header with functionality
    create_tailadmin_header(
        title="Interactive Dashboard Demo",
        subtitle="Real-time analytics with enhanced user interactions",
        user_info={"name": "Sarah Johnson", "role": "Store Manager", "avatar": "👩‍💼"},
        search_enabled=True,
        notifications=5,
    )

    # Enhanced Metrics Grid with click handlers
    st.markdown("### Interactive Metrics Grid")
    st.markdown("Click on any metric card to see interactions in action!")

    metrics = [
        {
            "icon": "💰",
            "value": "$42,750",
            "label": "Monthly Revenue",
            "change": "15.2%",
            "change_type": "positive",
            "target": "$50,000",
            "description": "85% of monthly target achieved",
        },
        {
            "icon": "🛒",
            "value": "1,247",
            "label": "Total Orders",
            "change": "8.7%",
            "change_type": "positive",
            "description": "Orders processed this month",
        },
        {
            "icon": "👥",
            "value": "3,892",
            "label": "Active Customers",
            "change": "12.3%",
            "change_type": "positive",
            "description": "Unique customers this period",
        },
        {
            "icon": "📈",
            "value": "94.2%",
            "label": "Customer Satisfaction",
            "change": "2.1%",
            "change_type": "positive",
            "description": "Based on recent surveys",
        },
    ]

    display_enhanced_metrics_grid(metrics, columns=4)

    # ========================================================================
    # STEP 4: INTERACTIVE CHART DEMONSTRATION
    # ========================================================================

    st.markdown("### Enhanced Chart Styling with TailAdmin Theme")

    # Generate sample data
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    sales_data = pd.DataFrame(
        {"Date": dates, "Sales": np.random.randint(2000, 8000, 30) + np.cumsum(np.random.randn(30) * 200)}
    )

    revenue_data = pd.DataFrame(
        {
            "Category": ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"],
            "Revenue": [45000, 32000, 28000, 19000, 12000],
        }
    )

    col1, col2 = st.columns(2)

    with col1:
        # Line chart
        fig_line = create_tailadmin_plotly_chart(
            data=sales_data, chart_type="line", title="Sales Trend", height=350, color_scheme="brand"
        )

        display_tailadmin_chart_card(
            fig=fig_line, title="Daily Sales Performance", description="Interactive sales trend over the last 30 days"
        )

    with col2:
        # Bar chart
        fig_bar = create_tailadmin_plotly_chart(
            data=revenue_data, chart_type="bar", title="Revenue by Category", height=350, color_scheme="multi"
        )

        display_tailadmin_chart_card(
            fig=fig_bar, title="Category Performance", description="Interactive revenue breakdown by category"
        )

    # ========================================================================
    # STEP 5: ENHANCED TABLE DEMONSTRATION
    # ========================================================================

    st.markdown("### Interactive Enhanced Data Tables")
    st.markdown("Try searching, sorting, and clicking actions in the table below!")

    # Sample table data
    products_data = pd.DataFrame(
        {
            "Product Name": [
                "iPhone 15 Pro",
                "Samsung Galaxy S24",
                "MacBook Pro M3",
                "Dell XPS 13",
                "iPad Air",
                "Surface Pro 9",
                "AirPods Pro",
                "Sony WH-1000XM5",
            ],
            "Category": [
                "Electronics",
                "Electronics",
                "Computers",
                "Computers",
                "Tablets",
                "Tablets",
                "Audio",
                "Audio",
            ],
            "Price": [999, 849, 1599, 1199, 599, 1299, 249, 399],
            "Stock": [45, 67, 23, 34, 78, 56, 123, 89],
            "Rating": [4.8, 4.6, 4.9, 4.5, 4.7, 4.4, 4.9, 4.8],
            "Status": ["In Stock", "In Stock", "Low Stock", "In Stock", "In Stock", "In Stock", "In Stock", "In Stock"],
        }
    )

    display_tailadmin_data_table(
        data=products_data,
        title="Interactive Product Inventory",
        searchable=True,
        sortable=True,
        pagination=True,
        actions=[
            {"text": "Add Product", "type": "primary", "icon": "➕"},
            {"text": "Export Data", "type": "secondary", "icon": "📊"},
            {"text": "Import", "type": "secondary", "icon": "📥"},
            {"text": "Refresh", "type": "secondary", "icon": "🔄"},
        ],
        row_actions=True,
        height=400,
    )

    # ========================================================================
    # STEP 6: ENHANCED NOTIFICATION SYSTEM
    # ========================================================================

    st.markdown("### Interactive Notification System")
    st.markdown("Notifications now support auto-dismiss and click interactions!")

    col1, col2 = st.columns(2)

    with col1:
        display_tailadmin_notification(
            title="Success!",
            message="Product inventory has been successfully updated with enhanced interactions.",
            notification_type="success",
            dismissible=True,
            duration=10,  # Auto-dismiss after 10 seconds
        )

        display_tailadmin_notification(
            title="Warning",
            message="Low stock alert: 3 products need restocking. Click to dismiss.",
            notification_type="warning",
            dismissible=True,
        )

    with col2:
        display_tailadmin_notification(
            title="Information",
            message="New interactive features have been added to the dashboard components.",
            notification_type="info",
            dismissible=True,
            duration=15,
        )

        display_tailadmin_notification(
            title="Error",
            message="Failed to sync with external inventory system. Please retry.",
            notification_type="error",
            dismissible=True,
        )

    # ========================================================================
    # STEP 7: IMPLEMENTATION CODE EXAMPLES
    # ========================================================================

    st.markdown("## 💻 Enhanced Implementation Code Examples")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Components HTML", "Interactive Features", "Chart Styling", "Theme Config", "Best Practices"]
    )

    with tab1:
        st.markdown("### Using st.components.v1.html")
        st.code(
            '''
# Import components
import streamlit.components.v1 as components
from components.tailadmin import inject_tailadmin_css

# Inject CSS using components (better isolation)
inject_tailadmin_css()

# Create interactive component
interactive_html = """
<div class="tailadmin-card" onclick="cardClicked()">
    <h3>Interactive Card</h3>
    <p>Click me for interaction!</p>
    <button class="tailadmin-btn tailadmin-btn-primary" onclick="buttonClicked()">
        Action Button
    </button>
</div>

<script>
    function cardClicked() {
        console.log('Card clicked');
        window.parent.postMessage({type: 'cardClicked'}, '*');
    }

    function buttonClicked() {
        alert('Button clicked!');
        window.parent.postMessage({type: 'buttonClicked'}, '*');
    }
</script>
"""

# Render with components.html for interactivity
components.html(interactive_html, height=200)
        ''',
            language="python",
        )

    with tab2:
        st.markdown("### Interactive Features Implementation")
        st.code(
            '''
# Enhanced component with JavaScript functionality
def create_interactive_metric_card(label, value, **kwargs):
    """Create an interactive metric card with click handlers."""

    card_html = f"""
    <div class="tailadmin-card" onclick="metricClicked('{label}')"
         style="cursor: pointer; transition: transform 0.2s ease;">
        <div class="metric-content">
            <h3>{label}</h3>
            <div class="metric-value">{value}</div>
        </div>
    </div>

    <script>
        function metricClicked(label) {{
            // Send message to Streamlit
            window.parent.postMessage({{
                type: 'metricClicked',
                metric: label,
                timestamp: new Date().toISOString()
            }}, '*');
        }}
    </script>
    """

    components.html(card_html, height=150)

# Listen for messages in main app
components.html("""
<script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'metricClicked') {
            console.log('Metric clicked:', event.data.metric);
            // Handle the interaction
        }
    });
</script>
""", height=0)
        ''',
            language="python",
        )

    with tab3:
        st.markdown("### Advanced Chart Styling")
        st.code(
            '''
# Create TailAdmin-styled interactive charts
def create_enhanced_chart(data, chart_type="line"):
    """Create an enhanced chart with TailAdmin styling."""

    fig = create_tailadmin_plotly_chart(
        data=data,
        chart_type=chart_type,
        title="Interactive Chart",
        color_scheme="brand",  # Uses TailAdmin brand colors
        height=400
    )

    # Add custom interactivity
    fig.update_layout(
        clickmode='event+select',
        title={
            'font': {'family': 'Outfit, sans-serif'},
            'x': 0
        }
    )

    # Display in enhanced card with actions
    display_tailadmin_chart_card(
        fig=fig,
        title="Enhanced Analytics",
        description="Click data points for details",
        actions="""
        <button class="tailadmin-btn tailadmin-btn-primary" onclick="exportChart()">
            📊 Export
        </button>
        <button class="tailadmin-btn tailadmin-btn-secondary" onclick="refreshChart()">
            🔄 Refresh
        </button>
        """
    )
        ''',
            language="python",
        )

    with tab4:
        st.markdown("### Streamlit Theme Configuration")
        st.code(
            """
# .streamlit/config.toml
[theme]
# TailAdmin brand primary color
primaryColor = "#465fff"

# TailAdmin background colors
backgroundColor = "#f9fafb"
secondaryBackgroundColor = "#ffffff"

# TailAdmin text color
textColor = "#344054"

# Font family (sans serif for Outfit fallback)
font = "sans serif"

# Light theme to match TailAdmin default
base = "light"

[server]
# Enable components functionality
enableCORS = false
enableXsrfProtection = false

# Custom theme variables for TailAdmin integration
[theme.custom]
tailadmin_brand_500 = "#465fff"
tailadmin_gray_50 = "#f9fafb"
tailadmin_gray_700 = "#344054"
tailadmin_success_500 = "#12b76a"
tailadmin_warning_500 = "#f79009"
tailadmin_error_500 = "#f04438"
        """,
            language="toml",
        )

    with tab5:
        st.markdown("### Enhanced Best Practices")
        st.markdown("""
        #### 🚀 TailAdmin + Components Best Practices

        1. **Use st.components.v1.html for Interactivity**
           - Better isolation of HTML/CSS/JavaScript
           - Enable click handlers and user interactions
           - Message passing between component and Streamlit

        2. **Component Height Management**
           - Set appropriate heights for components
           - Use scrolling=True for large content
           - Cap maximum heights to prevent layout issues

        3. **JavaScript Messaging**
           - Use window.parent.postMessage() to communicate with Streamlit
           - Listen for messages in main app
           - Pass structured data objects

        4. **Performance Considerations**
           - Inject CSS only once using components
           - Cache expensive component generation
           - Minimize component re-renders

        5. **Responsive Design with Components**
           - Use CSS media queries in component HTML
           - Test interactive elements on mobile
           - Ensure touch-friendly button sizes

        6. **Security and Safety**
           - Sanitize any user inputs in component HTML
           - Use HTTPS for external resources
           - Enable CORS settings carefully

        7. **Theme Integration**
           - Configure .streamlit/config.toml with TailAdmin colors
           - Ensure consistent color usage across components
           - Test with both light and dark mode support
        """)

    # ========================================================================
    # STEP 8: USAGE SUMMARY
    # ========================================================================

    st.markdown("## 🚀 Enhanced Quick Start Summary")

    summary_card = f"""
    <div class="tailadmin-card" style="
        background: {TAILADMIN_COLORS["brand"]["50"]};
        border-color: {TAILADMIN_COLORS["brand"]["200"]};
        cursor: pointer;
        transition: all 0.3s ease;
    " onclick="summaryClicked()"
       onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 12px 16px -4px rgba(16, 24, 40, 0.08)'"
       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0px 1px 3px 0px rgba(16, 24, 40, 0.1)'">
        <h3 style="color: {TAILADMIN_COLORS["brand"]["800"]}; margin-top: 0;">
            🎉 Enhanced TailAdmin Integration (Click me!)
        </h3>
        <ol style="color: {TAILADMIN_COLORS["brand"]["700"]};">
            <li><strong>Configure theme:</strong> Set up <code>.streamlit/config.toml</code> with TailAdmin colors</li>
            <li><strong>Import components:</strong> <code>import streamlit.components.v1 as components</code></li>
            <li><strong>Initialize styling:</strong> Call <code>initialize_tailadmin_app()</code></li>
            <li><strong>Use interactive components:</strong> Leverage <code>components.html()</code> for enhanced UX</li>
            <li><strong>Add JavaScript handlers:</strong> Enable user interactions and messaging</li>
            <li><strong>Customize as needed:</strong> Extend with your own interactive components</li>
        </ol>
        <p style="color: {TAILADMIN_COLORS["brand"]["600"]}; margin-bottom: 0;">
            This enhanced approach provides professional, interactive dashboard interfaces that fully match
            TailAdmin's design language while adding modern web functionality to Streamlit apps.
        </p>
    </div>

    <script>
        function summaryClicked() {{
            alert('🎉 Great! You\\'re ready to build amazing TailAdmin-styled Streamlit apps!');
            window.parent.postMessage({{type: 'summaryClicked', message: 'User completed guide'}}, '*');
        }}
    </script>
    """

    components.html(summary_card, height=280)

    # Finalize app
    finalize_tailadmin_app()


if __name__ == "__main__":
    main()
