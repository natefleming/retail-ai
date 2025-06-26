"""Implementation Guide for TailAdmin in Streamlit Apps."""

import streamlit as st
import streamlit.components.v1 as components

from components.tailadmin import (
    TAILADMIN_COLORS,
    create_tailadmin_button,
    create_tailadmin_card,
    create_tailadmin_metric_card,
    create_tailadmin_progress_bar,
    create_tailadmin_stat_widget,
    get_tailadmin_color,
    inject_tailadmin_css,
)


def show_implementation_guide():
    """Show the TailAdmin implementation guide."""

    inject_tailadmin_css()

    st.markdown("""
    # 🎨 TailAdmin Implementation Guide

    ## Complete guide to implementing TailAdmin design system in Streamlit applications

    This guide provides you with everything you need to create beautiful, modern Streamlit apps using the TailAdmin design system.
    """)

    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🏁 Getting Started", "🎨 Color System", "🧩 Components", "📐 Layout Patterns", "💡 Best Practices"]
    )

    with tab1:
        show_getting_started()

    with tab2:
        show_color_system()

    with tab3:
        show_components()

    with tab4:
        show_layout_patterns()

    with tab5:
        show_best_practices()


def show_getting_started():
    """Show getting started guide."""

    st.markdown("""
    ## 🏁 Getting Started with TailAdmin

    TailAdmin is a comprehensive design system that brings modern UI patterns to Streamlit applications.
    """)

    # Installation section
    install_card = create_tailadmin_card(
        title="📦 Installation",
        content="""
        <div style="margin-bottom: 1rem;">
            <h4 style="margin-bottom: 0.5rem; color: #374151;">1. Install Dependencies</h4>
            <pre style="background: #f3f4f6; padding: 1rem; border-radius: 0.5rem; overflow-x: auto;">
pip install streamlit plotly pandas
            </pre>
        </div>

        <div style="margin-bottom: 1rem;">
            <h4 style="margin-bottom: 0.5rem; color: #374151;">2. Import TailAdmin Components</h4>
            <pre style="background: #f3f4f6; padding: 1rem; border-radius: 0.5rem; overflow-x: auto;">
from components.tailadmin import (
    inject_tailadmin_css,
    create_tailadmin_card,
    create_tailadmin_metric_card,
    get_tailadmin_color
)
            </pre>
        </div>

        <div>
            <h4 style="margin-bottom: 0.5rem; color: #374151;">3. Initialize TailAdmin Styles</h4>
            <pre style="background: #f3f4f6; padding: 1rem; border-radius: 0.5rem; overflow-x: auto;">
# Add this at the start of your Streamlit app
inject_tailadmin_css()
            </pre>
        </div>
        """,
    )

    components.html(install_card, height=400)

    # Quick start example
    st.markdown("### 🚀 Quick Start Example")

    quick_start_code = """
import streamlit as st
from components.tailadmin import inject_tailadmin_css, create_tailadmin_metric_card

def main():
    st.set_page_config(page_title="My TailAdmin App", layout="wide")

    # Initialize TailAdmin styles
    inject_tailadmin_css()

    # Create a beautiful metric card
    metric_html = create_tailadmin_metric_card(
        icon="💰",
        value="$45.2K",
        label="Monthly Revenue",
        change="12.5%",
        change_type="positive"
    )

    st.components.v1.html(metric_html, height=200)

if __name__ == "__main__":
    main()
    """

    st.code(quick_start_code, language="python")

    # Architecture overview
    st.markdown("### 🏗️ Architecture Overview")

    architecture_card = create_tailadmin_card(
        title="System Architecture",
        content="""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div style="padding: 1rem; background: #f8fafc; border-radius: 0.5rem; border-left: 4px solid #3b82f6;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1e40af;">Core Components</h4>
                <ul style="margin: 0; padding-left: 1.25rem; color: #64748b;">
                    <li>Color System</li>
                    <li>Typography Scale</li>
                    <li>Shadow System</li>
                    <li>Component Library</li>
                </ul>
            </div>
            <div style="padding: 1rem; background: #f0fdf4; border-radius: 0.5rem; border-left: 4px solid #22c55e;">
                <h4 style="margin: 0 0 0.5rem 0; color: #16a34a;">Utilities</h4>
                <ul style="margin: 0; padding-left: 1.25rem; color: #64748b;">
                    <li>CSS Injection</li>
                    <li>HTML Generators</li>
                    <li>Helper Functions</li>
                    <li>Layout Tools</li>
                </ul>
            </div>
        </div>
        """,
    )

    components.html(architecture_card, height=200)


def show_color_system():
    """Show the color system documentation."""

    st.markdown("""
    ## 🎨 TailAdmin Color System

    TailAdmin provides a comprehensive color palette designed for modern web applications.
    """)

    # Color categories
    color_categories = [
        ("Brand Colors", "brand", "Primary brand identity colors"),
        ("Gray Scale", "gray", "Neutral colors for text and backgrounds"),
        ("Success Colors", "success", "For positive actions and states"),
        ("Warning Colors", "warning", "For caution and attention"),
        ("Error Colors", "error", "For errors and destructive actions"),
    ]

    for name, category, description in color_categories:
        st.markdown(f"### {name}")
        st.caption(description)

        # Create color swatches
        cols = st.columns(len(TAILADMIN_COLORS[category]))

        for i, (shade, color) in enumerate(TAILADMIN_COLORS[category].items()):
            if shade in ["white", "black", "dark"]:
                continue

            with cols[i % len(cols)]:
                swatch_html = f"""
                <div style="
                    background: {color};
                    height: 60px;
                    border-radius: 0.5rem;
                    margin-bottom: 0.5rem;
                    border: 1px solid #e5e7eb;
                    display: flex;
                    align-items: end;
                    justify-content: center;
                    padding: 0.5rem;
                ">
                    <span style="
                        background: rgba(255,255,255,0.9);
                        padding: 0.25rem 0.5rem;
                        border-radius: 0.25rem;
                        font-size: 0.75rem;
                        font-weight: 600;
                        color: #1f2937;
                    ">{shade}</span>
                </div>
                <div style="text-align: center;">
                    <code style="font-size: 0.75rem;">{color}</code>
                </div>
                """
                components.html(swatch_html, height=100)

        st.divider()

    # Usage examples
    st.markdown("### 🛠️ Using Colors in Your App")

    color_usage_code = '''
from components.tailadmin import get_tailadmin_color, TAILADMIN_COLORS

# Get a specific color
primary_color = get_tailadmin_color("brand", "500")  # Returns #465fff
success_color = get_tailadmin_color("success")       # Returns #12b76a (default shade)

# Use colors in HTML
st.markdown(f"""
<div style="
    background: {get_tailadmin_color("brand", "50")};
    border: 1px solid {get_tailadmin_color("brand", "200")};
    color: {get_tailadmin_color("brand", "700")};
    padding: 1rem;
    border-radius: 0.5rem;
">
    This is a branded info box!
</div>
""", unsafe_allow_html=True)

# Access color directly
brand_colors = TAILADMIN_COLORS["brand"]
white = TAILADMIN_COLORS["white"]
    '''

    st.code(color_usage_code, language="python")


def show_components():
    """Show component examples and documentation."""

    st.markdown("""
    ## 🧩 TailAdmin Components

    Pre-built components that maintain design consistency across your application.
    """)

    # Metric Cards
    st.markdown("### 📊 Metric Cards")
    st.caption("Display key performance indicators with trend indicators")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card_html = create_tailadmin_metric_card(
            icon="💰", value="$45.2K", label="Monthly Revenue", change="12.5%", change_type="positive"
        )
        components.html(metric_card_html, height=180)

    with col2:
        metric_card_html = create_tailadmin_metric_card(
            icon="👥", value="2,847", label="Active Users", change="8.3%", change_type="positive"
        )
        components.html(metric_card_html, height=180)

    with col3:
        metric_card_html = create_tailadmin_metric_card(
            icon="📦", value="156", label="Inventory Items", change="2.1%", change_type="negative"
        )
        components.html(metric_card_html, height=180)

    # Show code example
    metric_code = """
from components.tailadmin import create_tailadmin_metric_card
import streamlit.components.v1 as components

metric_html = create_tailadmin_metric_card(
    icon="💰",
    value="$45.2K",
    label="Monthly Revenue",
    change="12.5%",
    change_type="positive"
)

components.html(metric_html, height=180)
    """

    st.code(metric_code, language="python")

    st.divider()

    # Cards
    st.markdown("### 🃏 Cards")
    st.caption("Flexible containers for grouping related content")

    col1, col2 = st.columns(2)

    with col1:
        simple_card = create_tailadmin_card(
            title="Simple Card",
            content="""
            <p style="margin: 0; color: #6b7280; line-height: 1.5;">
                This is a simple card with a title and content. Perfect for displaying
                information in an organized, visually appealing way.
            </p>
            """,
        )
        components.html(simple_card, height=150)

    with col2:
        card_with_actions = create_tailadmin_card(
            title="Card with Actions",
            content="""
            <p style="margin: 0 0 1rem 0; color: #6b7280; line-height: 1.5;">
                This card includes action buttons in the header area.
            </p>
            <button style="
                background: #3b82f6;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 0.375rem;
                font-size: 0.875rem;
                cursor: pointer;
            ">Action Button</button>
            """,
            actions="""
            <button style="
                background: #f3f4f6;
                color: #6b7280;
                border: 1px solid #e5e7eb;
                padding: 0.25rem 0.75rem;
                border-radius: 0.375rem;
                font-size: 0.75rem;
                cursor: pointer;
            ">Options</button>
            """,
        )
        components.html(card_with_actions, height=150)

    # Card code example
    card_code = """
from components.tailadmin import create_tailadmin_card
import streamlit.components.v1 as components

card_html = create_tailadmin_card(
    title="My Card Title",
    content="<p>Your card content goes here.</p>",
    actions="<button>Action</button>"  # Optional
)

components.html(card_html, height=200)
    """

    st.code(card_code, language="python")


def show_layout_patterns():
    """Show layout pattern examples."""

    st.markdown("""
    ## 📐 Layout Patterns

    Common layout patterns and best practices for organizing content.
    """)

    # Dashboard Layout
    st.markdown("### 📊 Dashboard Layout")
    st.caption("Typical layout for executive dashboards")

    # Simulate dashboard layout
    header_html = f"""
    <div style="
        background: linear-gradient(135deg, {get_tailadmin_color("brand", "600")} 0%, {get_tailadmin_color("brand", "700")} 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
    ">
        <h2 style="margin: 0 0 0.5rem 0; font-size: 1.75rem; font-weight: 700;">
            Dashboard Header
        </h2>
        <p style="margin: 0; opacity: 0.9;">
            Welcome message and key information
        </p>
    </div>
    """

    components.html(header_html, height=140)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        {"icon": "💰", "value": "$45.2K", "label": "Revenue"},
        {"icon": "👥", "value": "2,847", "label": "Users"},
        {"icon": "📦", "value": "156", "label": "Products"},
        {"icon": "⭐", "value": "4.8", "label": "Rating"},
    ]

    for i, metric in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            metric_html = create_tailadmin_metric_card(
                icon=metric["icon"], value=metric["value"], label=metric["label"], change="5.2%", change_type="positive"
            )
            components.html(metric_html, height=160)

    # Content area
    col1, col2 = st.columns([2, 1])

    with col1:
        chart_card = create_tailadmin_card(
            title="📈 Performance Chart",
            content="""
            <div style="
                height: 200px;
                background: #f8fafc;
                border-radius: 0.5rem;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #6b7280;
                font-style: italic;
            ">
                Chart visualization would go here
            </div>
            """,
        )
        components.html(chart_card, height=300)

    with col2:
        stats_card = create_tailadmin_card(
            title="📊 Quick Stats",
            content="""
            <div style="display: flex; flex-direction: column; gap: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #e5e7eb;">
                    <span style="color: #6b7280;">Active Sessions</span>
                    <strong style="color: #1f2937;">1,247</strong>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #e5e7eb;">
                    <span style="color: #6b7280;">Conversion Rate</span>
                    <strong style="color: #1f2937;">3.2%</strong>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;">
                    <span style="color: #6b7280;">Avg. Order Value</span>
                    <strong style="color: #1f2937;">$86.50</strong>
                </div>
            </div>
            """,
        )
        components.html(stats_card, height=300)

    # Layout code example
    layout_code = """
# Dashboard Layout Pattern
import streamlit as st
from components.tailadmin import inject_tailadmin_css, create_tailadmin_card

def create_dashboard():
    inject_tailadmin_css()

    # Header section
    st.markdown("# Dashboard Title")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    # Add metrics to each column...

    # Content area
    col1, col2 = st.columns([2, 1])
    with col1:
        # Main chart
        pass
    with col2:
        # Sidebar stats
        pass
    """

    st.code(layout_code, language="python")


def show_best_practices():
    """Show best practices and tips."""

    st.markdown("""
    ## 💡 Best Practices

    Guidelines for creating effective TailAdmin applications.
    """)

    # Best practices sections
    practices = [
        {
            "title": "🎨 Design Consistency",
            "points": [
                "Use the defined color palette consistently",
                "Maintain consistent spacing and typography",
                "Apply consistent border radius and shadows",
                "Use icons consistently across similar functions",
            ],
            "color": get_tailadmin_color("brand"),
        },
        {
            "title": "📱 Responsive Design",
            "points": [
                "Test layouts on different screen sizes",
                "Use flexible column layouts",
                "Ensure touch targets are appropriate size",
                "Consider mobile-first design approach",
            ],
            "color": get_tailadmin_color("success"),
        },
        {
            "title": "⚡ Performance",
            "points": [
                "Inject CSS only once per app",
                "Use appropriate component heights",
                "Minimize custom HTML when Streamlit components suffice",
                "Cache data and expensive computations",
            ],
            "color": get_tailadmin_color("warning"),
        },
        {
            "title": "♿ Accessibility",
            "points": [
                "Ensure sufficient color contrast",
                "Use semantic HTML structure",
                "Provide alt text for visual elements",
                "Make interactive elements keyboard accessible",
            ],
            "color": get_tailadmin_color("error"),
        },
    ]

    col1, col2 = st.columns(2)

    for i, practice in enumerate(practices):
        with col1 if i % 2 == 0 else col2:
            practice_card = create_tailadmin_card(
                title=practice["title"],
                content=f"""
                <ul style="margin: 0; padding-left: 1.25rem; color: #6b7280; line-height: 1.6;">
                    {"".join([f"<li>{point}</li>" for point in practice["points"]])}
                </ul>
                """,
                card_class="practice-card",
            )

            # Add border color styling
            styled_card = practice_card.replace(
                'class="tailadmin-card practice-card"',
                f'class="tailadmin-card practice-card" style="border-left: 4px solid {practice["color"]};"',
            )

            components.html(styled_card, height=200)

    # Common mistakes section
    st.markdown("### ⚠️ Common Mistakes to Avoid")

    mistakes_html = f"""
    <div style="
        background: {get_tailadmin_color("error", "50")};
        border: 1px solid {get_tailadmin_color("error", "200")};
        border-left: 4px solid {get_tailadmin_color("error")};
        border-radius: 0.5rem;
        padding: 1.5rem;
    ">
        <h4 style="margin: 0 0 1rem 0; color: {get_tailadmin_color("error", "700")};">
            ❌ Things to Avoid
        </h4>
        <ul style="margin: 0; padding-left: 1.25rem; color: {get_tailadmin_color("error", "700")}; line-height: 1.6;">
            <li>Mixing TailAdmin colors with custom colors inconsistently</li>
            <li>Overusing custom HTML when Streamlit components work</li>
            <li>Ignoring mobile responsiveness</li>
            <li>Creating overly complex component hierarchies</li>
            <li>Not testing with different amounts of data</li>
        </ul>
    </div>
    """

    components.html(mistakes_html, height=180)

    # Final tips
    st.markdown("### 🎯 Pro Tips")

    tips_html = f"""
    <div style="
        background: {get_tailadmin_color("success", "50")};
        border: 1px solid {get_tailadmin_color("success", "200")};
        border-left: 4px solid {get_tailadmin_color("success")};
        border-radius: 0.5rem;
        padding: 1.5rem;
    ">
        <h4 style="margin: 0 0 1rem 0; color: {get_tailadmin_color("success", "700")};">
            ✅ Pro Tips for Success
        </h4>
        <ul style="margin: 0; padding-left: 1.25rem; color: {get_tailadmin_color("success", "700")}; line-height: 1.6;">
            <li>Start with a design mockup or wireframe</li>
            <li>Use the component library as building blocks</li>
            <li>Test your app with real data and edge cases</li>
            <li>Get feedback from actual users early and often</li>
            <li>Keep components modular and reusable</li>
        </ul>
    </div>
    """

    components.html(tips_html, height=180)
