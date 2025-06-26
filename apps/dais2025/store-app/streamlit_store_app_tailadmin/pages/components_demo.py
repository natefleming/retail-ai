"""Components Demo page showcasing all TailAdmin components."""

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


def show_components_demo():
    """Show all TailAdmin components with interactive examples."""

    inject_tailadmin_css()

    st.markdown("""
    # 📋 TailAdmin Components Demo

    ## Interactive showcase of all available TailAdmin components

    This page demonstrates all TailAdmin components with live examples and customization options.
    """)

    # Component sections
    demo_sections = [
        ("📊 Metric Cards", demo_metric_cards),
        ("🃏 Cards", demo_cards),
        ("🔘 Buttons", demo_buttons),
        ("📊 Progress Bars", demo_progress_bars),
        ("📈 Stat Widgets", demo_stat_widgets),
        ("🎨 Color Utilities", demo_colors),
        ("📝 Typography", demo_typography),
        ("🚨 Alerts & Badges", demo_alerts_badges),
    ]

    for title, demo_func in demo_sections:
        st.markdown(f"## {title}")
        demo_func()
        st.divider()


def demo_metric_cards():
    """Demo metric cards with customization options."""

    st.markdown("### Interactive Metric Card Builder")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Customize Your Card")

        icon = st.selectbox("Icon", ["💰", "👥", "📦", "⭐", "📈", "📉", "🚀", "🎯", "💡", "🔥"], index=0)

        value = st.text_input("Value", "45.2K")
        label = st.text_input("Label", "Monthly Revenue")

        change = st.number_input("Change (%)", value=12.5, step=0.1)
        change_type = st.selectbox("Change Type", ["positive", "negative"])

        if st.button("Generate Card"):
            st.session_state.custom_metric = {
                "icon": icon,
                "value": value,
                "label": label,
                "change": f"{change}%",
                "change_type": change_type,
            }

    with col2:
        st.markdown("#### Preview")

        if "custom_metric" in st.session_state:
            metric = st.session_state.custom_metric
        else:
            metric = {
                "icon": "💰",
                "value": "45.2K",
                "label": "Monthly Revenue",
                "change": "12.5%",
                "change_type": "positive",
            }

        metric_html = create_tailadmin_metric_card(
            icon=metric["icon"],
            value=metric["value"],
            label=metric["label"],
            change=metric["change"],
            change_type=metric["change_type"],
        )
        components.html(metric_html, height=200)

    # Pre-built examples
    st.markdown("### Pre-built Examples")

    col1, col2, col3, col4 = st.columns(4)

    examples = [
        {"icon": "💰", "value": "$45.2K", "label": "Revenue", "change": "12.5%", "change_type": "positive"},
        {"icon": "👥", "value": "2,847", "label": "Users", "change": "8.3%", "change_type": "positive"},
        {"icon": "📦", "value": "156", "label": "Products", "change": "2.1%", "change_type": "negative"},
        {"icon": "⭐", "value": "4.8", "label": "Rating", "change": "0.2", "change_type": "positive"},
    ]

    for i, example in enumerate(examples):
        with [col1, col2, col3, col4][i]:
            metric_html = create_tailadmin_metric_card(**example)
            components.html(metric_html, height=180)


def demo_cards():
    """Demo card components."""

    st.markdown("### Card Variations")

    col1, col2 = st.columns(2)

    with col1:
        # Simple card
        simple_card = create_tailadmin_card(
            title="Simple Card",
            content="""
            <p style="margin: 0; color: #6b7280; line-height: 1.5;">
                This is a basic card with title and content. Perfect for displaying
                information in a clean, organized way.
            </p>
            """,
        )
        components.html(simple_card, height=150)

        # Card with list content
        list_card = create_tailadmin_card(
            title="📋 Task List",
            content="""
            <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #22c55e;">✅</span>
                    <span style="color: #6b7280;">Complete project setup</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #22c55e;">✅</span>
                    <span style="color: #6b7280;">Design components</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #f59e0b;">⏳</span>
                    <span style="color: #6b7280;">Test functionality</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #9ca3af;">⭕</span>
                    <span style="color: #9ca3af;">Deploy to production</span>
                </div>
            </div>
            """,
        )
        components.html(list_card, height=200)

    with col2:
        # Card with actions
        action_card = create_tailadmin_card(
            title="Card with Actions",
            content="""
            <p style="margin: 0 0 1rem 0; color: #6b7280; line-height: 1.5;">
                This card includes action buttons in the header and footer areas.
            </p>
            <div style="display: flex; gap: 0.5rem;">
                <button style="
                    background: #3b82f6;
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 0.375rem;
                    font-size: 0.875rem;
                    cursor: pointer;
                ">Primary Action</button>
                <button style="
                    background: transparent;
                    color: #6b7280;
                    border: 1px solid #e5e7eb;
                    padding: 0.5rem 1rem;
                    border-radius: 0.375rem;
                    font-size: 0.875rem;
                    cursor: pointer;
                ">Secondary</button>
            </div>
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
            ">⚙️ Options</button>
            """,
        )
        components.html(action_card, height=200)

        # Status card
        status_card = create_tailadmin_card(
            title="🚦 System Status",
            content="""
            <div style="display: flex; flex-direction: column; gap: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #6b7280;">Database</span>
                    <span style="color: #22c55e; font-weight: 600;">🟢 Operational</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #6b7280;">API Server</span>
                    <span style="color: #22c55e; font-weight: 600;">🟢 Operational</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #6b7280;">CDN</span>
                    <span style="color: #f59e0b; font-weight: 600;">🟡 Degraded</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #6b7280;">Monitoring</span>
                    <span style="color: #22c55e; font-weight: 600;">🟢 Operational</span>
                </div>
            </div>
            """,
        )
        components.html(status_card, height=200)


def demo_buttons():
    """Demo button components."""

    st.markdown("### Button Styles")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Basic Buttons")

        button_types = ["primary", "secondary", "success", "warning", "error"]

        for btn_type in button_types:
            btn_html = create_tailadmin_button(text=f"{btn_type.title()} Button", button_type=btn_type, size="medium")
            components.html(btn_html, height=60)

    with col2:
        st.markdown("#### Buttons with Icons")

        icon_buttons = [
            ("Download", "📥", "primary"),
            ("Upload", "📤", "secondary"),
            ("Save", "💾", "success"),
            ("Edit", "✏️", "warning"),
            ("Delete", "🗑️", "error"),
        ]

        for text, icon, btn_type in icon_buttons:
            btn_html = create_tailadmin_button(text=text, button_type=btn_type, icon=icon, size="medium")
            components.html(btn_html, height=60)

    # Size demonstration
    st.markdown("#### Button Sizes")

    for size in ["small", "medium", "large"]:
        btn_html = create_tailadmin_button(text=f"{size.title()} Button", button_type="primary", size=size)
        components.html(btn_html, height=70)


def demo_progress_bars():
    """Demo progress bar components."""

    st.markdown("### Progress Bars")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Different Progress Levels")

        progress_examples = [
            (25, "Project Setup", "brand"),
            (60, "Development", "success"),
            (85, "Testing", "warning"),
            (95, "Deployment", "error"),
        ]

        for percentage, label, color in progress_examples:
            progress_html = create_tailadmin_progress_bar(
                percentage=percentage, label=label, color=color, show_percentage=True
            )
            components.html(progress_html, height=80)

    with col2:
        st.markdown("#### Interactive Progress")

        progress_value = st.slider("Progress Value", 0, 100, 65)
        progress_label = st.text_input("Progress Label", "Current Progress")
        progress_color = st.selectbox("Color Theme", ["brand", "success", "warning", "error"])
        show_percentage = st.checkbox("Show Percentage", True)

        progress_html = create_tailadmin_progress_bar(
            percentage=progress_value, label=progress_label, color=progress_color, show_percentage=show_percentage
        )
        components.html(progress_html, height=80)


def demo_stat_widgets():
    """Demo stat widget components."""

    st.markdown("### Statistics Widgets")

    col1, col2 = st.columns(2)

    with col1:
        # Sales stats widget
        sales_stats = [
            {"label": "Today's Sales", "value": "$12,450", "change": "8.2%", "change_type": "positive"},
            {"label": "This Week", "value": "$76,200", "change": "12.5%", "change_type": "positive"},
            {"label": "This Month", "value": "$324,150", "change": "5.3%", "change_type": "negative"},
            {"label": "This Quarter", "value": "$1.2M", "change": "18.7%", "change_type": "positive"},
        ]

        sales_widget = create_tailadmin_stat_widget(title="💰 Sales Performance", stats=sales_stats)
        components.html(sales_widget, height=350)

    with col2:
        # User engagement stats
        engagement_stats = [
            {"label": "Active Users", "value": "2,847", "change": "156 new", "change_type": "positive"},
            {"label": "Session Duration", "value": "4m 32s", "change": "12s", "change_type": "positive"},
            {"label": "Bounce Rate", "value": "32.4%", "change": "2.1%", "change_type": "negative"},
            {"label": "Conversion Rate", "value": "3.8%", "change": "0.5%", "change_type": "positive"},
        ]

        engagement_widget = create_tailadmin_stat_widget(title="👥 User Engagement", stats=engagement_stats)
        components.html(engagement_widget, height=350)


def demo_colors():
    """Demo color utilities."""

    st.markdown("### Color System Utilities")

    # Color picker demo
    st.markdown("#### Interactive Color Picker")

    col1, col2, col3 = st.columns(3)

    with col1:
        color_category = st.selectbox("Color Category", ["brand", "gray", "success", "warning", "error"])

    with col2:
        available_shades = [k for k in TAILADMIN_COLORS[color_category].keys() if k not in ["white", "black", "dark"]]
        color_shade = st.selectbox("Color Shade", available_shades, index=4)  # Default to middle shade

    with col3:
        selected_color = get_tailadmin_color(color_category, color_shade)
        st.color_picker("Selected Color", selected_color, disabled=True)

    # Usage example
    st.markdown("#### Usage Example")

    usage_html = f"""
    <div style="
        background: {get_tailadmin_color(color_category, "50")};
        border: 2px solid {selected_color};
        color: {get_tailadmin_color(color_category, "700")};
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    ">
        <h4 style="margin: 0 0 0.5rem 0;">
            Selected Color: {color_category.title()} {color_shade.title()}
        </h4>
        <p style="margin: 0; font-family: monospace;">
            Color Code: {selected_color}
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.875rem;">
            This demonstrates how the selected color looks in practice with proper contrast.
        </p>
    </div>
    """

    components.html(usage_html, height=140)

    # Color palette overview
    st.markdown("#### Full Color Palette")

    for category_name, colors in TAILADMIN_COLORS.items():
        if category_name in ["white", "black"]:
            continue

        st.markdown(f"**{category_name.title()} Colors**")

        palette_html = """
        <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
        """

        for shade, color in colors.items():
            if shade in ["white", "black", "dark"]:
                continue

            palette_html += f"""
            <div style="
                background: {color};
                width: 60px;
                height: 60px;
                border-radius: 0.5rem;
                border: 1px solid #e5e7eb;
                display: flex;
                align-items: end;
                justify-content: center;
                padding: 0.25rem;
                position: relative;
            ">
                <span style="
                    background: rgba(255,255,255,0.9);
                    padding: 0.125rem 0.25rem;
                    border-radius: 0.25rem;
                    font-size: 0.625rem;
                    font-weight: 600;
                    color: #1f2937;
                ">{shade}</span>
            </div>
            """

        palette_html += "</div>"
        components.html(palette_html, height=80)


def demo_typography():
    """Demo typography styles."""

    st.markdown("### Typography System")

    col1, col2 = st.columns(2)

    with col1:
        typography_html = f"""
        <div style="font-family: 'Outfit', sans-serif;">
            <h1 style="font-size: 2.25rem; font-weight: 800; color: {get_tailadmin_color("gray", "900")}; margin: 0 0 1rem 0;">
                Display Large
            </h1>
            <h2 style="font-size: 1.875rem; font-weight: 700; color: {get_tailadmin_color("gray", "800")}; margin: 0 0 1rem 0;">
                Heading 1
            </h2>
            <h3 style="font-size: 1.5rem; font-weight: 600; color: {get_tailadmin_color("gray", "700")}; margin: 0 0 1rem 0;">
                Heading 2
            </h3>
            <h4 style="font-size: 1.25rem; font-weight: 600; color: {get_tailadmin_color("gray", "700")}; margin: 0 0 1rem 0;">
                Heading 3
            </h4>
            <p style="font-size: 1rem; color: {get_tailadmin_color("gray", "600")}; margin: 0 0 1rem 0; line-height: 1.5;">
                Body text with proper line height and spacing for optimal readability.
            </p>
            <p style="font-size: 0.875rem; color: {get_tailadmin_color("gray", "500")}; margin: 0;">
                Small text for captions and secondary information.
            </p>
        </div>
        """
        components.html(typography_html, height=300)

    with col2:
        font_weights_html = f"""
        <div style="font-family: 'Outfit', sans-serif;">
            <h3 style="margin: 0 0 1rem 0; color: {get_tailadmin_color("gray", "700")};">Font Weights</h3>
            <p style="font-weight: 100; margin: 0 0 0.5rem 0; color: {get_tailadmin_color("gray", "600")};">
                Thin (100)
            </p>
            <p style="font-weight: 300; margin: 0 0 0.5rem 0; color: {get_tailadmin_color("gray", "600")};">
                Light (300)
            </p>
            <p style="font-weight: 400; margin: 0 0 0.5rem 0; color: {get_tailadmin_color("gray", "600")};">
                Regular (400)
            </p>
            <p style="font-weight: 500; margin: 0 0 0.5rem 0; color: {get_tailadmin_color("gray", "600")};">
                Medium (500)
            </p>
            <p style="font-weight: 600; margin: 0 0 0.5rem 0; color: {get_tailadmin_color("gray", "600")};">
                Semi-bold (600)
            </p>
            <p style="font-weight: 700; margin: 0 0 0.5rem 0; color: {get_tailadmin_color("gray", "600")};">
                Bold (700)
            </p>
            <p style="font-weight: 800; margin: 0 0 0.5rem 0; color: {get_tailadmin_color("gray", "600")};">
                Extra-bold (800)
            </p>
            <p style="font-weight: 900; margin: 0; color: {get_tailadmin_color("gray", "600")};">
                Black (900)
            </p>
        </div>
        """
        components.html(font_weights_html, height=300)


def demo_alerts_badges():
    """Demo alert and badge components."""

    st.markdown("### Alerts & Badges")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Alert Messages")

        alerts = [
            ("success", "✅ Operation completed successfully!"),
            ("warning", "⚠️ Please review your input before proceeding."),
            ("error", "❌ An error occurred. Please try again."),
            ("info", "ℹ️ New features are now available."),
        ]

        for alert_type, message in alerts:
            color = get_tailadmin_color(alert_type)
            bg_color = get_tailadmin_color(alert_type, "50")

            alert_html = f"""
            <div style="
                background: {bg_color};
                border: 1px solid {get_tailadmin_color(alert_type, "200")};
                border-left: 4px solid {color};
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                color: {get_tailadmin_color(alert_type, "700")};
            ">
                {message}
            </div>
            """
            components.html(alert_html, height=70)

    with col2:
        st.markdown("#### Status Badges")

        badges = [
            ("Active", "success"),
            ("Pending", "warning"),
            ("Inactive", "error"),
            ("Draft", "gray"),
            ("Premium", "brand"),
        ]

        badges_html = """
        <div style="display: flex; flex-direction: column; gap: 1rem;">
        """

        for label, badge_type in badges:
            color = get_tailadmin_color(badge_type, "700")
            bg_color = get_tailadmin_color(badge_type, "100")

            badges_html += f"""
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="color: {get_tailadmin_color("gray", "600")}; width: 80px;">
                    Status:
                </span>
                <span style="
                    background: {bg_color};
                    color: {color};
                    padding: 0.25rem 0.75rem;
                    border-radius: 9999px;
                    font-size: 0.875rem;
                    font-weight: 500;
                    text-transform: uppercase;
                    letter-spacing: 0.025em;
                ">
                    {label}
                </span>
            </div>
            """

        badges_html += "</div>"
        components.html(badges_html, height=200)
