"""
Enhanced TailAdmin Components for Streamlit

This module demonstrates how to apply TailAdmin's visual styles to Streamlit components,
using both custom HTML/CSS and styled Streamlit widgets.

Features:
- Direct HTML injection with TailAdmin styling
- Custom wrappers for Streamlit widgets
- Responsive design patterns
- Interactive components with TailAdmin aesthetics
- Reusable component library
"""

from typing import Dict, List, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from .tailadmin_styles import (
    TAILADMIN_COLORS,
    create_tailadmin_button,
    create_tailadmin_card,
    create_tailadmin_metric_card,
    create_tailadmin_progress_bar,
    inject_tailadmin_css,
)

# ============================================================================
# INITIALIZATION AND SETUP
# ============================================================================


def initialize_tailadmin_app():
    """
    Initialize a Streamlit app with TailAdmin styling.
    Call this at the beginning of your Streamlit app.
    """
    # Configure page
    st.set_page_config(
        page_title="TailAdmin Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject TailAdmin CSS
    inject_tailadmin_css()

    # Add app wrapper using components
    components.html('<div class="tailadmin-app">', height=0)


# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================


def create_tailadmin_header(
    title: str,
    subtitle: Optional[str] = None,
    user_info: Optional[Dict] = None,
    search_enabled: bool = False,
    notifications: int = 0,
):
    """
    Create a TailAdmin-styled header with title, user info, and controls.

    Args:
        title: Main page title
        subtitle: Optional subtitle
        user_info: Dict with user info (name, avatar, role)
        search_enabled: Whether to show search bar
        notifications: Number of notifications
    """

    # User info section
    user_section = ""
    if user_info:
        avatar = user_info.get("avatar", "👤")
        name = user_info.get("name", "User")
        role = user_info.get("role", "")

        notification_badge = ""
        if notifications > 0:
            notification_badge = f"""
                <span style="
                    position: absolute;
                    top: -0.25rem;
                    right: -0.25rem;
                    background-color: {TAILADMIN_COLORS["error"]["500"]};
                    color: white;
                    border-radius: 9999px;
                    width: 1.25rem;
                    height: 1.25rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 0.75rem;
                    font-weight: 600;
                ">{notifications}</span>
            """

        user_section = f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 1rem;
            ">
                <div style="position: relative;">
                    <button onclick="showNotifications()" style="
                        background: {TAILADMIN_COLORS["gray"]["100"]};
                        border: none;
                        border-radius: 0.5rem;
                        padding: 0.5rem;
                        cursor: pointer;
                        font-size: 1.25rem;
                    ">🔔</button>
                    {notification_badge}
                </div>
                
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    padding: 0.5rem 1rem;
                    background: {TAILADMIN_COLORS["gray"]["50"]};
                    border-radius: 0.75rem;
                ">
                    <span style="font-size: 2rem;">{avatar}</span>
                    <div>
                        <div style="
                            font-weight: 600;
                            color: {TAILADMIN_COLORS["gray"]["900"]};
                            font-size: 0.875rem;
                        ">{name}</div>
                        <div style="
                            color: {TAILADMIN_COLORS["gray"]["500"]};
                            font-size: 0.75rem;
                        ">{role}</div>
                    </div>
                </div>
            </div>
        """

    # Search section with functionality
    search_section = ""
    if search_enabled:
        search_section = f"""
            <div style="
                flex: 1;
                max-width: 400px;
                position: relative;
            ">
                <input 
                    type="text" 
                    placeholder="Search..."
                    class="tailadmin-input"
                    style="padding-left: 2.5rem;"
                    oninput="handleSearch(this.value)"
                />
                <div style="
                    position: absolute;
                    left: 0.75rem;
                    top: 50%;
                    transform: translateY(-50%);
                    color: {TAILADMIN_COLORS["gray"]["400"]};
                    font-size: 1.25rem;
                ">🔍</div>
            </div>
        """

    # Subtitle section
    subtitle_html = ""
    if subtitle:
        subtitle_html = f"""
            <p style="
                color: {TAILADMIN_COLORS["gray"]["600"]};
                margin: 0.5rem 0 0 0;
                font-size: 1rem;
                font-weight: 400;
            ">{subtitle}</p>
        """

    header_html = f"""
    <div class="tailadmin-header" style="margin-bottom: 2rem;">
        <div style="flex: 1;">
            <h1 style="
                font-size: 2.25rem;
                font-weight: 800;
                color: {TAILADMIN_COLORS["gray"]["900"]};
                margin: 0;
                line-height: 1.2;
            ">{title}</h1>
            {subtitle_html}
        </div>
        
        <div style="
            display: flex;
            align-items: center;
            gap: 1rem;
            flex: none;
        ">
            {search_section}
            {user_section}
        </div>
    </div>
    
    <script>
        function showNotifications() {{
            alert('Notifications: You have {notifications} new notifications');
        }}
        
        function handleSearch(value) {{
            console.log('Searching for:', value);
            // Add search functionality here
        }}
    </script>
    """

    # Use components.html for interactive header
    components.html(header_html, height=120)


def create_tailadmin_sidebar(
    logo: Optional[str] = None,
    menu_items: Optional[List[Dict]] = None,
    user_profile: Optional[Dict] = None,
    collapsed: bool = False,
):
    """
    Create a TailAdmin-styled sidebar with navigation using components.

    Args:
        logo: Logo HTML/text
        menu_items: List of menu item dicts with keys: icon, label, url, active, children
        user_profile: User profile dict
        collapsed: Whether sidebar is collapsed
    """

    # Build menu items HTML
    menu_html = ""
    if menu_items:
        for i, item in enumerate(menu_items):
            icon = item.get("icon", "📁")
            label = item.get("label", "Menu Item")
            active = item.get("active", False)
            children = item.get("children", [])

            # Main menu item
            active_class = "active" if active else ""
            menu_html += f"""
                <div class="tailadmin-menu-item {active_class}" 
                     onclick="selectMenuItem({i})"
                     style="margin-bottom: 0.5rem; cursor: pointer;">
                    <span style="font-size: 1.25rem;">{icon}</span>
                    <span style="font-weight: 500;">{label}</span>
                </div>
            """

            # Sub-menu items
            if children and active:
                for j, child in enumerate(children):
                    child_icon = child.get("icon", "•")
                    child_label = child.get("label", "Sub Item")
                    child_active = child.get("active", False)

                    child_class = "active" if child_active else ""
                    menu_html += f"""
                        <div class="tailadmin-menu-item {child_class}" 
                             onclick="selectSubMenuItem({i}, {j})"
                             style="margin-left: 2rem; margin-bottom: 0.25rem; cursor: pointer;">
                            <span style="font-size: 0.875rem;">{child_icon}</span>
                            <span style="font-size: 0.875rem;">{child_label}</span>
                        </div>
                    """

    # Logo section
    logo_html = ""
    if logo:
        logo_html = f"""
            <div style="
                padding: 1rem 0 2rem 0;
                text-align: center;
                border-bottom: 1px solid {TAILADMIN_COLORS["gray"]["200"]};
                margin-bottom: 1.5rem;
            ">
                {logo}
            </div>
        """

    # User profile section
    profile_html = ""
    if user_profile:
        name = user_profile.get("name", "User")
        email = user_profile.get("email", "user@example.com")
        avatar = user_profile.get("avatar", "👤")

        profile_html = f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 1rem;
                background: {TAILADMIN_COLORS["gray"]["50"]};
                border-radius: 0.75rem;
                margin-top: 2rem;
                cursor: pointer;
            " onclick="showProfile()">
                <span style="font-size: 2.5rem;">{avatar}</span>
                <div style="flex: 1; min-width: 0;">
                    <div style="
                        font-weight: 600;
                        color: {TAILADMIN_COLORS["gray"]["900"]};
                        font-size: 0.875rem;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                    ">{name}</div>
                    <div style="
                        color: {TAILADMIN_COLORS["gray"]["500"]};
                        font-size: 0.75rem;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                    ">{email}</div>
                </div>
            </div>
        """

    width = "80px" if collapsed else "290px"

    sidebar_html = f"""
    <div class="tailadmin-sidebar" style="width: {width};">
        {logo_html}
        {menu_html}
        {profile_html}
    </div>
    
    <script>
        function selectMenuItem(index) {{
            console.log('Selected menu item:', index);
            // Add navigation logic here
            window.parent.postMessage({{type: 'menuSelected', index: index}}, '*');
        }}
        
        function selectSubMenuItem(parentIndex, childIndex) {{
            console.log('Selected sub menu item:', parentIndex, childIndex);
            // Add navigation logic here
            window.parent.postMessage({{type: 'subMenuSelected', parent: parentIndex, child: childIndex}}, '*');
        }}
        
        function showProfile() {{
            console.log('Show profile clicked');
            // Add profile logic here
            window.parent.postMessage({{type: 'showProfile'}}, '*');
        }}
    </script>
    """

    # Render sidebar in Streamlit sidebar
    with st.sidebar:
        components.html(sidebar_html, height=800, scrolling=True)


# ============================================================================
# ENHANCED METRIC COMPONENTS
# ============================================================================


def display_enhanced_metrics_grid(metrics: List[Dict], columns: int = 4):
    """
    Display an enhanced grid of TailAdmin metric cards with animations and interactions.

    Args:
        metrics: List of metric dicts with enhanced features
        columns: Number of columns in the grid
    """

    # Create responsive columns
    cols = st.columns(columns)

    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            icon = metric.get("icon", "📊")
            value = metric.get("value", "0")
            label = metric.get("label", "Metric")
            change = metric.get("change")
            change_type = metric.get("change_type", "positive")
            trend_data = metric.get("trend_data")
            target = metric.get("target")
            description = metric.get("description")

            # Enhanced metric card with additional features
            enhanced_content = create_tailadmin_metric_card(
                icon=icon,
                value=value,
                label=label,
                change=change,
                change_type=change_type,
                trend_data=trend_data,
            )

            # Add target progress if provided
            if target:
                progress_value = float(
                    str(value).replace("%", "").replace("$", "").replace(",", "")
                )
                target_value = float(
                    str(target).replace("%", "").replace("$", "").replace(",", "")
                )
                progress_percent = min((progress_value / target_value) * 100, 100)

                progress_html = create_tailadmin_progress_bar(
                    percentage=progress_percent,
                    label=f"Target: {target}",
                    color="brand" if change_type == "positive" else "warning",
                )

                enhanced_content = enhanced_content.replace(
                    "</div>",
                    f"<div style='margin-top: 1rem;'>{progress_html}</div></div>",
                    1,
                )

            # Add description if provided
            if description:
                desc_html = f"""
                <div style="
                    margin-top: 0.75rem;
                    padding-top: 0.75rem;
                    border-top: 1px solid {TAILADMIN_COLORS["gray"]["100"]};
                    font-size: 0.75rem;
                    color: {TAILADMIN_COLORS["gray"]["500"]};
                ">{description}</div>
                """
                enhanced_content = enhanced_content.replace(
                    "</div>", f"{desc_html}</div>", 1
                )

            # Add click functionality with components
            metric_html = f"""
            <div onclick="metricClicked('{label}')" style="cursor: pointer;">
                {enhanced_content}
            </div>
            
            <script>
                function metricClicked(label) {{
                    console.log('Metric clicked:', label);
                    window.parent.postMessage({{type: 'metricClicked', metric: label}}, '*');
                }}
            </script>
            """

            components.html(metric_html, height=200)


# ============================================================================
# CHART COMPONENTS WITH TAILADMIN STYLING
# ============================================================================


def create_tailadmin_plotly_chart(
    data: Union[pd.DataFrame, Dict],
    chart_type: str = "line",
    title: str = "Chart",
    subtitle: Optional[str] = None,
    height: int = 400,
    color_scheme: str = "brand",
    show_toolbar: bool = False,
    **kwargs,
) -> go.Figure:
    """
    Create a Plotly chart with TailAdmin styling.

    Args:
        data: Chart data
        chart_type: Chart type (line, bar, pie, scatter, area)
        title: Chart title
        subtitle: Chart subtitle
        height: Chart height
        color_scheme: Color scheme to use
        show_toolbar: Whether to show Plotly toolbar
        **kwargs: Additional Plotly parameters

    Returns:
        Styled Plotly figure
    """

    # Color palettes based on TailAdmin colors
    color_palettes = {
        "brand": [
            TAILADMIN_COLORS["brand"][shade]
            for shade in ["500", "400", "600", "300", "700"]
        ],
        "multi": [
            TAILADMIN_COLORS["brand"]["500"],
            TAILADMIN_COLORS["success"]["500"],
            TAILADMIN_COLORS["warning"]["500"],
            TAILADMIN_COLORS["error"]["500"],
            TAILADMIN_COLORS["gray"]["500"],
        ],
        "gradient": [
            TAILADMIN_COLORS["brand"][shade]
            for shade in ["200", "300", "400", "500", "600"]
        ],
    }

    colors = color_palettes.get(color_scheme, color_palettes["brand"])

    # Create figure based on chart type
    if chart_type == "line":
        if isinstance(data, pd.DataFrame):
            fig = px.line(
                data,
                x=data.columns[0],
                y=data.columns[1],
                color_discrete_sequence=colors,
                **kwargs,
            )
        else:
            fig = go.Figure(
                data=go.Scatter(
                    x=data.get("x", []),
                    y=data.get("y", []),
                    mode="lines",
                    line_color=colors[0],
                )
            )

    elif chart_type == "bar":
        if isinstance(data, pd.DataFrame):
            fig = px.bar(
                data,
                x=data.columns[0],
                y=data.columns[1],
                color_discrete_sequence=colors,
                **kwargs,
            )
        else:
            fig = go.Figure(
                data=go.Bar(
                    x=data.get("x", []), y=data.get("y", []), marker_color=colors[0]
                )
            )

    elif chart_type == "pie":
        if isinstance(data, pd.DataFrame):
            fig = px.pie(
                data,
                values=data.columns[1],
                names=data.columns[0],
                color_discrete_sequence=colors,
                **kwargs,
            )
        else:
            fig = go.Figure(
                data=go.Pie(
                    labels=data.get("labels", []),
                    values=data.get("values", []),
                    marker_colors=colors,
                )
            )

    elif chart_type == "scatter":
        if isinstance(data, pd.DataFrame):
            fig = px.scatter(
                data,
                x=data.columns[0],
                y=data.columns[1],
                color_discrete_sequence=colors,
                **kwargs,
            )
        else:
            fig = go.Figure(
                data=go.Scatter(
                    x=data.get("x", []),
                    y=data.get("y", []),
                    mode="markers",
                    marker_color=colors[0],
                )
            )

    elif chart_type == "area":
        if isinstance(data, pd.DataFrame):
            fig = px.area(
                data,
                x=data.columns[0],
                y=data.columns[1],
                color_discrete_sequence=colors,
                **kwargs,
            )
        else:
            fig = go.Figure(
                data=go.Scatter(
                    x=data.get("x", []),
                    y=data.get("y", []),
                    fill="tonexty",
                    fillcolor=colors[0],
                )
            )

    # Apply TailAdmin styling
    fig.update_layout(
        title={
            "text": title,
            "font": {
                "family": "Outfit, sans-serif",
                "size": 20,
                "color": TAILADMIN_COLORS["gray"]["900"],
            },
            "x": 0,
            "xanchor": "left",
        },
        height=height,
        margin=dict(l=0, r=0, t=60, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Outfit, sans-serif",
        font_color=TAILADMIN_COLORS["gray"]["600"],
        showlegend=chart_type == "pie",
    )

    # Style axes for non-pie charts
    if chart_type != "pie":
        fig.update_xaxes(
            gridcolor=TAILADMIN_COLORS["gray"]["100"],
            linecolor=TAILADMIN_COLORS["gray"]["200"],
            tickcolor=TAILADMIN_COLORS["gray"]["200"],
            title_font_color=TAILADMIN_COLORS["gray"]["700"],
        )
        fig.update_yaxes(
            gridcolor=TAILADMIN_COLORS["gray"]["100"],
            linecolor=TAILADMIN_COLORS["gray"]["200"],
            tickcolor=TAILADMIN_COLORS["gray"]["200"],
            title_font_color=TAILADMIN_COLORS["gray"]["700"],
        )

    return fig


def display_tailadmin_chart_card(
    fig: go.Figure,
    title: str,
    description: Optional[str] = None,
    actions: Optional[str] = None,
):
    """
    Display a Plotly chart wrapped in a TailAdmin card.

    Args:
        fig: Plotly figure
        title: Card title
        description: Optional description
        actions: Optional action buttons HTML
    """

    # Chart placeholder for content
    chart_placeholder = "CHART_PLACEHOLDER"

    # Description section
    desc_html = ""
    if description:
        desc_html = f"""
        <p style="
            color: {TAILADMIN_COLORS["gray"]["600"]};
            margin-bottom: 1.5rem;
            font-size: 0.875rem;
        ">{description}</p>
        """

    # Create card with chart placeholder
    card_html = create_tailadmin_card(
        content=f"{desc_html}{chart_placeholder}", title=title, actions=actions
    )

    # Split card HTML to insert chart
    card_parts = card_html.split(chart_placeholder)

    # Display first part
    st.markdown(card_parts[0], unsafe_allow_html=True)

    # Display chart
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Display second part
    st.markdown(card_parts[1], unsafe_allow_html=True)


# ============================================================================
# ENHANCED TABLE COMPONENTS
# ============================================================================


def display_tailadmin_data_table(
    data: pd.DataFrame,
    title: str = "Data Table",
    searchable: bool = True,
    sortable: bool = True,
    pagination: bool = True,
    actions: Optional[List[Dict]] = None,
    row_actions: bool = False,
    height: Optional[int] = None,
):
    """
    Display an enhanced data table with TailAdmin styling using components.

    Args:
        data: DataFrame to display
        title: Table title
        searchable: Enable search functionality
        sortable: Enable column sorting
        pagination: Enable pagination
        actions: List of action button dicts
        row_actions: Enable per-row actions
        height: Fixed table height
    """

    # Table actions
    actions_html = ""
    if actions:
        actions_html = "<div style='display: flex; gap: 0.5rem;'>"
        for i, action in enumerate(actions):
            btn_html = f"""
                <button class="tailadmin-btn tailadmin-btn-{action.get("type", "secondary")}" 
                        onclick="actionClicked('{action.get("text", "Action")}')"
                        style="padding: 0.5rem 1rem; font-size: 0.875rem;">
                    {action.get("icon", "")} {action.get("text", "Action")}
                </button>
            """
            actions_html += btn_html
        actions_html += "</div>"

    # Search bar
    search_html = ""
    if searchable:
        search_html = f"""
        <div style="margin-bottom: 1rem;">
            <div style="position: relative; max-width: 300px;">
                <input 
                    type="text" 
                    id="tableSearch"
                    placeholder="Search table..."
                    class="tailadmin-input"
                    style="padding-left: 2.5rem; font-size: 0.875rem;"
                    oninput="filterTable(this.value)"
                />
                <div style="
                    position: absolute;
                    left: 0.75rem;
                    top: 50%;
                    transform: translateY(-50%);
                    color: {TAILADMIN_COLORS["gray"]["400"]};
                ">🔍</div>
            </div>
        </div>
        """

    # Generate table HTML with interactive features
    table_html = """
    <table class="tailadmin-table" id="dataTable">
        <thead>
            <tr>
    """

    # Table headers with sorting
    for col in data.columns:
        sort_icon = " ↕️" if sortable else ""
        sort_onclick = f"onclick='sortTable(\"{col}\")'" if sortable else ""
        table_html += f"<th {sort_onclick} style='cursor: {'pointer' if sortable else 'default'};'>{col}{sort_icon}</th>"

    if row_actions:
        table_html += "<th>Actions</th>"

    table_html += """
            </tr>
        </thead>
        <tbody id="tableBody">
    """

    # Table rows
    for idx, row in data.iterrows():
        table_html += f"<tr data-index='{idx}'>"
        for col in data.columns:
            value = row[col]
            # Format value based on type
            if isinstance(value, (int, float)):
                if value >= 1000:
                    formatted_value = f"{value:,}"
                else:
                    formatted_value = str(value)
            else:
                formatted_value = str(value)

            table_html += f"<td>{formatted_value}</td>"

        if row_actions:
            actions_cell = f"""
            <td>
                <button class="tailadmin-btn tailadmin-btn-secondary" 
                        onclick="rowAction({idx})"
                        style="padding: 0.25rem 0.5rem; font-size: 0.75rem;">
                    ⋯
                </button>
            </td>
            """
            table_html += actions_cell

        table_html += "</tr>"

    table_html += """
        </tbody>
    </table>
    """

    # Pagination
    pagination_html = ""
    if pagination:
        pagination_html = f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid {TAILADMIN_COLORS["gray"]["200"]};
        ">
            <div id="paginationInfo" style="color: {TAILADMIN_COLORS["gray"]["600"]}; font-size: 0.875rem;">
                Showing 1 to {len(data)} of {len(data)} entries
            </div>
            <div style="display: flex; gap: 0.25rem;">
                <button class="tailadmin-btn tailadmin-btn-secondary" 
                        onclick="previousPage()"
                        style="padding: 0.5rem; font-size: 0.875rem;">←</button>
                <button class="tailadmin-btn tailadmin-btn-primary" 
                        style="padding: 0.5rem 0.75rem; font-size: 0.875rem;">1</button>
                <button class="tailadmin-btn tailadmin-btn-secondary" 
                        onclick="nextPage()"
                        style="padding: 0.5rem; font-size: 0.875rem;">→</button>
            </div>
        </div>
        """

    # JavaScript for interactivity
    script = (
        """
    <script>
        let currentSort = {column: null, direction: 'asc'};
        let filteredData = null;
        
        function filterTable(searchTerm) {
            const table = document.getElementById('dataTable');
            const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');
            
            for (let i = 0; i < rows.length; i++) {
                const row = rows[i];
                const cells = row.getElementsByTagName('td');
                let found = false;
                
                for (let j = 0; j < cells.length - ("""
        + str(1 if row_actions else 0)
        + """); j++) {
                    if (cells[j].textContent.toLowerCase().includes(searchTerm.toLowerCase())) {
                        found = true;
                        break;
                    }
                }
                
                row.style.display = found ? '' : 'none';
            }
        }
        
        function sortTable(column) {
            console.log('Sorting by:', column);
            // Add sorting logic here
            window.parent.postMessage({type: 'sortTable', column: column}, '*');
        }
        
        function actionClicked(actionName) {
            console.log('Action clicked:', actionName);
            window.parent.postMessage({type: 'actionClicked', action: actionName}, '*');
        }
        
        function rowAction(rowIndex) {
            console.log('Row action for index:', rowIndex);
            window.parent.postMessage({type: 'rowAction', index: rowIndex}, '*');
        }
        
        function previousPage() {
            console.log('Previous page');
            window.parent.postMessage({type: 'previousPage'}, '*');
        }
        
        function nextPage() {
            console.log('Next page');
            window.parent.postMessage({type: 'nextPage'}, '*');
        }
    </script>
    """
    )

    # Combine all parts
    table_container_style = f"height: {height}px; overflow-y: auto;" if height else ""
    full_table_html = f"""
    <div style="{table_container_style}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: {TAILADMIN_COLORS["gray"]["900"]}; font-size: 1.25rem; font-weight: 600;">{title}</h3>
            {actions_html}
        </div>
        {search_html}
        {table_html}
        {pagination_html}
    </div>
    {script}
    """

    # Calculate height for component
    component_height = height if height else (len(data) * 50 + 300)
    component_height = min(component_height, 800)  # Cap at 800px

    components.html(full_table_html, height=component_height, scrolling=True)


# ============================================================================
# FORM COMPONENTS WITH TAILADMIN STYLING
# ============================================================================


def create_tailadmin_form_field(
    label: str,
    field_type: str = "text",
    placeholder: str = "",
    required: bool = False,
    help_text: Optional[str] = None,
    options: Optional[List] = None,
    **kwargs,
) -> str:
    """
    Create a TailAdmin-styled form field.

    Args:
        label: Field label
        field_type: Field type (text, email, password, select, textarea)
        placeholder: Placeholder text
        required: Whether field is required
        help_text: Optional help text
        options: Options for select fields
        **kwargs: Additional field attributes

    Returns:
        HTML string for the form field
    """

    required_indicator = " *" if required else ""
    field_id = label.lower().replace(" ", "_")

    # Label HTML
    label_html = f"""
    <label for="{field_id}" style="
        display: block;
        font-weight: 500;
        color: {TAILADMIN_COLORS["gray"]["700"]};
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
    ">
        {label}{required_indicator}
    </label>
    """

    # Field HTML based on type
    if field_type == "select" and options:
        options_html = ""
        for option in options:
            if isinstance(option, dict):
                value = option.get("value", "")
                text = option.get("text", value)
            else:
                value = text = str(option)
            options_html += f'<option value="{value}">{text}</option>'

        field_html = f"""
        <select id="{field_id}" class="tailadmin-input" {" ".join(f'{k}="{v}"' for k, v in kwargs.items())}>
            <option value="">{placeholder}</option>
            {options_html}
        </select>
        """

    elif field_type == "textarea":
        field_html = f"""
        <textarea 
            id="{field_id}" 
            class="tailadmin-input" 
            placeholder="{placeholder}"
            rows="4"
            {" ".join(f'{k}="{v}"' for k, v in kwargs.items())}
        ></textarea>
        """

    else:
        field_html = f"""
        <input 
            type="{field_type}" 
            id="{field_id}" 
            class="tailadmin-input" 
            placeholder="{placeholder}"
            {" ".join(f'{k}="{v}"' for k, v in kwargs.items())}
        />
        """

    # Help text
    help_html = ""
    if help_text:
        help_html = f"""
        <p style="
            margin-top: 0.5rem;
            font-size: 0.75rem;
            color: {TAILADMIN_COLORS["gray"]["500"]};
        ">{help_text}</p>
        """

    return f"""
    <div style="margin-bottom: 1.5rem;">
        {label_html}
        {field_html}
        {help_html}
    </div>
    """


def create_tailadmin_form(
    title: str,
    fields: List[Dict],
    submit_text: str = "Submit",
    cancel_text: Optional[str] = None,
    form_id: str = "tailadmin_form",
) -> str:
    """
    Create a complete TailAdmin-styled form.

    Args:
        title: Form title
        fields: List of field configuration dicts
        submit_text: Submit button text
        cancel_text: Cancel button text (optional)
        form_id: Form ID attribute

    Returns:
        HTML string for the complete form
    """

    # Generate form fields
    fields_html = ""
    for field in fields:
        field_html = create_tailadmin_form_field(**field)
        fields_html += field_html

    # Form buttons
    cancel_button = ""
    if cancel_text:
        cancel_button = create_tailadmin_button(
            text=cancel_text, button_type="secondary", size="medium"
        )

    submit_button = create_tailadmin_button(
        text=submit_text, button_type="primary", size="medium"
    )

    buttons_html = f"""
    <div style="
        display: flex;
        gap: 1rem;
        justify-content: flex-end;
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid {TAILADMIN_COLORS["gray"]["200"]};
    ">
        {cancel_button}
        {submit_button}
    </div>
    """

    # Complete form
    form_html = f"""
    <form id="{form_id}">
        {fields_html}
        {buttons_html}
    </form>
    """

    return create_tailadmin_card(content=form_html, title=title)


# ============================================================================
# NOTIFICATION AND ALERT COMPONENTS
# ============================================================================


def display_tailadmin_notification(
    message: str,
    notification_type: str = "info",
    title: Optional[str] = None,
    dismissible: bool = True,
    duration: Optional[int] = None,
):
    """
    Display a TailAdmin-styled notification/toast using components.

    Args:
        message: Notification message
        notification_type: Type (success, warning, error, info)
        title: Optional notification title
        dismissible: Whether notification can be dismissed
        duration: Auto-dismiss duration in seconds
    """

    icons = {"success": "✅", "warning": "⚠️", "error": "❌", "info": "ℹ️"}

    colors = {
        "success": TAILADMIN_COLORS["success"],
        "warning": TAILADMIN_COLORS["warning"],
        "error": TAILADMIN_COLORS["error"],
        "info": TAILADMIN_COLORS["brand"],
    }

    color_set = colors.get(notification_type, colors["info"])
    icon = icons.get(notification_type, "ℹ️")

    # Title section
    title_html = ""
    if title:
        title_html = f"""
        <div style="
            font-weight: 600;
            color: {color_set["700"]};
            margin-bottom: 0.25rem;
            font-size: 0.875rem;
        ">{title}</div>
        """

    # Dismiss button
    dismiss_html = ""
    if dismissible:
        dismiss_html = f"""
        <button onclick="dismissNotification()" style="
            background: none;
            border: none;
            color: {color_set["600"]};
            cursor: pointer;
            font-size: 1.5rem;
            padding: 0;
            margin-left: auto;
            line-height: 1;
        ">×</button>
        """

    # Auto-dismiss script
    autodismiss_script = ""
    if duration:
        autodismiss_script = f"""
        setTimeout(function() {{
            dismissNotification();
        }}, {duration * 1000});
        """

    notification_html = f"""
    <div id="tailadmin-notification" class="tailadmin-alert tailadmin-alert-{notification_type}" 
         style="
             display: flex;
             align-items: flex-start;
             gap: 0.75rem;
             margin-bottom: 1rem;
             transition: opacity 0.3s ease;
         ">
        <span style="font-size: 1.25rem; flex-shrink: 0;">{icon}</span>
        <div style="flex: 1;">
            {title_html}
            <div style="color: {color_set["700"]}; font-size: 0.875rem;">
                {message}
            </div>
        </div>
        {dismiss_html}
    </div>
    
    <script>
        function dismissNotification() {{
            const notification = document.getElementById('tailadmin-notification');
            notification.style.opacity = '0';
            setTimeout(() => {{
                notification.style.display = 'none';
            }}, 300);
            
            window.parent.postMessage({{type: 'notificationDismissed', notificationType: '{notification_type}'}}, '*');
        }}
        
        {autodismiss_script}
    </script>
    """

    components.html(notification_html, height=80)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def finalize_tailadmin_app():
    """
    Finalize TailAdmin app (call at the end of your app).
    """
    components.html("</div>", height=0)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================


def show_tailadmin_showcase():
    """
    Demonstrate all TailAdmin components in a showcase page.
    """

    initialize_tailadmin_app()

    # Header
    create_tailadmin_header(
        title="TailAdmin Showcase",
        subtitle="Complete demonstration of TailAdmin styling in Streamlit",
        user_info={"name": "John Doe", "role": "Administrator", "avatar": "👨‍💼"},
        search_enabled=True,
        notifications=3,
    )

    # Sample metrics
    metrics = [
        {
            "icon": "💰",
            "value": "$45,231",
            "label": "Total Revenue",
            "change": "12.5%",
            "change_type": "positive",
            "target": "$50,000",
            "description": "Monthly revenue target progress",
        },
        {
            "icon": "👥",
            "value": "2,345",
            "label": "Active Users",
            "change": "8.2%",
            "change_type": "positive",
            "description": "Users active in the last 30 days",
        },
        {
            "icon": "📦",
            "value": "123",
            "label": "Orders",
            "change": "3.1%",
            "change_type": "negative",
            "description": "Orders processed today",
        },
        {
            "icon": "⭐",
            "value": "4.8",
            "label": "Rating",
            "change": "0.3",
            "change_type": "positive",
            "description": "Average customer rating",
        },
    ]

    display_enhanced_metrics_grid(metrics)

    # Notifications
    display_tailadmin_notification(
        title="Success!",
        message="Your data has been successfully updated.",
        notification_type="success",
        dismissible=True,
    )

    # Sample chart
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    chart_data = pd.DataFrame(
        {
            "Date": dates,
            "Sales": np.random.randint(1000, 5000, 30)
            + np.cumsum(np.random.randn(30) * 100),
        }
    )

    fig = create_tailadmin_plotly_chart(
        data=chart_data, chart_type="line", title="Sales Trend", color_scheme="brand"
    )

    display_tailadmin_chart_card(
        fig=fig,
        title="Sales Analytics",
        description="Daily sales performance over the last 30 days",
        actions=create_tailadmin_button("Export", "secondary", "📊", size="small"),
    )

    # Sample table
    sample_data = pd.DataFrame(
        {
            "Product": ["Widget A", "Widget B", "Widget C", "Widget D"],
            "Sales": [1250, 890, 2340, 1680],
            "Revenue": ["$25,000", "$17,800", "$46,800", "$33,600"],
            "Status": ["Active", "Active", "Pending", "Active"],
        }
    )

    display_tailadmin_data_table(
        data=sample_data,
        title="Product Performance",
        searchable=True,
        sortable=True,
        pagination=True,
        actions=[
            {"text": "Add Product", "type": "primary", "icon": "➕"},
            {"text": "Export", "type": "secondary", "icon": "📊"},
        ],
        row_actions=True,
    )

    finalize_tailadmin_app()


if __name__ == "__main__":
    show_tailadmin_showcase()
