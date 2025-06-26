"""
Component Showcase - Demonstration of Enhanced Streamlit Components

This page showcases the powerful third-party Streamlit components that have been
integrated into the store operations app to enhance user experience and functionality.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import requests

# Third-party component imports with better error handling
try:
    from streamlit_extras.colored_header import colored_header
    from streamlit_extras.metric_cards import style_metric_cards
    from streamlit_extras.add_vertical_space import add_vertical_space
    from streamlit_extras.dataframe_explorer import dataframe_explorer
    try:
        from streamlit_extras.buy_me_a_coffee import button as coffee_button
        COFFEE_AVAILABLE = True
    except ImportError:
        COFFEE_AVAILABLE = False
    EXTRAS_AVAILABLE = True
except ImportError:
    EXTRAS_AVAILABLE = False
    
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False
    
try:
    import extra_streamlit_components as stx
    STX_AVAILABLE = True
except ImportError:
    STX_AVAILABLE = False
    
try:
    from streamlit_option_menu import option_menu
    OPTION_MENU_AVAILABLE = True
except ImportError:
    OPTION_MENU_AVAILABLE = False
    
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False
    
try:
    from streamlit_echarts import st_echarts
    ECHARTS_AVAILABLE = True
except ImportError:
    ECHARTS_AVAILABLE = False
    
try:
    from streamlit_ace import st_ace
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False
    
try:
    from streamlit_image_select import image_select
    IMAGE_SELECT_AVAILABLE = True
except ImportError:
    IMAGE_SELECT_AVAILABLE = False
    
try:
    from streamlit_tags import st_tags
    TAGS_AVAILABLE = True
except ImportError:
    TAGS_AVAILABLE = False
    
try:
    from streamlit_toggle_switch import st_toggle_switch
    TOGGLE_SWITCH_AVAILABLE = True
except ImportError:
    TOGGLE_SWITCH_AVAILABLE = False
    
try:
    from streamlit_calendar import calendar
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False

try:
    import streamlit_shadcn_ui as ui
    SHADCN_AVAILABLE = True
except ImportError:
    SHADCN_AVAILABLE = False

# Check if all components are available
COMPONENTS_AVAILABLE = all([
    EXTRAS_AVAILABLE, AGGRID_AVAILABLE, STX_AVAILABLE, OPTION_MENU_AVAILABLE,
    LOTTIE_AVAILABLE, ECHARTS_AVAILABLE, ACE_AVAILABLE, IMAGE_SELECT_AVAILABLE,
    TAGS_AVAILABLE, TOGGLE_SWITCH_AVAILABLE, CALENDAR_AVAILABLE
])


def load_lottie_url(url: str):
    """Load Lottie animation from URL."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


def create_sample_data():
    """Create sample data for demonstrations."""
    # Sample sales data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    # Build data lists separately to avoid multiple comprehensions in dict
    date_values = list(dates)
    sales_values = [1000 + i*50 + (i%7)*200 for i in range(len(dates))]
    store_values = ['Store A' if i%2==0 else 'Store B' for i in range(len(dates))]
    category_values = [['Electronics', 'Clothing', 'Home', 'Sports'][i%4] for i in range(len(dates))]
    
    sales_data = pd.DataFrame({
        'Date': date_values,
        'Sales': sales_values,
        'Store': store_values,
        'Category': category_values
    })
    
    # Sample inventory data
    inventory_data = pd.DataFrame({
        'Product': ['iPhone 15', 'Samsung Galaxy', 'Nike Shoes', 'Levi Jeans', 'Coffee Maker'],
        'Stock': [25, 30, 45, 60, 15],
        'Price': [999, 899, 120, 89, 159],
        'Category': ['Electronics', 'Electronics', 'Sports', 'Clothing', 'Home'],
        'Status': ['Low Stock', 'In Stock', 'In Stock', 'In Stock', 'Critical']
    })
    
    return sales_data, inventory_data


def show_enhanced_navigation():
    """Demonstrate enhanced navigation components."""
    if not EXTRAS_AVAILABLE:
        st.error("❌ streamlit-extras not available")
        return
        
    colored_header(
        label="🚀 Enhanced Navigation Components",
        description="Beautiful menus and navigation options",
        color_name="blue-70"
    )
    
    # Option Menu - horizontal
    if OPTION_MENU_AVAILABLE:
        selected = option_menu(
            menu_title="Store Sections",
            options=["Dashboard", "Inventory", "Sales", "Staff", "Reports"],
            icons=["house", "box", "graph-up", "people", "file-text"],
            menu_icon="shop",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "green"},
            }
        )
        st.info(f"Selected: {selected}")
    else:
        st.warning("⚠️ streamlit-option-menu not available")
    
    # Tab Bar from extra-streamlit-components
    if STX_AVAILABLE:
        chosen_id = stx.tab_bar(data=[
            stx.TabBarItemData(id=1, title="Today", description="Today's metrics"),
            stx.TabBarItemData(id=2, title="Week", description="Weekly overview"),
            stx.TabBarItemData(id=3, title="Month", description="Monthly stats"),
        ], default=1)
        
        st.success(f"Selected time period: {['Today', 'Weekly', 'Monthly'][chosen_id-1]}")
    else:
        st.warning("⚠️ extra-streamlit-components not available")


def show_enhanced_data_display():
    """Demonstrate enhanced data display components."""
    colored_header(
        label="📊 Enhanced Data Display Components", 
        description="Advanced tables and data visualization",
        color_name="green-70"
    )
    
    sales_data, inventory_data = create_sample_data()
    
    # AgGrid - Advanced data table
    st.subheader("🔥 AgGrid - Advanced Interactive Table")
    st.write("Features: sorting, filtering, selection, editing, and more!")
    
    gb = GridOptionsBuilder.from_dataframe(inventory_data)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    
    gridOptions = gb.build()
    
    grid_response = AgGrid(
        inventory_data,
        gridOptions=gridOptions,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=True,
        height=400,
        width='100%',
        reload_data=False
    )
    
    if grid_response['selected_rows']:
        st.success(f"Selected {len(grid_response['selected_rows'])} items")
        
    # Dataframe Explorer from streamlit-extras
    st.subheader("🔍 Interactive Dataframe Explorer")
    filtered_df = dataframe_explorer(sales_data, case=False)
    st.dataframe(filtered_df, use_container_width=True)


def show_enhanced_widgets():
    """Demonstrate enhanced input widgets."""
    if not EXTRAS_AVAILABLE:
        st.error("❌ streamlit-extras not available")
        return
        
    colored_header(
        label="🎛️ Enhanced Input Widgets",
        description="Beautiful and functional input components",
        color_name="orange-70"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tags Input")
        if TAGS_AVAILABLE:
            keywords = st_tags(
                label='# Enter Keywords:',
                text='Press enter to add more',
                value=['retail', 'management', 'streamlit'],
                suggestions=['inventory', 'sales', 'customers', 'analytics', 'reports'],
                maxtags=10,
                key='tags_demo'
            )
            st.write("Selected tags:", keywords)
        else:
            st.warning("⚠️ streamlit-tags not available")
        
        st.subheader("Toggle Switches")
        if TOGGLE_SWITCH_AVAILABLE:
            notifications = st_toggle_switch(
                label="Enable Notifications",
                key="notifications_toggle",
                default_value=True,
                label_after=False,
                inactive_color='#D3D3D3',
                active_color="#11FF00",
                track_color="#29B5E8"
            )
            st.write(f"Notifications: {'ON' if notifications else 'OFF'}")
            
            maintenance_mode = st_toggle_switch(
                label="Maintenance Mode",
                key="maintenance_toggle", 
                default_value=False
            )
            st.write(f"Maintenance: {'ON' if maintenance_mode else 'OFF'}")
        else:
            st.warning("⚠️ streamlit-toggle-switch not available")
            # Fallback to regular checkboxes
            notifications = st.checkbox("Enable Notifications", value=True)
            maintenance_mode = st.checkbox("Maintenance Mode", value=False)
            st.write(f"Notifications: {'ON' if notifications else 'OFF'}")
            st.write(f"Maintenance: {'ON' if maintenance_mode else 'OFF'}")
    
    with col2:
        st.subheader("Image Selection")
        if IMAGE_SELECT_AVAILABLE:
            img = image_select(
                label="Select store layout:",
                images=[
                    "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=200",
                    "https://images.unsplash.com/photo-1516975080664-ed2fc6a32937?w=200", 
                    "https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=200",
                ],
                captions=["Traditional Layout", "Modern Layout", "Compact Layout"],
                index=0,
                key="layout_select"
            )
            st.write(f"Selected layout: {['Traditional', 'Modern', 'Compact'][img]}")
        else:
            st.warning("⚠️ streamlit-image-select not available")
            # Fallback to regular selectbox
            layout = st.selectbox("Select store layout:", ["Traditional Layout", "Modern Layout", "Compact Layout"])
            st.write(f"Selected layout: {layout}")
        
        st.subheader("Code Editor")
        if ACE_AVAILABLE:
            content = st_ace(
                value='# Enter SQL query for reports\nSELECT * FROM sales WHERE date >= "2024-01-01"',
                language='sql',
                theme='monokai',
                key='sql_editor',
                height=150,
                annotations=None,
                markers=None,
                wrap=False,
                font_size=14,
                tab_size=4,
                show_gutter=True,
                show_print_margin=True,
            )
        else:
            st.warning("⚠️ streamlit-ace not available")
            # Fallback to text area
            content = st.text_area(
                "SQL Query Editor:",
                value='# Enter SQL query for reports\nSELECT * FROM sales WHERE date >= "2024-01-01"',
                height=150,
                key='sql_editor_fallback'
            )


def show_enhanced_charts():
    """Demonstrate enhanced chart components."""
    colored_header(
        label="📈 Enhanced Chart Components",
        description="Beautiful interactive charts with ECharts",
        color_name="violet-70"
    )
    
    sales_data, inventory_data = create_sample_data()
    
    # ECharts integration
    st.subheader("📊 ECharts - Professional Charts")
    
    # Prepare data for ECharts
    chart_data = sales_data.groupby('Category')['Sales'].sum().reset_index()
    
    options = {
        "title": {"text": "Sales by Category", "left": "center"},
        "tooltip": {"trigger": "item"},
        "legend": {"orient": "vertical", "left": "left"},
        "series": [
            {
                "name": "Sales",
                "type": "pie",
                "radius": "50%",
                "data": [
                    {"value": row['Sales'], "name": row['Category']} 
                    for _, row in chart_data.iterrows()
                ],
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                }
            }
        ]
    }
    
    st_echarts(
        options=options, 
        height="400px",
        key="sales_pie_chart"
    )
    
    # Gauge chart for KPI
    st.subheader("🎯 KPI Gauge")
    gauge_options = {
        "series": [
            {
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "center": ["50%", "75%"],
                "radius": "90%",
                "min": 0,
                "max": 100,
                "splitNumber": 8,
                "axisLine": {
                    "lineStyle": {
                        "width": 6,
                        "color": [
                            [0.25, "#FF6E76"],
                            [0.5, "#FDDD60"], 
                            [0.75, "#58D9F9"],
                            [1, "#7CFFB2"]
                        ]
                    }
                },
                "pointer": {
                    "icon": "path://M12.8,0.7l12,40.1H0.7L12.8,0.7z",
                    "length": "12%",
                    "width": 20,
                    "offsetCenter": [0, "-60%"],
                    "itemStyle": {"color": "auto"}
                },
                "axisTick": {"length": 12, "lineStyle": {"color": "auto", "width": 2}},
                "splitLine": {"length": 20, "lineStyle": {"color": "auto", "width": 5}},
                "axisLabel": {
                    "color": "#464646",
                    "fontSize": 20,
                    "distance": -60,
                    "rotate": "tangential",
                },
                "title": {"offsetCenter": [0, "-10%"], "fontSize": 20},
                "detail": {
                    "fontSize": 30,
                    "offsetCenter": [0, "-35%"],
                    "valueAnimation": True,
                    "color": "auto"
                },
                "data": [{"value": 85, "name": "Store Performance"}]
            }
        ]
    }
    
    st_echarts(options=gauge_options, height="300px", key="performance_gauge")


def show_enhanced_animations():
    """Demonstrate Lottie animations."""
    colored_header(
        label="✨ Enhanced Animations & Visual Elements",
        description="Lottie animations and visual enhancements",
        color_name="red-70"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📦 Loading Animation")
        lottie_loading = load_lottie_url("https://lottie.host/4f64e42e-7b7d-4ddb-9baa-89e37c94b48e/UtZfT2SuJH.json")
        if lottie_loading:
            st_lottie(lottie_loading, height=200, key="loading")
    
    with col2:
        st.subheader("✅ Success Animation")
        lottie_success = load_lottie_url("https://lottie.host/017f3479-a351-41d3-8c29-14dced6ebdc0/TgGBRLYJyF.json")
        if lottie_success:
            st_lottie(lottie_success, height=200, key="success")
    
    with col3:
        st.subheader("🏪 Store Animation")
        lottie_store = load_lottie_url("https://lottie.host/8b2e55dc-7b52-4ac3-bf5d-9092b2dfa7de/YqYPZKbPKP.json")
        if lottie_store:
            st_lottie(lottie_store, height=200, key="store")


def show_shadcn_components():
    """Demonstrate shadcn-ui components."""
    if not SHADCN_AVAILABLE:
        st.warning("⚠️ streamlit-shadcn-ui not available")
        return
        
    if not EXTRAS_AVAILABLE:
        st.error("❌ streamlit-extras not available for headers")
        return
        
    colored_header(
        label="🎨 Modern shadcn/ui Components",
        description="Beautiful modern UI components based on shadcn/ui",
        color_name="violet-70"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔘 Modern Buttons")
        
        # Different button variants
        if ui.button("Primary Button", key="primary_btn", variant="default"):
            st.success("Primary button clicked!")
            
        if ui.button("Secondary Button", key="secondary_btn", variant="secondary"):
            st.info("Secondary button clicked!")
            
        if ui.button("Destructive Button", key="destructive_btn", variant="destructive"):
            st.error("Destructive button clicked!")
            
        if ui.button("Outline Button", key="outline_btn", variant="outline"):
            st.info("Outline button clicked!")
        
        st.subheader("📝 Input Components")
        
        # Input field
        name = ui.input(label="Full Name", placeholder="Enter your name", key="name_input")
        if name:
            st.write(f"Hello, {name}!")
            
        # Textarea
        feedback = ui.textarea(
            label="Feedback", 
            placeholder="Share your thoughts...", 
            key="feedback_input"
        )
        if feedback:
            st.write(f"Feedback: {feedback}")
    
    with col2:
        st.subheader("🎛️ Form Controls")
        
        # Switch component
        notifications = ui.switch(label="Enable Notifications", key="notif_switch")
        st.write(f"Notifications: {'Enabled' if notifications else 'Disabled'}")
        
        # Select component
        store_location = ui.select(
            options=["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
            label="Store Location",
            placeholder="Choose a location",
            key="store_select"
        )
        if store_location:
            st.write(f"Selected store: {store_location}")
        
        # Slider component
        inventory_threshold = ui.slider(
            label="Inventory Alert Threshold",
            min_value=0,
            max_value=100,
            value=20,
            key="threshold_slider"
        )
        st.write(f"Alert when stock below: {inventory_threshold} units")
        
        st.subheader("💳 Card Components")
        
        # Card component
        with ui.card(key="info_card"):
            ui.card_content([
                "📊 **Store Performance**",
                "",
                "Today's sales: **$12,543**",
                "Active customers: **247**", 
                "Inventory status: **Healthy**"
            ])


def show_enhanced_calendar():
    """Demonstrate calendar component."""
    if not CALENDAR_AVAILABLE:
        st.warning("⚠️ streamlit-calendar not available")
        return
        
    if not EXTRAS_AVAILABLE:
        st.error("❌ streamlit-extras not available for headers")
        return
        
    colored_header(
        label="📅 Enhanced Calendar Component",
        description="Interactive calendar for scheduling and events",
        color_name="blue-green-70"
    )
    
    # Calendar events
    events = [
        {
            "title": "Store Meeting",
            "start": "2024-01-15T10:00:00",
            "end": "2024-01-15T11:00:00",
            "color": "#FF6B6B",
        },
        {
            "title": "Inventory Check", 
            "start": "2024-01-16T14:00:00",
            "end": "2024-01-16T16:00:00",
            "color": "#4ECDC4",
        },
        {
            "title": "Staff Training",
            "start": "2024-01-18T09:00:00", 
            "end": "2024-01-18T17:00:00",
            "color": "#45B7D1",
        }
    ]
    
    calendar_options = {
        "editable": "true",
        "navLinks": "true",
        "selectable": "true",
    }
    
    custom_css = """
        .fc-event-past {
            opacity: 0.8;
        }
        .fc-event-time {
            font-style: italic;
        }
        .fc-event-title {
            font-weight: 700;
        }
        .fc-toolbar-title {
            font-size: 2rem;
        }
    """
    
    calendar_component = calendar(
        events=events,
        options=calendar_options,
        custom_css=custom_css,
        key="store_calendar"
    )
    
    if calendar_component.get('eventsSet'):
        st.success("Calendar events loaded successfully!")
    
    if calendar_component.get('dateClick'):
        st.info(f"Date clicked: {calendar_component['dateClick']['date']}")


def show_enhanced_metrics():
    """Demonstrate enhanced metrics display."""
    colored_header(
        label="📊 Enhanced Metrics Display",
        description="Beautiful metric cards with styling",
        color_name="green-blue-70"
    )
    
    # Create metrics with streamlit-extras styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Daily Sales",
            value="$12,543",
            delta="8.5%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Active Customers", 
            value="2,847",
            delta="156",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Inventory Turnover",
            value="7.2x",
            delta="-0.3x",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Staff Efficiency",
            value="94%",
            delta="2%",
            delta_color="normal" 
        )
    
    # Apply styling to metrics
    style_metric_cards(
        background_color="#FFFFFF",
        border_left_color="#686664", 
        border_color="#1f66bd",
        box_shadow="#F71938"
    )


def main():
    """Main function for the component showcase."""
    st.set_page_config(
        page_title="Component Showcase",
        page_icon="🚀",
        layout="wide"
    )
    
    # Main header
    st.title("🚀 Streamlit Component Showcase")
    st.markdown("""
    **Welcome to the enhanced store operations app!** This page demonstrates the powerful 
    third-party Streamlit components that have been integrated to provide a superior user experience.
    """)
    
    # Component availability status
    st.markdown("### 📦 Component Availability Status")
    
    component_status = {
        "streamlit-extras": EXTRAS_AVAILABLE,
        "streamlit-aggrid": AGGRID_AVAILABLE, 
        "extra-streamlit-components": STX_AVAILABLE,
        "streamlit-option-menu": OPTION_MENU_AVAILABLE,
        "streamlit-lottie": LOTTIE_AVAILABLE,
        "streamlit-echarts": ECHARTS_AVAILABLE,
        "streamlit-ace": ACE_AVAILABLE,
        "streamlit-image-select": IMAGE_SELECT_AVAILABLE,
        "streamlit-tags": TAGS_AVAILABLE,
        "streamlit-toggle-switch": TOGGLE_SWITCH_AVAILABLE,
        "streamlit-calendar": CALENDAR_AVAILABLE,
        "streamlit-shadcn-ui": SHADCN_AVAILABLE,
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (component, available) in enumerate(component_status.items()):
        col = [col1, col2, col3][i % 3]
        with col:
            status_icon = "✅" if available else "❌"
            st.write(f"{status_icon} {component}")
    
    missing_components = [comp for comp, avail in component_status.items() if not avail]
    
    if missing_components:
        st.warning(f"⚠️ {len(missing_components)} components are missing. Some demonstrations will show fallbacks.")
        with st.expander("🔧 Installation Instructions"):
            st.markdown("To install missing components, run:")
            st.code("uv sync", language="bash")
            st.markdown("Missing components:")
            for comp in missing_components:
                st.markdown(f"- `{comp}`")
    else:
        st.success("🎉 All components are available!")
    
    # Add vertical space
    if EXTRAS_AVAILABLE:
        add_vertical_space(2)
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Show all component demonstrations
    show_enhanced_navigation()
    
    if EXTRAS_AVAILABLE:
        add_vertical_space(3)
    else:
        st.markdown("---")
    
    show_enhanced_metrics()
    
    if EXTRAS_AVAILABLE:
        add_vertical_space(3)
    else:
        st.markdown("---")
    
    show_enhanced_data_display()
    
    if EXTRAS_AVAILABLE:
        add_vertical_space(3)
    else:
        st.markdown("---")
    
    show_enhanced_widgets()
    
    if EXTRAS_AVAILABLE:
        add_vertical_space(3)
    else:
        st.markdown("---")
    
    show_enhanced_charts()
    
    if EXTRAS_AVAILABLE:
        add_vertical_space(3)
    else:
        st.markdown("---")
    
    show_enhanced_animations()
    
    if EXTRAS_AVAILABLE:
        add_vertical_space(3)
    else:
        st.markdown("---")
    
    show_shadcn_components()
    
    if EXTRAS_AVAILABLE:
        add_vertical_space(3)
    else:
        st.markdown("---")
    
    show_enhanced_calendar()
    
    if EXTRAS_AVAILABLE:
        add_vertical_space(3)
    else:
        st.markdown("---")
    
    # Buy me a coffee button (fun extra)
    st.markdown("---")
    st.markdown("### 💝 Support the Development")
    if COFFEE_AVAILABLE:
        coffee_button(username="your-username", floating=False, width=221)
    else:
        st.info("☕ Buy me a coffee component not available")


if __name__ == "__main__":
    main() 