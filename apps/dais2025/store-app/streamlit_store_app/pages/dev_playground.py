"""Development Playground - Create and test visual components in isolation."""

import importlib.util
import sys
from pathlib import Path
import pandas as pd

import streamlit as st

# Remove the global sys.path modification to prevent interference with main app imports


def main():
    """Main development playground page."""
    st.set_page_config(
        page_title="Development Playground",
        page_icon="🧪",
        layout="wide",
    )

    # Custom CSS for development environment
    st.markdown(
        """
    <style>
    .dev-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .component-selector {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 20px;
    }
    
    .dev-section {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .preview-area {
        border: 2px solid #007bff;
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9ff;
        margin-top: 20px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        """
    <div class="dev-header">
        <h1 style="margin: 0; font-size: 28px; font-weight: 600;">
            🧪 Development Playground
        </h1>
        <p style="margin: 8px 0 0 0; font-size: 16px; opacity: 0.9;">
            Create, test, and preview visual components in isolation
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Main layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="dev-section">', unsafe_allow_html=True)
        st.subheader("📁 Component Manager")
        
        # Component selector
        component_files = get_available_components()
        
        if component_files:
            selected_component = st.selectbox(
                "Select Component to View:",
                options=component_files,
                format_func=lambda x: x.replace('.py', '').replace('_', ' ').title(),
                key="component_selector"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 Reload Component", use_container_width=True):
                    st.rerun()
                    
            with col_b:
                if st.button("📝 Edit Component", use_container_width=True):
                    st.session_state.editing_component = selected_component
        else:
            st.info("No components found. Create your first component below!")
            selected_component = None

        st.markdown('</div>', unsafe_allow_html=True)
        
        # Component creation
        st.markdown('<div class="dev-section">', unsafe_allow_html=True)
        st.subheader("➕ Create New Component")
        
        new_component_name = st.text_input(
            "Component Name:",
            placeholder="my_awesome_component",
            help="Use lowercase with underscores"
        )
        
        if st.button("🎨 Create Component", use_container_width=True):
            if new_component_name:
                create_new_component(new_component_name)
                st.success(f"Created {new_component_name}.py!")
                st.rerun()
            else:
                st.error("Please enter a component name")
                
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick actions
        st.markdown('<div class="dev-section">', unsafe_allow_html=True)
        st.subheader("⚡ Quick Actions")
        
        if st.button("📋 View All Components", use_container_width=True):
            st.session_state.show_component_list = True
            
        if st.button("🗑️ Clear Preview", use_container_width=True):
            st.session_state.clear_preview = True
            st.rerun()
            
        if st.button("💡 Show Examples", use_container_width=True):
            st.session_state.show_examples = True
            
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Preview area
        st.markdown('<div class="preview-area">', unsafe_allow_html=True)
        st.subheader("👁️ Component Preview")
        
        # Handle component preview
        if 'clear_preview' in st.session_state and st.session_state.clear_preview:
            st.info("Preview cleared. Select a component to view.")
            del st.session_state.clear_preview
        elif component_files and selected_component:
            try:
                # Load and display the selected component
                st.markdown(f"**Displaying:** `{selected_component}`")
                st.divider()
                
                # Import and run the component
                component_module = import_component(selected_component)
                if hasattr(component_module, 'show_component'):
                    component_module.show_component()
                elif hasattr(component_module, 'main'):
                    component_module.main()
                else:
                    st.error(f"Component {selected_component} must have a 'show_component()' or 'main()' function")
                    
            except Exception as e:
                st.error(f"Error loading component: {str(e)}")
                st.code(str(e), language="python")
        else:
            st.info("🎯 Select or create a component to see it here!")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Show component list if requested
    if st.session_state.get('show_component_list', False):
        show_component_list()
        st.session_state.show_component_list = False

    # Show examples if requested  
    if st.session_state.get('show_examples', False):
        show_examples()
        st.session_state.show_examples = False


def get_available_components():
    """Get list of available component files."""
    components_dir = Path(__file__).parent.parent / "dev" / "components"
    if not components_dir.exists():
        return []
    
    return [f.name for f in components_dir.glob("*.py") if f.name != "__init__.py"]


def import_component(component_file):
    """Import a component module dynamically from dev/components."""
    module_name = component_file.replace('.py', '')
    
    # Use importlib.util to directly load from file path
    file_path = Path(__file__).parent.parent / "dev" / "components" / component_file
    
    if not file_path.exists():
        raise ImportError(f"Component file {component_file} not found")
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    
    # Execute the module
    spec.loader.exec_module(module)
    
    return module


def create_new_component(name):
    """Create a new component file with template code."""
    components_dir = Path(__file__).parent.parent / "dev" / "components"
    components_dir.mkdir(exist_ok=True)
    
    # Use string formatting instead of f-strings to avoid nesting issues
    component_title = name.replace('_', ' ').title()
    
    template = '''"""
{title} Component

Created in the Development Playground.
"""

import streamlit as st
import plotly.express as px
import pandas as pd


def show_component():
    """Main function to display this component."""
    st.header("🎨 {title}")
    
    st.markdown("""
    Welcome to your new component! 
    
    **Getting Started:**
    1. Edit this file in your IDE
    2. Use the reload button to see changes
    3. Build amazing visualizations!
    """)
    
    # Example interactive element
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sample Chart")
        # Sample data
        data = pd.DataFrame({{
            'x': range(10),
            'y': [i**2 for i in range(10)]
        }})
        
        fig = px.line(data, x='x', y='y', title='Sample Interactive Chart')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎛️ Controls")
        
        number = st.slider("Pick a number", 0, 100, 50)
        st.metric("Your Number", number, delta=number-50)
        
        color = st.color_picker("Pick a color", "#00f900")
        st.markdown(f'<div style="background-color: {{color}}; padding: 20px; border-radius: 10px; text-align: center;">Your color!</div>', unsafe_allow_html=True)
    
    # Add your component logic here
    st.info("💡 Edit this file to customize your component!")


if __name__ == "__main__":
    show_component()
'''.format(title=component_title)
    
    file_path = components_dir / f"{name}.py"
    file_path.write_text(template)


def show_component_list():
    """Show a list of all components with details."""
    st.subheader("📋 All Components")
    
    components_dir = Path(__file__).parent.parent / "dev" / "components"
    if not components_dir.exists():
        st.info("No components directory found.")
        return
    
    components = list(components_dir.glob("*.py"))
    
    if not components:
        st.info("No components found.")
        return
    
    for component in components:
        if component.name == "__init__.py":
            continue
            
        with st.expander(f"📄 {component.stem}"):
            st.code(f"File: {component.name}")
            st.code(f"Path: {component}")
            
            # Show file size and last modified
            stat = component.stat()
            st.text(f"Size: {stat.st_size} bytes")
            st.text(f"Modified: {pd.Timestamp.fromtimestamp(stat.st_mtime)}")


def show_examples():
    """Show example component templates."""
    st.subheader("💡 Component Examples")
    
    examples = {
        "Dashboard Card": """
def show_component():
    st.markdown('''
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 20px; border-radius: 10px;">
        <h3>📊 Dashboard Card</h3>
        <p>Revenue: $45,230</p>
        <small>↗️ +12% vs last month</small>
    </div>
    ''', unsafe_allow_html=True)
        """,
        
        "Interactive Chart": """
def show_component():
    data = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D'],
        'Values': [23, 45, 56, 78]
    })
    
    chart_type = st.selectbox("Chart Type", ["bar", "line", "pie"])
    
    if chart_type == "bar":
        fig = px.bar(data, x='Category', y='Values')
    elif chart_type == "line":
        fig = px.line(data, x='Category', y='Values')
    else:
        fig = px.pie(data, values='Values', names='Category')
    
    st.plotly_chart(fig, use_container_width=True)
        """,
        
        "Metric Grid": """
def show_component():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sales", "$25,430", "+12%")
    with col2:
        st.metric("Orders", "1,234", "+5%")
    with col3:
        st.metric("Customers", "567", "+8%")
    with col4:
        st.metric("Revenue", "$45,230", "+15%")
        """
    }
    
    for title, code in examples.items():
        with st.expander(f"📖 {title}"):
            st.code(code, language="python")


if __name__ == "__main__":
    main() 