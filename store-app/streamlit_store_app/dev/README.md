# Development Playground

This directory provides a sandbox environment for creating and testing visual components without affecting the main application.

## 🏗️ Directory Structure

```
dev/
├── components/          # Your custom components go here
├── examples/           # Example components for reference
├── README.md          # This file
└── __init__.py        # Package initialization
```

## 🚀 Getting Started

1. **Access the Playground**: Navigate to `/dev_playground` in your Streamlit app
2. **Create a Component**: Use the "Create New Component" section
3. **Edit Your Component**: Edit the generated file in your IDE
4. **Preview**: Use the "Reload Component" button to see changes
5. **Iterate**: Make changes and reload as needed

## 📝 Component Structure

Every component should have a `show_component()` function:

```python
def show_component():
    """Main function to display this component."""
    st.header("My Component")
    # Your component code here
```

## 🎯 Features

- **🔄 Hot Reload**: Changes are reflected immediately with the reload button
- **📁 Component Manager**: Easy selection and management of components
- **🧪 Isolated Environment**: Test without affecting the main app
- **💡 Examples**: Built-in examples and templates
- **📋 Component List**: View all created components with metadata

## 📖 Examples

Check the `examples/` directory for sample components showing:
- Dashboard layouts
- Interactive charts
- Metric cards
- Data visualizations

## 🛠️ Development Workflow

1. **Create**: Use the playground to generate a new component
2. **Edit**: Open the file in your preferred IDE
3. **Test**: Use the reload button to see changes
4. **Iterate**: Refine your component with instant feedback
5. **Deploy**: Once ready, integrate into the main app

## 📂 File Locations

- **Components**: `dev/components/your_component.py`
- **Examples**: `dev/examples/sample_dashboard.py`
- **Playground**: Navigate to `/dev_playground` in the app

## 💡 Tips

- Use descriptive names for your components
- Follow the `show_component()` function pattern
- Test with different data and edge cases
- Check the examples for inspiration
- Use the reload button frequently during development

## 🔗 Integration

When your component is ready for production:
1. Move it to the main `components/` directory
2. Import it in the relevant homepage modules
3. Add it to the main application navigation

Happy coding! 🎉 