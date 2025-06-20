# 🎨 TailAdmin Store Operations Dashboard

A modern, multi-page Streamlit application showcasing the TailAdmin design system for retail store operations management.

## 🌟 Features

- **🏠 Homepage**: Role-based dashboard with store metrics and activities
- **📊 VP Dashboard (Clean)**: Executive-level performance dashboard with TailAdmin styling
- **📈 VP Dashboard (Enhanced)**: Advanced analytics with interactive controls
- **📋 Components Demo**: Interactive showcase of all TailAdmin components  
- **📖 Implementation Guide**: Complete documentation and best practices

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- UV package manager

### Installation

1. **Clone and navigate to the project**
   ```bash
   cd streamlit_store_app_tailadmin
   ```

2. **Install dependencies**
   ```bash
   make install
   ```

3. **Start the application**
   ```bash
   make start
   ```

4. **Access the app**
   Open your browser to `http://localhost:8501`

## 🛠️ Available Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install all dependencies |
| `make start` | Start the TailAdmin app |
| `make start-debug` | Start with debug logging |
| `make format` | Format code with ruff |
| `make lint` | Lint code with ruff |
| `make test` | Run tests |
| `make clean` | Clean build artifacts |
| `make export-requirements` | Export requirements.txt |

## 📁 Project Structure

```
streamlit_store_app_tailadmin/
├── app.py                      # Main application with navigation
├── components/
│   └── tailadmin/
│       ├── __init__.py         # Component exports
│       └── tailadmin_styles.py # Core TailAdmin components
├── pages/
│   ├── homepage.py             # Store homepage
│   ├── vp_dashboard_clean.py   # Clean VP dashboard
│   ├── vp_dashboard_enhanced.py # Enhanced VP dashboard
│   ├── components_demo.py      # Component showcase
│   └── implementation_guide.py # Documentation
├── pyproject.toml              # Project configuration
├── Makefile                    # Build commands
└── README.md                   # This file
```

## 🎨 TailAdmin Design System

### Core Components

- **📊 Metric Cards**: KPI displays with trend indicators
- **🃏 Cards**: Flexible content containers
- **🔘 Buttons**: Styled action buttons with variants
- **📊 Progress Bars**: Animated progress indicators
- **📈 Stat Widgets**: Multi-metric statistics displays

### Color System

- **Brand Colors**: Primary identity colors
- **Gray Scale**: Neutral text and background colors
- **Success/Warning/Error**: Semantic state colors

### Typography

- **Font Family**: 'Outfit' - Modern, clean typeface
- **Scales**: Consistent sizing from captions to displays
- **Weights**: Complete range from thin (100) to black (900)

## 📱 Pages Overview

### 🏠 Homepage
Role-based dashboard showing:
- Store information and status
- Key performance metrics
- Recent activities and tasks
- Quick action buttons

### 📊 VP Dashboard (Clean)
Executive dashboard featuring:
- High-level KPI metrics
- Revenue vs target charts
- Regional performance breakdown
- AI-powered insights

### 📈 VP Dashboard (Enhanced)
Advanced analytics with:
- Interactive controls and filters
- Geographic performance mapping
- Predictive insights
- Multi-level data drill-down

### 📋 Components Demo
Interactive showcase of:
- All TailAdmin components
- Customization options
- Live preview and code examples
- Color system utilities

### 📖 Implementation Guide
Complete documentation including:
- Getting started guide
- Component usage examples
- Best practices and patterns
- Common pitfalls to avoid

## 🔧 Customization

### Adding New Components

1. Add component function to `components/tailadmin/tailadmin_styles.py`
2. Export in `components/tailadmin/__init__.py`
3. Document in the implementation guide

### Creating New Pages

1. Create page file in `pages/` directory
2. Add page function that returns content
3. Import and add to navigation in `app.py`

### Modifying Colors

Update color definitions in `TAILADMIN_COLORS` dictionary in `tailadmin_styles.py`.

## 🏗️ Development

### Code Style

- **Formatter**: Ruff (configured in pyproject.toml)
- **Linter**: Ruff with custom rules
- **Type Hints**: Encouraged for better code quality

### Testing

```bash
make test
```

### Building for Production

```bash
make build
```

This will:
- Install dependencies
- Format and lint code  
- Run tests
- Validate configuration

## 🚀 Deployment

### Databricks Apps

1. Set your app name:
   ```bash
   export DATABRICKS_APP_NAME=your-tailadmin-app
   ```

2. Deploy:
   ```bash
   make deploy
   ```

### Other Platforms

The app can be deployed to any platform supporting Streamlit:
- Streamlit Cloud
- Heroku
- AWS/GCP/Azure
- Docker containers

## 🤝 Contributing

1. **Format your code**: `make format`
2. **Run tests**: `make test`
3. **Check linting**: `make lint`
4. Follow the established patterns and component structure

## 📄 License

This project is part of the retail store operations demo and follows the same licensing terms.

## 🆘 Support

For issues or questions:
1. Check the Implementation Guide in the app
2. Review component examples in the demo page
3. Examine the source code for usage patterns

## 🔗 Related

- **Main Store App**: `../streamlit_store_app/` - Original store operations app
- **Components**: Full TailAdmin component library
- **Documentation**: In-app implementation guide

---

Built with ❤️ using Streamlit and the TailAdmin design system. 