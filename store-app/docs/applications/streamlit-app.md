# Store Companion - Streamlit Application

The Store Companion is a Streamlit-based retail store management application with AI assistance, providing a comprehensive interface for store operations.

## 🌟 Features

- **Multi-store Management**: Manage multiple retail locations from a single interface
- **Role-based Access Control**: Different views for Store Associates and Managers
- **Real-time Inventory Tracking**: Live inventory updates and stock monitoring
- **Order Management**: Complete order lifecycle management
- **Staff Management**: Employee scheduling and management tools
- **AI-powered Chat**: Integrated AI assistant for store operations

## 🏗️ Application Structure

```
streamlit_store_app/
├── app.py                 # Main application entry point
├── config.yaml           # Centralized configuration
├── components/          # Reusable UI components
│   ├── chat.py         # AI chat widget
│   ├── metrics.py      # Store metrics display
│   ├── navigation.py   # Navigation components
│   └── styles.py       # UI styling utilities
├── pages/              # Application pages
│   ├── 1_📦_Orders.py
│   ├── 2_📊_Inventory.py
│   └── 3_👥_Staff.py
├── utils/              # Utility functions
│   ├── config.py       # Configuration management
│   ├── database.py     # Database operations
│   └── model_serving.py # AI model integration
└── tests/              # Test suite
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Access to the Retail AI model endpoint
- Databricks workspace credentials

### Installation

1. **Navigate to the Streamlit app directory**
   ```bash
   cd streamlit_store_app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The application will be available at `http://localhost:8501`.

## ⚙️ Configuration

All configuration is centralized in `config.yaml`:

### Application Settings

```yaml
app:
  title: "Store Companion"
  page_title: "Retail Store Management"
  page_icon: "🏪"
  layout: "wide"
```

### Store Information

```yaml
stores:
  - id: "store_001"
    name: "Downtown Location"
    address: "123 Main St, City, State"
    manager: "John Smith"
  - id: "store_002"
    name: "Mall Location"
    address: "456 Mall Blvd, City, State"
    manager: "Jane Doe"
```

### User Roles

```yaml
roles:
  store_associate:
    permissions:
      - view_inventory
      - process_orders
      - chat_assistance
  store_manager:
    permissions:
      - view_inventory
      - process_orders
      - manage_staff
      - view_analytics
      - chat_assistance
```

### Chat Interface

```yaml
chat:
  model_endpoint: "retail_ai_agent"
  max_messages: 50
  welcome_message: "Hello! I'm your AI store assistant. How can I help you today?"
```

### Model Serving

```yaml
model_serving:
  endpoint_url: "https://your-databricks-workspace.cloud.databricks.com"
  token: "${DATABRICKS_TOKEN}"
  model_name: "retail_ai_agent"
```

### Database Settings

```yaml
database:
  connection_string: "${DATABASE_URL}"
  pool_size: 10
  timeout: 30
```

## 📱 Application Pages

### Main Dashboard (`app.py`)

The main dashboard provides:

- **Store Selection**: Choose which store location to manage
- **Role Selection**: Switch between Store Associate and Manager views
- **Key Metrics**: Overview of store performance
- **Quick Actions**: Common tasks and shortcuts
- **AI Chat**: Integrated assistant for immediate help

### Orders Page (`1_📦_Orders.py`)

Order management features:

- **Order List**: View all orders with filtering and sorting
- **Order Details**: Detailed view of individual orders
- **Status Updates**: Update order status and tracking
- **Customer Information**: Access customer details and history
- **Fulfillment**: Manage picking, packing, and shipping

### Inventory Page (`2_📊_Inventory.py`)

Inventory management tools:

- **Stock Levels**: Real-time inventory across all locations
- **Low Stock Alerts**: Automated alerts for items needing restock
- **Product Search**: Find products by SKU, name, or category
- **Inventory Adjustments**: Record stock adjustments and transfers
- **Reporting**: Generate inventory reports and analytics

### Staff Page (`3_👥_Staff.py`)

Staff management capabilities:

- **Employee Directory**: View all staff members and their roles
- **Scheduling**: Create and manage work schedules
- **Performance Metrics**: Track employee performance
- **Training Records**: Manage training and certifications
- **Communication**: Internal messaging and announcements

## 🤖 AI Chat Integration

The integrated AI chat provides:

### Capabilities

- **Product Information**: Get details about products and inventory
- **Order Assistance**: Help with order processing and customer inquiries
- **Policy Questions**: Answer questions about store policies and procedures
- **Troubleshooting**: Assist with common issues and problems
- **Analytics**: Provide insights and recommendations

### Usage Examples

```python
# Example chat interactions
"What's the current stock level for SKU ABC123?"
"How do I process a return for order #12345?"
"What's our policy on price matching?"
"Show me sales trends for the last month"
```

### Implementation

The chat component (`components/chat.py`) integrates with the Retail AI model:

```python
import streamlit as st
from utils.model_serving import get_model_response

def chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about store operations..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        response = get_model_response(prompt, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
```

## 🎨 UI Components

### Reusable Components

The `components/` directory contains reusable UI elements:

#### Metrics Display (`metrics.py`)

```python
def display_store_metrics(store_id):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Daily Sales", "$12,345", "5.2%")
    with col2:
        st.metric("Orders", "89", "12")
    with col3:
        st.metric("Inventory Items", "1,234", "-23")
    with col4:
        st.metric("Staff On Duty", "8", "2")
```

#### Navigation (`navigation.py`)

```python
def sidebar_navigation():
    with st.sidebar:
        st.title("Store Companion")
        
        # Store selection
        store = st.selectbox("Select Store", get_stores())
        
        # Role selection
        role = st.selectbox("Role", ["Store Associate", "Store Manager"])
        
        # Navigation menu
        page = st.radio("Navigate", ["Dashboard", "Orders", "Inventory", "Staff"])
        
        return store, role, page
```

#### Styling (`styles.py`)

```python
def apply_custom_styles():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
```

## 🧪 Development

### Development Setup

1. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run tests**
   ```bash
   pytest tests/
   ```

3. **Format code**
   ```bash
   black .
   isort .
   ```

4. **Type checking**
   ```bash
   mypy .
   ```

### Testing

The test suite covers:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality
- **UI Tests**: Streamlit component testing
- **API Tests**: Model serving integration

### Code Quality

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **pytest**: Testing framework
- **pre-commit**: Git hooks for quality checks

## 🚀 Deployment

### Docker Deployment

The application can be deployed using Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t store-companion .
docker run -p 8501:8501 store-companion
```

### Cloud Deployment

Deploy to cloud platforms:

- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Using the Heroku CLI
- **AWS/Azure/GCP**: Using container services
- **Databricks**: As a Databricks app

## 🔧 Troubleshooting

### Common Issues

1. **Model Connection Errors**
   - Check Databricks token and endpoint URL
   - Verify model serving endpoint is running
   - Check network connectivity

2. **Configuration Issues**
   - Validate `config.yaml` syntax
   - Check environment variable values
   - Verify file permissions

3. **Performance Issues**
   - Monitor memory usage
   - Check database connection pool
   - Optimize query performance

### Debug Mode

Enable debug mode for troubleshooting:

```python
# In app.py
if st.checkbox("Debug Mode"):
    st.write("Session State:", st.session_state)
    st.write("Configuration:", st.session_state.config)
```

## 📄 License

This application is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

The Store Companion provides a comprehensive, user-friendly interface for retail store management with integrated AI assistance, making store operations more efficient and effective. 