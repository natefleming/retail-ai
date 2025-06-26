# Installation Guide

This guide provides detailed installation instructions for the Retail AI system across different environments.

## System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space
- **Network**: Internet connection for package downloads

### Databricks Requirements
- **Databricks Workspace**: Access to a Databricks workspace
- **Unity Catalog**: Enabled and configured
- **SQL Warehouse**: Running SQL warehouse for function execution
- **Permissions**: Ability to create functions and access data

## Installation Methods

### Method 1: Standard Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/retail-ai.git
   cd retail-ai
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Method 2: Development Installation

For contributors and developers:

1. **Clone with Development Dependencies**
   ```bash
   git clone https://github.com/your-org/retail-ai.git
   cd retail-ai
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

### Method 3: Docker Installation

1. **Build Docker Image**
   ```bash
   docker build -t retail-ai .
   ```

2. **Run Container**
   ```bash
   docker run -p 8501:8501 retail-ai
   ```

## Environment Configuration

### 1. Databricks Configuration

Create a `.env` file in the project root:

```env
# Databricks Configuration
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-personal-access-token
DATABRICKS_WAREHOUSE_ID=your-sql-warehouse-id

# Optional: Specific endpoint configurations
DATABRICKS_CLUSTER_ID=your-cluster-id
DATABRICKS_WORKSPACE_ID=your-workspace-id
```

### 2. Model Configuration

Edit `model_config.yaml` to match your environment:

```yaml
catalog_name: your_catalog_name
database_name: your_database_name
volume_name: your_volume_name

# Update function names to match your catalog/database
functions:
  find_product_by_sku:
    name: your_catalog.your_database.find_product_by_sku
  # ... other functions
```

### 3. Vector Search Configuration (Optional)

If using vector search capabilities:

```yaml
vector_stores:
  products_vector_store:
    index_name: your_catalog.your_database.product_description_index
    endpoint_name: your-vector-search-endpoint
```

## Verification

### 1. Test Python Installation

```bash
python --version  # Should be 3.9+
pip list | grep -E "(databricks|langchain|mlflow)"
```

### 2. Test Databricks Connection

```python
from databricks.sdk import WorkspaceClient

try:
    w = WorkspaceClient()
    print("✅ Databricks connection successful")
    print(f"Workspace URL: {w.config.host}")
except Exception as e:
    print(f"❌ Databricks connection failed: {e}")
```

### 3. Test Unity Catalog Access

```python
from databricks.sdk import WorkspaceClient
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

try:
    w = WorkspaceClient()
    client = DatabricksFunctionClient(client=w)
    print("✅ Unity Catalog client initialized")
except Exception as e:
    print(f"❌ Unity Catalog access failed: {e}")
```

## Platform-Specific Instructions

### Windows

1. **Install Python from Microsoft Store or python.org**
2. **Use PowerShell or Command Prompt**
3. **Virtual Environment Activation**:
   ```cmd
   .venv\Scripts\activate
   ```

### macOS

1. **Install Python via Homebrew (recommended)**:
   ```bash
   brew install python@3.9
   ```
2. **Or use pyenv for version management**:
   ```bash
   brew install pyenv
   pyenv install 3.9.18
   pyenv global 3.9.18
   ```

### Linux (Ubuntu/Debian)

1. **Install Python and dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3.9 python3.9-venv python3.9-dev
   ```

2. **Install build tools if needed**:
   ```bash
   sudo apt install build-essential
   ```

## Troubleshooting Installation

### Common Issues

**Python version conflicts**
```bash
# Check Python version
python --version
python3 --version

# Use specific Python version
python3.9 -m venv .venv
```

**Permission errors on Windows**
```cmd
# Run as administrator or use:
pip install --user -r requirements.txt
```

**SSL certificate errors**
```bash
# Upgrade certificates
pip install --upgrade certifi
# Or use trusted hosts
pip install --trusted-host pypi.org --trusted-host pypi.python.org -r requirements.txt
```

**Memory errors during installation**
```bash
# Install packages one by one
pip install --no-cache-dir -r requirements.txt
```

**Databricks authentication issues**
```bash
# Check token validity
databricks auth login --host https://your-workspace.cloud.databricks.com

# Or set environment variables
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-token"
```

### Package-Specific Issues

**LangChain installation**
```bash
# If you encounter issues with LangChain
pip install --upgrade langchain langchain-community
```

**MLflow installation**
```bash
# For MLflow tracking issues
pip install --upgrade mlflow[extras]
```

**Databricks SDK issues**
```bash
# Ensure latest SDK version
pip install --upgrade databricks-sdk
```

## Next Steps

After successful installation:

1. **Configure your environment** - [Configuration Guide](configuration.md)
2. **Run the quick start** - [Quick Start Guide](quick-start.md)
3. **Set up Unity Catalog functions** - Run `python 04_unity_catalog_tools.py`
4. **Test the system** - Run `python 05a_run_examples.py`

## Getting Help

If you encounter installation issues:

1. **Check the troubleshooting section above**
2. **Review system requirements**
3. **Check GitHub issues** for similar problems
4. **Create a new issue** with:
   - Operating system and version
   - Python version
   - Error messages
   - Installation method used

## Uninstallation

To completely remove the installation:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf .venv

# Remove cloned repository
cd ..
rm -rf retail-ai
``` 