# Quick Start Guide

This guide will help you get the Retail AI system up and running quickly.

## Prerequisites

- Python 3.9 or higher
- Access to a Databricks workspace
- Unity Catalog enabled
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/retail-ai.git
cd retail-ai
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file with your Databricks configuration:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-access-token
DATABRICKS_WAREHOUSE_ID=your-warehouse-id
```

### 4. Configure Model Settings

Edit `model_config.yaml` to match your environment:

```yaml
catalog_name: your_catalog
database_name: your_database
```

## Initial Setup

### 1. Create Unity Catalog Functions

Run the setup script to create all necessary Unity Catalog functions:

```bash
python 04_unity_catalog_tools.py
```

This will create the following functions:
- `find_product_by_sku`
- `find_product_by_upc`
- `find_inventory_by_sku`
- `find_inventory_by_upc`
- `find_store_inventory_by_sku`
- `find_store_inventory_by_upc`

### 2. Set Up Vector Search (Optional)

If you want to use semantic search capabilities:

```bash
python 02_provision-vector-search.py
```

### 3. Generate Test Data (Optional)

Create evaluation data for testing:

```bash
python 03_generate_evaluation_data.py
```

## Testing the Installation

### 1. Test Unity Catalog Functions

```python
from databricks.sdk import WorkspaceClient
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

# Initialize clients
w = WorkspaceClient()
client = DatabricksFunctionClient(client=w)

# Test a function
result = client.execute_function(
    function_name="your_catalog.your_database.find_product_by_sku",
    parameters={"sku": ["TEST-SKU-001"]}
)
print(result.value)
```

### 2. Test the Agent System

```python
python 05a_run_examples.py
```

### 3. Run the Streamlit App

```bash
cd streamlit_store_app
streamlit run app.py
```

## Basic Usage Examples

### Product Lookup

```python
from retail_ai.tools import create_find_product_by_sku_tool

# Create tool
tool = create_find_product_by_sku_tool(warehouse_id="your-warehouse-id")

# Use tool
result = tool.invoke({"skus": ["STB-KCP-001"]})
print(result)
```

### Product Comparison

```python
from retail_ai.tools import create_product_comparison_tool
from langchain_community.llms import Databricks

# Initialize LLM
llm = Databricks(endpoint_name="your-llm-endpoint")

# Create comparison tool
comparison_tool = create_product_comparison_tool(llm)

# Compare products
result = comparison_tool.invoke({
    "products": [
        {"product_id": "1", "name": "Product A", "price": 10.99},
        {"product_id": "2", "name": "Product B", "price": 12.99}
    ]
})
print(result)
```

### Vector Search

```python
from retail_ai.tools import find_product_details_by_description_tool

# Create search tool
search_tool = find_product_details_by_description_tool(
    endpoint_name="your-endpoint",
    index_name="your_catalog.your_database.product_description_index",
    columns=["sku", "product_name", "description"]
)

# Search for products
results = search_tool.invoke({
    "content": "medium roast coffee pods"
})
print(results)
```

## Next Steps

Now that you have the system running:

1. **Explore the Tools**: Check out the [Tools Reference](../agents-and-tools/overview.md) to understand all available capabilities
2. **Try Examples**: Run through the [Examples](../examples/basic-usage.md) to see common use cases
3. **Customize**: Follow the [Developer Guide](../development/setup.md) to add your own tools
4. **Deploy**: Use the [Deployment Guide](../development/deployment.md) to deploy to production

## Troubleshooting

### Common Issues

**Function not found error**
```
Solution: Ensure Unity Catalog functions are created and you have proper permissions
```

**Vector search errors**
```
Solution: Check that vector search index exists and endpoint is accessible
```

**Authentication errors**
```
Solution: Verify your Databricks token and workspace URL in .env file
```

**Import errors**
```
Solution: Ensure all dependencies are installed with pip install -r requirements.txt
```

### Getting Help

If you encounter issues:

1. Check the [troubleshooting section](../development/troubleshooting.md)
2. Review the logs for error details
3. Open an issue on GitHub with error details
4. Contact the development team

## Configuration Reference

For detailed configuration options, see:
- [Configuration Guide](configuration.md)
- [Model Configuration](../api/models.md)
- [Environment Variables](../development/environment.md) 