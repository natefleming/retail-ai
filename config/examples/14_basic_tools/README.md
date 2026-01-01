# 14. Basic Tools

**Simple tool integrations for common data operations**

This category demonstrates straightforward single-agent applications using basic tool integrations like SQL execution, file operations, and other utility tools.

## Examples

| File | Description | Prerequisites |
|------|-------------|---------------|
| `sql_tool_example.yaml` | SQL execution tools for inventory analysis | Databricks SQL warehouse, hardware store tables |

## What You'll Learn

- **SQL execution tools** - Pre-configured queries against Databricks SQL warehouses
- **Single-agent patterns** - Simple agent configurations with multiple tools
- **Data analysis** - Querying and analyzing structured data
- **Tool factories** - Creating tools with factory functions

## Quick Start

### Test SQL tool example
```bash
# Set your warehouse ID
export DATABRICKS_WAREHOUSE_ID="your-warehouse-id"

# Make sure the hardware store tables exist (see data/hardware_store/)
# Then chat with the inventory analyst
dao-ai chat -c config/examples/14_basic_tools/sql_tool_example.yaml
```

Example questions:
- *"How many products are in the catalog?"*
- *"Show me products by department"*
- *"What items have low inventory?"*
- *"Which popular products should we focus on?"*

## Tool Patterns

### SQL Execution Tools
Pre-configured SQL statements executed by the agent:
- **Fixed queries**: SQL is defined at configuration time, not by the LLM
- **Warehouse execution**: Runs against Databricks SQL warehouses
- **Structured results**: Returns formatted table data to the agent

```yaml
tools:
  my_sql_tool:
    name: my_sql_tool
    function:
      type: factory
      name: dao_ai.tools.sql.create_execute_statement_tool
      args:
        warehouse: *my_warehouse
        statement: |
          SELECT * FROM catalog.schema.table
          WHERE condition = true
        description: "Description of what this query does"
```

## Prerequisites

### For SQL Tool Example
- Databricks SQL warehouse (serverless or provisioned)
- Unity Catalog with `retail_consumer_goods.hardware_store` schema
- Tables: `products` and `inventory`
- Data loaded from `data/hardware_store/` directory

To set up the hardware store data, run the SQL scripts in:
```
data/hardware_store/products.sql
data/hardware_store/inventory.sql
```

## Security Best Practices

🔒 **Warehouse access** - Ensure agents only have access to appropriate warehouses

**Best practices:**
- Use service principals for production deployments
- Grant minimum required permissions on Unity Catalog tables
- Use SQL execution tools (pre-configured queries) instead of dynamic SQL generation
- Monitor query execution and costs

**Example warehouse configuration:**
```yaml
warehouses:
  analytics_warehouse:
    name: analytics_warehouse
    warehouse_id:
      env: DATABRICKS_WAREHOUSE_ID  # Store ID in environment variable
```

## Next Steps

After mastering basic tools:

👉 **02_mcp/** - Integrate with external services via MCP  
👉 **04_genie/** - Use Genie for natural language to SQL  
👉 **15_complete_applications/** - Build complex multi-agent systems

## Troubleshooting

**"Warehouse not found"**
- Verify `DATABRICKS_WAREHOUSE_ID` is set correctly
- Ensure the warehouse exists and is running
- Check you have access to the warehouse

**"Table or view not found"**
- Confirm Unity Catalog and schema names are correct
- Verify tables exist: `SHOW TABLES IN catalog.schema`
- Check you have SELECT permissions on the tables

**"Query execution timeout"**
- Increase warehouse size for complex queries
- Optimize SQL queries (add WHERE clauses, LIMIT results)
- Check for table statistics: `ANALYZE TABLE catalog.schema.table COMPUTE STATISTICS`

## Related Documentation

- [Tool Development Guide](../../../docs/contributing.md#adding-a-new-tool)
- [SQL Tool Implementation](../../../src/dao_ai/tools/sql.py)
- [Configuration Reference](../../../docs/configuration-reference.md)
