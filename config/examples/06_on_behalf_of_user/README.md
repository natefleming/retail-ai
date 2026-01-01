# On-Behalf-Of (OBO) User Authentication

This directory demonstrates how to configure agents to use On-Behalf-Of (OBO) user authentication with Databricks resources.

## 📋 Overview

On-Behalf-Of (OBO) authentication allows your agents to access Databricks resources using the end user's credentials rather than a service principal. This provides:

- **Fine-grained access control**: Users can only access data they have permission to view
- **Audit trails**: Actions are attributed to individual users, not a service account
- **Compliance**: Meets data governance requirements for user-level tracking
- **Security**: No need to grant broad permissions to service principals

## 📁 Examples in This Directory

### `obo_basic.yaml`
A simple single-agent configuration demonstrating OBO with:
- **UC Functions**: Execute functions as the end user
- **Genie Spaces**: Query data with user's permissions
- **LLM Requests**: Track which user made which requests

**Key Configuration Points:**
```yaml
resources:
  llms:
    default_llm:
      on_behalf_of_user: true  # Enable OBO for LLM
  
  genie_rooms:
    retail_genie:
      on_behalf_of_user: true  # Enable OBO for Genie

tools:
  inventory_lookup:
    function:
      on_behalf_of_user: true  # Enable OBO for UC Functions
```

## 🔧 How OBO Works

### 1. Request Flow
```
User Request → Agent → Databricks Resource (as user)
                ↓
         User's Token Passed Through
```

### 2. Configuration Levels
OBO can be enabled at multiple levels:
- **Resource level**: LLMs, Genie rooms, tables, functions, etc.
- **Tool level**: Override resource settings for specific tools
- **Global level**: Default behavior for all resources

### 3. Token Propagation
When OBO is enabled:
- User's OAuth token is automatically propagated to Databricks APIs
- Resources are accessed with the user's permissions
- Failed requests return permission errors if user lacks access

## 🚀 Deployment Scenarios

### Databricks Apps
```python
# OBO is automatic - no additional configuration needed
# User tokens are passed through the Apps platform
dao-ai deploy -c obo_basic.yaml
```

### Model Serving
```python
# Pass user token in API request
import requests

response = requests.post(
    "https://<workspace>.cloud.databricks.com/serving-endpoints/<endpoint>/invocations",
    headers={
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json"
    },
    json={"messages": [{"role": "user", "content": "Find product SKU 12345"}]}
)
```

### Local Development
```bash
# Uses your personal credentials from databricks-cli
dao-ai chat -c config/examples/06_on_behalf_of_user/obo_basic.yaml
```

## 🔒 Access Control Setup

### Unity Catalog Permissions
Users need appropriate permissions on the resources they'll access:

```sql
-- Grant function execution permissions
GRANT EXECUTE ON FUNCTION catalog.schema.function_name TO user@example.com;

-- Grant schema usage permissions
GRANT USE CATALOG ON CATALOG catalog_name TO user@example.com;
GRANT USE SCHEMA ON SCHEMA catalog.schema TO user@example.com;

-- Grant table access (for Genie queries)
GRANT SELECT ON TABLE catalog.schema.table_name TO user@example.com;
```

### Genie Space Permissions
```sql
-- Grant Genie space access via Databricks UI or API
-- Users need "Can Use" or "Can Manage" permission on the Genie space
```

## 🧪 Testing OBO

### Test with Different Users
1. **Admin User**: Should have full access to all resources
2. **Read-Only User**: Should only access permitted tables
3. **Restricted User**: Should see permission errors for restricted data

### Verify OBO is Working
```python
# Check audit logs to confirm user attribution
# Databricks Audit Logs will show:
# - serviceName: "unityCatalog"
# - actionName: "execute" (for functions)
# - userIdentity: actual_user@example.com (not service_principal)
```

## 📊 Benefits vs. Service Principal Auth

| Feature | Service Principal | On-Behalf-Of User |
|---------|------------------|-------------------|
| Access Control | Broad permissions required | User's individual permissions |
| Audit Trail | Shows service principal | Shows actual end user |
| Compliance | May not meet requirements | Meets user-level tracking requirements |
| Multi-tenancy | Requires manual filtering | Automatic based on permissions |
| Setup Complexity | Simple | Moderate |
| Use Case | Internal tools, batch jobs | Production apps, user-facing tools |

## ⚠️ Important Considerations

### 1. Performance
- Each user's token is cached by Databricks
- No significant performance impact for most use cases
- Token refresh is handled automatically

### 2. Fallback Behavior
If OBO token is not available (e.g., local development), the system falls back to:
1. Service principal credentials (if configured)
2. Personal access token (if available)
3. Databricks CLI authentication

### 3. Compatibility
OBO works with:
- ✅ Unity Catalog Functions
- ✅ Genie Spaces
- ✅ SQL Warehouses
- ✅ Vector Search
- ✅ LLM Serving Endpoints
- ✅ Tables and Views

## 🔗 Related Examples

- **[01_getting_started](../01_getting_started/)**: Basic agent setup without OBO
- **[04_genie](../04_genie/)**: More Genie examples
- **[08_guardrails](../08_guardrails/)**: Combine OBO with guardrails for secure AI

## 📚 Additional Resources

- [Databricks OBO Documentation](https://docs.databricks.com/dev-tools/auth.html#on-behalf-of-authentication)
- [Unity Catalog Access Control](https://docs.databricks.com/data-governance/unity-catalog/manage-privileges/index.html)
- [Model Serving Authentication](https://docs.databricks.com/machine-learning/model-serving/authentication.html)
