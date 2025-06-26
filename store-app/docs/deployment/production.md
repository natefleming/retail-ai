# Production Deployment

This guide covers deploying the Retail AI system to production environments.

## 📋 Prerequisites

Before deploying to production, ensure you have:

- **Python 3.12+**: Required runtime environment
- **Databricks Workspace**: With appropriate permissions and resources
- **Unity Catalog**: Enabled and configured
- **Model Endpoints**: Access to required LLM and embedding endpoints

### Required Databricks Resources

- **Unity Catalog**: Data governance and function management
- **Model Serving**: For hosting LLM endpoints
- **Vector Search**: For semantic search capabilities
- **Genie**: Natural language to SQL conversion
- **SQL Warehouse**: For query execution

### Default Model Endpoints

- **LLM Endpoint**: `databricks-meta-llama-3-3-70b-instruct`
- **Embedding Model**: `databricks-gte-large-en`

## 🚀 Deployment Process

### Step 1: Environment Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/retail-ai.git
   cd retail-ai
   ```

2. **Create Virtual Environment**
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   uv sync
   ```

3. **Configure Environment**
   ```bash
   # Copy configuration template
   cp model_config.yaml.template model_config.yaml
   
   # Update with your Databricks workspace details
   # See Configuration section below
   ```

### Step 2: Data Setup

Run the data preparation notebooks in order:

```bash
# 1. Ingest and transform data
python 01_ingest-and-transform.py

# 2. Provision vector search
python 02_provision-vector-search.py

# 3. Generate evaluation data (optional)
python 03_generate_evaluation_data.py
```

### Step 3: Model Development and Registration

```bash
# Develop, log, and register the model
python 05_agent_as_code_driver.py
```

This notebook will:
- Build the agent graph
- Log the model to MLflow
- Register the model in the MLflow Model Registry

### Step 4: Model Evaluation

```bash
# Run formal evaluation
python 06_evaluate_agent.py
```

This provides:
- Performance metrics
- Quality assessments
- Evaluation reports

### Step 5: Production Deployment

```bash
# Deploy to production
python 07_deploy_agent.py
```

This notebook handles:
- Model alias management (Champion)
- Endpoint deployment
- Permissions configuration

## ⚙️ Configuration

Configuration is managed through `model_config.yaml`. Key sections include:

### Catalog and Database

```yaml
catalog_name: "your_catalog"
database_name: "your_database"
```

### Model Endpoints

```yaml
resources:
  endpoints:
    llm_endpoint: "databricks-meta-llama-3-3-70b-instruct"
    embedding_endpoint: "databricks-gte-large-en"
```

### Vector Search

```yaml
vector_search:
  endpoint_name: "your_vector_search_endpoint"
  index_name: "your_vector_index"
```

### Genie Configuration

```yaml
genie:
  space_id: "your_genie_space_id"
```

### Application Settings

```yaml
application:
  name: "retail_ai_agent"
  version: "1.0.0"
  description: "Retail AI conversational agent"
```

## 🔧 Production Usage

### REST API Endpoint

Once deployed, the agent can be called via REST API:

```python
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
response = client.predict(
    endpoint="retail_ai_agent",
    inputs={
        "messages": [
            {"role": "user", "content": "Can you recommend a lamp to match my oak side tables?"}
        ],
        "custom_inputs": {
            "configurable": {
                "thread_id": "1",
                "tone": "friendly"
            }
        }
    }
)
```

### Streamlit Application

Deploy the store management interface:

```bash
cd streamlit_store_app
streamlit run app.py
```

## 📊 Monitoring and Observability

### MLflow Tracking

Enable comprehensive tracking:

```python
import mlflow
mlflow.set_tracking_uri("databricks")

# View traces and metrics in MLflow UI
```

### Debug Logging

Enable debug logging for troubleshooting:

```python
import logging
logging.getLogger("retail_ai").setLevel(logging.DEBUG)
```

### Performance Monitoring

Monitor key metrics:
- **Response Time**: End-to-end latency
- **Accuracy**: Response quality scores
- **Usage**: Request volume and patterns
- **Errors**: Error rates and types

## 🔒 Security Considerations

### Access Control

- **Unity Catalog Permissions**: Ensure proper data access controls
- **Model Serving Permissions**: Restrict endpoint access
- **API Authentication**: Implement proper authentication
- **Network Security**: Configure VPC and firewall rules

### Data Privacy

- **PII Handling**: Implement data anonymization
- **Audit Logging**: Enable comprehensive audit trails
- **Encryption**: Ensure data encryption at rest and in transit
- **Compliance**: Meet regulatory requirements

## 🚨 Troubleshooting

### Common Issues

1. **Tool Not Found**
   - Verify tool registration in agent configuration
   - Check Unity Catalog function permissions

2. **Type Errors**
   - Validate Pydantic model definitions
   - Check field types and constraints

3. **Database Errors**
   - Verify Unity Catalog permissions
   - Check function names and schemas

4. **Vector Search Issues**
   - Verify endpoint status
   - Check index configuration and permissions

### Debug Steps

1. **Check Configuration**
   ```bash
   # Validate model_config.yaml
   python -c "import yaml; print(yaml.safe_load(open('model_config.yaml')))"
   ```

2. **Test Endpoints**
   ```python
   # Test LLM endpoint
   from databricks.sdk import WorkspaceClient
   w = WorkspaceClient()
   # Test endpoint connectivity
   ```

3. **Verify Permissions**
   ```sql
   -- Check Unity Catalog permissions
   SHOW GRANTS ON CATALOG your_catalog;
   SHOW GRANTS ON SCHEMA your_catalog.your_schema;
   ```

## 📈 Scaling Considerations

### Performance Optimization

- **Endpoint Scaling**: Configure auto-scaling for model endpoints
- **Caching**: Implement response caching for common queries
- **Load Balancing**: Distribute traffic across multiple endpoints
- **Resource Allocation**: Optimize compute resources

### Cost Optimization

- **Endpoint Management**: Scale down during low usage periods
- **Query Optimization**: Optimize SQL queries and vector searches
- **Resource Monitoring**: Track and optimize resource usage
- **Cost Alerts**: Set up cost monitoring and alerts

## 🔄 Continuous Deployment

### CI/CD Pipeline

Set up automated deployment:

```yaml
# .github/workflows/deploy.yml
name: Deploy Retail AI
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Databricks
        run: |
          # Run deployment scripts
          python 07_deploy_agent.py
```

### Model Updates

For model updates:

1. **Development**: Update model in development environment
2. **Testing**: Run evaluation and quality checks
3. **Staging**: Deploy to staging environment
4. **Production**: Promote to production with alias update

This deployment guide ensures a robust, secure, and scalable production deployment of the Retail AI system. 