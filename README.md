# Retail AI Agent

A conversational AI system for retail operations built on Databricks, providing recommendation, inventory management, and product information capabilities through a unified agent architecture.

## Overview

This project implements a LangGraph-based conversational AI agent for retail operations that can:

- Answer questions about product inventory
- Provide product recommendations based on customer preferences
- Look up specific product details
- Answer general retail-related questions

The system uses Databricks Vector Search, Unity Catalog, and LLMs to provide accurate, context-aware responses.

## Architecture

### Overview

The Retail AI system is built as a sophisticated agent-based architecture that routes queries to specialized agents based on the nature of the request. This approach enables domain-specific handling while maintaining a unified interface.

### Core Components

#### Message Routing and Processing

- **Message Validation**: Validates incoming requests against required configuration parameters
- **Router Agent**: Analyzes user queries and routes them to the appropriate specialized agent
- **Factuality Check**: Ensures responses are factually accurate through iterative refinement

#### Specialized Agents

1. **General Agent**: Handles general inquiries about store policies and basic information
2. **Product Agent**: Provides detailed product specifications, availability, and compatibility
3. **Inventory Agent**: Offers real-time inventory checks and stock availability across locations
4. **Recommendation Agent**: Suggests products based on user preferences and purchase history
5. **Orders Agent**: Manages order status inquiries, tracking, and order history
6. **Comparison Agent**: Compares different products to help customers make informed decisions
7. **DIY Agent**: Offers project advice, tutorials, and step-by-step instructions for DIY projects

### Guardrails and Quality Control

- **Factuality Judge**: Evaluates responses for factual accuracy and triggers refinement when needed
- **Configuration Validation**: Ensures all required parameters are provided before processing
- **Retry Mechanism**: Implements intelligent retry logic when responses don't meet quality thresholds

### Technical Implementation

The system is implemented using:

- **LangGraph**: For workflow orchestration and state management
- **LangChain**: For LLM interactions and chain composition
- **MLflow**: For model deployment and serving
- **Databricks LLM APIs**: As the foundation models for natural language processing

The architecture follows a graph-based state machine pattern:

1. User messages enter through validation
2. Messages are routed by the router agent
3. Specialized agents process domain-specific requests
4. Responses undergo factuality checking
5. If needed, responses are refined until they meet quality thresholds

![Agent Architecture](docs/architecture.png)

## Prerequisites

- Python 3.12+
- Databricks workspace with access to:
  - Unity Catalog
  - Model Serving
  - Vector Search
  - Genie
- Databricks model endpoints:
  - LLM endpoint (default: databricks-meta-llama-3-3-70b-instruct)
  - Embedding model endpoint (default: databricks-gte-large-en)

## Setup

1. Clone this repository
2. Install dependencies:

```bash
# Create and activate a Python virtual environment 
uv venv
source  .venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
uv sync
```

## Configuration

Configuration is managed through `model_config.yaml`, which includes:

- Catalog and database information
- LLM model endpoints
- Vector search settings
- Genie space ID
- Application deployment details

## Notebooks

The project includes several notebooks for setup and execution:

1. `01_ingest-and-transform.py`: Ingests product data and prepares it for vector search
2. `02_provision-vector-search.py`: Creates vector search endpoint and index
3. `03_generate_evaluation_data.py`: Prepares evaluation data for the agent
4. `04_unity_catalog_tools.py`: Creates Unity Catalog functions
5. `05_agent_as_code_driver.py`: Deploys the agent as a model

## Development

### Project Structure

- `retail_ai/`: Core package containing agent implementation
  - `agents.py`: Agent implementation 
  - `catalog.py`: Unity Catalog integration
  - `graph.py`: LangGraph workflow definition
  - `models.py`: MLflow model integration
  - `nodes.py`: Agent node definitions
  - `tools.py`: Tool definitions for the agent
  - `vector_search.py`: Vector search utilities

### Building the Package

```bash
make dist
```

## Deployment

The agent can be deployed as a model endpoint using MLflow:

```python
# Register the model
mlflow.register_model(
    name="retail_ai_agent",
    model_uri=logged_agent_info.model_uri
)

# Deploy as an endpoint
agents.deploy(
  model_name="retail_ai_agent", 
  model_version=latest_version,
  endpoint_name="retail_ai_agent"
)
```

## Usage

Once deployed, the agent can be called with:

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.