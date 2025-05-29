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

The Retail AI system is built as a component-based agent architecture that routes queries to specialized agents based on the nature of the request. This approach enables domain-specific handling while maintaining a unified interface.

[View Architecture Diagram](docs/architecture.png)

### Core Components

#### Configuration Components

All components are defined in `model_config.yaml` using a modular approach:

- **Schemas**: Define database and catalog structures
- **Resources**: Configure infrastructure components like LLMs, vector stores, catalogs, warehouses, and databases
- **Tools**: Define functions that agents can use to perform tasks (dictionary-based with keys as tool names)
- **Agents**: Specialized AI assistants configured for specific domains (dictionary-based with keys as agent names)
- **Guardrails**: Quality control mechanisms to ensure accurate responses
- **Retrievers**: Configuration for vector search and retrieval
- **Evaluation**: Configuration for model evaluation and testing
- **Datasets**: Configuration for training and evaluation datasets
- **App**: Overall application configuration including orchestration and logging

#### Message Processing Flow

The system uses a LangGraph-based workflow with the following key nodes:

- **Message Validation**: Validates incoming requests (`message_validation_node`)
- **Agent Routing**: Routes messages to appropriate specialized agents
- **Agent Execution**: Processes requests using specialized agents with their configured tools
- **Response Generation**: Returns structured responses to users

#### Specialized Agents

Agents are dynamically configured from the `model_config.yaml` file and can include:
- Custom LLM models and parameters
- Specific sets of available tools
- Domain-specific system prompts
- Guardrails for response quality

### Technical Implementation

The system is implemented using:

- **LangGraph**: For workflow orchestration and state management
- **LangChain**: For LLM interactions and tool integration
- **MLflow**: For model tracking and deployment
- **Databricks**: LLM APIs, Vector Search, Unity Catalog, and Model Serving
- **Pydantic**: For configuration validation and schema management

## Prerequisites

- Python 3.12+
- Databricks workspace with access to:
  - Unity Catalog
  - Model Serving
  - Vector Search
  - Genie (optional)
- Databricks CLI configured with appropriate permissions
- Databricks model endpoints for LLMs and embeddings

## Setup

1. Clone this repository
2. Install dependencies:

```bash
# Create and activate a Python virtual environment 
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies using Makefile
make install
```

## Configuration

Configuration is managed through `model_config.yaml`. This file defines all components of the Retail AI system, including resources, tools, agents, and the overall application setup.

### Basic Structure of `model_config.yaml`

The `model_config.yaml` is organized into several top-level keys:

```yaml
# filepath: /Users/nate/development/retail-ai/config/model_config.yaml
schemas:
  # ... schema definitions ...

resources:
  # ... resource definitions (LLMs, vector stores, etc.) ...

tools:
  # ... tool definitions ...

agents:
  # ... agent definitions ...

app:
  # ... application configuration ...

# Other sections like guardrails, retrievers, evaluation, datasets
```

### Developing and Configuring Tools

Tools are functions that agents can use to interact with external systems or perform specific tasks. They are defined under the `tools` key in `model_config.yaml`. The key for each tool entry serves as its unique name.

There are four types of tools supported:

#### 1. Python Tools (`type: python`)
   These tools directly map to Python functions. The `name` field should correspond to a function defined typically in `retail_ai/tools.py` or another importable module.

   **Configuration Example:**
   ```yaml
   # filepath: /Users/nate/development/retail-ai/config/model_config.yaml
   tools:
     get_product_details_by_id:
       function:
         type: python
         name: get_product_details_by_id # Corresponds to a Python function
         # Optional: schema for the function if it's a UC function or needs specific metadata
         schema: 
           function_name: get_product_details_by_id 
           # ... other schema details
   ```
   **Development:**
   Implement the Python function (e.g., `get_product_details_by_id`) in `retail_ai/tools.py`. This function will be called when the tool is invoked.

#### 2. Factory Tools (`type: factory`)
   Factory tools are used when a tool requires more complex initialization or setup. The `name` field points to a factory function (e.g., in `retail_ai/tools.py`) that returns an initialized LangChain `BaseTool` instance.

   **Configuration Example:**
   ```yaml
   # filepath: /Users/nate/development/retail-ai/config/model_config.yaml
   tools:
     product_vector_search:
       function:
         type: factory
         name: create_product_vector_search_tool # Corresponds to a factory function
         # Optional: schema for the function
         schema:
           function_name: create_product_vector_search_tool
   ```
   **Development:**
   Implement the factory function (e.g., `create_product_vector_search_tool`) in `retail_ai/tools.py`. This function should return a fully configured `BaseTool` object.

#### 3. Unity Catalog Tools (`type: unity_catalog`)
   These tools represent SQL functions registered in Databricks Unity Catalog. The agent can call these functions directly.

   **Configuration Example:**
   ```yaml
   # filepath: /Users/nate/development/retail-ai/config/model_config.yaml
   tools:
     check_stock_level:
       function:
         type: unity_catalog
         name: check_stock_level_uc # Name of the UC function
         schema: # UC schema details
           catalog_name: main
           schema_name: retail_data
           function_name: check_stock_level_uc
   ```
   **Development:**
   Create the corresponding SQL function (e.g., `main.retail_data.check_stock_level_uc`) in your Databricks Unity Catalog.

#### 4. MCP (Model Context Protocol) Tools (`type: mcp`)
   MCP tools allow interaction with external services or models that adhere to the Model Context Protocol.

   **Configuration Example:**
   ```yaml
   # filepath: /Users/nate/development/retail-ai/config/model_config.yaml
   tools:
     external_pricing_service:
       function:
         type: mcp
         name: get_latest_price # Name of the MCP function/endpoint
         # MCP-specific configuration might go here or be handled by the tool's implementation
         schema:
           function_name: get_latest_price
   ```
   **Development:**
   Ensure the MCP service is running and accessible. The tool implementation will handle communication with this service.

### Configuring New Agents

Agents are specialized AI assistants. They are defined under the `agents` key in `model_config.yaml`. The key for each agent entry serves as its unique name.

**Configuration Example:**
```yaml
# filepath: /Users/nate/development/retail-ai/config/model_config.yaml
agents:
  product_inquiry_agent: # Unique name for the agent
    name: "Product Inquiry Agent" # Display name
    description: "Handles specific questions about product features, availability, and price."
    model: *llama3_70b_instruct # Reference to an LLM defined in resources
    tools: # List of tool names (keys from the 'tools' section) this agent can use
      - get_product_details_by_id
      - check_stock_level
      - product_vector_search 
    system_prompt: |
      You are a helpful assistant specializing in product information.
      Use the available tools to answer questions accurately.
    # Optional: guardrails specific to this agent
    guardrails:
      - ensure_politeness 
```
**To configure a new agent:**
1.  Add a new entry under the `agents` section with a unique key (e.g., `product_inquiry_agent`).
2.  Define its `name`, `description`, `model` (referencing an LLM from `resources` using a YAML anchor like `*tool_calling_llm`), and `system_prompt`.
3.  List the tools (by their keys from the `tools` section, e.g., `get_product_details_by_id`) that this agent is allowed to use under its `tools` array.

### Assigning Tools to Agents

As shown in the agent configuration example above, tools are assigned to an agent by listing their unique names (the keys used in the `tools` section) within the `tools` array of that agent's definition.

```yaml
# filepath: /Users/nate/development/retail-ai/config/model_config.yaml
agents:
  some_agent:
    # ... other agent properties ...
    tools:
      - tool_key_1 # Defined under tools.tool_key_1
      - tool_key_2 # Defined under tools.tool_key_2
```

### Assigning Agents to the Application and Configuring Orchestration

Agents are made available to the application by listing their YAML anchors (defined in the `agents:` section) within the `agents` array under the `app` section. The `app.orchestration` section defines how these agents interact.

**Orchestration Configuration:**

The `orchestration` block within the `app` section allows you to define the interaction pattern. Your current configuration primarily uses a **Supervisor** pattern.

```yaml
# filepath: /Users/nate/development/retail-ai/config/model_config.yaml
# ...
# app:
#   ...
#   agents:
#     - *orders
#     - *diy
#     - *product
#     # ... other agents referenced by their anchors
#     - *general
#   orchestration:
#     supervisor:
#       model: *tool_calling_llm # LLM for the supervisor agent
#       default_agent: *general   # Agent to handle tasks if no specific agent is chosen
#     # swarm: # Example of how a swarm might be configured if activated
#     #   model: *tool_calling_llm
# ...
```

**Orchestration Patterns:**

1.  **Supervisor Pattern (Currently Active)**
    *   Your configuration defines a `supervisor` block under `app.orchestration`.
    *   `model`: Specifies the LLM (e.g., `*tool_calling_llm`) that the supervisor itself will use for its decision-making and routing logic.
    *   `default_agent`: Specifies an agent (e.g., `*general`) that the supervisor will delegate to if it cannot determine a more specialized agent from the `app.agents` list or if the query is general.
    *   The supervisor is responsible for receiving the initial user query, deciding which specialized agent (from the `app.agents` list) is best suited to handle it, and then passing the query to that agent. If no specific agent is a clear match, or if the query is general, it falls back to the `default_agent`.

2.  **Swarm Pattern (Commented Out)**
    *   Your configuration includes a commented-out `swarm` block. If activated, this would imply a different interaction model.
    *   In a swarm, agents might collaborate more directly or work in parallel on different aspects of a query. The `model` under `swarm` would likely define the LLM used by the agents within the swarm or by a coordinating element of the swarm.
    *   The specific implementation of how a swarm pattern behaves would be defined in your `retail_ai/graph.py` and `retail_ai/nodes.py`.


## Development

### Project Structure

- `retail_ai/`: Core package
  - `config.py`: Pydantic configuration models with full validation
  - `graph.py`: LangGraph workflow definition
  - `nodes.py`: Agent node factories and implementations
  - `tools.py`: Tool creation and factory functions, implementations for Python tools
  - `vector_search.py`: Vector search utilities
  - `state.py`: State management for conversations
- `tests/`: Test suite with configuration fixtures
- `schemas/`: JSON schemas for configuration validation
- `notebooks/`: Jupyter notebooks for setup and experimentation
- `docs/`: Documentation files, including architecture diagrams.
- `config/`: Contains `model_config.yaml`.

### Building the Package

```bash
# Install development dependencies
make install-dev

# Build the package
make build

# Run tests
make test

# Run linting
make lint

# Format code
make format
```

## Deployment with Databricks Bundle CLI

The agent can be deployed using the existing Databricks Bundle CLI configuration:

1. Ensure Databricks CLI is installed and configured:
   ```bash
   pip install databricks-cli
   databricks configure
   ```

2. Deploy using the existing `databricks.yml`:
   ```bash
   databricks bundle deploy
   ```

3. Check deployment status:
   ```bash
   databricks bundle status
   ```

## Usage

Once deployed, interact with the agent:

```python
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
response = client.predict(
  endpoint="retail_ai_agent", # Matches endpoint_name in model_config.yaml
  inputs={
    "messages": [
      {"role": "user", "content": "Can you recommend a lamp for my oak side tables?"}
    ]
  }
)

print(response["message"]["content"])
```

## Customization

To customize the agent:

1. **Update `config/model_config.yaml`**:
   - Add tools in the `tools` section
   - Create agents in the `agents` section
   - Configure resources (LLMs, vector stores, etc.)
   - Adjust orchestration patterns as described above.

2. **Implement new tools** in `retail_ai/tools.py` (for Python and Factory tools) or in Unity Catalog (for UC tools).

3. **Extend workflows** in `retail_ai/graph.py` to support the chosen orchestration patterns and agent interactions.

## Testing

```bash
# Run all tests
make test

# Run specific tests
make test ARGS="tests/test_config.py"

# Run with coverage
make test-cov
```

## Logging

The primary log level for the application is configured in `config/model_config.yaml` under the `app.log_level` field.

**Configuration Example:**
```yaml
# filepath: /Users/nate/development/retail-ai/config/model_config.yaml
app:
  log_level: INFO  # Supported levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
  # ... other app configurations ...
```

This setting controls the verbosity of logs produced by the `retail_ai` package.

The system also includes:
- **MLflow tracing** for request tracking.
- **Structured logging** is used internally.

To temporarily override logging levels in your development environment for specific modules, you can use standard Python logging:
```python
import logging
# Example: Set retail_ai logger to DEBUG
logging.getLogger("retail_ai").setLevel(logging.DEBUG) 
# Example: Set a specific module's logger to DEBUG
logging.getLogger("retail_ai.tools").setLevel(logging.DEBUG)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.