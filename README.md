# Multi-Agent AI Orchestration Framework

A modular, multi-agent orchestration framework for building sophisticated AI workflows on Databricks. While this implementation provides a complete retail AI reference architecture, the framework is designed to support any domain or use case requiring agent coordination, tool integration, and dynamic configuration.

## Overview

This project implements a LangGraph-based multi-agent orchestration framework that can:

- **Route queries** to specialized agents based on content and context
- **Coordinate multiple AI agents** working together on complex tasks
- **Integrate diverse tools** including databases, APIs, vector search, and external services
- **Support flexible orchestration patterns** (supervisor, swarm, and custom workflows)
- **Provide dynamic configuration** through YAML-based agent and tool definitions
- **Enable domain-specific specialization** while maintaining a unified interface

**Retail Reference Implementation**: This repository includes a complete retail AI system demonstrating:
- Product inventory management and search
- Customer recommendation engines  
- Order tracking and management
- Product classification and information retrieval

The system uses Databricks Vector Search, Unity Catalog, and LLMs to provide accurate, context-aware responses across any domain.

## Architecture

### Overview

The Multi-Agent AI system is built as a component-based agent architecture that routes queries to specialized agents based on the nature of the request. This approach enables domain-specific handling while maintaining a unified interface that can be adapted to any industry or use case.

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
- **Agent Routing**: Routes messages to appropriate specialized agents using supervisor or swarm patterns
- **Agent Execution**: Processes requests using specialized agents with their configured tools
- **Response Generation**: Returns structured responses to users

#### Specialized Agents

Agents are dynamically configured from the `model_config.yaml` file and can include:
- Custom LLM models and parameters
- Specific sets of available tools (Python functions, Unity Catalog functions, factory tools, MCP services)
- Domain-specific system prompts
- Guardrails for response quality
- Handoff prompts for agent coordination

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

## Command Line Interface

The framework includes a comprehensive CLI for managing, validating, and visualizing your multi-agent system:

### Schema Generation
Generate JSON schema for configuration validation and IDE autocompletion:
```bash
retail-ai schema > schema.json
```

### Configuration Validation
Validate your configuration file for syntax and semantic correctness:
```bash
# Validate default configuration
retail-ai validate

# Validate specific configuration file
retail-ai validate -c config/production.yaml
```

### Graph Visualization
Generate visual representations of your agent workflow:
```bash
# Generate architecture diagram
retail-ai graph -o architecture.png

# Generate diagram from specific config
retail-ai graph -o workflow.png -c config/custom.yaml
```

### Deployment
Deploy your multi-agent system using Databricks Asset Bundles:
```bash
# Deploy the system
retail-ai bundle --deploy

# Run the deployed system
retail-ai bundle --run

# Use specific Databricks profile
retail-ai bundle --deploy --run --profile my-profile
```

### Verbose Output
Add `-v`, `-vv`, `-vvv`, or `-vvvv` flags for increasing levels of verbosity (ERROR, WARNING, INFO, DEBUG, TRACE).

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

Tools are functions that agents can use to interact with external systems or perform specific tasks. They are defined under the `tools` key in `model_config.yaml`. Each tool has a unique name and contains a `function` specification.

There are four types of tools supported:

#### 1. Python Tools (`type: python`)
   These tools directly map to Python functions. The `name` field should correspond to a function that can be imported and called directly.

   **Configuration Example:**
   ```yaml
   tools:
     my_python_tool:
       name: my_python_tool
       function:
         type: python
         name: retail_ai.tools.my_function_name
         schema: # Optional schema definition
           name: my_function_name
   ```
   **Development:**
   Implement the Python function in the specified module (e.g., `retail_ai/tools.py`). The function will be imported and called directly when the tool is invoked.

#### 2. Factory Tools (`type: factory`)
   Factory tools use factory functions that return initialized LangChain `BaseTool` instances. This is useful for tools requiring complex initialization or configuration.

   **Configuration Example:**
   ```yaml
   tools:
     vector_search_tool:
       name: vector_search
       function:
         type: factory
         name: retail_ai.tools.create_vector_search_tool
         args:
           retriever: *products_retriever
           name: product_vector_search_tool
           description: "Search for products using vector search"
   ```
   **Development:**
   Implement the factory function (e.g., `create_vector_search_tool`) in `retail_ai/tools.py`. This function should accept the specified `args` and return a fully configured `BaseTool` object.

#### 3. Unity Catalog Tools (`type: unity_catalog`)
   These tools represent SQL functions registered in Databricks Unity Catalog. They reference functions by their Unity Catalog schema and name.

   **Configuration Example:**
   ```yaml
   tools:
     find_product_by_sku_uc_tool:
       name: find_product_by_sku_uc
       function:
         type: unity_catalog
         name: find_product_by_sku
         schema:
           name: retail_schema
   ```
   **Development:**
   Create the corresponding SQL function in your Databricks Unity Catalog using the specified schema and function name. The tool will automatically generate the appropriate function signature and documentation.

#### 4. MCP (Model Context Protocol) Tools (`type: mcp`)
   MCP tools allow interaction with external services that implement the Model Context Protocol, supporting both HTTP and stdio transports.

   **Configuration Example:**
   ```yaml
   tools:
     weather_tool_mcp:
       name: weather
       function:
         type: mcp
         name: weather
         transport: streamable_http
         url: http://localhost:8000/mcp
   ```
   **Development:**
   Ensure the MCP service is running and accessible at the specified URL or command. The framework will handle the MCP protocol communication automatically.

### Configuring New Agents

Agents are specialized AI assistants defined under the `agents` key in `model_config.yaml`. Each agent has a unique name and specific configuration.

**Configuration Example:**
```yaml
agents:
  general:
    name: general
    description: "General retail store assistant for home improvement and hardware store inquiries"
    model: *tool_calling_llm
    tools:
      - *find_product_details_by_description_tool
      - *vector_search_tool
    guardrails: []
    checkpointer: *checkpointer
    prompt: |
      You are a helpful retail store assistant for a home improvement and hardware store.
      You have access to search tools to find current information about products, pricing, and store policies.
      
      #### CRITICAL INSTRUCTION: ALWAYS USE SEARCH TOOLS FIRST
      Before answering ANY question:
      - ALWAYS use your available search tools to find the most current and accurate information
      - Search for specific details about store policies, product availability, pricing, and services
```

**Agent Configuration Fields:**
- `name`: Unique identifier for the agent
- `description`: Human-readable description of the agent's purpose
- `model`: Reference to an LLM model (using YAML anchors like `*tool_calling_llm`)
- `tools`: Array of tool references (using YAML anchors like `*search_tool`)
- `guardrails`: Array of guardrail references (can be empty `[]`)
- `checkpointer`: Reference to a checkpointer for conversation state (optional)
- `prompt`: System prompt that defines the agent's behavior and instructions

**To configure a new agent:**
1. Add a new entry under the `agents` section with a unique key
2. Define the required fields: `name`, `description`, `model`, `tools`, and `prompt`
3. Optionally configure `guardrails` and `checkpointer`
4. Reference the agent in the application configuration using YAML anchors

### Assigning Tools to Agents

Tools are assigned to agents by referencing them using YAML anchors in the agent's `tools` array. Each tool must be defined in the `tools` section with an anchor (using `&tool_name`), then referenced in the agent configuration (using `*tool_name`).

**Example:**
```yaml
tools:
  search_tool: &search_tool
    name: search
    function:
      type: factory
      name: retail_ai.tools.search_tool
      args: {}

  genie_tool: &genie_tool
    name: genie
    function:
      type: factory
      name: retail_ai.tools.create_genie_tool
      args:
        genie_room: *retail_genie_room

agents:
  general:
    name: general
    description: "General retail store assistant"
    model: *tool_calling_llm
    tools:
      - *search_tool    # Reference to the search_tool anchor
      - *genie_tool     # Reference to the genie_tool anchor
    # ... other agent configuration
```

This YAML anchor system allows for:
- **Reusability**: The same tool can be assigned to multiple agents
- **Maintainability**: Tool configuration is centralized in one place
- **Consistency**: Tools are guaranteed to have the same configuration across agents

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