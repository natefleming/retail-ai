# Declarative Agent Orchestration (DAO) Framework

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

## Key Features

- **Multi-Modal Interface**: CLI commands and Python API for development and deployment
- **Agent Lifecycle Management**: Create, deploy, and monitor agents programmatically
- **Vector Search Integration**: Built-in support for Databricks Vector Search with retrieval tools
- **Configuration-Driven**: YAML-based configuration with validation and IDE support
- **MLflow Integration**: Automatic model packaging, versioning, and deployment
- **Monitoring & Evaluation**: Built-in assessment and monitoring capabilities

## Architecture

### Overview

The Multi-Agent AI system is built as a component-based agent architecture that routes queries to specialized agents based on the nature of the request. This approach enables domain-specific handling while maintaining a unified interface that can be adapted to any industry or use case.

![View Architecture Diagram](./docs/hardware_store/retail_supervisor.png)

### Core Components

#### Configuration Components

All components are defined from the provided [`model_config.yaml`](config/hardware_store/supervisor_postgres.yaml) using a modular approach:

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

Agents are dynamically configured from the provided [`model_config.yaml`](config/hardware_store/supervisor_postgres.yaml) file and can include:
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

3. Configure Databricks CLI with appropriate workspace access

## Quick Start

### Option 1: Using Python API (Recommended for Development)

```python
from retail_ai.config import AppConfig

# Load your configuration
config = AppConfig.from_file("config/hardware_store/supervisor_postgres.yaml")

# Create vector search infrastructure
for name, vector_store in config.resources.vector_stores.items():
    vector_store.create()

# Create and deploy your agent
config.create_agent()
config.deploy_agent()

```

### Option 2: Using CLI Commands

```bash
# Validate configuration
dao-ai validate -c config/hardware_store/supervisor_postgres.yaml

# Generate workflow diagram
dao-ai graph -o architecture.png

# Deploy using Databricks Asset Bundles
dao-ai bundle --deploy --run

# Deploy using Databricks Asset Bundles with specific configuration
dao-ai -vvvv bundle --deploy --run --target dev --config config/hardware_store/supervisor_postgres.yaml --profile DEFAULT
```

See the [Python API](#python-api) section for detailed programmatic usage, or [Command Line Interface](#command-line-interface) for CLI usage.

## Command Line Interface

The framework includes a comprehensive CLI for managing, validating, and visualizing your multi-agent system:

### Schema Generation
Generate JSON schema for configuration validation and IDE autocompletion:
```bash
dao-ai schema > schema.json
```

### Configuration Validation
Validate your configuration file for syntax and semantic correctness:
```bash
# Validate default configuration (config/hardware_store/supervisor_postgres.yaml)
dao-ai validate

# Validate specific configuration file
dao-ai validate -c config/production.yaml
```

### Graph Visualization
Generate visual representations of your agent workflow:
```bash
# Generate architecture diagram (using default config/hardware_store/supervisor_postgres.yaml)
dao-ai graph -o architecture.png

# Generate diagram from specific config
dao-ai graph -o workflow.png -c config/custom.yaml
```

### Deployment
Deploy your multi-agent system using Databricks Asset Bundles:
```bash
# Deploy the system
dao-ai bundle --deploy

# Run the deployed system
dao-ai bundle --run

# Use specific Databricks profile
dao-ai bundle --deploy --run --profile my-profile
```

### Verbose Output
Add `-v`, `-vv`, `-vvv`, or `-vvvv` flags for increasing levels of verbosity (ERROR, WARNING, INFO, DEBUG, TRACE).

## Python API

The framework provides a comprehensive Python API for programmatic access to all functionality. The main entry point is the `AppConfig` class, which provides methods for agent lifecycle management, vector search operations, and configuration utilities.

### Quick Start

```python
from retail_ai.config import AppConfig

# Load configuration from file
config = AppConfig.from_file(path="config/hardware_store/supervisor_postgres.yaml")
```

### Agent Lifecycle Management

#### Creating Agents
Package and register your multi-agent system as an MLflow model:

```python
# Create agent with default settings
config.create_agent()

# Create agent with additional requirements and code paths
config.create_agent(
    additional_pip_reqs=["custom-package==1.0.0"],
    additional_code_paths=["./custom_modules"]
)
```

#### Deploying Agents
Deploy your registered agent to a Databricks serving endpoint:

```python
# Deploy agent to serving endpoint
config.deploy_agent()
```

The deployment process:
1. Retrieves the latest model version from MLflow
2. Creates or updates a Databricks model serving endpoint
3. Configures scaling, environment variables, and permissions
4. Sets up proper authentication and resource access

### Vector Search Operations

#### Creating Vector Search Infrastructure
Create vector search endpoints and indexes from your configuration:

```python
# Access vector stores from configuration
vector_stores = config.resources.vector_stores

# Create all vector stores
for name, vector_store in vector_stores.items():
    print(f"Creating vector store: {name}")
    vector_store.create()
```

#### Using Vector Search
Query your vector search indexes for retrieval-augmented generation:

```python
# Method 1: Direct index access
from retail_ai.config import RetrieverModel

question = "What products do you have in stock?"

for name, retriever in config.retrievers.items():
    # Get the vector search index
    index = retriever.vector_store.as_index()
    
    # Perform similarity search
    results = index.similarity_search(
        query_text=question,
        columns=retriever.columns,
        **retriever.search_parameters.model_dump()
    )
    
    chunks = results.get('result', {}).get('data_array', [])
    print(f"Found {len(chunks)} relevant results")
```

```python
# Method 2: LangChain integration
from databricks_langchain import DatabricksVectorSearch

for name, retriever in config.retrievers.items():
    # Create LangChain vector store
    vector_search = DatabricksVectorSearch(
        endpoint=retriever.vector_store.endpoint.name,
        index_name=retriever.vector_store.index.full_name,
        columns=retriever.columns,
    )
    
    # Search using LangChain interface
    documents = vector_search.similarity_search(
        query=question,
        **retriever.search_parameters.model_dump()
    )
    
    print(f"Found {len(documents)} documents")
```

### Configuration Utilities

The `AppConfig` class provides helper methods to find and filter configuration components:

#### Finding Agents
```python
# Get all agents
all_agents = config.find_agents()

# Find agents with specific criteria
def has_vector_search(agent):
    return any("vector_search" in tool.name.lower() for tool in agent.tools)

vector_agents = config.find_agents(predicate=has_vector_search)
```

#### Finding Tools and Guardrails
```python
# Get all tools
all_tools = config.find_tools()

# Get all guardrails
all_guardrails = config.find_guardrails()

# Find tools by type
def is_python_tool(tool):
    return tool.function.type == "python"

python_tools = config.find_tools(predicate=is_python_tool)
```

### Visualization

Generate and save workflow diagrams:

```python
# Display graph in notebook
config.display_graph()

# Save architecture diagram
config.save_image("docs/my_architecture.png")
```

### Complete Example

See [`notebooks/05_agent_as_code_driver.py`](notebooks/05_agent_as_code_driver.py) for a complete example:

```python
from retail_ai.config import AppConfig
from pathlib import Path

# Load configuration
config = AppConfig.from_file("config/hardware_store/supervisor_postgres.yaml")

# Visualize the workflow
config.display_graph()

# Save architecture diagram
path = Path("docs") / f"{config.app.name}_architecture.png"
config.save_image(path)

# Create and deploy the agent
config.create_agent()
config.deploy_agent()
```

For vector search examples, see [`notebooks/02_provision_vector_search.py`](notebooks/02_provision_vector_search.py).

### Available Notebooks

The framework includes several example notebooks demonstrating different aspects:

| Notebook | Description | Key Methods Demonstrated |
|----------|-------------|-------------------------|
| [`01_ingest_and_transform.py`](notebooks/01_ingest_and_transform.py) | Data ingestion and transformation | Dataset creation and SQL execution |
| [`02_provision_vector_search.py`](notebooks/02_provision_vector_search.py) | Vector search setup and usage | `vector_store.create()`, `as_index()` |
| [`03_generate_evaluation_data.py`](notebooks/03_generate_evaluation_data.py) | Generate synthetic evaluation datasets | Data generation and evaluation setup |
| [`04_unity_catalog_tools.py`](notebooks/04_unity_catalog_tools.py) | Unity Catalog function deployment | SQL function creation and testing |
| [`05_agent_as_code_driver.py`](notebooks/05_agent_as_code_driver.py) | **Complete agent lifecycle** | `create_agent()`, `deploy_agent()` |
| [`06_run_evaluation.py`](notebooks/06_run_evaluation.py) | Agent evaluation and testing | Evaluation framework usage |
| [`08_run_examples.py`](notebooks/08_run_examples.py) | End-to-end example queries | Agent interaction and testing |

## Configuration

Configuration is managed through [`model_config.yaml`](config/hardware_store/supervisor_postgres.yaml). This file defines all components of the Retail AI system, including resources, tools, agents, and the overall application setup.

**Note**: The configuration file location is configurable throughout the framework. You can specify a different configuration file using the `-c` or `--config` flag in CLI commands, or by setting the appropriate parameters in the Python API.

### Basic Structure of [`model_config.yaml`](config/hardware_store/supervisor_postgres.yaml)

The [`model_config.yaml`](config/hardware_store/supervisor_postgres.yaml) is organized into several top-level keys:

```yaml
# filepath: /Users/nate/development/dao-ai/config/hardware_store/supervisor_postgres.yaml
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

### Loading and Using Configuration

The configuration can be loaded and used programmatically through the `AppConfig` class:

```python
from retail_ai.config import AppConfig

# Load configuration from file
config = AppConfig.from_file("config/hardware_store/supervisor_postgres.yaml")

# Access different configuration sections
print(f"Available agents: {list(config.agents.keys())}")
print(f"Available tools: {list(config.tools.keys())}")
print(f"Vector stores: {list(config.resources.vector_stores.keys())}")

# Use configuration methods for deployment
config.create_agent()          # Package as MLflow model
config.deploy_agent()          # Deploy to serving endpoint
```

The configuration supports both CLI and programmatic workflows, with the Python API providing more flexibility for complex deployment scenarios.

### Developing and Configuring Tools

Tools are functions that agents can use to interact with external systems or perform specific tasks. They are defined under the `tools` key in [`model_config.yaml`](config/hardware_store/supervisor_postgres.yaml). Each tool has a unique name and contains a `function` specification.

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
         schema: *retail_schema # Optional schema definition
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
         schema: *retail_schema
   ```
   **Development:**
   Create the corresponding SQL function in your Databricks Unity Catalog using the specified schema and function name. The tool will automatically generate the appropriate function signature and documentation.

### Developing Unity Catalog Functions

Unity Catalog functions provide the backbone for data access in the multi-agent system. The framework automatically deploys these functions from SQL DDL files during system initialization.

#### Function Deployment Configuration

Unity Catalog functions are defined in the `unity_catalog_functions` section of [`model_config.yaml`](config/hardware_store/supervisor_postgres.yaml). Each function specification includes:

- **Function metadata**: Schema and name for Unity Catalog registration
- **DDL file path**: Location of the SQL file containing the function definition
- **Test parameters**: Optional test data for function validation

**Configuration Example from [`model_config.yaml`](config/hardware_store/supervisor_postgres.yaml):**
```yaml
unity_catalog_functions:
  - function:
      schema: *retail_schema               # Reference to schema configuration
      name: find_product_by_sku           # Function name in Unity Catalog
    ddl: ../functions/retail/find_product_by_sku.sql  # Path to SQL DDL file
    test:                                 # Optional test configuration
      parameters:
        sku: ["00176279"]                 # Test parameters for validation
  - function:
      schema: *retail_schema
      name: find_store_inventory_by_sku
    ddl: ../functions/retail/find_store_inventory_by_sku.sql
    test:
      parameters:
        store: "35048"                    # Multiple parameters for complex functions
        sku: ["00176279"]
```

#### SQL Function Structure

SQL files should follow this structure for proper deployment:

**File Structure Example** (`functions/retail/find_product_by_sku.sql`):
```sql
-- Function to find product details by SKU
CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_product_by_sku(
  sku ARRAY<STRING> COMMENT 'One or more unique identifiers for retrieve. SKU values are between 5-8 alpha numeric characters'
)
RETURNS TABLE(
  product_id BIGINT COMMENT 'Unique identifier for each product in the catalog',
  sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code',
  upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification',
  brand_name STRING COMMENT 'Name of the manufacturer or brand that produces the product',
  product_name STRING COMMENT 'Display name of the product as shown to customers',
  -- ... additional columns
)
READS SQL DATA
COMMENT 'Retrieves detailed information about a specific product by its SKU. This function is designed for product information retrieval in retail applications.'
RETURN 
SELECT 
  product_id,
  sku,
  upc,
  brand_name,
  product_name
  -- ... additional columns
FROM products
WHERE ARRAY_CONTAINS(find_product_by_sku.sku, products.sku);
```

**Key Requirements:**
- Use `{catalog_name}.{schema_name}` placeholders - these are automatically replaced during deployment
- Include comprehensive `COMMENT` attributes for all parameters and return columns
- Provide a clear function-level comment describing purpose and use cases
- Use `READS SQL DATA` for functions that query data
- Follow consistent naming conventions for parameters and return values

#### Test Configuration

The optional `test` section allows you to define test parameters for automatic function validation:

```yaml
test:
  parameters:
    sku: ["00176279"]                     # Single parameter
    # OR for multi-parameter functions:
    store: "35048"                        # Multiple parameters
    sku: ["00176279"]
```

**Test Benefits:**
- **Validation**: Ensures functions work correctly after deployment
- **Documentation**: Provides example usage for other developers
- **CI/CD Integration**: Enables automated testing in deployment pipelines

**Note**: Test parameters should use realistic data from your datasets to ensure meaningful validation. The framework will execute these tests automatically during deployment to verify function correctness.

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

Agents are specialized AI assistants defined under the `agents` key in [`model_config.yaml`](config/hardware_store/supervisor_postgres.yaml). Each agent has a unique name and specific configuration.

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
# filepath: /Users/nate/development/dao-ai/config/hardware_store/supervisor_postgres.yaml
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

## Integration Hooks

The DAO framework provides several hook integration points that allow you to customize agent behavior and application lifecycle. These hooks enable you to inject custom logic at key points in the system without modifying the core framework code.

### Hook Types

#### Agent-Level Hooks

**Agent hooks** are defined at the individual agent level and allow you to customize specific agent behavior:

##### `create_agent_hook`
Used to provide a completely custom agent implementation. When this is provided all other configuration is ignored. See: **Hook Implementation**

```yaml
agents:
  custom_agent:
    name: custom_agent
    description: "Agent with custom initialization"
    model: *tool_calling_llm
    create_agent_hook: my_package.hooks.initialize_custom_agent
    # ... other agent configuration
```

##### `pre_agent_hook`
Executed before an agent processes a message. Ideal for request preprocessing, logging, validation, or context injection. See: **Hook Implementation**

```yaml
agents:
  logging_agent:
    name: logging_agent
    description: "Agent with request logging"
    model: *tool_calling_llm
    pre_agent_hook: my_package.hooks.log_incoming_request
    # ... other agent configuration
```

##### `post_agent_hook`
Executed after an agent completes processing a message. Perfect for response post-processing, logging, metrics collection, or cleanup operations. See: **Hook Implementation**

```yaml
agents:
  analytics_agent:
    name: analytics_agent
    description: "Agent with response analytics"
    model: *tool_calling_llm
    post_agent_hook: my_package.hooks.collect_response_metrics
    # ... other agent configuration
```

#### Application-Level Hooks

**Application hooks** operate at the global application level and affect the entire system lifecycle:

##### `initialization_hooks`
Executed when the application starts up via `AppConfig.from_file()`. Use these for system initialization, resource setup, database connections, or external service configuration. See: **Hook Implementation**

```yaml
app:
  name: my_retail_app
  initialization_hooks:
    - my_package.hooks.setup_database_connections
    - my_package.hooks.initialize_external_apis
    - my_package.hooks.setup_monitoring
  # ... other app configuration
```

##### `shutdown_hooks`
Executed when the application shuts down (registered via `atexit`). Essential for cleanup operations, closing connections, saving state, or performing final logging. See: **Hook Implementation**

```yaml
app:
  name: my_retail_app
  shutdown_hooks:
    - my_package.hooks.cleanup_database_connections
    - my_package.hooks.save_session_data
    - my_package.hooks.send_shutdown_metrics
  # ... other app configuration
```

##### `message_hooks`
Executed for every message processed by the system. Useful for global logging, authentication, rate limiting, or message transformation. See: **Hook Implementation**

```yaml
app:
  name: my_retail_app
  message_hooks:
    - my_package.hooks.authenticate_user
    - my_package.hooks.apply_rate_limiting
    - my_package.hooks.transform_message_format
  # ... other app configuration
```

### Hook Implementation

Hooks can be implemented as either:

1. **Python Functions**: Direct function references
   ```yaml
   initialization_hooks: my_package.hooks.setup_function
   ```

2. **Factory Functions**: Functions that return configured tools or handlers
   ```yaml
   initialization_hooks:
     type: factory
     name: my_package.hooks.create_setup_handler
     args:
       config_param: "value"
   ```

3. **Hook Lists**: Multiple hooks executed in sequence
   ```yaml
   initialization_hooks:
     - my_package.hooks.setup_database
     - my_package.hooks.setup_cache
     - my_package.hooks.setup_monitoring
   ```

### Hook Function Signatures

Each hook type expects specific function signatures:

#### Agent Hooks
```python
# create_agent_hook
def initialize_custom_agent(state: dict, config: dict) -> dict:
    """Custom agent initialization logic"""
    pass

# pre_agent_hook  
def log_incoming_request(state: dict, config: dict) -> dict:
    """Pre-process incoming request"""
    return state

# post_agent_hook
def collect_response_metrics(state: dict, config: dict) -> dict:
    """Post-process agent response"""
    return state
```

#### Application Hooks
```python
# initialization_hooks
def setup_database_connections(config: AppConfig) -> None:
    """Initialize database connections"""
    pass

# shutdown_hooks  
def cleanup_resources(config: AppConfig) -> None:
    """Clean up resources on shutdown"""
    pass

# message_hooks
def authenticate_user(state: dict, config: dict) -> dict:
    """Authenticate and authorize user requests"""
    return state
```

### Use Cases and Examples

#### Common Hook Patterns

**Logging and Monitoring**:
```python
def log_agent_performance(state: dict, config: AppConfig) -> dict:
    """Log agent response times and quality metrics"""
    start_time = state.get('start_time')
    if start_time:
        duration = time.time() - start_time
        logger.info(f"Agent response time: {duration:.2f}s")
    return state
```

**Authentication and Authorization**:
```python
def validate_user_permissions(state: dict, config: AppConfig) -> dict:
    """Validate user has permission for requested operation"""
    user_id = state.get('user_id')
    if not has_permission(user_id, state.get('operation')):
        raise UnauthorizedError("Insufficient permissions")
    return state
```

**Resource Management**:
```python
def initialize_vector_search(config: AppConfig) -> None:
    """Initialize vector search connections during startup"""
    for vs_name, vs_config in config.resources.vector_stores.items():
        vs_config.create()
        logger.info(f"Vector store {vs_name} initialized")
```

**State Enrichment**:
```python
def enrich_user_context(state: dict, config: AppConfig) -> dict:
    """Add user profile and preferences to state"""
    user_id = state.get('user_id')
    if user_id:
        user_profile = get_user_profile(user_id)
        state['user_context'] = user_profile
    return state
```

### Best Practices

1. **Keep hooks lightweight**: Avoid heavy computations that could slow down message processing
2. **Handle errors gracefully**: Use try-catch blocks to prevent hook failures from breaking the system
3. **Use appropriate hook types**: Choose agent-level vs application-level hooks based on scope
4. **Maintain state immutability**: Return modified copies of state rather than mutating in-place
5. **Log hook execution**: Include logging for troubleshooting and monitoring
6. **Test hooks independently**: Write unit tests for hook functions separate from the main application


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
- `config/`: Contains [`model_config.yaml`](config/hardware_store/supervisor_postgres.yaml).

### Building the Package

```bash
# Install development dependencies
make depends

# Build the package
make install

# Run tests
make test

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

### Advanced Configuration

You can also pass additional configuration parameters to customize the agent's behavior:

```python
response = client.predict(
  endpoint="retail_ai_agent",
  inputs={
    "messages": [
      {"role": "user", "content": "Can you recommend a lamp for my oak side tables?"}
    ],
    "configurable": {
      "thread_id": "1",
      "user_id": "my_user_id", 
      "store_num": 87887
    }
  }
)
```

The `configurable` section supports:
- **`thread_id`**: Unique identifier for conversation threading and state management
- **`user_id`**: User identifier for personalization and tracking
- **`store_num`**: Store number for location-specific recommendations and inventory

## Customization

To customize the agent:

1. **Update [`model_config.yaml`](config/hardware_store/supervisor_postgres.yaml)**:
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
```

## Logging

The primary log level for the application is configured in [`model_config.yaml`](config/hardware_store/supervisor_postgres.yaml) under the `app.log_level` field.

**Configuration Example:**
```yaml
# filepath: /Users/nate/development/dao-ai/config/hardware_store/supervisor_postgres.yaml
app:
  log_level: INFO  # Supported levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
  # ... other app configurations ...
```

This setting controls the verbosity of logs produced by the `retail_ai` package.

The system also includes:
- **MLflow tracing** for request tracking.
- **Structured logging** is used internally.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
