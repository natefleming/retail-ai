# AI Agents

AI Agents are autonomous, multi-step reasoning components that orchestrate complex retail workflows using specialized tools and domain knowledge. These agents provide conversational interfaces and intelligent decision-making capabilities for retail operations.

## Overview

AI Agents in the Retail AI system:
- Provide autonomous decision making and multi-step reasoning
- Orchestrate complex workflows across multiple tools
- Enable natural language conversational interfaces
- Include built-in guardrails for safety and quality control
- Specialize in specific retail domains (products, inventory, comparison)

## Agent Architecture

### Core Components

**LangGraph Framework**: Provides the foundation for agent state management and workflow orchestration
**ReAct Pattern**: Reasoning and Acting pattern for tool selection and execution
**State Management**: Persistent conversation context and user session data
**Guardrails**: Content safety, quality control, and business rule enforcement

### Agent State

All agents share a common state structure:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    store_num: str

class AgentConfig(TypedDict):
    configurable: dict[str, Any]
```

## Specialized Agents

### Product Agent

Handles product discovery, information retrieval, and recommendations.

**Capabilities:**
- Product lookup by SKU, UPC, or description
- Semantic product search using vector embeddings
- Product feature extraction and analysis
- Cross-selling and upselling recommendations

**Tools Available:**
- `find_product_by_sku` - Unity Catalog function for exact SKU lookup
- `find_product_by_upc` - Unity Catalog function for exact UPC lookup
- `find_product_details_by_description` - Vector search for semantic product discovery

**Usage Example:**
```python
from retail_ai.nodes import product_node
from retail_ai.state import AgentState, AgentConfig

# Create product agent
product_agent = product_node(model_config)

# Agent state
state = AgentState(
    messages=[HumanMessage(content="I'm looking for wireless headphones")],
    user_id="user123",
    store_num="101"
)

config = AgentConfig(configurable={})

# Execute agent
response = product_agent.invoke(state, config)
print(response["messages"][-1].content)
```

**Configuration:**
```yaml
# model_config.yaml
agents:
  product:
    model:
      name: "databricks-meta-llama-3-3-70b-instruct"
    prompt: |
      You are a helpful product specialist at BrickMart store {store_num}.
      Help customers find products using the available tools.
      Always provide accurate product information and helpful recommendations.
    guardrails:
      - type: "content_safety"
        config: {"block_harmful": true}
```

### Inventory Agent

Manages inventory queries, stock level checks, and availability information.

**Capabilities:**
- Real-time inventory checking across stores and warehouses
- Stock level monitoring and alerts
- Product availability by location
- Inventory movement tracking

**Tools Available:**
- `find_inventory_by_sku` - Global inventory lookup by SKU
- `find_inventory_by_upc` - Global inventory lookup by UPC
- `find_store_inventory_by_sku` - Store-specific inventory by SKU
- `find_store_inventory_by_upc` - Store-specific inventory by UPC
- `find_product_details_by_description` - Product discovery for inventory checks

**Usage Example:**
```python
from retail_ai.nodes import inventory_node

# Create inventory agent
inventory_agent = inventory_node(model_config)

# Check inventory for specific product
state = AgentState(
    messages=[HumanMessage(content="Do you have SKU ABC123 in stock at store 101?")],
    user_id="user123",
    store_num="101"
)

response = inventory_agent.invoke(state, config)
```

**Specialized Queries:**
```python
# Multi-store inventory check
query = "Check inventory for wireless headphones across all stores"

# Low stock alerts
query = "Which products are running low in store 101?"

# Warehouse vs store availability
query = "Is product XYZ available in store or do we need to order from warehouse?"
```

### Comparison Agent

Provides detailed product comparisons and recommendation analysis.

**Capabilities:**
- Side-by-side product comparisons
- Feature analysis and scoring
- Pros and cons evaluation
- Personalized recommendations based on user needs

**Tools Available:**
- `find_product_by_sku` - Product lookup for comparison
- `find_product_by_upc` - Alternative product lookup
- `find_product_details_by_description` - Find similar products to compare
- `product_comparison` - LLM-powered comparison analysis

**Usage Example:**
```python
from retail_ai.nodes import comparison_node

# Create comparison agent
comparison_agent = comparison_node(model_config)

# Compare products
state = AgentState(
    messages=[HumanMessage(content="Compare iPhone 15 vs Samsung Galaxy S24")],
    user_id="user123", 
    store_num="101"
)

response = comparison_agent.invoke(state, config)
```

**Comparison Features:**
- **Feature Analysis**: Detailed breakdown of product specifications
- **Pros and Cons**: Balanced evaluation of each product
- **Use Case Recommendations**: Best product for specific needs
- **Price Comparison**: Value analysis and cost considerations

### DIY Agent

General-purpose agent for diverse queries and exploratory tasks.

**Capabilities:**
- Web search for external information
- General product discovery
- Open-ended customer assistance
- Fallback for queries outside other agent specializations

**Tools Available:**
- `search_tool` - DuckDuckGo web search
- `find_product_details_by_description` - Product discovery
- Additional tools based on configuration

**Usage Example:**
```python
from retail_ai.nodes import diy_node

# Create DIY agent
diy_agent = diy_node(model_config)

# Handle general query
state = AgentState(
    messages=[HumanMessage(content="What are the latest trends in smart home technology?")],
    user_id="user123",
    store_num="101"
)

response = diy_agent.invoke(state, config)
```

---

## Tools for Building Agents

Agents leverage different types of tools to access data, perform analysis, and provide intelligent responses. Here's an overview of the available tool categories:

### Unity Catalog Functions

Unity Catalog functions provide direct SQL-based access to product and inventory data. These functions are created and managed through the `04_unity_catalog_tools.py` notebook and offer high-performance database operations.

**Overview:**
The Unity Catalog functions are SQL functions that:
- Provide direct access to product and inventory tables
- Support batch operations with array parameters
- Return structured tabular data
- Integrate seamlessly with Databricks SQL Warehouse
- Offer the fastest query performance for exact matches

**Key Functions:**

#### Product Lookup Functions
- **`find_product_by_sku`** - Retrieves detailed product information by SKU identifiers
- **`find_product_by_upc`** - Retrieves detailed product information by UPC identifiers

#### Inventory Management Functions
- **`find_inventory_by_sku`** - Retrieves inventory information across all stores for specific SKUs
- **`find_inventory_by_upc`** - Retrieves inventory information across all stores for specific UPCs
- **`find_store_inventory_by_sku`** - Retrieves inventory information for a specific store and SKUs
- **`find_store_inventory_by_upc`** - Retrieves inventory information for a specific store and UPCs

**Usage Example:**
```python
from databricks.sdk import WorkspaceClient
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient(client=WorkspaceClient())
result = client.execute_function(
    function_name="catalog.database.find_product_by_sku",
    parameters={"sku": ["STB-KCP-001", "DUN-KCP-001"]}
)
```

**Performance Characteristics:**
- **High Performance**: Direct SQL execution in Databricks SQL Warehouse
- **Batch Operations**: Support for multiple identifiers in single call
- **Structured Output**: Consistent tabular format
- **Type Safety**: Strong typing with SQL schema validation

### Vector Search Tools

Vector Search tools provide semantic search capabilities using vector embeddings for intelligent product discovery and document retrieval. These tools enable natural language queries to find relevant products and content based on meaning rather than exact keyword matches.

**Overview:**
Vector Search tools in the Retail AI system:
- Perform semantic similarity matching using vector embeddings
- Scale to large datasets with efficient indexing
- Rank results by relevance and similarity scores
- Integrate with MLflow for model serving and observability
- Support real-time and batch search operations

**Core Components:**

#### Vector Search Index
The foundation of semantic search, containing:
- **Product Embeddings**: Vector representations of product descriptions
- **Metadata**: Product attributes, categories, and structured data
- **Search Endpoints**: Databricks Vector Search endpoints for query processing

#### Embedding Models
Models used to convert text to vector representations:
- **Text Embedding Models**: Transform product descriptions to vectors
- **Multimodal Models**: Handle text, images, and other content types
- **Domain-Specific Models**: Optimized for retail and product data

**Key Functions:**

#### `find_product_details_by_description`
Performs semantic search over product data to find items matching natural language descriptions.

**Usage Example:**
```python
from retail_ai.tools import find_product_details_by_description_tool

# Create the tool
search_tool = find_product_details_by_description_tool(
    endpoint_name="vs_endpoint_name",
    index_name="products_index",
    columns=["product_name", "description", "category", "price"],
    k=10
)

# Search for products
results = search_tool.invoke("wireless bluetooth headphones with noise cancellation")

# Process results
for doc in results:
    print(f"Product: {doc.metadata['product_name']}")
    print(f"Description: {doc.page_content}")
    print(f"Score: {doc.metadata.get('score', 'N/A')}")
    print("---")
```

#### `create_vector_search_tool`
Creates a configurable Vector Search tool for retrieving documents from any Databricks Vector Search index.

**Usage Example:**
```python
from retail_ai.tools import create_vector_search_tool

# Create product search tool
product_search = create_vector_search_tool(
    name="product_search",
    description="Search for products using natural language descriptions",
    index_name="catalog.schema.products_vs_index",
    primary_key="product_id",
    text_column="description",
    doc_uri="product_url",
    columns=["product_id", "name", "description", "category", "price"],
    search_parameters={"num_results": 15}
)
```

### LangChain Tools

LangChain tools in the Retail AI system:
- Use large language models for intelligent processing
- Support flexible input/output formats
- Return structured output using Pydantic models
- Provide creative and analytical capabilities
- Integrate seamlessly with the agent framework

**Key Tools:**

#### Product Analysis Tools

**`product_comparison`** - Compares multiple products and provides structured analysis of their features, specifications, pros, cons, and recommendations.

**Usage Example:**
```python
from retail_ai.tools import create_product_comparison_tool
from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(model="databricks-meta-llama-3-3-70b-instruct")
comparison_tool = create_product_comparison_tool(llm)

products = [
    {
        "product_id": 1,
        "product_name": "Wireless Headphones A",
        "price": 99.99,
        "battery_life": "20 hours",
        "noise_cancellation": True
    },
    {
        "product_id": 2,
        "product_name": "Wireless Headphones B", 
        "price": 149.99,
        "battery_life": "30 hours",
        "noise_cancellation": True
    }
]

result = comparison_tool.invoke(products)
print(result.summary)
print(result.recommendations)
```

#### Text Processing Tools

**`sku_extraction`** - Extracts product SKUs from natural language text for product identification.

**Usage Example:**
```python
from retail_ai.tools import create_sku_extraction_tool

sku_tool = create_sku_extraction_tool(llm)

text = "I'm looking for information about SKU ABC123 and also need details on product DUN-KCP-001"
skus = sku_tool.invoke(text)
print(skus)  # Output: ['ABC123', 'DUN-KCP-001']
```

#### Classification Tools

**`product_classification`** - Classifies product descriptions into predefined categories using natural language understanding.

**Usage Example:**
```python
from retail_ai.tools import create_product_classification_tool

categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
classification_tool = create_product_classification_tool(
    llm=llm,
    allowable_classifications=categories,
    k=2
)

description = "Wireless bluetooth headphones with noise cancellation"
classifications = classification_tool.invoke(description)
print(classifications)  # Output: ['Electronics']
```

#### External Integration Tools

**`genie_tool`** - Interfaces with Databricks Genie for natural language to SQL query translation and execution.

**`search_tool`** - Provides web search capabilities using DuckDuckGo for external information retrieval.

---

## Agent Orchestration

### Multi-Agent Workflows

Agents can be chained together for complex workflows:

```python
def multi_agent_workflow(user_query: str, user_id: str, store_num: str):
    """Orchestrate multiple agents for comprehensive assistance."""
    
    # Step 1: Product discovery
    product_state = AgentState(
        messages=[HumanMessage(content=user_query)],
        user_id=user_id,
        store_num=store_num
    )
    
    product_response = product_agent.invoke(product_state, config)
    
    # Step 2: Inventory check for found products
    inventory_query = f"Check inventory for products mentioned in: {product_response['messages'][-1].content}"
    inventory_state = AgentState(
        messages=[HumanMessage(content=inventory_query)],
        user_id=user_id,
        store_num=store_num
    )
    
    inventory_response = inventory_agent.invoke(inventory_state, config)
    
    # Step 3: Comparison if multiple products found
    if "compare" in user_query.lower():
        comparison_state = AgentState(
            messages=[HumanMessage(content=f"Compare: {product_response['messages'][-1].content}")],
            user_id=user_id,
            store_num=store_num
        )
        
        comparison_response = comparison_agent.invoke(comparison_state, config)
        return comparison_response
    
    return inventory_response
```

### Agent Router

Intelligent routing to the most appropriate agent:

```python
def route_to_agent(query: str) -> str:
    """Route user query to the most appropriate agent."""
    
    query_lower = query.lower()
    
    # Product-focused queries
    if any(keyword in query_lower for keyword in ["product", "find", "search", "recommend"]):
        return "product"
    
    # Inventory-focused queries  
    elif any(keyword in query_lower for keyword in ["stock", "inventory", "available", "quantity"]):
        return "inventory"
    
    # Comparison queries
    elif any(keyword in query_lower for keyword in ["compare", "vs", "versus", "difference"]):
        return "comparison"
    
    # General queries
    else:
        return "diy"

def smart_agent_dispatch(query: str, user_id: str, store_num: str):
    """Dispatch query to appropriate agent."""
    
    agent_type = route_to_agent(query)
    
    agents = {
        "product": product_agent,
        "inventory": inventory_agent, 
        "comparison": comparison_agent,
        "diy": diy_agent
    }
    
    selected_agent = agents[agent_type]
    
    state = AgentState(
        messages=[HumanMessage(content=query)],
        user_id=user_id,
        store_num=store_num
    )
    
    return selected_agent.invoke(state, config)
```

## Guardrails and Safety

### Content Safety

Agents include built-in guardrails for content safety and quality control:

```python
# Reflection guardrail for content quality
def reflection_guardrail(guardrail_definition: dict):
    """Create reflection-based guardrail for agent responses."""
    
    @mlflow.trace()
    def guardrail_check(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        
        # Check content safety
        if guardrail_definition.get("content_safety"):
            safety_check = check_content_safety(last_message.content)
            if not safety_check.is_safe:
                return {"messages": [AIMessage(content="I apologize, but I cannot provide that information.")]}
        
        # Check response quality
        if guardrail_definition.get("quality_check"):
            quality_score = assess_response_quality(last_message.content)
            if quality_score < 0.7:
                return {"messages": [AIMessage(content="Let me provide a better response...")]}
        
        return state
    
    return guardrail_check
```

### Business Rules

Enforce business logic and policies:

```python
def business_rules_guardrail(state: AgentState) -> dict:
    """Enforce business rules and policies."""
    
    last_message = state["messages"][-1].content
    
    # Price disclosure rules
    if "price" in last_message.lower():
        if not has_price_disclaimer(last_message):
            enhanced_message = add_price_disclaimer(last_message)
            return {"messages": [AIMessage(content=enhanced_message)]}
    
    # Inventory accuracy warnings
    if "in stock" in last_message.lower():
        disclaimer = "\n\n*Inventory levels are updated in real-time but may vary."
        enhanced_message = last_message + disclaimer
        return {"messages": [AIMessage(content=enhanced_message)]}
    
    return state
```

## Performance Optimization

### Agent Caching

Cache agent responses for frequently asked questions:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=500)
def cached_agent_response(query_hash: str, agent_type: str) -> str:
    """Cache agent responses for common queries."""
    # Implementation would store and retrieve cached responses
    pass

def agent_with_cache(agent, query: str, user_id: str, store_num: str):
    """Execute agent with intelligent caching."""
    
    # Create cache key
    cache_key = hashlib.md5(f"{query}_{agent_type}_{store_num}".encode()).hexdigest()
    
    # Try cache first
    try:
        cached_response = cached_agent_response(cache_key, agent.__name__)
        if cached_response:
            return {"messages": [AIMessage(content=cached_response)]}
    except:
        pass
    
    # Execute agent if no cache hit
    state = AgentState(
        messages=[HumanMessage(content=query)],
        user_id=user_id,
        store_num=store_num
    )
    
    return agent.invoke(state, config)
```

### Parallel Tool Execution

Execute independent tools in parallel for better performance:

```python
import asyncio

async def parallel_tool_execution(tools: list, queries: list):
    """Execute multiple tools in parallel."""
    
    tasks = []
    for tool, query in zip(tools, queries):
        task = asyncio.create_task(tool.ainvoke(query))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

## Monitoring and Analytics

### Agent Performance Tracking

Monitor agent performance and user satisfaction:

```python
@mlflow.trace(span_type="AGENT", name="agent_execution")
def monitored_agent_execution(agent, state: AgentState, config: AgentConfig):
    """Execute agent with comprehensive monitoring."""
    
    start_time = time.time()
    
    try:
        # Execute agent
        response = agent.invoke(state, config)
        
        # Log performance metrics
        execution_time = time.time() - start_time
        mlflow.log_metrics({
            "agent_execution_time": execution_time,
            "message_count": len(response["messages"]),
            "user_satisfaction": calculate_satisfaction_score(response)
        })
        
        # Log agent usage
        mlflow.log_params({
            "agent_type": agent.__name__,
            "user_id": state["user_id"],
            "store_num": state["store_num"]
        })
        
        return response
        
    except Exception as e:
        mlflow.log_metric("agent_errors", 1)
        logger.error(f"Agent execution failed: {e}")
        raise
```

### Conversation Analytics

Track conversation patterns and user behavior:

```python
def analyze_conversation(messages: Sequence[BaseMessage]):
    """Analyze conversation for insights."""
    
    analytics = {
        "conversation_length": len(messages),
        "user_messages": len([m for m in messages if isinstance(m, HumanMessage)]),
        "agent_messages": len([m for m in messages if isinstance(m, AIMessage)]),
        "tool_calls": len([m for m in messages if isinstance(m, ToolMessage)]),
        "topics": extract_conversation_topics(messages),
        "sentiment": analyze_conversation_sentiment(messages)
    }
    
    # Log analytics
    mlflow.log_metrics(analytics)
    
    return analytics
```

## Integration Patterns

### Streamlit Integration

Integrate agents with Streamlit for web interfaces:

```python
import streamlit as st

def streamlit_agent_chat():
    """Streamlit chat interface for agents."""
    
    st.title("BrickMart AI Assistant")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Route to appropriate agent
        agent_type = route_to_agent(prompt)
        agent = get_agent(agent_type)
        
        # Execute agent
        state = AgentState(
            messages=[HumanMessage(content=prompt)],
            user_id=st.session_state.get("user_id", "anonymous"),
            store_num=st.session_state.get("store_num", "101")
        )
        
        response = agent.invoke(state, config)
        
        # Add agent response
        agent_message = response["messages"][-1].content
        st.session_state.messages.append({"role": "assistant", "content": agent_message})
        
        # Display new message
        with st.chat_message("assistant"):
            st.markdown(agent_message)
```

### API Integration

Expose agents through REST APIs:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    user_id: str
    store_num: str
    agent_type: str = "auto"

class ChatResponse(BaseModel):
    response: str
    agent_used: str
    execution_time: float

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for agent interaction."""
    
    start_time = time.time()
    
    try:
        # Route to agent
        if request.agent_type == "auto":
            agent_type = route_to_agent(request.message)
        else:
            agent_type = request.agent_type
        
        agent = get_agent(agent_type)
        
        # Execute agent
        state = AgentState(
            messages=[HumanMessage(content=request.message)],
            user_id=request.user_id,
            store_num=request.store_num
        )
        
        response = agent.invoke(state, config)
        
        return ChatResponse(
            response=response["messages"][-1].content,
            agent_used=agent_type,
            execution_time=time.time() - start_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Best Practices

### Agent Design

1. **Single Responsibility**: Each agent should have a clear, focused purpose
2. **Tool Selection**: Provide agents with appropriate tools for their domain
3. **Prompt Engineering**: Craft clear, specific prompts for consistent behavior
4. **Error Handling**: Implement robust error handling and graceful degradation

### Performance

1. **Caching**: Cache frequent queries and responses appropriately
2. **Parallel Execution**: Use async/await for independent operations
3. **Tool Optimization**: Choose the fastest appropriate tools for each task
4. **State Management**: Keep agent state minimal and efficient

### User Experience

1. **Clear Communication**: Provide clear, helpful responses
2. **Context Awareness**: Maintain conversation context across interactions
3. **Personalization**: Adapt responses to user preferences and history
4. **Feedback Integration**: Collect and incorporate user feedback

## Troubleshooting

### Common Issues

**Agent Not Responding**
```python
# Debug agent execution
def debug_agent_execution(agent, state, config):
    try:
        logger.info(f"Executing agent: {agent.__name__}")
        logger.info(f"State: {state}")
        
        response = agent.invoke(state, config)
        
        logger.info(f"Response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise
```

**Tool Selection Issues**
```python
# Debug tool selection
def debug_tool_selection(agent_state):
    last_message = agent_state["messages"][-1].content
    
    print(f"User query: {last_message}")
    print(f"Available tools: {[tool.name for tool in agent.tools]}")
    
    # Check tool descriptions
    for tool in agent.tools:
        print(f"Tool: {tool.name}")
        print(f"Description: {tool.description}")
        print("---")
```

**Performance Issues**
```python
# Profile agent performance
import cProfile

def profile_agent_execution(agent, state, config):
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        response = agent.invoke(state, config)
        return response
    finally:
        profiler.disable()
        profiler.print_stats(sort='cumulative')
```

For more troubleshooting help, see the [Development Guide](../development/contributing.md#troubleshooting).

## Next Steps

- [Tools Reference](references/tools-reference.md) - Complete technical specifications for all tools
- [Agent Reference](references/agent-reference.md) - Detailed agent specifications and configurations
- [Agent Development Patterns](agents/agent-development-patterns.md) - Implementation patterns and best practices
- [Development Guide](../development/contributing.md) - Creating custom agents and tools 