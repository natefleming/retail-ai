# Agent Development Patterns

This guide covers common patterns and best practices for developing retail AI agents, including code structures, state management, and guardrails implementation.

## Common Agent Structure

### Basic Agent Template

```python
def my_agent_node(model_config: ModelConfig) -> AgentCallable:
    # Extract configuration
    model = model_config.get("agents").get("my_agent").get("model").get("name")
    prompt = model_config.get("agents").get("my_agent").get("prompt")
    guardrails = model_config.get("agents").get("my_agent").get("guardrails") or []
    
    @mlflow.trace()
    def my_agent(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        # Initialize LLM
        llm = ChatDatabricks(model=model, temperature=0.1)
        
        # Format prompt with context
        prompt_template = PromptTemplate.from_template(prompt)
        system_prompt = prompt_template.format(
            user_id=state["user_id"],
            store_num=state["store_num"]
        )
        
        # Configure tools
        tools = [
            # Add relevant tools for this agent
        ]
        
        # Create agent
        agent = create_react_agent(
            model=llm,
            prompt=system_prompt,
            tools=tools
        )
        
        # Apply guardrails
        for guardrail_definition in guardrails:
            guardrail = reflection_guardrail(guardrail_definition)
            agent = with_guardrails(agent, guardrail)
        
        return agent
    
    return my_agent
```

### Specialized Agent Example

```python
def customer_service_node(model_config: ModelConfig) -> AgentCallable:
    """Customer service agent with order lookup and policy tools."""
    
    # Configuration
    model = model_config.get("agents").get("customer_service").get("model").get("name")
    warehouse_id = model_config.get("warehouse_id")
    
    @mlflow.trace()
    def customer_service_agent(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        llm = ChatDatabricks(model=model, temperature=0.1)
        
        # Context-aware prompt
        prompt = """You are a helpful customer service representative at BrickMart store {store_num}.
        Help customers with orders, returns, and general inquiries.
        Always be polite and provide accurate information.
        
        Current customer: {user_id}
        Store location: {store_num}
        """
        
        system_prompt = prompt.format(
            store_num=state["store_num"],
            user_id=state["user_id"]
        )
        
        # Tool selection for customer service
        tools = [
            # Order management
            create_uc_tools(["catalog.database.find_order_by_id"]),
            
            # Product questions
            find_product_details_by_description_tool(
                endpoint_name="vs_endpoint",
                index_name="products_index",
                columns=["product_name", "description", "price"]
            ),
            
            # Policy search
            create_vector_search_tool(
                name="policy_search",
                description="Search store policies and procedures",
                index_name="policies_index"
            )
        ]
        
        agent = create_react_agent(
            model=llm,
            prompt=system_prompt,
            tools=tools
        )
        
        return {"messages": [agent.invoke(state)]}
    
    return customer_service_agent
```

---

## Agent State Management

### AgentState Structure

```python
class AgentState(MessagesState):
    """Extended state for retail AI agents."""
    context: Sequence[Document]  # Retrieved documents
    route: str                   # Current routing decision
    is_valid_config: bool       # Configuration validation
    user_id: str                # User identifier
    store_num: str              # Store context
    session_data: dict          # Session-specific data
    preferences: dict           # User preferences
```

### State Initialization

```python
def initialize_agent_state(
    user_message: str,
    user_id: str,
    store_num: str,
    session_data: dict = None
) -> AgentState:
    """Initialize agent state with context."""
    
    return AgentState(
        messages=[HumanMessage(content=user_message)],
        user_id=user_id,
        store_num=store_num,
        context=[],
        route="",
        is_valid_config=True,
        session_data=session_data or {},
        preferences={}
    )
```

### State Updates

```python
def update_agent_state(
    state: AgentState,
    new_message: BaseMessage,
    context: Sequence[Document] = None,
    route: str = None
) -> AgentState:
    """Update agent state with new information."""
    
    updated_state = state.copy()
    updated_state["messages"].append(new_message)
    
    if context:
        updated_state["context"] = context
    
    if route:
        updated_state["route"] = route
    
    return updated_state
```

---

## Guardrails Implementation

### Basic Guardrail Pattern

```python
def reflection_guardrail(guardrail_definition: dict):
    """Create a guardrail based on configuration."""
    
    @mlflow.trace()
    def guardrail_check(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        
        # Content safety check
        if guardrail_definition.get("content_safety"):
            safety_check = check_content_safety(last_message.content)
            if not safety_check.is_safe:
                return {
                    "messages": [
                        AIMessage(content="I apologize, but I cannot provide that information.")
                    ]
                }
        
        # Quality check
        if guardrail_definition.get("quality_check"):
            quality_score = assess_response_quality(last_message.content)
            if quality_score < 0.7:
                return {
                    "messages": [
                        AIMessage(content="Let me provide a better response...")
                    ]
                }
        
        # Business rules check
        if guardrail_definition.get("business_rules"):
            rules_check = validate_business_rules(last_message.content, state)
            if not rules_check.is_valid:
                return {
                    "messages": [
                        AIMessage(content=rules_check.fallback_message)
                    ]
                }
        
        return state
    
    return guardrail_check
```

### Content Safety Guardrail

```python
def content_safety_guardrail():
    """Guardrail for content safety and appropriateness."""
    
    @mlflow.trace(span_type="GUARDRAIL", name="content_safety")
    def safety_check(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        
        # Check for inappropriate content
        safety_result = content_safety_classifier(last_message.content)
        
        if safety_result.risk_level > 0.8:
            return {
                "messages": [
                    AIMessage(
                        content="I'm here to help with retail-related questions. "
                               "Please let me know how I can assist you with products, "
                               "inventory, or store information."
                    )
                ]
            }
        
        return state
    
    return safety_check
```

### Business Rules Guardrail

```python
def business_rules_guardrail(rules_config: dict):
    """Guardrail for business rules and policies."""
    
    @mlflow.trace(span_type="GUARDRAIL", name="business_rules")
    def rules_check(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        
        # Price disclosure rules
        if "price" in last_message.content.lower():
            if not validate_price_disclosure(last_message.content):
                return {
                    "messages": [
                        AIMessage(
                            content="Prices are subject to change and may vary by location. "
                                   "Please check with your local store for current pricing."
                        )
                    ]
                }
        
        # Inventory accuracy rules
        if "in stock" in last_message.content.lower():
            if not validate_inventory_disclaimer(last_message.content):
                updated_content = last_message.content + "\n\n*Inventory levels are updated in real-time but may vary."
                return {
                    "messages": [
                        AIMessage(content=updated_content)
                    ]
                }
        
        return state
    
    return rules_check
```

---

## Tool Integration Patterns

### Tool Selection Strategy

```python
def select_tools_for_agent(agent_type: str, capabilities: list[str]) -> list:
    """Select appropriate tools based on agent type and capabilities."""
    
    tool_mapping = {
        "product": {
            "lookup": [create_find_product_by_sku_tool, create_find_product_by_upc_tool],
            "search": [find_product_details_by_description_tool],
            "analysis": [create_product_comparison_tool]
        },
        "inventory": {
            "lookup": [create_find_inventory_by_sku_tool],
            "store_specific": [create_find_store_inventory_by_sku_tool],
            "search": [find_product_details_by_description_tool]
        },
        "customer_service": {
            "orders": [create_order_lookup_tool],
            "policies": [create_policy_search_tool],
            "products": [find_product_details_by_description_tool]
        }
    }
    
    tools = []
    agent_tools = tool_mapping.get(agent_type, {})
    
    for capability in capabilities:
        if capability in agent_tools:
            tools.extend(agent_tools[capability])
    
    return tools
```

### Dynamic Tool Loading

```python
def load_tools_dynamically(config: dict, warehouse_id: str) -> list:
    """Load tools based on configuration."""
    
    tools = []
    
    # Unity Catalog tools
    if config.get("unity_catalog_tools"):
        for function_name in config["unity_catalog_tools"]:
            tool = create_uc_function_tool(warehouse_id, function_name)
            tools.append(tool)
    
    # Vector search tools
    if config.get("vector_search"):
        vs_config = config["vector_search"]
        tool = find_product_details_by_description_tool(
            endpoint_name=vs_config["endpoint"],
            index_name=vs_config["index"],
            columns=vs_config["columns"]
        )
        tools.append(tool)
    
    # LangChain tools
    if config.get("langchain_tools"):
        llm = ChatDatabricks(model=config["model"])
        for tool_name in config["langchain_tools"]:
            tool = create_langchain_tool(tool_name, llm)
            tools.append(tool)
    
    return tools
```

---

## Agent Orchestration Patterns

### Sequential Agent Chain

```python
def create_agent_chain(agents: list[AgentCallable]) -> AgentCallable:
    """Create a chain of agents that process sequentially."""
    
    @mlflow.trace(span_type="AGENT_CHAIN")
    def agent_chain(state: AgentState, config: AgentConfig) -> dict:
        current_state = state
        
        for agent in agents:
            result = agent.invoke(current_state, config)
            current_state = update_agent_state(
                current_state,
                result["messages"][-1]
            )
        
        return {"messages": current_state["messages"]}
    
    return agent_chain
```

### Parallel Agent Execution

```python
import asyncio

async def execute_agents_parallel(
    agents: list[AgentCallable],
    state: AgentState,
    config: AgentConfig
) -> list[dict]:
    """Execute multiple agents in parallel."""
    
    tasks = [
        agent.ainvoke(state, config) for agent in agents
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and return successful results
    successful_results = [
        result for result in results 
        if not isinstance(result, Exception)
    ]
    
    return successful_results
```

### Conditional Agent Routing

```python
def create_conditional_router(routing_rules: dict) -> AgentCallable:
    """Create a router that selects agents based on conditions."""
    
    @mlflow.trace(span_type="ROUTER")
    def conditional_router(state: AgentState, config: AgentConfig) -> dict:
        user_message = state["messages"][-1].content.lower()
        
        # Apply routing rules
        for condition, agent_name in routing_rules.items():
            if condition in user_message:
                selected_agent = get_agent_by_name(agent_name)
                return selected_agent.invoke(state, config)
        
        # Default fallback
        default_agent = get_agent_by_name("general")
        return default_agent.invoke(state, config)
    
    return conditional_router
```

---

## Error Handling Patterns

### Graceful Degradation

```python
def create_resilient_agent(
    primary_agent: AgentCallable,
    fallback_agent: AgentCallable
) -> AgentCallable:
    """Create an agent with fallback capabilities."""
    
    @mlflow.trace(span_type="RESILIENT_AGENT")
    def resilient_agent(state: AgentState, config: AgentConfig) -> dict:
        try:
            # Try primary agent
            return primary_agent.invoke(state, config)
        
        except Exception as e:
            logger.warning(f"Primary agent failed: {e}")
            
            # Fallback to simpler agent
            try:
                return fallback_agent.invoke(state, config)
            except Exception as fallback_error:
                logger.error(f"Fallback agent also failed: {fallback_error}")
                
                # Final fallback - simple response
                return {
                    "messages": [
                        AIMessage(
                            content="I'm experiencing technical difficulties. "
                                   "Please try again or contact customer service."
                        )
                    ]
                }
    
    return resilient_agent
```

### Retry Logic

```python
def create_agent_with_retry(
    agent: AgentCallable,
    max_retries: int = 3,
    backoff_factor: float = 1.0
) -> AgentCallable:
    """Add retry logic to an agent."""
    
    @mlflow.trace(span_type="RETRY_AGENT")
    def retry_agent(state: AgentState, config: AgentConfig) -> dict:
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return agent.invoke(state, config)
            
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    time.sleep(sleep_time)
                    logger.warning(f"Agent attempt {attempt + 1} failed, retrying in {sleep_time}s")
        
        # All retries failed
        logger.error(f"Agent failed after {max_retries} attempts: {last_exception}")
        raise last_exception
    
    return retry_agent
```

---

## Related Documentation

- [Agent Reference](../references/agent-reference.md) - Detailed agent specifications
- [Agent Performance](agent-performance.md) - Performance metrics and optimization
- [Best Practices](agent-best-practices.md) - Guidelines for agent development
- [Tools Reference](../references/tools-reference.md) - Available tools and their usage 