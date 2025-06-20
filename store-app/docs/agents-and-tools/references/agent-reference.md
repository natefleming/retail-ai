# Agent Reference

This reference provides detailed information about all implemented retail AI agents, their capabilities, configurations, and usage examples.

## Product Agent
**File:** `retail_ai/nodes.py` → `product_node()`

**Specialization:** Product discovery, details, and recommendations

**Capabilities:**
- Exact product lookup by SKU or UPC
- Semantic product search using natural language
- Product feature extraction and analysis
- Cross-selling and upselling recommendations

**Tools Configuration:**
```python
tools = [
    find_product_details_by_description_tool(
        endpoint_name=endpoint_name,
        index_name=index_name,
        columns=columns,
        k=num_results,
    ),
    create_find_product_by_sku_tool(warehouse_id=warehouse_id),
    create_find_product_by_upc_tool(warehouse_id=warehouse_id),
]
```

**Example Usage:**
```python
from retail_ai.nodes import product_node

# Create agent
product_agent = product_node(model_config)

# Execute
state = AgentState(
    messages=[HumanMessage(content="I'm looking for wireless headphones")],
    user_id="user123",
    store_num="101"
)
response = product_agent.invoke(state, config)
```

**Sample Interactions:**
```
User: "Tell me about SKU STB-KCP-001"
Agent: "That's our Stanley 20oz Tumbler in Charcoal. It features double-wall vacuum insulation, keeps drinks cold for 11 hours or hot for 7 hours. Currently priced at $39.95 with high popularity rating."

User: "I need wireless headphones under $200"
Agent: "I found several great options! The Sony WH-CH720N offers excellent noise cancellation for $149.99, while the JBL Tune 760NC provides 35-hour battery life for $179.99."
```

---

## Inventory Agent
**File:** `retail_ai/nodes.py` → `inventory_node()`

**Specialization:** Stock levels, availability, and inventory management

**Capabilities:**
- Real-time inventory checking across stores and warehouses
- Stock level monitoring and alerts
- Product availability by location
- Inventory movement tracking

**Tools Configuration:**
```python
tools = [
    find_product_details_by_description_tool(...),
    create_find_inventory_by_sku_tool(warehouse_id=warehouse_id),
    create_find_inventory_by_upc_tool(warehouse_id=warehouse_id),
    create_find_store_inventory_by_sku_tool(warehouse_id=warehouse_id),
    create_find_store_inventory_by_upc_tool(warehouse_id=warehouse_id),
]
```

**Sample Interactions:**
```
User: "Do you have the Nintendo Switch OLED in stock at store 101?"
Agent: "Yes! Store 101 has 3 Nintendo Switch OLED units in stock. They're located in Electronics, aisle E-12. We also have 15 units available in our warehouse."

User: "Which stores have the Milwaukee M18 drill in stock?"
Agent: "The Milwaukee M18 drill is available at: Store 101 (5 units), Store 105 (2 units), and Store 110 (8 units). Store 103 is currently out of stock but expecting delivery tomorrow."
```

---

## Comparison Agent
**File:** `retail_ai/nodes.py` → `comparison_node()`

**Specialization:** Product comparisons and recommendation analysis

**Capabilities:**
- Side-by-side product comparisons
- Feature analysis and scoring
- Pros and cons evaluation
- Personalized recommendations based on user needs

**Tools Configuration:**
```python
tools = [
    find_product_details_by_description_tool(...),
    create_find_product_by_sku_tool(warehouse_id=warehouse_id),
    create_find_product_by_upc_tool(warehouse_id=warehouse_id),
]
```

**Sample Interactions:**
```
User: "Compare the DeWalt and Milwaukee cordless drills"
Agent: "Great question! Here's a detailed comparison:

**DeWalt 20V MAX (DCD771C2)**
Pros: Lighter weight (3.6 lbs), longer warranty (3 years), $20 less expensive
Cons: Lower torque (300 in-lbs), shorter battery life

**Milwaukee M18 (2804-20)**  
Pros: Higher torque (725 in-lbs), better battery technology, more durable
Cons: Heavier (4.2 lbs), higher price point

**Recommendation:** For heavy-duty work, choose Milwaukee. For general home use, DeWalt offers better value."
```

---

## DIY Agent
**File:** `retail_ai/nodes.py` → `diy_node()`

**Specialization:** Project guidance, tutorials, and general assistance

**Capabilities:**
- Web search for external information
- General product discovery
- Open-ended customer assistance
- Fallback for queries outside other agent specializations

**Tools Configuration:**
```python
tools = [
    search_tool(model_config),
    find_product_details_by_description_tool(...),
]
```

**Sample Interactions:**
```
User: "How do I install crown molding in my living room?"
Agent: "Here's a step-by-step guide for crown molding installation:

**Tools Needed:** Miter saw, nail gun, measuring tape, stud finder, caulk gun
**Materials:** Crown molding, finishing nails, wood filler, caulk

**Steps:**
1. Measure room perimeter and add 10% for waste
2. Cut inside corners at 45° angles using miter saw
3. Locate wall studs and mark positions
4. Install molding starting with longest wall
5. Use nail gun to secure into studs every 16 inches

Would you like product recommendations for any of these tools?"
```

---

## General Agent
**File:** `retail_ai/nodes.py` → `general_node()`

**Specialization:** Store policies, hours, and general customer service

**Capabilities:**
- Store information and policies
- General customer service inquiries
- Fallback for unrouted queries
- Policy and procedure guidance

**Tools Configuration:**
```python
tools = [
    find_product_details_by_description_tool(...),
]
```

**Sample Interactions:**
```
User: "What are your store hours and return policy?"
Agent: "Our store hours are Monday-Saturday 6AM-10PM, Sunday 8AM-8PM. 

**Return Policy:**
- 90 days for most items with receipt
- 365 days for major appliances
- Original packaging preferred but not required
- Special orders may have different terms

Is there a specific item you're looking to return or exchange?"
```

---

## Recommendation Agent
**File:** `retail_ai/nodes.py` → `recommendation_node()`

**Specialization:** Product recommendations and suggestions

**Capabilities:**
- Personalized product recommendations
- Cross-selling and upselling
- Product discovery based on preferences
- Recommendation explanations

**Tools Configuration:**
```python
tools = [
    find_product_details_by_description_tool(...),
]
```

---

## Agent Orchestration

### Router Agent
**File:** `retail_ai/nodes.py` → `router_node()`

The router agent intelligently directs user queries to the most appropriate specialized agent based on query analysis.

**Routing Logic:**
```yaml
# From model_config.yaml
allowed_routes:
  - product      # Product details, features, specifications
  - comparison   # Comparing multiple products
  - recommendation # Product suggestions and recommendations
  - inventory    # Stock levels and availability
  - diy         # How-to instructions and tutorials
  - orders      # Order tracking and delivery
  - general     # Store policies and general inquiries
```

**Example Routing:**
```python
def route_to_agent(query: str) -> str:
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in ["product", "find", "search"]):
        return "product"
    elif any(keyword in query_lower for keyword in ["stock", "inventory", "available"]):
        return "inventory"
    elif any(keyword in query_lower for keyword in ["compare", "vs", "versus"]):
        return "comparison"
    # ... additional routing logic
```

---

## Related Documentation

- [Agent Development Patterns](../agents/agent-development-patterns.md) - Common patterns for building agents
- [Agent Performance](../agents/agent-performance.md) - Performance metrics and optimization
- [Best Practices](../agents/agent-best-practices.md) - Guidelines for agent development
- [Tools Reference](tools-reference.md) - Detailed tool specifications 