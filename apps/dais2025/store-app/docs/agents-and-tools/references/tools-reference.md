# Tools Reference Guide

This guide provides detailed technical information about all available tools in the Retail AI system, including input/output specifications, implementation details, and usage examples.

## Tools by Category

### Product Discovery & Lookup

#### `find_product_by_sku`
**Type:** Unity Catalog Function  
**Purpose:** Exact product lookup by SKU identifier

**Input:**
```python
{
    "sku": ["STB-KCP-001", "DUN-KCP-001"]  # Array of SKU strings
}
```

**Output:**
```python
[
    {
        "product_id": 12345,
        "sku": "STB-KCP-001",
        "upc": "012345678901",
        "brand_name": "Stanley",
        "product_name": "Stanley 20oz Tumbler",
        "short_description": "Insulated tumbler with lid",
        "long_description": "Double-wall vacuum insulated tumbler...",
        "merchandise_class": "Kitchen & Dining",
        "class_cd": "KD001",
        "department_name": "Home",
        "category_name": "Drinkware",
        "subcategory_name": "Tumblers",
        "base_price": 39.95,
        "msrp": 44.99
    }
]
```

**Implementation:**
```python
# Direct Unity Catalog execution
result = client.execute_function(
    function_name="catalog.database.find_product_by_sku",
    parameters={"sku": ["STB-KCP-001"]}
)

# LangChain tool wrapper
tool = create_find_product_by_sku_tool(warehouse_id)
result = tool.invoke({"sku": ["STB-KCP-001"]})
```

---

#### `find_product_by_upc`
**Type:** Unity Catalog Function  
**Purpose:** Exact product lookup by UPC barcode

**Input:**
```python
{
    "upc": ["012345678901", "123456789012"]  # Array of UPC strings
}
```

**Output:** Same structure as `find_product_by_sku`

---

#### `find_product_details_by_description`
**Type:** Vector Search Tool  
**Purpose:** Semantic product search using natural language

**Input:**
```python
"wireless bluetooth headphones with noise cancellation"
```

**Output:**
```python
[
    {
        "page_content": "Sony WH-CH720N Wireless Noise Canceling Headphones",
        "metadata": {
            "product_id": "67890",
            "sku": "SONY-WH720N",
            "brand_name": "Sony",
            "price": 149.99,
            "category": "Electronics",
            "score": 0.89  # Similarity score
        }
    },
    {
        "page_content": "JBL Tune 760NC Wireless Over-Ear Headphones",
        "metadata": {
            "product_id": "67891",
            "sku": "JBL-760NC",
            "brand_name": "JBL",
            "price": 179.99,
            "category": "Electronics", 
            "score": 0.85
        }
    }
]
```

**Implementation:**
```python
vector_tool = find_product_details_by_description_tool(
    endpoint_name="vs_endpoint",
    index_name="products_index",
    columns=["product_name", "description", "category", "price"],
    k=10
)
results = vector_tool.invoke("wireless bluetooth headphones")
```

---

### Inventory Management

#### `find_inventory_by_sku`
**Type:** Unity Catalog Function  
**Purpose:** Global inventory lookup across all stores

**Input:**
```python
{
    "sku": ["STB-KCP-001"]
}
```

**Output:**
```python
[
    {
        "inventory_id": 98765,
        "sku": "STB-KCP-001",
        "upc": "012345678901",
        "product_id": 12345,
        "store_id": 101,
        "store_quantity": 15,
        "warehouse": "WH-CENTRAL",
        "warehouse_quantity": 250,
        "retail_amount": 39.95,
        "popularity_rating": "High",
        "department": "Home",
        "aisle_location": "H-12",
        "is_closeout": false
    },
    {
        "inventory_id": 98766,
        "sku": "STB-KCP-001", 
        "upc": "012345678901",
        "product_id": 12345,
        "store_id": 105,
        "store_quantity": 8,
        "warehouse": "WH-CENTRAL",
        "warehouse_quantity": 250,
        "retail_amount": 39.95,
        "popularity_rating": "High",
        "department": "Home",
        "aisle_location": "H-15",
        "is_closeout": false
    }
]
```

---

#### `find_store_inventory_by_sku`
**Type:** Unity Catalog Function  
**Purpose:** Store-specific inventory lookup

**Input:**
```python
{
    "store_id": 101,
    "sku": ["STB-KCP-001", "DUN-KCP-001"]
}
```

**Output:** Same structure as `find_inventory_by_sku` but filtered to specified store

---

### Product Analysis & Comparison

#### `product_comparison`
**Type:** LangChain Tool  
**Purpose:** AI-powered product comparison and analysis

**Input:**
```python
[
    {
        "product_id": "A",
        "product_name": "DeWalt 20V MAX Drill",
        "price": 129.99,
        "torque": "300 in-lbs",
        "weight": "3.6 lbs",
        "battery_life": "2 hours",
        "warranty": "3 years"
    },
    {
        "product_id": "B", 
        "product_name": "Milwaukee M18 Drill",
        "price": 149.99,
        "torque": "725 in-lbs",
        "weight": "4.2 lbs", 
        "battery_life": "1.5 hours",
        "warranty": "5 years"
    }
]
```

**Output:**
```python
{
    "products": [
        {
            "product_id": "A",
            "product_name": "DeWalt 20V MAX Drill",
            "attributes": [
                {
                    "feature": "Weight",
                    "value": "3.6 lbs",
                    "rating": 8,
                    "pros": ["Lightweight", "Easy to handle"],
                    "cons": ["Less robust feel"]
                }
            ],
            "overall_rating": 7,
            "price_value_ratio": 8,
            "summary": "Great for home use and light projects"
        }
    ],
    "key_features": [
        {
            "name": "Torque",
            "description": "Drilling and driving power",
            "importance": 9
        }
    ],
    "winner": "B",
    "best_value": "A",
    "comparison_summary": "Milwaukee offers superior torque for heavy-duty tasks, while DeWalt provides better value for home users.",
    "recommendations": [
        "Choose Milwaukee for professional use",
        "Choose DeWalt for home projects and budget-conscious buyers"
    ]
}
```

**Implementation:**
```python
comparison_tool = create_product_comparison_tool(llm)
result = comparison_tool.invoke([product1, product2])
```

---

#### `sku_extraction`
**Type:** LangChain Tool  
**Purpose:** Extract SKU identifiers from natural language text

**Input:**
```python
"I'm looking for information about SKU ABC123 and also need details on product DUN-KCP-001"
```

**Output:**
```python
["ABC123", "DUN-KCP-001"]
```

**Implementation:**
```python
sku_tool = create_sku_extraction_tool(llm)
skus = sku_tool.invoke("I need info about SKU ABC123")
```

---

#### `product_classification`
**Type:** LangChain Tool  
**Purpose:** Classify products into predefined categories

**Input:**
```python
"wireless bluetooth headphones with noise cancellation"
```

**Output:**
```python
["Electronics"]
```

**Configuration:**
```python
categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
classification_tool = create_product_classification_tool(
    llm=llm,
    allowable_classifications=categories,
    k=1
)
```

---

### Data Access & Queries

#### `genie_tool`
**Type:** Databricks Genie  
**Purpose:** Natural language to SQL query translation

**Input:**
```python
"What are the top 5 selling products this month?"
```

**Output:**
```python
{
    "content": "Here are the top 5 selling products this month:\n1. Stanley Tumbler - 1,250 units\n2. Milwaukee Drill - 890 units...",
    "sql_query": "SELECT product_name, SUM(quantity_sold) as total_sold FROM sales WHERE month = CURRENT_MONTH() GROUP BY product_name ORDER BY total_sold DESC LIMIT 5",
    "data": [
        {"product_name": "Stanley Tumbler", "total_sold": 1250},
        {"product_name": "Milwaukee Drill", "total_sold": 890}
    ]
}
```

**Implementation:**
```python
genie = create_genie_tool(space_id="workspace_id")
response = genie.invoke("What are the top selling products?")
```

---

#### `search_tool`
**Type:** External API (DuckDuckGo)  
**Purpose:** Web search for external information

**Input:**
```python
"latest smartphone reviews 2024"
```

**Output:**
```python
[
    "iPhone 15 Pro review: The best iPhone yet with titanium design...",
    "Samsung Galaxy S24 Ultra review: S Pen and AI features shine...",
    "Google Pixel 8 Pro review: Camera excellence continues..."
]
```

---

## Tool Implementation Patterns

### Unity Catalog Functions

**Characteristics:**
- Fastest execution (direct SQL)
- Exact match queries
- Structured tabular output
- Built-in governance

**When to Use:**
- Known SKUs or UPCs
- Real-time inventory checks
- High-volume queries
- Exact data lookups

**Implementation Pattern:**
```python
def create_uc_function_tool(warehouse_id: str, function_name: str):
    @tool
    def uc_function_tool(params: dict) -> tuple:
        w = WorkspaceClient()
        
        # Build SQL statement
        param_str = format_parameters(params)
        statement = f"SELECT * FROM {function_name}({param_str})"
        
        # Execute
        response = w.statement_execution.execute_statement(
            statement=statement, 
            warehouse_id=warehouse_id
        )
        
        # Wait for completion
        while response.status.state in [StatementState.PENDING, StatementState.RUNNING]:
            response = w.statement_execution.get_statement(response.statement_id)
        
        return response.result.data_array
    
    return uc_function_tool
```

---

### Vector Search Tools

**Characteristics:**
- Semantic similarity matching
- Scalable to large datasets
- Relevance-based ranking
- MLflow integration

**When to Use:**
- Natural language queries
- "Find similar" requests
- Content discovery
- Recommendation systems

**Implementation Pattern:**
```python
def create_vector_search_tool(
    endpoint_name: str,
    index_name: str,
    columns: Sequence[str],
    k: int = 10
):
    @tool
    @mlflow.trace(span_type="RETRIEVER", name="vector_search")
    def vector_search(query: str) -> Sequence[Document]:
        vector_store = DatabricksVectorSearch(
            endpoint=endpoint_name,
            index_name=index_name,
            columns=columns
        )
        
        documents = vector_store.similarity_search(
            query=query, 
            k=k, 
            filter={}
        )
        
        return documents
    
    return vector_search
```

---

### LangChain Tools

**Characteristics:**
- Natural language understanding
- Flexible input/output
- Structured output (Pydantic)
- Creative and analytical

**When to Use:**
- Complex analysis tasks
- Text processing
- Classification
- Comparison and reasoning

**Implementation Pattern:**
```python
def create_langchain_tool(llm: LanguageModelLike):
    class OutputModel(BaseModel):
        result: str = Field(description="Tool output")
    
    @tool
    def langchain_tool(input: str) -> str:
        llm_with_structure = llm.with_structured_output(OutputModel)
        result = llm_with_structure.invoke(input)
        return result.result
    
    return langchain_tool
```

---

## Performance Comparison

| Tool Type | Avg Latency | Throughput | Accuracy | Use Case |
|-----------|-------------|------------|----------|----------|
| **Unity Catalog** | 200ms | 1000 req/s | 99% | Exact lookups |
| **Vector Search** | 300ms | 500 req/s | 95% | Semantic search |
| **LangChain** | 1.5s | 100 req/s | 92% | Analysis tasks |
| **External APIs** | 2s | 50 req/s | 88% | Real-time data |

---

## Tool Selection Guide

### Decision Matrix

| Query Type | Primary Tool | Fallback Tool | Example |
|------------|--------------|---------------|---------|
| **Exact SKU** | Unity Catalog | Vector Search | "Tell me about SKU ABC123" |
| **Product Description** | Vector Search | LangChain | "wireless headphones" |
| **Comparison** | LangChain | Vector Search | "compare iPhone vs Samsung" |
| **How-to Questions** | External Search | Vector Search | "how to install crown molding" |
| **Inventory Check** | Unity Catalog | - | "stock levels for store 101" |

### Performance Optimization

```python
def optimized_tool_selection(query: str, context: dict):
    # Extract SKUs first (fastest path)
    skus = extract_skus(query)
    if skus:
        return unity_catalog_lookup(skus)
    
    # Check for comparison keywords
    if is_comparison_query(query):
        products = vector_search(query)
        return langchain_comparison(products)
    
    # Default to semantic search
    return vector_search(query)
```

---

## Advanced Usage Patterns

### Tool Chaining
```python
def product_discovery_chain(user_query: str):
    # Step 1: Extract any mentioned SKUs
    skus = sku_extraction_tool(user_query)
    
    if skus:
        # Step 2a: Direct lookup for known SKUs
        products = find_product_by_sku_tool(skus)
    else:
        # Step 2b: Semantic search for descriptions
        products = vector_search_tool(user_query)
    
    # Step 3: Check inventory for found products
    inventory = find_inventory_by_sku_tool([p['sku'] for p in products])
    
    return combine_product_and_inventory(products, inventory)
```

### Parallel Execution
```python
import asyncio

async def parallel_search(query: str):
    tasks = [
        vector_search_tool.ainvoke(query),
        web_search_tool.ainvoke(query),
        classification_tool.ainvoke(query)
    ]
    
    results = await asyncio.gather(*tasks)
    return combine_results(results)
```

### Caching Strategy
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_product_lookup(sku: str):
    return find_product_by_sku_tool([sku])

def smart_product_lookup(sku: str):
    # Try cache first
    try:
        return cached_product_lookup(sku)
    except Exception:
        # Fallback to direct lookup
        return find_product_by_sku_tool([sku])
```

---

## Troubleshooting

### Common Issues

**Tool Not Found**
```python
# Check tool registration
available_tools = [tool.name for tool in agent.tools]
print(f"Available: {available_tools}")
```

**Performance Issues**
```python
# Enable detailed logging
import logging
logging.getLogger("retail_ai.tools").setLevel(logging.DEBUG)

# Profile tool execution
import time
start = time.time()
result = tool.invoke(input)
print(f"Execution time: {time.time() - start:.2f}s")
```

**Authentication Errors**
```python
# Test Databricks connection
from databricks.sdk import WorkspaceClient
try:
    w = WorkspaceClient()
    print(f"Connected as: {w.current_user.me().user_name}")
except Exception as e:
    print(f"Connection failed: {e}")
```

---

## Related Documentation

- [Agent Overview](../overview.md) - How agents use these tools
- [AI Agents Implementation](../ai-agents.md) - Advanced agent patterns with tools
- [Development Guide](../../development/contributing.md) - Building custom tools 