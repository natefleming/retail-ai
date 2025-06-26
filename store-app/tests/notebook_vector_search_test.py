# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search Integration Test
# MAGIC 
# MAGIC This notebook tests the complete vector search integration including:
# MAGIC - Product search by natural language descriptions  
# MAGIC - Store search by location/name
# MAGIC - Full agent workflow integration
# MAGIC - Direct vector search functionality

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Imports

# COMMAND ----------

from typing import Any
from rich import print as pprint
from agent_as_code import app, config
from retail_ai.models import process_messages
from retail_ai.tools.product import find_product_details_by_description_tool
from retail_ai.tools.store import find_store_details_by_location_tool

print("🧪 Vector Search Integration Test Suite")
print("=" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 1: Demo Examples Integration

# COMMAND ----------

# Get examples from config
examples: dict[str, Any] = config.get("app").get("examples")

# Test 1: Initial Inventory Check
print("📦 Test 1: Initial Inventory Check")
print("-" * 30)
input_example = examples.get("demo_initial_inventory_check")
pprint(input_example)

response = process_messages(app=app, **input_example)
print(f"\n✅ Response: {response.choices[0].message.content}")

# COMMAND ----------

# Test 2: Recommendation Request  
print("🎯 Test 2: Recommendation Request")
print("-" * 30)
input_example = examples.get("demo_recommendation_request")
pprint(input_example)

response = process_messages(app=app, **input_example)
print(f"\n✅ Response: {response.choices[0].message.content}")

# COMMAND ----------

# Test 3: Cross-Store Check
print("🏪 Test 3: Cross-Store Inventory Check")
print("-" * 30)
input_example = examples.get("demo_cross_store_check")
pprint(input_example)

response = process_messages(app=app, **input_example)
print(f"\n✅ Response: {response.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 2: Direct Vector Search

# COMMAND ----------

# Get vector search configuration
vector_stores = config.get("resources").get("vector_stores")
products_config = vector_stores.get("products_vector_store")
stores_config = vector_stores.get("store_vector_store")

print("🔍 Vector Search Configuration")
print("-" * 30)
print(f"Product Endpoint: {products_config.get('endpoint_name')}")
print(f"Product Index: {products_config.get('index_name')}")
print(f"Product Columns: {products_config.get('columns')}")
print(f"Store Endpoint: {stores_config.get('endpoint_name')}")
print(f"Store Index: {stores_config.get('index_name')}")
print(f"Store Columns: {stores_config.get('columns')}")

# COMMAND ----------

# Test Product Search
print("👟 Product Vector Search Test")
print("-" * 30)

product_search_tool = find_product_details_by_description_tool(
    endpoint_name=products_config.get("endpoint_name"),
    index_name=products_config.get("index_name"),
    columns=products_config.get("columns"),
    k=5,
)

test_queries = [
    "Adidas Gazelle sneakers",
    "black athletic shoes", 
    "casual footwear",
]

for query in test_queries:
    print(f"\n🔍 Searching for: '{query}'")
    try:
        results = product_search_tool(query)
        print(f"✅ Found {len(results)} products:")
        
        for i, doc in enumerate(results[:3], 1):
            content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  {i}. {content}")
            
    except Exception as e:
        print(f"❌ Error searching for '{query}': {str(e)}")

# COMMAND ----------

# Test Store Search
print("🏪 Store Vector Search Test")
print("-" * 30)

store_search_tool = find_store_details_by_location_tool(
    endpoint_name=stores_config.get("endpoint_name"),
    index_name=stores_config.get("index_name"),
    columns=stores_config.get("columns"),
    k=5,
)

store_queries = [
    "Downtown Market",
    "San Francisco stores",
    "Marina location",
]

for query in store_queries:
    print(f"\n🔍 Searching for: '{query}'")
    try:
        results = store_search_tool(query)
        print(f"✅ Found {len(results)} stores:")
        
        for i, doc in enumerate(results[:3], 1):
            content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  {i}. {content}")
            
    except Exception as e:
        print(f"❌ Error searching for '{query}': {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 3: Custom Queries

# COMMAND ----------

# Test custom queries
print("🎨 Custom Query Tests")
print("-" * 30)

custom_tests = [
    {
        "name": "Product Name to SKU Lookup",
        "query": "What is the SKU for Adidas Gazelle?",
        "expected": "Should find SKU ADI-GAZ-001"
    },
    {
        "name": "Store Name to ID Lookup", 
        "query": "What is the store ID for Downtown Market?",
        "expected": "Should find store ID 101"
    },
    {
        "name": "Natural Language Inventory",
        "query": "Do you have any Adidas sneakers at the Marina store?",
        "expected": "Should search products and check Marina Market inventory"
    }
]

for test in custom_tests:
    print(f"\n🧪 {test['name']}")
    print(f"Query: {test['query']}")
    print(f"Expected: {test['expected']}")
    
    try:
        test_input = {
            "messages": [{"role": "user", "content": test["query"]}],
            "custom_inputs": {
                "configurable": {
                    "thread_id": "test",
                    "user_id": "test.user",
                    "store_num": 101
                }
            }
        }
        
        response = process_messages(app=app, **test_input)
        
        if response and response.choices:
            response_content = response.choices[0].message.content
            print(f"\n✅ Response: {response_content}")
        else:
            print("❌ No response received")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("=" * 50)

# COMMAND ----------

print("✅ All vector search integration tests completed!") 