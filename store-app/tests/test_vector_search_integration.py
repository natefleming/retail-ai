#!/usr/bin/env python3
"""
Vector Search Integration Test Suite

This script tests the complete vector search integration including:
- Product search by natural language descriptions
- Store search by location/name
- Full agent workflow integration
- Direct vector search functionality

Usage:
    # In a Databricks notebook:
    %run tests/test_vector_search_integration.py
    
    # As a standalone script:
    python tests/test_vector_search_integration.py
    
    # Run specific test sections:
    python tests/test_vector_search_integration.py --test demo_examples
    python tests/test_vector_search_integration.py --test vector_search
    python tests/test_vector_search_integration.py --test all
"""

import sys
import argparse
from typing import Any, Sequence
from rich import print as pprint
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

try:
    from agent_as_code import app, config
    from retail_ai.models import process_messages
    from retail_ai.tools.product import find_product_details_by_description_tool
    from retail_ai.tools.store import find_store_details_by_location_tool
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the retail-ai directory with the environment activated.")
    sys.exit(1)

console = Console()


def print_header(title: str, emoji: str = "🧪") -> None:
    """Print a formatted header for test sections."""
    console.print(Panel(f"{emoji} {title}", style="bold blue"))


def print_subheader(title: str, emoji: str = "📋") -> None:
    """Print a formatted subheader for test subsections."""
    console.print(f"\n{emoji} {title}", style="bold yellow")
    console.print("-" * 50, style="dim")


def test_demo_examples() -> None:
    """Test the main demo examples from the configuration."""
    print_header("Demo Examples Integration Test", "🎭")
    
    examples: dict[str, Any] = config.get("app").get("examples")
    
    demo_tests = [
        ("demo_initial_inventory_check", "📦 Initial Inventory Check", 
         "Tests product and store lookup with inventory check"),
        ("demo_recommendation_request", "🎯 Recommendation Request", 
         "Tests product recommendations with vector similarity"),
        ("demo_cross_store_check", "🏪 Cross-Store Inventory Check", 
         "Tests inventory lookup across multiple stores"),
    ]
    
    results = []
    
    for test_key, test_name, description in demo_tests:
        print_subheader(test_name)
        console.print(f"Description: {description}", style="italic")
        
        try:
            input_example = examples.get(test_key)
            if not input_example:
                console.print(f"❌ Test example '{test_key}' not found in config", style="red")
                results.append((test_name, "FAILED", "Example not found"))
                continue
                
            console.print("\nInput:", style="bold")
            pprint(input_example)
            
            console.print("\nProcessing...", style="yellow")
            response = process_messages(app=app, **input_example)
            
            if response and response.choices:
                response_content = response.choices[0].message.content
                console.print(f"\n✅ Response:", style="green bold")
                console.print(response_content, style="green")
                results.append((test_name, "PASSED", "Response received"))
            else:
                console.print("❌ No response received", style="red")
                results.append((test_name, "FAILED", "No response"))
                
        except Exception as e:
            console.print(f"❌ Error: {str(e)}", style="red")
            results.append((test_name, "FAILED", str(e)))
        
        console.print("\n" + "="*70)
    
    # Print summary table
    print_subheader("Demo Examples Summary", "📊")
    table = Table(title="Test Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="white")
    
    for test_name, status, details in results:
        status_style = "green" if status == "PASSED" else "red"
        table.add_row(test_name, f"[{status_style}]{status}[/{status_style}]", details)
    
    console.print(table)


def test_vector_search_direct() -> None:
    """Test vector search functionality directly."""
    print_header("Direct Vector Search Test", "🔍")
    
    try:
        # Get vector search configuration
        vector_stores = config.get("resources").get("vector_stores")
        products_config = vector_stores.get("products_vector_store")
        stores_config = vector_stores.get("store_vector_store")
        
        print_subheader("Configuration Check", "⚙️")
        
        # Display configuration
        config_table = Table(title="Vector Search Configuration")
        config_table.add_column("Component", style="cyan")
        config_table.add_column("Value", style="white")
        
        config_table.add_row("Product Endpoint", products_config.get("endpoint_name"))
        config_table.add_row("Product Index", products_config.get("index_name"))
        config_table.add_row("Product Columns", str(products_config.get("columns")))
        config_table.add_row("Store Endpoint", stores_config.get("endpoint_name"))
        config_table.add_row("Store Index", stores_config.get("index_name"))
        config_table.add_row("Store Columns", str(stores_config.get("columns")))
        
        console.print(config_table)
        
        # Test Product Search
        print_subheader("Product Vector Search", "👟")
        
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
            console.print(f"\n🔍 Searching for: '{query}'", style="bold")
            try:
                results = product_search_tool(query)
                console.print(f"✅ Found {len(results)} products:", style="green")
                
                for i, doc in enumerate(results[:3], 1):
                    content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    console.print(f"  {i}. {content}", style="dim")
                    
            except Exception as e:
                console.print(f"❌ Error searching for '{query}': {str(e)}", style="red")
        
        # Test Store Search
        print_subheader("Store Vector Search", "🏪")
        
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
            console.print(f"\n🔍 Searching for: '{query}'", style="bold")
            try:
                results = store_search_tool(query)
                console.print(f"✅ Found {len(results)} stores:", style="green")
                
                for i, doc in enumerate(results[:3], 1):
                    content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    console.print(f"  {i}. {content}", style="dim")
                    
            except Exception as e:
                console.print(f"❌ Error searching for '{query}': {str(e)}", style="red")
                
    except Exception as e:
        console.print(f"❌ Configuration Error: {str(e)}", style="red")


def test_custom_queries() -> None:
    """Test custom queries to validate the system."""
    print_header("Custom Query Test", "🎨")
    
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
        print_subheader(test["name"], "🧪")
        console.print(f"Query: {test['query']}", style="bold")
        console.print(f"Expected: {test['expected']}", style="italic")
        
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
                console.print(f"\n✅ Response:", style="green bold")
                console.print(response_content, style="green")
            else:
                console.print("❌ No response received", style="red")
                
        except Exception as e:
            console.print(f"❌ Error: {str(e)}", style="red")
        
        console.print("\n" + "="*70)


def run_all_tests() -> None:
    """Run all test suites."""
    console.print(Panel("🚀 Vector Search Integration Test Suite", style="bold green"))
    console.print("Testing complete vector search integration...\n")
    
    test_demo_examples()
    test_vector_search_direct()
    test_custom_queries()
    
    console.print(Panel("✅ All tests completed!", style="bold green"))


def main():
    """Main function to run tests based on command line arguments."""
    parser = argparse.ArgumentParser(description="Vector Search Integration Tests")
    parser.add_argument(
        "--test", 
        choices=["demo_examples", "vector_search", "custom_queries", "all"],
        default="all",
        help="Which test suite to run"
    )
    
    args = parser.parse_args()
    
    if args.test == "demo_examples":
        test_demo_examples()
    elif args.test == "vector_search":
        test_vector_search_direct()
    elif args.test == "custom_queries":
        test_custom_queries()
    else:
        run_all_tests()


if __name__ == "__main__":
    main() 