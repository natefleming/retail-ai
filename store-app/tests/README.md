# Vector Search Integration Tests

This directory contains comprehensive test suites for the vector search integration functionality.

## Test Files

### 1. `test_vector_search_integration.py`
**Comprehensive CLI Test Suite**

A full-featured test script with rich formatting and command-line options.

**Features:**
- ✅ Demo examples integration testing
- ✅ Direct vector search functionality testing  
- ✅ Custom query validation
- ✅ Rich console output with tables and panels
- ✅ Command-line argument support
- ✅ Error handling and reporting

**Usage:**
```bash
# Run all tests
python tests/test_vector_search_integration.py

# Run specific test suites
python tests/test_vector_search_integration.py --test demo_examples
python tests/test_vector_search_integration.py --test vector_search
python tests/test_vector_search_integration.py --test custom_queries
```

### 2. `notebook_vector_search_test.py`
**Databricks Notebook Version**

A notebook-friendly version formatted for Databricks with markdown cells and command separators.

**Usage:**
```python
# In Databricks notebook:
%run tests/notebook_vector_search_test.py

# Or copy individual cells as needed
```

## What These Tests Validate

### 🎭 Demo Examples Integration
Tests the main demo scenarios:
- **Initial Inventory Check**: Product + store lookup with inventory verification
- **Recommendation Request**: Vector similarity-based product recommendations  
- **Cross-Store Check**: Multi-location inventory queries

### 🔍 Direct Vector Search
Tests the underlying vector search functionality:
- **Product Search**: Natural language to product matching
- **Store Search**: Location/name to store matching
- **Configuration Validation**: Endpoint and index verification

### 🎨 Custom Queries
Tests specific use cases:
- **Product Name → SKU**: "Adidas Gazelle" → "ADI-GAZ-001"
- **Store Name → ID**: "Downtown Market" → Store ID 101
- **Natural Language Inventory**: Complex multi-step queries

## Expected Results

### ✅ Successful Test Indicators
- Vector search finds relevant products/stores
- Agents route queries correctly
- SKU and store ID lookups work
- Inventory checks return data
- No column mismatch errors

### ❌ Common Issues to Watch For
- **Column Mismatch**: "Requested columns not present in index"
- **Index Not Found**: Vector search endpoints not provisioned
- **Timeout Errors**: Database queries taking too long
- **Import Errors**: Missing dependencies or wrong directory

## Configuration Dependencies

These tests rely on:
- ✅ Vector search indexes provisioned (`02_provision-vector-search.py`)
- ✅ Database tables populated with data
- ✅ Model configuration aligned (`model_config.yaml`)
- ✅ Agent tools properly configured

## Troubleshooting

### Vector Search Errors
```bash
# Re-provision vector search indexes
python 02_provision-vector-search.py
```

### Column Mismatch Errors
Check that `model_config.yaml` retrievers match actual index columns:
- Products: `product_id`, `sku`, `product_name`, `long_description`
- Stores: `store_id`, `store_name`, `store_address`, `store_city`, `store_state`, `store_details_text`

### Import Errors
```bash
# Ensure you're in the right directory
cd /path/to/retail-ai

# Activate environment
source .venv/bin/activate  # or conda activate retail-ai
```

## Test Output Examples

### Successful Product Search
```
🔍 Searching for: 'Adidas Gazelle sneakers'
✅ Found 5 products:
  1. 12.0, 'ADI-GAZ-001', 'Gazelle Classic Sneakers', 'The Adidas Gazelle is a timeless classic...
  2. 8.0, 'ADI-STS-001', 'Stan Smith Classic Sneakers', 'The Adidas Stan Smith is one of the most...
```

### Successful Agent Response
```
✅ Response: The Adidas Gazelle sneakers in black are available at the Downtown Market location. 
The current stock level is 9 units, and they can be found in Aisle 5A. The retail price is $89.99.
```

## Integration with CI/CD

These tests can be integrated into automated testing pipelines:

```yaml
# Example GitHub Actions step
- name: Run Vector Search Tests
  run: |
    python tests/test_vector_search_integration.py --test all
```

## Contributing

When adding new tests:
1. Add test cases to the appropriate function
2. Update this README with new test descriptions
3. Ensure tests are self-contained and don't require manual setup
4. Include both positive and negative test cases 