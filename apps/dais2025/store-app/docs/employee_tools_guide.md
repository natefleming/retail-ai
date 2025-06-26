# Employee Tools Guide

This guide explains how to use the new employee performance tools to answer questions like "who are the top employees in X department?" and identify the best associates for personal shopping appointments.

## Overview

The employee tools provide AI agents with the ability to:
- Find top performing employees by department
- Identify best personal shopping associates
- Extract department names from natural language queries
- Support manager decision-making for staffing and recognition

## Tools Available

### 1. `find_top_employees_by_department`

**Purpose**: Find the highest performing employees within a specific department.

**Parameters**:
- `department` (str): Department name (e.g., "Electronics", "Footwear", "Customer Service", "Womens Fashion")
- `limit` (int, optional): Maximum number of employees to return (default: 5)

**Returns**: Employee performance data including:
- Employee ID, name, position title, store name
- Overall performance score and department ranking
- Sales achievement percentage and customer satisfaction score
- Task completion rate, attendance rate
- Personal shopping sessions (for fashion departments)
- Recognition points and awards

**Example Usage**:
```python
# Find top 3 employees in Electronics department
result = find_top_employees_by_department.invoke({
    "department": "Electronics", 
    "limit": 3
})
```

### 2. `find_personal_shopping_associates`

**Purpose**: Identify the best associates for personal shopping appointments.

**Parameters**:
- `store_id` (str, optional): Specific store ID to search within ("101", "102", "103")
- `expertise_level` (str, optional): Filter by expertise level ("Expert", "Advanced", "Intermediate", "Beginner")

**Returns**: Personal shopping associate data including:
- Store and employee information
- Personal shopping sessions completed
- Customer satisfaction and product knowledge scores
- Comprehensive expertise score and ranking
- Expertise level classification

**Example Usage**:
```python
# Find expert personal shopping associates at Downtown Market
result = find_personal_shopping_associates.invoke({
    "store_id": "101", 
    "expertise_level": "Expert"
})
```

### 3. `department_extraction`

**Purpose**: Extract department names from natural language text using LLM.

**Parameters**:
- `input` (str): Natural language text that may contain department names

**Returns**: List of extracted department names

**Example Usage**:
```python
# Extract departments from user query
result = department_extraction.invoke({
    "input": "Who are the best employees in electronics and footwear?"
})
# Returns: ["Electronics", "Footwear"]
```

## Integration with ToolFactory

The employee tools are automatically included when using the `ToolFactory`:

```python
from retail_ai.tools import ToolFactory
from mlflow.models import ModelConfig

# Initialize factory
config = ModelConfig({"catalog_name": "your_catalog", "database_name": "your_db"})
factory = ToolFactory(config)

# Create all tools including employee tools
tools = factory.create_all_tools(
    llm=your_llm,
    warehouse_id="your_warehouse_id",
    product_endpoint_name="your_endpoint",
    product_index_name="your_index", 
    product_columns=["col1", "col2"]
)

# Access employee tools
employee_tools = tools["employee"]
top_employees_tool = employee_tools["find_top_employees_by_department"]
personal_shopping_tool = employee_tools["find_personal_shopping_associates"]
```

## Sample Queries and Responses

### Query: "Who are the top employees in Women's Fashion?"

**Tool Used**: `find_top_employees_by_department`
**Parameters**: `{"department": "Womens Fashion", "limit": 5}`

**Sample Response**:
```json
[
  {
    "employee_id": "EMP-016",
    "employee_name": "Isabella Rodriguez", 
    "position_title": "Fashion Department Lead",
    "store_name": "Downtown Market",
    "overall_performance_score": 4.90,
    "performance_ranking_in_department": 1,
    "sales_achievement_percentage": 116.40,
    "customer_satisfaction_score": 4.9,
    "personal_shopping_sessions": 18,
    "employee_of_month_awards": 3
  },
  {
    "employee_id": "EMP-017", 
    "employee_name": "Olivia Chen",
    "position_title": "Senior Style Consultant",
    "store_name": "Downtown Market",
    "overall_performance_score": 4.80,
    "performance_ranking_in_department": 2,
    "sales_achievement_percentage": 111.43,
    "customer_satisfaction_score": 4.8,
    "personal_shopping_sessions": 16,
    "employee_of_month_awards": 2
  }
]
```

### Query: "Who can help with a personal shopping appointment?"

**Tool Used**: `find_personal_shopping_associates`
**Parameters**: `{}`

**Sample Response**:
```json
[
  {
    "store_id": "101",
    "store_name": "Downtown Market", 
    "employee_id": "EMP-018",
    "employee_name": "Grace Williams",
    "position_title": "Personal Stylist",
    "personal_shopping_sessions": 22,
    "customer_satisfaction_score": 4.8,
    "product_knowledge_score": 4.7,
    "comprehensive_score": 95.2,
    "expertise_level": "Expert",
    "overall_rank": 1
  }
]
```

## Database Requirements

### Tables Required:
- `employee_performance`: Main performance tracking table
- Views: `top_employees_by_department`, `top_personal_shopping_associates_all_stores`

### Key Performance Metrics:
- **Overall Performance Score**: Calculated composite score (1.0-5.0)
- **Sales Achievement**: Percentage of sales target achieved
- **Customer Satisfaction**: Average customer rating (1.0-5.0)
- **Task Completion Rate**: Percentage of tasks completed on time
- **Personal Shopping Sessions**: Number of personal shopping appointments conducted
- **Product Knowledge Score**: Assessment score for product expertise

## Use Cases

### For Store Managers:
- **Recognition Programs**: Identify top performers for employee of the month
- **Promotion Decisions**: Find employees ready for advancement
- **Training Needs**: Identify areas where employees need development
- **Staffing Decisions**: Assign best performers to high-priority tasks

### For Personal Shopping Services:
- **Appointment Scheduling**: Match customers with best-suited associates
- **Expertise Matching**: Find associates with specific product knowledge
- **Quality Assurance**: Ensure high customer satisfaction scores
- **Cross-Store Recommendations**: Find best associates across all locations

### For HR and Analytics:
- **Performance Analysis**: Track department-level performance trends
- **Benchmarking**: Compare employee performance across stores
- **Retention**: Identify high performers for retention programs
- **Development Planning**: Create targeted development plans

## Configuration

Ensure your `model_config.yaml` includes:

```yaml
catalog_name: "your_catalog_name"
database_name: "your_database_name"
```

The tools will automatically use these values to query the correct Unity Catalog location.

## Error Handling

The tools include comprehensive error handling:
- Invalid department names return empty results
- Missing store IDs are handled gracefully  
- Database connection issues are logged and return empty tuples
- SQL query failures are captured and logged

## Performance Considerations

- Queries use pre-built views for optimal performance
- Results are limited to prevent overwhelming responses
- Indexes on department, store_id, and performance scores recommended
- Consider caching for frequently accessed data 