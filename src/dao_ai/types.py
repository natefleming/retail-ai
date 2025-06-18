from typing import Any, Callable, TypeAlias

from dao_ai.state import AgentConfig, AgentState

# Define a type alias for agent node functions in the LangGraph workflow
AgentCallable: TypeAlias = Callable[[AgentState, AgentConfig], dict[str, Any]]
"""
Type definition for agent node functions used in the retail AI LangGraph workflow.

This type alias represents a callable function that:
1. Takes two parameters:
   - AgentState: The current state of the agent conversation workflow
   - AgentConfig: Runtime configuration parameters for agent customization
   
2. Returns a dictionary mapping state keys to updated values
   - Keys are state field names (e.g., "messages", "route", "context")
   - Values are the new content for those fields
   
These functions form the nodes in the agent graph, each specializing in a 
particular task like routing, vector search, database queries, or response generation.
Each node modifies the state and passes it to the next node in the workflow.

Example usage:
```python
@mlflow.trace()
def route_question(state: AgentState, config: AgentConfig) -> dict[str, str]:
    # Process the state
    # ...
    return {"route": "vector_search"}  # Return updates to state
```
"""
