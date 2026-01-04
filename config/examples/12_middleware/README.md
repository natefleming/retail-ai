# Middleware Examples

Middleware allows you to add cross-cutting concerns to your agents such as validation, logging, authentication, rate limiting, and monitoring. Middleware runs before and after agent execution, providing a powerful way to enhance agent behavior without modifying agent code.

## 📚 What is Middleware?

Middleware are functions that wrap around agent execution to:
- **Validate inputs** before processing
- **Log requests** for debugging and auditing  
- **Monitor performance** and track metrics
- **Enforce rate limits** and quotas
- **Add authentication** and authorization
- **Handle errors** gracefully
- **Transform inputs/outputs** as needed

## 🗂️ Examples in This Directory

### 1. [`custom_field_validation.yaml`](custom_field_validation.yaml)
**Input validation for required context fields**

Learn how to validate that required fields (like `store_num`, `tenant_id`, `api_key`) are provided in custom inputs before agent execution.

**Key Concepts:**
- Required vs optional fields
- Custom error messages
- Multi-tenant validation
- API key validation

**Use Cases:**
- Multi-location businesses requiring store context
- Enterprise apps with tenant isolation
- APIs requiring authentication tokens
- Any scenario requiring custom context validation

**Example:**
```yaml
middleware:
  store_validation: &store_validation
    name: dao_ai.middleware.create_custom_field_validation_middleware
    args:
      fields:
        - name: store_num
          description: "Your store number for inventory lookups"
          example_value: "12345"
        - name: user_id
          description: "Your unique user identifier"
          required: false
          example_value: "user_abc123"

agents:
  my_agent:
    middleware:
      - *store_validation
```

---

### 2. [`logging_middleware.yaml`](logging_middleware.yaml)
**Comprehensive logging patterns**

Demonstrates different logging strategies including request logging, performance monitoring, and audit trails.

**Key Concepts:**
- Request/response logging
- Performance metrics
- Audit trails for compliance
- Sensitive data masking

**Use Cases:**
- Debugging agent behavior
- Performance optimization
- Compliance and auditing
- Security monitoring
- Cost tracking

**Example:**
```yaml
middleware:
  request_logger: &request_logger
    name: dao_ai.middleware.create_logging_middleware
    args:
      log_level: INFO
      log_inputs: true
      log_outputs: false
      include_metadata: true
      message_prefix: "[REQUEST]"
```

---

### 3. [`combined_middleware.yaml`](combined_middleware.yaml)
**Complete middleware stack for production**

Shows how to combine multiple middleware components into a comprehensive processing pipeline.

**Key Concepts:**
- Middleware execution order
- Production vs development stacks
- Error handling across middleware
- Performance considerations

**Use Cases:**
- Production-ready agents
- Environment-specific configurations
- Multi-layer security
- Comprehensive monitoring

**Example:**
```yaml
agents:
  production_agent:
    middleware:
      - *input_validation      # 1. Validate first
      - *request_logging       # 2. Log requests
      - *rate_limiting         # 3. Enforce limits
      - *performance_tracking  # 4. Monitor speed
      - *audit_logging         # 5. Create audit trail
```

---

### 4. [`limit_middleware.yaml`](limit_middleware.yaml)
**Tool and model call limiting**

Demonstrates how to prevent excessive API calls and runaway loops by limiting tool and LLM calls.

**Key Concepts:**
- Global vs tool-specific limits
- Thread limits (per conversation) vs run limits (per message)
- Exit behaviors: `continue`, `error`, `end`
- Model call limits for LLM cost control

**Use Cases:**
- Budget control for expensive API calls
- Preventing infinite loops in agent execution
- Resource management for database queries
- Controlling LLM API costs

**Example:**
```yaml
middleware:
  # Global limit across all tools
  global_limit: &global_limit
    name: dao_ai.middleware.create_tool_call_limit_middleware
    args:
      thread_limit: 50
      run_limit: 15
      exit_behavior: continue

  # Tool-specific limit using YAML alias
  genie_limit: &genie_limit
    name: dao_ai.middleware.create_tool_call_limit_middleware
    args:
      tool: *genie_tool         # Reference tool directly
      run_limit: 3
      exit_behavior: continue

  # Model call limit
  model_limit: &model_limit
    name: dao_ai.middleware.create_model_call_limit_middleware
    args:
      run_limit: 20
      exit_behavior: end        # Only "end" or "error" for model limits
```

---

### 5. [`retry_middleware.yaml`](retry_middleware.yaml)
**Automatic retry with exponential backoff**

Demonstrates how to add automatic retry logic for transient failures in tool and model calls.

**Key Concepts:**
- Exponential backoff configuration
- Jitter to prevent thundering herd
- Tool-specific retry configuration
- Model retry for LLM API resilience

**Use Cases:**
- Recovering from transient network errors
- Handling rate limits from external APIs
- Building resilient production agents
- Graceful degradation during outages

**Example:**
```yaml
middleware:
  # Tool retry with backoff
  tool_retry: &tool_retry
    name: dao_ai.middleware.create_tool_retry_middleware
    args:
      max_retries: 3
      backoff_factor: 2.0
      initial_delay: 1.0
      max_delay: 30.0
      jitter: true
      tools:
        - *genie_tool           # Reference tool directly
        - search_web            # Or use string name

  # Model retry for LLM calls
  model_retry: &model_retry
    name: dao_ai.middleware.create_model_retry_middleware
    args:
      max_retries: 3
      backoff_factor: 2.0
      jitter: true
```

---

### 6. [`context_management.yaml`](context_management.yaml)
**Context editing for long conversations**

Demonstrates how to manage conversation context by clearing older tool outputs to prevent token limit issues.

**Key Concepts:**
- Token threshold triggering
- Preserving recent tool outputs
- Excluding specific tools from clearing
- Approximate vs model-based token counting

**Use Cases:**
- Preventing context window overflow
- Reducing token costs in long conversations
- Managing memory-intensive agents
- Optimizing production performance

**Example:**
```yaml
middleware:
  context_editor: &context_editor
    name: dao_ai.middleware.create_context_editing_middleware
    args:
      trigger: 100000           # Clear when exceeding ~100k tokens
      keep: 3                   # Keep last 3 tool outputs
      exclude_tools:
        - *time_tool            # Never clear time outputs
      placeholder: "[cleared]"
```

---

### 7. [`pii_middleware.yaml`](pii_middleware.yaml)
**Personally Identifiable Information protection**

Demonstrates how to detect and handle sensitive personal information for privacy compliance.

**Key Concepts:**
- Built-in PII types (email, phone, SSN, credit card, etc.)
- Strategies: redact, mask, hash, block
- Input vs output vs tool result protection
- Compliance considerations (GDPR, HIPAA, PCI-DSS)

**Use Cases:**
- GDPR/CCPA compliance
- Healthcare applications (HIPAA)
- Financial services data protection
- Enterprise employee data handling

**Example:**
```yaml
middleware:
  # Redact email addresses
  email_protection: &email_protection
    name: dao_ai.middleware.create_pii_middleware
    args:
      pii_type: email
      strategy: redact
      apply_to_input: true
      apply_to_output: true

  # Block SSN entirely
  ssn_protection: &ssn_protection
    name: dao_ai.middleware.create_pii_middleware
    args:
      pii_type: ssn
      strategy: block           # Reject messages with SSN
      apply_to_input: true

  # Mask credit cards
  cc_protection: &cc_protection
    name: dao_ai.middleware.create_pii_middleware
    args:
      pii_type: credit_card
      strategy: mask
      apply_to_tool_results: true
```

## 🚀 Quick Start

### Step 1: Define Middleware

Define reusable middleware at the app level:

```yaml
middleware:
  my_middleware: &my_middleware
    name: dao_ai.middleware.my_middleware_factory
    args:
      key: value
```

### Step 2: Apply to Agents

Reference middleware in agent configurations:

```yaml
agents:
  my_agent:
    name: my_agent
    model: *llm
    middleware:
      - *my_middleware
    prompt: |
      Your agent prompt here
```

### Step 3: Test

Run your agent with required inputs:

```bash
dao-ai chat -c config/examples/12_middleware/custom_field_validation.yaml
```

## 📋 Common Middleware Patterns

### Input Validation
**When to Use:** Always validate required context fields
```yaml
middleware:
  validation: &validation
    name: dao_ai.middleware.create_custom_field_validation_middleware
    args:
      fields:
        - name: required_field
          description: "Field description"
          example_value: "example"
```

### Request Logging
**When to Use:** Debug issues, track usage, audit access
```yaml
middleware:
  logging: &logging
    name: dao_ai.middleware.create_logging_middleware
    args:
      log_level: INFO
      log_inputs: true
      log_outputs: true
```

### Performance Monitoring
**When to Use:** Optimize slow agents, track SLAs
```yaml
middleware:
  performance: &performance
    name: dao_ai.middleware.create_performance_middleware
    args:
      threshold_ms: 1000
      include_tool_timing: true
```

### Rate Limiting
**When to Use:** Prevent abuse, control costs
```yaml
middleware:
  rate_limit: &rate_limit
    name: dao_ai.middleware.create_rate_limit_middleware
    args:
      max_requests_per_minute: 60
      rate_limit_by: user_id
```

### Audit Trail
**When to Use:** Compliance, security, investigation
```yaml
middleware:
  audit: &audit
    name: dao_ai.middleware.create_audit_middleware
    args:
      log_user_info: true
      log_tool_calls: true
      mask_sensitive_fields: true
```

## 🎯 Best Practices

### 1. **Order Matters**
Middleware executes in the order defined. Put validation first, logging early, and expensive operations last:

```yaml
middleware:
  - *validation        # Fail fast
  - *logging           # Capture everything
  - *rate_limiting     # Before expensive ops
  - *performance       # Around main execution
  - *audit             # Comprehensive tracking
```

### 2. **Environment-Specific Stacks**
Use different middleware for different environments:

```yaml
# Development
agents:
  dev_agent:
    middleware:
      - *logging         # Logging only

# Production
agents:
  prod_agent:
    middleware:
      - *validation
      - *logging
      - *rate_limiting
      - *performance
      - *audit
```

### 3. **Error Handling**
Each middleware should:
- Handle errors gracefully
- Return clear error messages
- Not block requests on non-critical failures
- Log errors appropriately

### 4. **Performance**
Keep middleware lightweight:
- Avoid blocking operations
- Cache expensive checks
- Use async where possible
- Monitor middleware overhead

### 5. **Security**
Protect sensitive data:
- Mask PII in logs
- Validate all inputs
- Rate limit by user/tenant
- Audit sensitive operations

## 🔧 Creating Custom Middleware

You can create custom middleware by implementing a factory function. **All middleware factories return `list[AgentMiddleware]`** for composability:

```python
# my_package/middleware.py
from typing import Callable, Any
from langgraph.types import StateSnapshot
from langchain.agents import AgentMiddleware

def create_my_middleware(**kwargs) -> list[AgentMiddleware]:
    """
    Factory function that creates middleware.
    
    Returns a list for composability - factories can return multiple
    middleware instances when needed (e.g., one per tool).
    """
    
    class MyMiddleware(AgentMiddleware):
        def __call__(
            self,
            state: StateSnapshot,
            next_fn: Callable,
            config: dict[str, Any]
        ) -> Any:
            """The actual middleware function."""
            
            # Pre-processing
            print(f"Before: {state}")
            
            # Call next middleware or agent
            result = next_fn(state, config)
            
            # Post-processing
            print(f"After: {result}")
            
            return result
    
    return [MyMiddleware()]

# Combine multiple middleware lists
all_middleware = (
    create_my_middleware()
    + create_other_middleware()
)
```

Then use it in your config:

```yaml
middleware:
  custom: &custom
    name: my_package.middleware.create_my_middleware
    args:
      custom_arg: value

agents:
  my_agent:
    middleware:
      - *custom
```

## 📊 Middleware Execution Flow

```
Request Received
    ↓
┌─────────────────────┐
│ Middleware 1 (Pre)  │ ← Validation
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Middleware 2 (Pre)  │ ← Logging
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Middleware 3 (Pre)  │ ← Rate Limiting
└─────────────────────┘
    ↓
┌─────────────────────┐
│   Agent Execution   │
│   (Tools, LLM, etc) │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Middleware 3 (Post) │ ← Rate Limiting
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Middleware 2 (Post) │ ← Logging
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Middleware 1 (Post) │ ← Validation
└─────────────────────┘
    ↓
Response Returned
```

## 🐛 Debugging Middleware

### Check Middleware Execution
Set log level to DEBUG to see middleware execution:

```yaml
app:
  log_level: DEBUG
```

### Test Validation Errors
Try sending requests without required fields to see error messages:

```bash
# Missing store_num - should return validation error
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "custom_inputs": {
      "configurable": {}
    }
  }'
```

### Monitor Performance
Enable performance logging to see timing:

```yaml
middleware:
  perf:
    name: dao_ai.middleware.create_performance_middleware
    args:
      log_level: INFO
      threshold_ms: 100  # Log if > 100ms
```

## 📖 Related Documentation

- **Hardware Store Example**: See [`15_complete_applications/hardware_store.yaml`](../15_complete_applications/hardware_store.yaml) for production usage
- **Human-in-the-Loop**: See [`07_human_in_the_loop/`](../07_human_in_the_loop/) for interrupt handling
- **Guardrails**: See [`08_guardrails/`](../08_guardrails/) for safety checks

## 💡 Common Use Cases

### Multi-Location Retail
Validate store context for location-specific queries:
- Store number validation
- Region-specific inventory
- Local pricing and availability

### Enterprise SaaS
Tenant isolation and access control:
- Tenant ID validation
- Workspace scoping
- Role-based permissions

### API Integration
Secure third-party service access:
- API key validation
- Region selection
- Rate limiting per API

### Compliance & Audit
Track all agent interactions:
- Full audit trails
- PII masking
- Security monitoring
- Cost attribution

### Performance Optimization
Identify and fix slow agents:
- Execution time tracking
- Tool performance monitoring
- Bottleneck identification
- SLA enforcement

## 🎓 Learning Path

1. **Start Here:** [`custom_field_validation.yaml`](custom_field_validation.yaml)
   - Understand basic validation
   - Learn error handling
   - Practice with required fields

2. **Next:** [`logging_middleware.yaml`](logging_middleware.yaml)
   - Add request logging
   - Monitor performance
   - Create audit trails

3. **Advanced:** [`combined_middleware.yaml`](combined_middleware.yaml)
   - Combine multiple middleware
   - Build production stacks
   - Optimize performance

4. **Production:** [`../10_complete_applications/hardware_store.yaml`](../10_complete_applications/hardware_store.yaml)
   - See real-world usage
   - Learn best practices
   - Apply to your use case

---

## 📞 Need Help?

- Review examples in this directory
- Check the hardware store example for production patterns
- Create custom middleware for your specific needs
- Test thoroughly before deploying to production

**Pro Tip:** Start with simple validation middleware, add logging for debugging, then build up your middleware stack as you move toward production! 🚀

---

# 📖 Detailed Middleware Reference

## Tool Call Limit Middleware

The Tool Call Limit middleware helps prevent excessive tool usage and runaway agent loops by limiting the number of tool calls either globally or for specific tools. It's built on LangChain's `ToolCallLimitMiddleware` and provides graceful termination options.

### Factory Function

```python
from dao_ai.middleware import create_tool_call_limit_middleware

# Returns list[ToolCallLimitMiddleware] for composability
middlewares = create_tool_call_limit_middleware(
    tool: str | ToolModel | dict[str, Any] | None = None,
    thread_limit: int | None = None,
    run_limit: int | None = None,
    exit_behavior: Literal["continue", "error", "end"] = "continue",
)
```

**Note**: All middleware factories return a list of middleware instances. This enables:
- Consistent composition with list concatenation
- Factories that create multiple middleware (e.g., one per tool)
- Uniform handling in agent configuration

### Parameters

- **`tool`** (str | ToolModel | dict[str, Any] | None): Tool to limit. Can be a string name, `ToolModel` instance, or dictionary config. If `None`, limits apply to all tools globally.
  - Accepts: string tool name, `ToolModel` instance, or dictionary config
  - If a `ToolModel` with multiple tools is provided, returns multiple middleware instances
  - Default: `None` (applies globally)

- **`thread_limit`** (Optional[int]): Maximum tool calls across all runs in a thread (conversation).
  - Persists across multiple invocations with the same thread ID
  - Requires a checkpointer to maintain state
  - `None` means no thread limit
  - Default: `None`

- **`run_limit`** (Optional[int]): Maximum tool calls per single invocation (one user message → response cycle).
  - Resets with each new user message
  - `None` means no run limit
  - Default: `None`

- **`exit_behavior`** (Literal["continue", "error", "end"]): Behavior when limit is reached.
  - **`"continue"`** (default): Block exceeded tool calls with error messages, let other tools and the model continue. The model decides when to end based on the error messages. **Allows graceful termination**.
  - **`"error"`**: Raise a `ToolCallLimitExceededError` exception, stopping execution immediately. Use for strict enforcement.
  - **`"end"`**: Stop execution immediately with a ToolMessage and AI message. Only works when limiting a single tool; raises `NotImplementedError` if other tools have pending calls.
  - Default: `"continue"`

**Note**: At least one of `thread_limit` or `run_limit` must be specified.

### Exit Behaviors

#### `continue` - Graceful Termination (Recommended)

The **default** and **recommended** behavior. When a limit is reached:
1. The exceeded tool call is blocked with an error message
2. The error message is passed back to the model
3. The agent can see the error and try alternative approaches
4. Other tools can still be used
5. The model decides when to stop based on the error feedback

**Best for**: Most scenarios where you want the agent to gracefully handle limits and potentially recover.

```python
middlewares = create_tool_call_limit_middleware(
    tool="search_web",
    run_limit=3,
    exit_behavior="continue",  # Agent can try other tools/approaches
)
```

#### `error` - Strict Enforcement

When a limit is reached, execution stops immediately with an exception. **No recovery possible**.

**Best for**: Critical operations where exceeding limits should halt the entire agent.

```python
middlewares = create_tool_call_limit_middleware(
    tool="execute_sql",
    run_limit=2,
    exit_behavior="error",  # Hard stop if exceeded
)
```

#### `end` - Clean Exit (Single-Tool Only)

When a limit is reached, execution stops with a clean ToolMessage + AI message. Only works for single-tool scenarios.

**Best for**: Workflows where only one tool is used and you want a graceful exit.

```python
middlewares = create_tool_call_limit_middleware(
    tool="web_scraper",
    run_limit=5,
    exit_behavior="end",  # Clean termination
)
```

### YAML Configuration Examples

#### Global Limit

```yaml
middleware:
  - &global_limiter
    name: dao_ai.middleware.create_tool_call_limit_middleware
    args:
      thread_limit: 20  # Max 20 tool calls per conversation
      run_limit: 10     # Max 10 tool calls per invocation
      exit_behavior: continue
```

#### Tool-Specific Limit

```yaml
middleware:
  - &search_limiter
    name: dao_ai.middleware.create_tool_call_limit_middleware
    args:
      tool: search_web
      thread_limit: 5   # Max 5 searches per conversation
      run_limit: 3      # Max 3 searches per invocation
      exit_behavior: continue  # Allow agent to try other approaches
```

#### Strict Limit for Critical Operations

```yaml
middleware:
  - &database_limiter
    name: dao_ai.middleware.create_tool_call_limit_middleware
    args:
      tool: query_database
      run_limit: 2      # Max 2 queries per invocation
      exit_behavior: error  # Hard stop if exceeded
```

### Python Usage

```python
from dao_ai.middleware import create_tool_call_limit_middleware
from langchain.agents import create_agent

# All factories return list[Middleware] for composability
global_limiters = create_tool_call_limit_middleware(
    thread_limit=20,
    run_limit=10,
)

search_limiters = create_tool_call_limit_middleware(
    tool="search_web",
    run_limit=3,
)

# Combine middleware lists using concatenation
all_middleware = global_limiters + search_limiters

# Create agent with combined middleware
agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool],
    middleware=all_middleware,
    checkpointer=checkpointer,  # Required for thread_limit
)
```

### Thread Limit Requirements

**Important**: The `thread_limit` parameter requires a checkpointer to be configured on the agent to maintain state across invocations.

```python
from langgraph.checkpoint.memory import MemorySaver

# Create checkpointer
checkpointer = MemorySaver()

# Create limiters (returns list)
limiters = create_tool_call_limit_middleware(
    thread_limit=20,  # Requires checkpointer
    run_limit=10,
)

# Create agent with thread limit support
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=limiters,
    checkpointer=checkpointer,  # Required for thread limits
)
```

### Tool Call Limit Best Practices

1. **Use `continue` for most cases**: Default behavior allows graceful recovery
2. **Combine global and tool-specific limits**: Global limit as safety net, specific limits for expensive operations
3. **Start with generous limits**: Adjust based on observed behavior
4. **Use `error` behavior sparingly**: Only for truly critical operations that must halt
5. **Consider run vs thread limits**: 
   - `run_limit` for per-invocation control
   - `thread_limit` for conversation-wide control
6. **Test limits with checkpointer**: If using thread limits, ensure checkpointer is configured

### Troubleshooting

#### `ValueError: At least one of thread_limit or run_limit must be specified`

**Cause**: Neither `thread_limit` nor `run_limit` was provided.

**Solution**: Specify at least one limit:

```python
middlewares = create_tool_call_limit_middleware(run_limit=5)
```

#### Thread limit not working

**Cause**: No checkpointer configured on agent.

**Solution**: Add checkpointer to agent configuration:

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[...],
    checkpointer=MemorySaver(),  # Required for thread limits
)
```

#### `NotImplementedError` with `exit_behavior="end"`

**Cause**: Using `"end"` behavior with multiple tools having pending calls.

**Solution**: Use `"end"` only for single-tool scenarios, or switch to `"continue"`:

```python
# Single tool - OK
middlewares = create_tool_call_limit_middleware(
    tool="single_tool",
    run_limit=5,
    exit_behavior="end",
)

# Multiple tools - use "continue" instead
middlewares = create_tool_call_limit_middleware(
    run_limit=5,
    exit_behavior="continue",  # Works with multiple tools
)
```

---

## Model Call Limit Middleware

The Model Call Limit middleware limits the number of LLM API calls to control costs and prevent runaway agent loops.

### Factory Function

```python
from dao_ai.middleware import create_model_call_limit_middleware

middlewares = create_model_call_limit_middleware(
    thread_limit: int | None = None,
    run_limit: int | None = None,
    exit_behavior: Literal["end", "error"] = "end",
)
```

### Parameters

- **`thread_limit`**: Maximum LLM calls per thread (conversation)
- **`run_limit`**: Maximum LLM calls per invocation
- **`exit_behavior`**: Only `"end"` or `"error"` are valid (no `"continue"`)

### YAML Example

```yaml
middleware:
  model_limit: &model_limit
    name: dao_ai.middleware.create_model_call_limit_middleware
    args:
      run_limit: 20
      exit_behavior: end
```

---

## Tool Retry Middleware

The Tool Retry middleware adds automatic retry logic with exponential backoff for transient tool failures.

### Factory Function

```python
from dao_ai.middleware import create_tool_retry_middleware

middlewares = create_tool_retry_middleware(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float | None = None,
    jitter: bool = False,
    tools: list[str | ToolModel | dict[str, Any]] | None = None,
    on_failure: Literal["continue", "error"] = "continue",
)
```

### Parameters

- **`max_retries`**: Maximum retry attempts (default: 3)
- **`backoff_factor`**: Delay multiplier between retries (default: 2.0)
- **`initial_delay`**: Initial delay in seconds (default: 1.0)
- **`max_delay`**: Maximum delay cap (optional)
- **`jitter`**: Add randomness to prevent thundering herd (default: false)
- **`tools`**: List of tools to apply retry to (None = all tools)
- **`on_failure`**: Behavior when all retries exhausted

### YAML Example

```yaml
middleware:
  tool_retry: &tool_retry
    name: dao_ai.middleware.create_tool_retry_middleware
    args:
      max_retries: 3
      backoff_factor: 2.0
      initial_delay: 1.0
      max_delay: 30.0
      jitter: true
      tools:
        - *genie_tool
        - search_web
```

---

## Model Retry Middleware

The Model Retry middleware adds automatic retry logic for transient LLM API failures.

### Factory Function

```python
from dao_ai.middleware import create_model_retry_middleware

middlewares = create_model_retry_middleware(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float | None = None,
    jitter: bool = False,
    on_failure: Literal["continue", "error"] = "continue",
)
```

### YAML Example

```yaml
middleware:
  model_retry: &model_retry
    name: dao_ai.middleware.create_model_retry_middleware
    args:
      max_retries: 3
      backoff_factor: 2.0
      jitter: true
```

---

## Context Editing Middleware

The Context Editing middleware manages conversation context by clearing older tool outputs to prevent token limit issues.

### Factory Function

```python
from dao_ai.middleware import create_context_editing_middleware

middlewares = create_context_editing_middleware(
    trigger: int = 100000,
    keep: int = 3,
    clear_at_least: int = 0,
    clear_tool_inputs: bool = False,
    exclude_tools: list[str | ToolModel | dict[str, Any]] | None = None,
    placeholder: str = "[cleared]",
    token_count_method: Literal["approximate", "model"] = "approximate",
)
```

### Parameters

- **`trigger`**: Token threshold that triggers clearing (default: 100000)
- **`keep`**: Number of recent tool outputs to preserve (default: 3)
- **`clear_at_least`**: Minimum outputs to clear when triggered (default: 0)
- **`clear_tool_inputs`**: Also clear tool inputs, not just outputs (default: false)
- **`exclude_tools`**: Tools whose outputs should never be cleared
- **`placeholder`**: Text to replace cleared content (default: "[cleared]")
- **`token_count_method`**: "approximate" (fast) or "model" (accurate)

### YAML Example

```yaml
middleware:
  context_editor: &context_editor
    name: dao_ai.middleware.create_context_editing_middleware
    args:
      trigger: 100000
      keep: 3
      exclude_tools:
        - *time_tool
      placeholder: "[cleared]"
```

---

## PII Middleware

The PII middleware detects and handles Personally Identifiable Information for privacy compliance.

### Factory Function

```python
from dao_ai.middleware import create_pii_middleware

middlewares = create_pii_middleware(
    pii_type: str,
    strategy: Literal["redact", "mask", "hash", "block"] = "redact",
    apply_to_input: bool = True,
    apply_to_output: bool = False,
    apply_to_tool_results: bool = False,
)
```

### Parameters

- **`pii_type`**: Type of PII to detect (email, phone, ssn, credit_card, address, name, ip_address, date_of_birth)
- **`strategy`**: How to handle detected PII:
  - `"redact"`: Replace with [REDACTED_TYPE]
  - `"mask"`: Partial masking (j***@email.com)
  - `"hash"`: Cryptographic hash (reversible with key)
  - `"block"`: Reject entire message
- **`apply_to_input`**: Check user inputs (default: true)
- **`apply_to_output`**: Check agent outputs (default: false)
- **`apply_to_tool_results`**: Check tool results (default: false)

### YAML Example

```yaml
middleware:
  email_protection: &email_protection
    name: dao_ai.middleware.create_pii_middleware
    args:
      pii_type: email
      strategy: redact
      apply_to_input: true
      apply_to_output: true

  ssn_protection: &ssn_protection
    name: dao_ai.middleware.create_pii_middleware
    args:
      pii_type: ssn
      strategy: block
      apply_to_input: true
```

### Strategy Comparison

| Strategy | Privacy | Reversible | User Experience | Use Case |
|----------|---------|------------|-----------------|----------|
| redact   | High    | No         | Clear indicator | General protection |
| mask     | Medium  | No         | Partial visibility | Phone numbers |
| hash     | High    | Yes        | Opaque          | Audit/logging |
| block    | Highest | N/A        | Disruptive      | Critical PII (SSN) |
