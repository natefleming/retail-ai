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

You can create custom middleware by implementing a factory function:

```python
# my_package/middleware.py
from typing import Callable, Any
from langgraph.types import StateSnapshot

def create_my_middleware(**kwargs) -> Callable:
    """Factory function that creates middleware."""
    
    def middleware(
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
    
    return middleware
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
