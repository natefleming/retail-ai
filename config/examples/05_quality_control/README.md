# 05. Quality Control

**Production-grade safety, validation, and approval workflows**

Essential patterns for deploying agents safely in production environments.

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| `guardrails_basic.yaml` | Content filtering and safety | PII detection, bias mitigation |
| `human_in_the_loop.yaml` | Tool approval workflows | Sensitive operations |
| `structured_output.yaml` | Enforce response format | API integration, data validation |

## What You'll Learn

- **Guardrails** - Content filtering, PII detection, bias mitigation
- **HITL (Human-in-the-Loop)** - Approval workflows for critical actions
- **Structured output** - Enforce JSON schemas for consistent responses
- **Validation** - Input/output validation patterns

## Quick Start

### Test guardrails
```bash
dao-ai chat -c config/examples/05_quality_control/guardrails_basic.yaml
```

Try inputs with PII - they'll be detected and handled appropriately.

### Test HITL
```bash
dao-ai chat -c config/examples/05_quality_control/human_in_the_loop.yaml
```

Request a sensitive action - it will pause for approval.

### Test structured output
```bash
dao-ai chat -c config/examples/05_quality_control/structured_output.yaml
```

Responses will always match the defined JSON schema.

## Quality Control Patterns

### 1. Guardrails
**What**: Content filtering, safety checks, compliance
**When**: Always-on protection for all interactions
**Example**: Detect and redact PII before processing

```yaml
middleware:
  - name: safety_check
    type: guardrail
    guardrail_name: pii_detector
```

### 2. Human-in-the-Loop (HITL)
**What**: Pause execution for human approval
**When**: Sensitive operations (delete, spend money, external comms)
**Example**: Approve before sending email

```yaml
tools:
  send_email_tool:
    name: send_email
    function:
      type: python
      name: my_package.send_email_function
      human_in_the_loop:
        review_prompt: "Review this email before sending"
        allowed_decisions:
          - approve
          - edit
          - reject
```

### 3. Structured Output
**What**: Enforce response schema
**When**: API integrations, data pipelines, validation
**Example**: Always return `{status, message, data}`

```yaml
agents:
  my_agent:
    response_schema: MyResponseModel
    use_tool: true
```

## Prerequisites

### For Guardrails
- ✅ Guardrail service endpoint (Databricks Lakehouse Monitoring or external)
- ✅ Guardrail policies configured

### For HITL
- ✅ MLflow for state persistence
- ✅ Checkpointer configured (PostgreSQL or Lakebase)
- ✅ Approval UI or API

### For Structured Output
- ✅ Pydantic model or JSON schema defined
- ✅ LLM that supports structured output (Claude, GPT-4, etc.)

## Production Checklist

Before deploying to production:

- [ ] **Guardrails** enabled for all user-facing agents
- [ ] **HITL** configured for sensitive operations
- [ ] **Structured output** for API integrations
- [ ] **Input validation** for user queries
- [ ] **Output validation** for agent responses
- [ ] **Logging** enabled for audit trail
- [ ] **Rate limiting** configured
- [ ] **Error handling** for all failure modes

## Configuration Examples

### Comprehensive Safety Stack
```yaml
# 1. Guardrails at agent level
agents:
  safe_agent:
    name: safe_agent
    model: *llm
    guardrails:
      - *pii_detection_guardrail
      - *output_validation_guardrail
    tools:
      - *delete_tool
      - *send_tool

# 2. HITL configured per tool
tools:
  delete_tool:
    name: delete_data
    function:
      type: python
      name: my_package.delete_function
      human_in_the_loop:
        review_prompt: "Review this deletion"
        allowed_decisions: [approve, reject]
  
  send_tool:
    name: send_email
    function:
      type: python
      name: my_package.send_function
      human_in_the_loop:
        review_prompt: "Review this email"
        allowed_decisions: [approve, edit, reject]
```

## Guardrail Types

- **PII Detection**: Detect and redact personal information
- **Bias Detection**: Identify biased or discriminatory content
- **Toxicity**: Filter harmful or offensive language
- **Prompt Injection**: Detect adversarial prompts
- **Custom**: Define your own safety rules

## HITL Patterns

### Pattern 1: Basic Approval (Approve or Reject Only)
```yaml
tools:
  send_email_tool:
    name: send_email
    function:
      type: python
      name: my_package.send_email
      human_in_the_loop:
        review_prompt: "Review this email"
        allowed_decisions: [approve, reject]
```

### Pattern 2: Full Control (Approve, Edit, or Reject)
```yaml
tools:
  create_ticket_tool:
    name: create_ticket
    function:
      type: python
      name: my_package.create_ticket
      human_in_the_loop:
        review_prompt: "Review ticket details"
        allowed_decisions: [approve, edit, reject]
```

### Pattern 3: Minimal Configuration (All Decisions Allowed)
```yaml
tools:
  risky_operation:
    name: risky_op
    function:
      type: python
      name: my_package.risky_function
      human_in_the_loop:
        review_prompt: "Approve this operation?"
        # allowed_decisions defaults to [approve, edit, reject]
```

## Next Steps

👉 **06_prompt_engineering/** - Optimize prompts for safety  
👉 **08_complete_applications/** - See these patterns in production apps

## Troubleshooting

**"Guardrail service unavailable"**
- Check service endpoint is accessible
- Verify authentication credentials
- Fallback: Disable guardrail for testing only

**"HITL state lost"**
- Ensure checkpointer is configured
- Verify database connectivity
- Check thread_id is preserved across requests

**"Structured output validation failed"**
- Review schema definition
- Check LLM supports structured output
- Add schema examples to prompt

## Related Documentation

- [Guardrails Configuration](../../../docs/key-capabilities.md)
- [HITL Patterns](../../../docs/key-capabilities.md#human-in-the-loop)
- [Structured Output](../../../docs/key-capabilities.md#structured-output)

