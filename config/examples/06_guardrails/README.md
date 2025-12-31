# 06. Guardrails

**Automated safety, validation, and compliance checks**

Protect your agents with always-on content filtering, PII detection, output validation, and custom safety rules.

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| `guardrails_basic.yaml` | Content filtering and safety | PII detection, bias mitigation, toxicity filtering |
| `structured_output.yaml` | Enforce response format | API integration, data validation, schema compliance |

## What You'll Learn

- **Content Filtering** - Detect and handle PII, toxic content, bias
- **Output Validation** - Enforce JSON schemas and response formats
- **Compliance** - Meet regulatory requirements automatically
- **Custom Rules** - Define your own safety guardrails

## Quick Start

### Test guardrails

```bash
dao-ai chat -c config/examples/06_guardrails/guardrails_basic.yaml
```

Try inputs with PII (like "My SSN is 123-45-6789") - they'll be detected and handled appropriately.

### Test structured output

```bash
dao-ai chat -c config/examples/06_guardrails/structured_output.yaml
```

All responses will conform to the defined JSON schema, regardless of the model's natural output format.

## Configuration Patterns

### Basic Guardrail

```yaml
middleware:
  - name: pii_detection
    type: guardrail
    guardrail_name: pii_detector
    action: block  # or redact, warn, log
```

### Structured Output

```yaml
agents:
  api_agent:
    name: api_agent
    model: *llm
    response_format:
      type: json_schema
      json_schema:
        name: ApiResponse
        schema:
          type: object
          properties:
            status: {type: string}
            message: {type: string}
            data: {type: object}
          required: [status, message]
```

## Guardrail Types

### PII Detection
Detect and handle personal information:
- Social Security Numbers
- Credit card numbers
- Email addresses
- Phone numbers
- Physical addresses

**Actions**: block, redact, warn, log

### Toxicity Filtering
Detect harmful content:
- Offensive language
- Hate speech
- Threats
- Harassment

**Actions**: block, warn, log

### Bias Detection
Identify discriminatory content:
- Gender bias
- Racial bias
- Age bias
- Other protected characteristics

**Actions**: warn, log, suggest alternatives

### Prompt Injection
Detect adversarial inputs:
- Jailbreak attempts
- System prompt leaks
- Instruction injection

**Actions**: block, log, alert

### Custom Guardrails
Define your own rules:
- Domain-specific compliance
- Business policy enforcement
- Industry regulations

## Structured Output Use Cases

### API Integration
Ensure consistent response format for downstream systems:

```yaml
response_format:
  type: json_schema
  json_schema:
    name: ProductLookup
    schema:
      type: object
      properties:
        product_id: {type: string}
        name: {type: string}
        price: {type: number}
        in_stock: {type: boolean}
```

### Data Validation
Enforce data types and constraints:

```yaml
response_format:
  type: json_schema
  json_schema:
    name: ValidationResult
    schema:
      type: object
      properties:
        valid: {type: boolean}
        errors: 
          type: array
          items: {type: string}
        field_values: {type: object}
```

### Multi-step Workflows
Parse agent decisions consistently:

```yaml
response_format:
  type: json_schema
  json_schema:
    name: AgentDecision
    schema:
      type: object
      properties:
        action: 
          type: string
          enum: [search, respond, escalate]
        reasoning: {type: string}
        confidence: {type: number, minimum: 0, maximum: 1}
```

## Prerequisites

### For Guardrails
- ✅ Guardrail service endpoint (Databricks Lakehouse Monitoring or external)
- ✅ Guardrail policies configured
- ✅ Authentication credentials

### For Structured Output
- ✅ Pydantic model or JSON schema defined
- ✅ LLM that supports structured output (Claude, GPT-4, etc.)

## Production Checklist

Before deploying to production:

- [ ] **Guardrails** enabled for all user-facing agents
- [ ] **PII detection** configured and tested
- [ ] **Toxicity filtering** appropriate for your domain
- [ ] **Structured output** for API integrations
- [ ] **Logging** enabled for audit trail
- [ ] **Fallback behavior** defined for guardrail failures
- [ ] **Performance testing** with guardrails enabled
- [ ] **Alert monitoring** for blocked content

## Comprehensive Safety Stack

Combine multiple guardrails for defense in depth:

```yaml
agents:
  safe_agent:
    name: safe_agent
    model: *llm
    middleware:
      - name: input_pii_detection
        type: guardrail
        guardrail_name: pii_detector_input
        stage: before_model
      - name: output_validation
        type: guardrail
        guardrail_name: output_validator
        stage: after_model
      - name: toxicity_filter
        type: guardrail
        guardrail_name: toxicity_detector
        stage: after_model
    response_format: *api_response_schema
```

## Best Practices

### Guardrails
1. **Layer defenses**: Use multiple complementary guardrails
2. **Test thoroughly**: Verify detection rates and false positives
3. **Monitor continuously**: Track guardrail activations
4. **Provide feedback**: Log why content was blocked
5. **Balance safety and UX**: Avoid over-blocking legitimate use

### Structured Output
1. **Define clear schemas**: Be explicit about required fields
2. **Add descriptions**: Help the model understand field purposes
3. **Use enums**: Constrain choices when possible
4. **Validate outputs**: Don't assume 100% schema compliance
5. **Provide examples**: Show the model what good outputs look like

## Troubleshooting

**"Guardrail service unavailable"**
- Check service endpoint is accessible
- Verify authentication credentials
- Check network connectivity
- Fallback: Disable guardrail for testing only (not production!)

**"False positives from PII detector"**
- Review and tune detection thresholds
- Whitelist known safe patterns
- Use redaction instead of blocking
- Collect examples for model improvement

**"Structured output validation failed"**
- Review schema definition for ambiguity
- Check LLM supports structured output mode
- Add schema examples to prompt
- Verify response parsing logic

**"Performance degradation with guardrails"**
- Optimize guardrail service latency
- Cache common checks
- Run non-critical checks async
- Consider sampling for high-volume scenarios

## Next Steps

👉 **07_prompt_engineering/** - Optimize prompts for safety and compliance  
👉 **08_orchestration/** - Apply guardrails to multi-agent systems  
👉 **09_complete_applications/** - See guardrails in production

## Related Documentation

- [Guardrails Configuration](../../../docs/key-capabilities.md#guardrails)
- [Structured Output](../../../docs/key-capabilities.md#structured-output)
- [Middleware](../../../docs/configuration-reference.md#middleware)
