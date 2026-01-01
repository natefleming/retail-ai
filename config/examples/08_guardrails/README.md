# 08. Guardrails

**Automated safety, validation, and compliance checks**

Protect your agents with always-on content filtering, PII detection, and custom safety rules.

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| `guardrails_basic.yaml` | Content filtering and safety | PII detection, bias mitigation, toxicity filtering |

## What You'll Learn

- **Content Filtering** - Detect and handle PII, toxic content, bias
- **Safety Checks** - Automated content moderation
- **Compliance** - Meet regulatory requirements automatically
- **Custom Rules** - Define your own safety guardrails

## Quick Start

### Test guardrails

```bash
dao-ai chat -c config/examples/08_guardrails/guardrails_basic.yaml
```

Try inputs with PII (like "My SSN is 123-45-6789") - they'll be detected and handled appropriately.

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

## Prerequisites

- ✅ Guardrail service endpoint (Databricks Lakehouse Monitoring or external)
- ✅ Guardrail policies configured
- ✅ Authentication credentials

## Production Checklist

Before deploying to production:

- [ ] **Guardrails** enabled for all user-facing agents
- [ ] **PII detection** configured and tested
- [ ] **Toxicity filtering** appropriate for your domain
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

👉 **09_structured_output/** - Enforce JSON schema responses  
👉 **11_prompt_engineering/** - Optimize prompts for safety and compliance  
👉 **15_complete_applications/** - See guardrails in production

## Related Documentation

- [Guardrails Configuration](../../../docs/key-capabilities.md#guardrails)
- [Structured Output](../../../docs/key-capabilities.md#structured-output)
- [Middleware](../../../docs/configuration-reference.md#middleware)
