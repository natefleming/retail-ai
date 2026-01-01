# 09. Structured Output

**Enforce JSON schema responses from LLMs**

Force agents to return responses in a specific JSON structure. This is essential for programmatic consumption, validation, and integration with downstream systems.

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| `structured_output.yaml` | Enforce JSON schema responses | Type-safe API responses, data extraction |

## What You'll Learn

- **Schema Definition** - Define Pydantic models for response structure
- **Response Validation** - Automatic validation of LLM outputs
- **Type Safety** - Ensure predictable, parseable responses
- **Error Handling** - Handle schema validation failures

## Quick Start

```bash
dao-ai chat -c config/examples/09_structured_output/structured_output.yaml
```

The agent will always return responses matching your defined schema.

## Configuration Pattern

```yaml
app:
  agents:
    main:
      model:
        name: databricks-meta-llama-3-1-70b-instruct
        structured_output:
          type: pydantic
          schema: MyResponseSchema
```

## Why Structured Output?

### Without Structured Output
- Hard to parse free-form text
- Inconsistent formats
- No type safety
- Difficult to validate

### With Structured Output
- Guaranteed JSON structure
- Consistent, predictable responses
- Type-safe programmatic access
- Automatic validation

## Use Cases

### 1. API Responses
Ensure agents return consistent JSON for REST APIs

### 2. Data Extraction
Extract structured data from unstructured text

### 3. Form Filling
Parse user input into structured forms

### 4. Classification
Categorize inputs with confidence scores

## Best Practices

1. **Keep schemas simple** - Avoid overly complex nesting
2. **Provide clear descriptions** - Help the LLM understand field purposes
3. **Use Optional for uncertain fields** - Don't force all fields to be required
4. **Validate programmatically** - Add custom validators when needed

## Performance

- **Latency**: +10-20% due to validation
- **Token Usage**: +5-10% for schema instructions
- **Accuracy**: 90%+ reduction in parsing errors

## Prerequisites

- ✅ Databricks workspace with LLM endpoint
- ✅ Model that supports structured output
- ✅ Basic understanding of JSON schema

## Next Steps

👉 **11_prompt_engineering/** - Optimize prompts for better structured output  
👉 **15_complete_applications/** - See structured output in production apps

## Related Documentation

- [Structured Output Guide](../../../docs/key-capabilities.md#structured-output)
- [Configuration Reference](../../../docs/configuration-reference.md#structured-output)
