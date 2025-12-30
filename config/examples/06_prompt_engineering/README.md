# 06. Prompt Engineering

**Prompt versioning, management, and automated optimization**

Advanced prompt management for production agents with version control and automated tuning.

## Examples

| File | Description | Approach |
|------|-------------|----------|
| `prompt_registry.yaml` | MLflow prompt registry integration | Version control, governance |
| `prompt_optimization.yaml` | Automated prompt tuning with GEPA | Evolutionary optimization |

## What You'll Learn

- **Prompt Registry** - Centralized prompt storage in MLflow
- **Versioning** - Track prompt changes and rollback
- **A/B Testing** - Compare prompt variations
- **Automated Optimization** - Use GEPA to improve prompts
- **Governance** - Review and approval workflows

## Quick Start

### Use prompt registry
```bash
dao-ai chat -c config/examples/06_prompt_engineering/prompt_registry.yaml
```

Prompts are loaded from MLflow registry, not hardcoded in YAML.

### Run prompt optimization
```bash
dao-ai chat -c config/examples/06_prompt_engineering/prompt_optimization.yaml
```

GEPA will iteratively improve prompts based on evaluation data.

## Prompt Registry Workflow

```
1. Define base prompt → 2. Register in MLflow → 3. Use in agent 
                         ↓
4. Deploy to production ← 3. Test and evaluate ← 2. Create variants
```

### Benefits
- ✅ **Version control**: Track all prompt changes
- ✅ **Rollback**: Quickly revert to previous version
- ✅ **A/B testing**: Compare prompt variants
- ✅ **Governance**: Review before production deployment
- ✅ **Centralized**: One source of truth for all prompts

## Automated Optimization (GEPA)

**GEPA** = Generative Evolution of Prompts and Agents

### How it works
1. **Start** with a base prompt and evaluation dataset
2. **Evaluate** current prompt performance
3. **Generate** variations using LLM
4. **Test** variations on evaluation data
5. **Select** best performing prompt
6. **Iterate** until convergence or max iterations

### Configuration
```yaml
prompt_optimization:
  training_data_path: "path/to/eval_data.json"
  max_iterations: 10
  population_size: 5
  evaluation_metrics:
    - accuracy
    - relevance
    - latency
```

## Prerequisites

### For Prompt Registry
- ✅ MLflow tracking server
- ✅ Prompt registry access
- ✅ Prompts registered in MLflow

### For Prompt Optimization
- ✅ Evaluation dataset with inputs and expected outputs
- ✅ MLflow experiment for tracking
- ✅ LLM for generating variations
- ✅ Time (optimization can take 30min-2hrs)

## Prompt Registry Setup

### 1. Register a prompt
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("prompt_name", "my_agent_prompt")
    mlflow.log_param("prompt_text", "You are a helpful assistant...")
    mlflow.log_param("version", "1.0")
```

### 2. Reference in config
```yaml
agents:
  my_agent:
    prompt:
      registry:
        model_name: "my_agent_prompt"
        model_version: "1"  # Or "latest"
```

## Optimization Best Practices

### Evaluation Data
- **Quality over quantity**: 20-50 high-quality examples > 1000 poor examples
- **Representative**: Cover key use cases and edge cases
- **Diverse**: Include various input types and expected behaviors
- **Labeled**: Clear expected outputs for each input

### Metrics
- **Accuracy**: Correctness of responses
- **Relevance**: Response addresses the query
- **Brevity**: Concise without losing information
- **Format**: Follows expected structure
- **Safety**: No hallucinations or harmful content

### Iteration Strategy
- **Start small**: 5-10 iterations with population of 3-5
- **Monitor progress**: Track metrics after each generation
- **Early stopping**: Stop if no improvement for 3 generations
- **Cost awareness**: Each iteration calls LLM multiple times

## Typical Results

| Metric | Before Optimization | After GEPA | Improvement |
|--------|---------------------|------------|-------------|
| Accuracy | 65% | 82% | +17% |
| Relevance | 70% | 88% | +18% |
| Latency | 2.3s | 2.1s | -9% |

*Results vary by use case and evaluation data quality*

## Next Steps

👉 **07_orchestration/** - Multi-agent prompt coordination  
👉 **08_complete_applications/** - Production prompt management

## Troubleshooting

**"Prompt not found in registry"**
- Verify prompt name and version
- Check MLflow tracking URI
- Ensure prompt is registered

**"Optimization not converging"**
- Review evaluation data quality
- Reduce population size
- Increase max iterations
- Try different evaluation metrics

**"High optimization costs"**
- Reduce max iterations (start with 5-10)
- Smaller population size (3-5)
- Cache LLM responses
- Use cheaper model for variation generation

## Related Documentation

- [MLflow Prompt Registry](https://mlflow.org/docs/latest/prompts.html)
- [GEPA Paper](https://arxiv.org/abs/2406.09769)
- [Prompt Engineering Guide](../../../docs/key-capabilities.md)

