# Prompt Optimization in dao-ai

This document describes the prompt optimization feature in dao-ai, which leverages [MLflow's prompt optimization capabilities](https://mlflow.org/docs/latest/genai/prompt-registry/rewrite-prompts/) to automatically rewrite prompts when migrating between language models.

## Overview

When migrating to a new language model (e.g., from GPT-4 to GPT-4o-mini for cost savings), your carefully crafted prompts often don't work as well with the new model. The prompt optimization feature helps you automatically rewrite prompts to maintain output quality when switching models, using your existing application's outputs as training data.

### Key Benefits

- **Model Migration**: Seamlessly switch between language models while maintaining output consistency
- **Automatic Optimization**: Automatically rewrites prompts based on your existing data
- **No Ground Truth Required**: No human labeling is required if you optimize prompts based on existing outputs
- **Trace-Aware**: Leverages MLflow tracing to understand prompt usage patterns
- **Configuration-Driven**: Define optimizations in YAML configuration files

## Requirements

- MLflow >= 3.5.0
- dao-ai with prompt optimization support

## Architecture

The prompt optimization feature consists of three main components:

### 1. Configuration Models

Defined in `src/dao_ai/config.py`, these models encapsulate all parameters needed for prompt optimization:

#### OptimizationsModel

Wrapper class for all prompt optimizations:

```python
class OptimizationsModel(BaseModel):
    prompt_optimizations: dict[str, PromptOptimizationModel]
    
    def optimize(self, w: WorkspaceClient | None = None) -> dict[str, PromptModel]:
        """Optimize all prompts and return mapping of results"""
```

#### PromptOptimizationModel

Individual prompt optimization configuration:

```python
class PromptOptimizationModel(BaseModel):
    name: str                                 # Unique name for this optimization
    prompt: PromptModel                       # The prompt to optimize
    agent: AgentModel | str                   # Agent using the target model
    dataset_name: str                         # MLflow dataset with training data
    reflection_model: Optional[LLMModel]      # Model for reflection (optional)
    num_candidates: Optional[int] = 5         # Number of candidate prompts
    max_steps: Optional[int] = 3              # Maximum optimization steps
    scorer_model: Optional[LLMModel]          # Model for scoring (optional)
    temperature: Optional[float] = 0.0        # Generation temperature
    
    def optimize(self, w: WorkspaceClient | None = None) -> PromptModel:
        """Optimize the prompt and return new version"""
```

### 2. Provider Method (`DatabricksProvider.optimize_prompt`)

Implemented in `src/dao_ai/providers/databricks.py`, this method handles the actual optimization:

```python
def optimize_prompt(
    self, 
    optimization: PromptOptimizationModel
) -> PromptModel:
    """
    Optimize a prompt using MLflow's prompt optimization.
    
    Args:
        optimization: PromptOptimizationModel containing configuration
        
    Returns:
        PromptModel: The optimized prompt with new URI
    """
```

The method:
1. Loads the dataset from MLflow
2. Creates a prediction function using the agent's LLM
3. Configures the GepaPromptOptimizer
4. Runs optimization using MLflow's `optimize_prompts()`
5. Returns a new `PromptModel` with the optimized prompt version

### 3. Optimization Notebook (`notebooks/10_optimize_prompts.py`)

A Databricks notebook that:
- Loads configuration from YAML
- Iterates through all defined optimizations
- Runs optimization for each
- Registers optimized prompts in MLflow
- Displays summary results

## Usage

### Step 1: Collect Training Data

First, collect outputs from your source model using MLflow tracing:

```python
import mlflow
from mlflow.genai.datasets import create_dataset

# Define prediction function with source model
@mlflow.trace
def predict_fn_source_model(text: str) -> str:
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-4",  # Source model
        messages=[{"role": "user", "content": prompt.format(text=text)}],
    )
    return completion.choices[0].message.content

# Collect traces
inputs = [
    {"inputs": {"text": "This movie was fantastic!"}},
    {"inputs": {"text": "The service was terrible."}},
    # ... more examples
]

with mlflow.start_run() as run:
    for record in inputs:
        predict_fn_source_model(**record["inputs"])

# Create dataset from traces
dataset = create_dataset(name="sentiment_migration_dataset")
traces = mlflow.search_traces(return_type="list", run_id=run.info.run_id)
dataset.merge_records(traces)
```

### Step 2: Configure Optimization

Define your optimization in a YAML configuration file:

```yaml
resources:
  llms:
    gpt_4:
      name: gpt-4
      temperature: 0.1
    
    gpt_4o_mini:
      name: gpt-4o-mini
      temperature: 0.1

prompts:
  sentiment_classifier:
    name: sentiment_classifier
    schema: *main_schema
    default_template: |
      Classify the sentiment. Answer 'positive' or 'negative' or 'neutral'.
      Text: {{text}}

agents:
  sentiment_agent:
    name: sentiment_agent
    model: *gpt_4o_mini  # Target model
    prompt: *sentiment_classifier

optimizations:
  prompt_optimizations:
    optimize_sentiment:
      name: optimize_sentiment
      prompt: *sentiment_classifier
      agent: *sentiment_agent
      dataset_name: sentiment_migration_dataset
      
      # Optional parameters
      reflection_model: *gpt_4      # Use source model for reflection
      num_candidates: 5             # Generate 5 candidate prompts
      max_steps: 3                  # Maximum 3 optimization steps
      scorer_model: *gpt_4          # Use source model for scoring
```

### Step 3: Run Optimization

Use the provided notebook:

```python
# In Databricks notebook or script
config = AppConfig.from_file("config/examples/prompt_optimization.yaml")

# Option 1: Optimize all prompts at once
if config.optimizations:
    results = config.optimizations.optimize()
    for opt_name, optimized_prompt in results.items():
        print(f"Optimized {opt_name}: {optimized_prompt.uri}")

# Option 2: Optimize individual prompts
if config.optimizations:
    for opt_name, optimization in config.optimizations.prompt_optimizations.items():
        print(f"Optimizing: {opt_name}")
        optimized_prompt = optimization.optimize()
        print(f"New version: {optimized_prompt.uri}")
```

Or run the notebook directly:
```bash
databricks notebook run notebooks/10_optimize_prompts.py \
  --config-path config/examples/prompt_optimization.yaml
```

### Step 4: Use Optimized Prompt

Update your configuration to use the optimized prompt version:

```yaml
prompts:
  sentiment_classifier:
    name: sentiment_classifier
    schema: *main_schema
    version: 2  # Use the optimized version
```

## Configuration Reference

### PromptOptimizationModel Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | Yes | - | Unique identifier for this optimization |
| `prompt` | `PromptModel` | Yes | - | The prompt to optimize |
| `agent` | `AgentModel \| str` | Yes | - | Agent using the target model |
| `dataset_name` | `str` | Yes | - | Name of MLflow dataset with training data |
| `reflection_model` | `LLMModel` | No | Agent's model | Model used for reflection during optimization |
| `num_candidates` | `int` | No | 5 | Number of candidate prompts to generate |
| `max_steps` | `int` | No | 3 | Maximum number of optimization steps |
| `scorer_model` | `LLMModel` | No | Agent's model | Model used to score outputs |
| `temperature` | `float` | No | 0.0 | Temperature for prompt generation |

## Best Practices

### 1. Collect Sufficient Data

For best results, collect outputs from at least 20-50 diverse examples:

```python
# Good: Diverse examples
inputs = [
    {"inputs": {"text": "Great product!"}},
    {"inputs": {"text": "The delivery was delayed by three days..."}},
    {"inputs": {"text": "It meets the basic requirements."}},
    # ... more varied examples
]

# Poor: Too few, too similar
inputs = [
    {"inputs": {"text": "Good"}},
    {"inputs": {"text": "Bad"}},
]
```

### 2. Use Representative Examples

Include edge cases and challenging inputs:

```python
inputs = [
    {"inputs": {"text": "Absolutely fantastic!"}},  # Clear positive
    {"inputs": {"text": "It's not bad, I guess."}},  # Ambiguous
    {"inputs": {"text": "Good food, terrible service."}},  # Mixed sentiment
]
```

### 3. Verify Results

Always evaluate optimized prompts before production:

```python
# Use dao-ai evaluation
config = AppConfig.from_file("config.yaml")
# Run evaluation notebook (07_run_evaluation.py)
```

### 4. Iterate if Needed

If results aren't satisfactory:
- Collect more diverse training data
- Adjust optimization parameters (`num_candidates`, `max_steps`)
- Try different reflection or scorer models

## Example Workflow

Complete workflow for optimizing prompts when migrating from GPT-4 to GPT-4o-mini:

```python
# 1. Collect data from source model
import mlflow
from mlflow.genai.datasets import create_dataset

@mlflow.trace
def predict_with_gpt4(text: str) -> str:
    return openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Classify: {text}"}]
    ).choices[0].message.content

with mlflow.start_run() as run:
    for example in training_examples:
        predict_with_gpt4(example["text"])

dataset = create_dataset(name="migration_data")
traces = mlflow.search_traces(return_type="list", run_id=run.info.run_id)
dataset.merge_records(traces)

# 2. Configure optimization
# (Edit config/examples/prompt_optimization.yaml)

# 3. Run optimization
config = AppConfig.from_file("config/examples/prompt_optimization.yaml")
optimized = config.optimizations.prompt_optimizations["optimize_sentiment"].optimize()

# 4. Evaluate
# (Use notebooks/07_run_evaluation.py)

# 5. Deploy
# (Update config with optimized version, use notebooks/05_deploy_agent.py)
```

## API Reference

### OptimizationsModel.optimize()

```python
def optimize(self, w: WorkspaceClient | None = None) -> dict[str, PromptModel]:
    """
    Optimize all prompts in this configuration.
    
    Args:
        w: Optional WorkspaceClient for Databricks operations
        
    Returns:
        dict[str, PromptModel]: Dictionary mapping optimization names to optimized prompts
    """
```

### PromptOptimizationModel.optimize()

```python
def optimize(self, w: WorkspaceClient | None = None) -> PromptModel:
    """
    Optimize the prompt using MLflow's prompt optimization.
    
    Args:
        w: Optional WorkspaceClient for Databricks operations
        
    Returns:
        PromptModel: The optimized prompt model with new URI
        
    Raises:
        ValueError: If agent is a string reference (not yet supported)
        Exception: If optimization fails
    """
```

### DatabricksProvider.optimize_prompt()

```python
def optimize_prompt(
    self, 
    optimization: PromptOptimizationModel
) -> PromptModel:
    """
    Optimize a prompt using MLflow's prompt optimization.
    
    Args:
        optimization: PromptOptimizationModel containing configuration
        
    Returns:
        PromptModel: The optimized prompt with new URI
        
    Raises:
        ValueError: If agent is a string reference
        Exception: If dataset loading or optimization fails
    """
```

## Troubleshooting

### Common Issues

**Issue**: "Dataset not found"
- **Solution**: Ensure you've created the dataset in MLflow using `create_dataset()` and `merge_records()`

**Issue**: "Agent reference by string not yet supported"
- **Solution**: Provide the full `AgentModel` object, not a string reference in the YAML

**Issue**: Optimization takes too long
- **Solution**: Reduce `num_candidates` or `max_steps` parameters

**Issue**: Optimized prompt doesn't improve quality
- **Solution**: 
  - Collect more diverse training examples
  - Ensure training data is representative of production use
  - Try different reflection/scorer models

## See Also

- [MLflow Prompt Optimization Documentation](https://mlflow.org/docs/latest/genai/prompt-registry/rewrite-prompts/)
- [MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/)
- [dao-ai Prompt Management](../README.md#prompt-management)
- [Example Configuration](../config/examples/prompt_optimization.yaml)

## Contributing

To extend the prompt optimization feature:

1. Add new optimizer types in `PromptOptimizationModel`
2. Implement custom scorers
3. Add support for agent string references
4. Enhance the optimization notebook with visualizations

For questions or issues, please open a GitHub issue.

