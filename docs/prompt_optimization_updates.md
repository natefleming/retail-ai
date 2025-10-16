# Prompt Optimization Updates

This document summarizes the changes made to support the new data model structure for prompt optimizations.

## Changes Overview

The prompt optimization feature has been updated to use a wrapper model (`OptimizationsModel`) that contains all prompt optimizations, providing better organization and a convenient method to optimize all prompts at once.

## Data Model Changes

### Before

```python
class AppConfig(BaseModel):
    # ... other fields
    optimizations: dict[str, PromptOptimizationModel] = Field(default_factory=dict)
```

### After

```python
class OptimizationsModel(BaseModel):
    prompt_optimizations: dict[str, PromptOptimizationModel] = Field(default_factory=dict)
    
    def optimize(self, w: WorkspaceClient | None = None) -> dict[str, PromptModel]:
        """Optimize all prompts and return mapping of results"""
        results: dict[str, PromptModel] = {}
        for name, optimization in self.prompt_optimizations.items():
            results[name] = optimization.optimize(w)
        return results

class AppConfig(BaseModel):
    # ... other fields
    optimizations: Optional[OptimizationsModel] = None
```

## Configuration File Changes

### Before

```yaml
optimizations:
  optimize_sentiment:
    name: optimize_sentiment
    prompt: *sentiment_prompt
    # ... other fields
```

### After

```yaml
optimizations:
  prompt_optimizations:
    optimize_sentiment:
      name: optimize_sentiment
      prompt: *sentiment_prompt
      # ... other fields
```

## Code Usage Changes

### Before

```python
# Access individual optimizations
config = AppConfig.from_file("config.yaml")
for opt_name, optimization in config.optimizations.items():
    result = optimization.optimize()
```

### After

```python
# Option 1: Optimize all at once
config = AppConfig.from_file("config.yaml")
if config.optimizations:
    results = config.optimizations.optimize()
    
# Option 2: Optimize individually
if config.optimizations:
    for opt_name, optimization in config.optimizations.prompt_optimizations.items():
        result = optimization.optimize()
```

## Files Updated

### 1. `src/dao_ai/config.py`
- Fixed bug in `OptimizationsModel.optimize()` method (was referencing wrong field name)
- Added proper return type and documentation
- Model validates correctly and passes linting

### 2. `notebooks/10_optimize_prompts.py`
- Updated to import `OptimizationsModel`
- Changed to access `config.optimizations.prompt_optimizations`
- Updated all references throughout the notebook
- Maintains same functionality with new structure

### 3. `config/examples/prompt_optimization.yaml`
- Wrapped prompt_optimizations in the `optimizations` parent key
- Added proper indentation for nested structure
- Configuration validates correctly

### 4. `docs/prompt_optimization.md`
- Updated architecture section to document both models
- Updated all code examples to use new structure
- Added API reference for `OptimizationsModel.optimize()`
- Updated usage examples showing both optimization options

## Benefits of New Structure

1. **Better Organization**: All prompt optimizations are grouped under a single model
2. **Batch Operations**: Can optimize all prompts with a single method call
3. **Extensibility**: Easy to add other types of optimizations in the future
4. **Consistency**: Follows the pattern of other optional grouping models in the codebase
5. **Type Safety**: Maintains full type hints and Pydantic validation

## Testing

All changes have been validated:

```bash
PASS: PromptOptimizationModel instantiation
PASS: OptimizationsModel instantiation  
PASS: OptimizationsModel.optimize() method exists
PASS: AppConfig loading with OptimizationsModel
PASS: Dictionary access to prompt_optimizations
PASS: No linting errors
```

## Migration Guide

If you have existing configurations using the old structure:

1. Wrap your optimizations in a `prompt_optimizations` key:

```yaml
# Old
optimizations:
  my_opt: {...}

# New  
optimizations:
  prompt_optimizations:
    my_opt: {...}
```

2. Update code accessing optimizations:

```python
# Old
config.optimizations["my_opt"].optimize()

# New
config.optimizations.prompt_optimizations["my_opt"].optimize()

# Or use batch method
config.optimizations.optimize()
```

## Summary

The new structure provides:
- Better code organization
- Convenient batch optimization
- Full backward compatibility with proper migration
- Maintained type safety and validation
- Consistent with framework patterns

All files have been updated and tested successfully!

