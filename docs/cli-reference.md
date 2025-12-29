# CLI Reference

## Validate Configuration

Check your configuration for errors:

```bash
dao-ai validate -c config/my_config.yaml
```

## Generate JSON Schema

Generate JSON schema for IDE support and validation:

```bash
dao-ai schema > schemas/model_config_schema.json
```

## Visualize Agent Workflow

Generate a diagram showing how your agent works:

```bash
dao-ai graph -c config/my_config.yaml -o workflow.png
```

## Deploy with Databricks Asset Bundles

Deploy your agent to Databricks:

```bash
dao-ai bundle --deploy --run -c config/my_config.yaml --profile DEFAULT
```

## Interactive Chat

Start an interactive chat session with your agent:

```bash
dao-ai chat -c config/my_config.yaml
```

## Verbose Output

Increase verbosity for debugging (use `-v` through `-vvvv`):

```bash
dao-ai -vvvv validate -c config/my_config.yaml
```

## Command Options

### Common Options

- `-c, --config`: Path to configuration file (required)
- `-v, --verbose`: Increase verbosity level (can be repeated up to 4 times)
- `--help`: Show help message

### Validate Options

```bash
dao-ai validate -c config/my_config.yaml [OPTIONS]
```

### Graph Options

```bash
dao-ai graph -c config/my_config.yaml -o output.png [OPTIONS]
```

- `-o, --output`: Output file path (supports .png, .pdf, .svg)

### Bundle Options

```bash
dao-ai bundle -c config/my_config.yaml [OPTIONS]
```

- `--deploy`: Deploy after bundling
- `--run`: Run after deployment
- `--profile`: Databricks profile to use (default: DEFAULT)

### Chat Options

```bash
dao-ai chat -c config/my_config.yaml [OPTIONS]
```

Starts an interactive REPL session where you can chat with your agent locally.

---

## Navigation

- [← Previous: Examples](examples.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: Python API →](python-api.md)

