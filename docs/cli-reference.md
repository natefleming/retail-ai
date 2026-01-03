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

Deploy your agent to Databricks. The CLI supports multi-cloud deployments with automatic cloud detection.

### Basic Deployment

```bash
# Deploy using default profile or environment
dao-ai bundle --deploy -c config/my_config.yaml
```

### Multi-Cloud Deployment

The CLI automatically detects the cloud provider from your Databricks workspace and selects the appropriate configuration (node types, etc.):

```bash
# Deploy to AWS workspace
dao-ai bundle --deploy -c config/my_config.yaml --profile aws-field-eng

# Deploy to Azure workspace
dao-ai bundle --deploy -c config/my_config.yaml --profile azure-retail

# Deploy to GCP workspace
dao-ai bundle --deploy -c config/my_config.yaml --profile gcp-analytics
```

### Deploy and Run

```bash
# Deploy and immediately run the job
dao-ai bundle --deploy --run -c config/my_config.yaml --profile aws-field-eng
```

### Explicit Cloud Override

If cloud auto-detection doesn't work, you can specify the cloud explicitly:

```bash
dao-ai bundle --deploy -c config/my_config.yaml --cloud aws
```

### Dry Run

Preview commands without executing:

```bash
dao-ai bundle --deploy -c config/my_config.yaml --profile aws-field-eng --dry-run
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

---

## Command Options

### Common Options

| Option | Description |
|--------|-------------|
| `-c, --config FILE` | Path to configuration file (required) |
| `-p, --profile NAME` | Databricks CLI profile to use |
| `-v, --verbose` | Increase verbosity (can be repeated up to 4 times) |
| `--help` | Show help message |

### Validate Options

```bash
dao-ai validate -c config/my_config.yaml [OPTIONS]
```

### Graph Options

```bash
dao-ai graph -c config/my_config.yaml -o output.png [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Output file path (supports .png, .pdf, .svg) |

### Bundle Options

```bash
dao-ai bundle -c config/my_config.yaml [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-d, --deploy` | Deploy the bundle to Databricks |
| `-r, --run` | Run the deployment job after deploying |
| `--destroy` | Destroy the deployed bundle |
| `-p, --profile NAME` | Databricks CLI profile to use |
| `--cloud {azure,aws,gcp}` | Cloud provider (auto-detected if not specified) |
| `-t, --target NAME` | Bundle target name (auto-generated if not specified) |
| `--dry-run` | Preview commands without executing |

### Chat Options

```bash
dao-ai chat -c config/my_config.yaml [OPTIONS]
```

Starts an interactive REPL session where you can chat with your agent locally.

---

## Multi-Cloud Support

DAO AI supports deploying to Azure, AWS, and GCP Databricks workspaces. The CLI handles cloud-specific configurations automatically.

### How It Works

1. **Cloud Detection**: When you specify a `--profile`, the CLI detects the cloud provider from the workspace URL
2. **Target Selection**: The CLI uses the profile name as the deployment target for per-profile isolation
3. **Node Types**: Cloud-appropriate compute node types are automatically selected:
   - Azure: `Standard_D4ads_v5`
   - AWS: `i3.xlarge`
   - GCP: `n1-standard-4`

### Profile Configuration

Profiles are configured in `~/.databrickscfg`:

```ini
[aws-field-eng]
host = https://my-workspace.cloud.databricks.com
token = dapi...

[azure-retail]
host = https://adb-123456789.azuredatabricks.net
token = dapi...

[gcp-analytics]
host = https://my-workspace.gcp.databricks.com
token = dapi...
```

### Deployment Isolation

Each profile gets its own isolated deployment state:

```
/.bundle/my_app/aws-field-eng/files    # AWS deployment
/.bundle/my_app/azure-retail/files     # Azure deployment
/.bundle/my_app/gcp-analytics/files    # GCP deployment
```

This allows you to deploy the same application to multiple workspaces without conflicts.

---

## Examples

### Deploy to Multiple Clouds

```bash
# Deploy to AWS
dao-ai bundle --deploy -c config/hardware_store.yaml --profile aws-prod

# Deploy same app to Azure
dao-ai bundle --deploy -c config/hardware_store.yaml --profile azure-prod

# Deploy same app to GCP
dao-ai bundle --deploy -c config/hardware_store.yaml --profile gcp-prod
```

### Development vs Production

```bash
# Deploy to development workspace
dao-ai bundle --deploy -c config/my_app.yaml --profile aws-dev

# Deploy to production workspace
dao-ai bundle --deploy -c config/my_app.yaml --profile aws-prod
```

### Full Deployment Pipeline

```bash
# Validate configuration
dao-ai validate -c config/my_app.yaml

# Generate workflow diagram
dao-ai graph -c config/my_app.yaml -o workflow.png

# Deploy and run
dao-ai bundle --deploy --run -c config/my_app.yaml --profile aws-field-eng
```

---

## Navigation

- [← Previous: Examples](examples.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: Python API →](python-api.md)

