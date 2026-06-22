# CLI Reference

## Validate Configuration

Check your configuration for errors:

```bash
dao-ai validate -c config/my_config.yaml

# With parameter overrides (repeatable)
dao-ai validate -c config/my_config.yaml --param catalog=main --param module_id=09
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

# With parameter overrides
dao-ai graph -c config/my_config.yaml -o workflow.png --param catalog=main
```

## Pipeline: Deploy and Run

The `dao-ai pipeline` subcommand deploys your agent to Databricks (under the hood it drives a Databricks Asset Bundle) and supports multi-cloud deployments with automatic cloud detection.

### Basic Deployment

```bash
# Deploy using default profile or environment
dao-ai pipeline --deploy -c config/my_config.yaml

# Deploy with parameter overrides
dao-ai pipeline --deploy -c config/my_config.yaml --param catalog=prod_catalog --param schema=prod_schema
```

`--param` (and the `--var` alias) flags are forwarded to the underlying `databricks bundle ...` invocation as `--var`, so Databricks Asset Bundles' own `${var.NAME}` substitution sees the same values when the names overlap.

### Multi-Cloud Deployment

The CLI automatically detects the cloud provider from your Databricks workspace and selects the appropriate configuration (node types, etc.):

```bash
# Deploy to AWS workspace
dao-ai pipeline --deploy -c config/my_config.yaml --profile aws-field-eng

# Deploy to Azure workspace
dao-ai pipeline --deploy -c config/my_config.yaml --profile azure-retail

# Deploy to GCP workspace
dao-ai pipeline --deploy -c config/my_config.yaml --profile gcp-analytics
```

### Deploy and Run

```bash
# Deploy and immediately run the job
dao-ai pipeline --deploy --run -c config/my_config.yaml --profile aws-field-eng
```

### Explicit Cloud Override

If cloud auto-detection doesn't work, you can specify the cloud explicitly:

```bash
dao-ai pipeline --deploy -c config/my_config.yaml --cloud aws
```

### Dry Run

Preview commands without executing:

```bash
dao-ai pipeline --deploy -c config/my_config.yaml --profile aws-field-eng --dry-run
```

## Generate Bundle

Generate a complete, deployable Databricks Apps bundle directory from a dao-ai config file. This is distinct from the `bundle` command -- while `bundle` wraps `databricks bundle deploy/run/destroy`, `generate-bundle` **creates** the bundle project itself.

When the source config uses `${param.NAME}` / `${var.NAME}` parameters or `${workspace.*}` references, the generated bundle writes the **resolved** config (all references substituted to literal values, `parameters:` block dropped) so the deployed app does not need the original `--param` flags or a runtime workspace lookup.

### Basic Usage

```bash
dao-ai generate-bundle -c config/retail.yaml -o ./my-bundle

# With parameter overrides baked into the generated bundle
dao-ai generate-bundle -c config/retail.yaml -o ./my-bundle --param catalog=prod_catalog
```

### What Gets Generated

The command creates a self-contained bundle directory with everything needed to deploy a Databricks App:

| File | Description |
|------|-------------|
| `databricks.yaml` | Bundle definition with app config, resources, and scopes |
| `<config>.yaml` | Copy of your dao-ai agent configuration (retains its original filename) |
| `pyproject.toml` | Python project with dao-ai dependency |
| `.gitignore` | Ignore patterns for build artifacts |
| `.python-version` | Python version pin (3.11) |
| `src/<package>/` | Stub package for custom code |

### Lock file: user-owned

`dao-ai generate-bundle` does **not** create `uv.lock`. The lock encodes URLs and version pins that depend on the index your environment uses, so the user owns it. After `generate-bundle`, run:

```bash
cd ./my-bundle
uv sync           # produces uv.lock from pyproject.toml against your default index
```

Apps' native uv support activates when both `pyproject.toml` and `uv.lock` are present and `requirements.txt` is absent — the BUILD phase runs `uv sync --locked --no-dev` for you.

> **Databricks-internal users**: if your local `uv` config defaults to the internal `pypi-proxy.dev.databricks.com` mirror, your generated lock will contain URLs that Apps containers cannot reach. Rewrite them before deploy (hashes don't change — proxy is a transparent mirror):
>
> ```bash
> sed -i '' \
>   -e 's|pypi-proxy\.dev\.databricks\.com/packages/|files.pythonhosted.org/packages/|g' \
>   -e 's|pypi-proxy\.dev\.databricks\.com/simple/|pypi.org/simple/|g' \
>   uv.lock
> ```

When `app.enable_chat_proxy` is `true` (the default), the deployed app automatically clones and builds the Databricks [e2e-chatbot-app-next](https://github.com/databricks/app-templates/tree/main/e2e-chatbot-app-next) chat UI at startup. The Apps runtime has Node.js pre-installed, so no Node.js is needed on your development machine. Set `enable_chat_proxy: false` to deploy without the chat UI.

### Overwriting Existing Files

If the output directory already contains generated files, they are skipped by default. Use `--force` to overwrite:

```bash
dao-ai generate-bundle -c config/retail.yaml -o ./my-bundle --force
```

### Using a Databricks Profile

If your config references workspace resources (Genie rooms, warehouses, etc.), specify a profile so they can be resolved during generation:

```bash
dao-ai generate-bundle -c config/retail.yaml -o ./my-bundle --profile my-workspace
```

### Development Mode

Use `--development` to bundle a local build of dao-ai instead of pulling from PyPI. This is useful when testing unreleased dao-ai changes in a deployed app.

```bash
dao-ai generate-bundle -c config/retail.yaml -o ./my-bundle --development
```

Development mode changes the generated bundle in several ways:

- **Local wheel**: Copies the dao-ai wheel from `dist/` into the bundle. If no wheel exists, one is built automatically via `uv build --wheel`.
- **Path dependency**: The generated `pyproject.toml` uses a `[tool.uv.sources]` path dependency pointing at the local wheel instead of pinning a PyPI version.
- **No artifacts block**: The `databricks.yaml` omits the `artifacts` section so the wheel uploads as a regular source file rather than being intercepted by the artifact system.
- **Adjusted .gitignore**: The `dist/` directory is not ignored, since the wheel must be included in the bundle.

### Next Steps

After generating the bundle, the command prints the next steps:

```bash
cd ./my-bundle
uv sync
databricks bundle deploy --target dev
databricks bundle run <app-name> --target dev
```

## Interactive Chat

Start an interactive chat session with your agent:

```bash
dao-ai chat -c config/my_config.yaml

# With parameter overrides
dao-ai chat -c config/my_config.yaml --param catalog=nfleming --param module_id=09
```

## List MCP Tools

Discover and inspect tools available from MCP (Model Context Protocol) servers configured in your application.

### Basic Usage

List all MCP tools with full descriptions and schemas:

```bash
dao-ai list-mcp-tools -c config/my_config.yaml

# With parameter overrides
dao-ai list-mcp-tools -c config/my_config.yaml --param catalog=main
```

### Show Only Filtered Tools

Use `--apply-filters` to see only the tools that will actually be loaded (respecting `include_tools` and `exclude_tools` configuration):

```bash
dao-ai list-mcp-tools -c config/my_config.yaml --apply-filters
```

### What It Shows

This command displays comprehensive information about each MCP server and its tools:

- **Server Information**: MCP server URL, transport type, and connection details
- **Filter Configuration**: `include_tools` and `exclude_tools` patterns
- **Tool Statistics**: Total available, included, and excluded tool counts
- **Tool Details** (for each included tool):
  - Full description (no truncation)
  - Parameters in readable format with:
    - Parameter names and types
    - Required vs optional indicators
    - Inline enum values
    - Parameter descriptions
    - Nested object structures
- **Exclusion Reasons**: Why tools are excluded (pattern matches, not in include list)

### Output Format

**Default view** (shows all tools with include/exclude status):
```
📦 Tool: search_tools
   Server: http://mcp-server.example.com
   Transport: stdio

   Filters:
     Include: search_*, query_*
     Exclude: *_deprecated

   Available Tools: 10 total
   ├─ ✓ Included: 7
   └─ ✗ Excluded: 3

   ✓ Included Tools (7):

     • search_web
       Description: Search the web for information...
       Parameters:
         query: string (required)
           └─ The search query to execute
         max_results: integer (optional)
           └─ Maximum number of results (default: 10)
         language: string (one of: en, es, fr, de) (optional)
           └─ Language for results

   ✗ Excluded Tools (3):
     • internal_api (not in include list)
     • legacy_search_deprecated (matches exclude pattern: *_deprecated)
```

**With `--apply-filters`** (shows only included tools):
```
📦 Tool: search_tools
   Server: http://mcp-server.example.com

   Available Tools: 7 (after filters)

   Tools (7):
     • search_web
       Description: Search the web for information...
       Parameters:
         query: string (required)
           └─ The search query to execute
```

### Use Cases

- **Discovery**: Find available tools before configuring agents
- **Documentation**: Review tool descriptions and parameter schemas
- **Debugging**: Verify filter configuration is working correctly
- **Validation**: Ensure MCP server connectivity
- **Planning**: Determine which tools to include in agent configuration

### Schema Format

Schemas are displayed in a concise, readable format (53% smaller than JSON):

- **Type-first**: Parameter types immediately visible
- **Clear indicators**: Required vs optional at a glance
- **Inline enums**: Allowed values shown directly
- **Proper nesting**: Hierarchical structure with indentation
- **No boilerplate**: Clean format without JSON syntax

## Inspect Declared Parameters

Print every parameter declared in a config's `parameters:` block, its current resolved value, and where that value came from.

```bash
dao-ai parameters list -c config/my_config.yaml

# With overrides to see how they resolve
dao-ai parameters list -c dao_ai.yaml --param module_id=09
```

`dao-ai vars` is kept as an alias for backward compatibility, and `--var` continues to work alongside `--param`.

Sample output:

```
NAME       REQUIRED  DEFAULT  RESOLVED  SOURCE   DESCRIPTION
------------------------------------------------------------
catalog    no        main     main      default  Unity Catalog catalog name
module_id  yes       -        09        --param  Workshop module identifier
```

Source values: `--param`, `env`, `default`, `inline-default`, `MISSING`.

Exit code is 1 if any required parameter is `MISSING`, 0 otherwise. This makes `parameters list` useful in CI pipelines to verify all overrides are wired up before deploying.

Any `${workspace.*}` references in a parameter's `default` are resolved before the table is rendered, so the listed `DEFAULT` reflects the live workspace user / host.

Full reference: [Parameters (Load-Time Substitution)](configuration-reference.md#parameters-load-time-substitution).

---

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
| `--param KEY=VALUE` | Override a `${param.KEY}` / `${var.KEY}` parameter in the config (repeatable). `--var` is kept as an alias. |
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

### Pipeline Options

```bash
dao-ai pipeline -c config/my_config.yaml [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-d, --deploy` | Deploy the pipeline to Databricks |
| `-r, --run` | Run the deployment job after deploying |
| `--destroy` | Destroy the deployed pipeline |
| `-p, --profile NAME` | Databricks CLI profile to use |
| `--cloud {azure,aws,gcp}` | Cloud provider (auto-detected if not specified) |
| `-t, --target NAME` | Bundle target name (auto-generated if not specified) |
| `--dry-run` | Preview commands without executing |

### Generate Bundle Options

```bash
dao-ai generate-bundle -c config/my_config.yaml -o ./my-bundle [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-c, --config FILE` | Path to the dao-ai configuration file (required) |
| `-o, --output-dir DIR` | Output directory for generated files (default: `.`) |
| `--force` | Overwrite existing files in the output directory |
| `--development` | Bundle a local dao-ai wheel instead of a PyPI dependency |
| `-p, --profile NAME` | Databricks profile for config loading and resource resolution |

### Chat Options

```bash
dao-ai chat -c config/my_config.yaml [OPTIONS]
```

Starts an interactive REPL session where you can chat with your agent locally.

### List MCP Tools Options

```bash
dao-ai list-mcp-tools -c config/my_config.yaml [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-c, --config FILE` | Path to configuration file (default: `./config/model_config.yaml`) |
| `--apply-filters` | Only show tools that pass include/exclude filters (hide excluded tools) |

Lists all available MCP tools with full descriptions and readable parameter schemas. Supports filtering to show only included tools.

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
dao-ai pipeline --deploy -c config/hardware_store.yaml --profile aws-prod

# Deploy same app to Azure
dao-ai pipeline --deploy -c config/hardware_store.yaml --profile azure-prod

# Deploy same app to GCP
dao-ai pipeline --deploy -c config/hardware_store.yaml --profile gcp-prod
```

### Development vs Production

```bash
# Deploy to development workspace
dao-ai pipeline --deploy -c config/my_app.yaml --profile aws-dev

# Deploy to production workspace
dao-ai pipeline --deploy -c config/my_app.yaml --profile aws-prod
```

### Full Deployment Pipeline

```bash
# Validate configuration
dao-ai validate -c config/my_app.yaml

# Generate workflow diagram
dao-ai graph -c config/my_app.yaml -o workflow.png

# Deploy and run
dao-ai pipeline --deploy --run -c config/my_app.yaml --profile aws-field-eng
```

---

## Navigation

- [← Previous: Examples](examples.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: Python API →](python-api.md)

