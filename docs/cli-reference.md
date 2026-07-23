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

## Deploy (direct)

`dao-ai deploy` deploys the agent **directly** from your machine — no Databricks
Asset Bundle, no job. It calls `AppConfig.create_agent()` + `deploy_agent()`
in-process: for Model Serving it registers the MLflow model and creates the
serving endpoint (`agents.deploy`); for Apps it uploads the config + source and
drives the Apps REST API. It works from a pip-installed `dao-ai` (no repo
checkout), auto-links the UC trace destination, and auto-grants the runtime
service principal the trace-write permissions (gated on `app.manage_permissions`).
On success it prints a link to the endpoint or app.

```bash
# Deploy to Model Serving (default when no target is set)
dao-ai deploy -c config/my_config.yaml --target model_serving --profile fevm

# Deploy as a Databricks App
dao-ai deploy -c config/my_config.yaml --target apps --profile fevm

# Both surfaces
dao-ai deploy -c config/my_config.yaml --target both --profile fevm

# Ship the local dao-ai wheel instead of the published PyPI package
dao-ai deploy -c config/my_config.yaml --development --profile fevm
```

Target resolution: `--target` > `app.deployment_target` in the config >
`model_serving`.

**When to use which:**

- **`dao-ai deploy`** — quickest path to a running agent (Model Serving or Apps),
  no bundle artifact. Best for iterating.
- **`dao-ai generate-agent` / `generate-mcp`** — emit an inspectable/committable
  Databricks App bundle you can deploy from CI or hand-tune (Apps only).
- **`dao-ai generate-workflow`** — provision the full backing infra (schemas,
  Vector Search, Lakebase, Genie, UC functions) *and* deploy the agent, as a
  multi-task Databricks Job. The job's deploy step runs the same
  `create_agent`/`deploy_agent` code `dao-ai deploy` runs directly.

## Pipeline: Deploy and Run

The `dao-ai generate-workflow` subcommand deploys your agent to Databricks (under the hood it drives a Databricks Asset Bundle) and supports multi-cloud deployments with automatic cloud detection.

### Basic Deployment

```bash
# Deploy using default profile or environment
dao-ai generate-workflow --deploy -c config/my_config.yaml

# Deploy with parameter overrides
dao-ai generate-workflow --deploy -c config/my_config.yaml --param catalog=prod_catalog --param schema=prod_schema
```

`--param` (and the `--var` alias) flags are forwarded to the underlying `databricks bundle ...` invocation as `--var`, so Databricks Asset Bundles' own `${var.NAME}` substitution sees the same values when the names overlap.

### Multi-Cloud Deployment

The CLI automatically detects the cloud provider from your Databricks workspace and selects the appropriate configuration (node types, etc.):

```bash
# Deploy to AWS workspace
dao-ai generate-workflow --deploy -c config/my_config.yaml --profile aws-field-eng

# Deploy to Azure workspace
dao-ai generate-workflow --deploy -c config/my_config.yaml --profile azure-retail

# Deploy to GCP workspace
dao-ai generate-workflow --deploy -c config/my_config.yaml --profile gcp-analytics
```

### Deploy and Run

```bash
# Deploy and immediately run the job
dao-ai generate-workflow --deploy --run -c config/my_config.yaml --profile aws-field-eng
```

### Explicit Cloud Override

If cloud auto-detection doesn't work, you can specify the cloud explicitly:

```bash
dao-ai generate-workflow --deploy -c config/my_config.yaml --cloud aws
```

### Dry Run

Preview commands without executing:

```bash
dao-ai generate-workflow --deploy -c config/my_config.yaml --profile aws-field-eng --dry-run
```

## Generate Bundle

Generate a complete, deployable Databricks Apps bundle directory from a dao-ai config file. This is distinct from the `bundle` command -- while `bundle` wraps `databricks bundle deploy/run/destroy`, `generate-agent` **creates** the bundle project itself.

When the source config uses `${param.NAME}` / `${var.NAME}` parameters or `${workspace.*}` references, the generated bundle writes the **resolved** config (all references substituted to literal values, `parameters:` block dropped) so the deployed app does not need the original `--param` flags or a runtime workspace lookup.

### Basic Usage

```bash
dao-ai generate-agent -c config/retail.yaml -o ./my-bundle

# With parameter overrides baked into the generated bundle
dao-ai generate-agent -c config/retail.yaml -o ./my-bundle --param catalog=prod_catalog
```

### Output location

All three generators (`generate-agent`, `generate-mcp`, `generate-workflow`)
resolve where to write the bundle in this order:

1. `-o/--output-dir <dir>` — used verbatim.
2. `DAO_AI_BUNDLE_DIR` env var — bundles land at `$DAO_AI_BUNDLE_DIR/<kind>/<app>`
   (`<kind>` is `agent`, `mcp`, or `workflow`). Set this for a central location,
   e.g. `export DAO_AI_BUNDLE_DIR=~/.dao-ai/bundle`.
3. Default — `./.dao-ai/bundle/<kind>/<app>` (per-app, so multiple configs never
   collide; gitignored).

The per-app `<kind>/<app>` structure is always appended to the env-var/default
base, so deploying many configs stays isolated. Each generated `databricks.yaml`
force-includes its own source via `sync.include`, so App deploys work even when
the bundle is staged under a git-ignored directory.

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

### Dependency install: `requirements.txt` + pip

`dao-ai generate-agent` writes a `requirements.txt` to the bundle. The Databricks Apps build phase installs dependencies by running `pip install -r requirements.txt` — no `uv.lock` is generated or required. Published mode (`--no-development`) pins `dao-ai==<version>` for reproducible redeploys; `--development` references the bundled local wheel (`./dist/<wheel>`) instead of PyPI.

> **Future direction:** an unmerged branch moves Apps deploys to `pyproject.toml` + a portable `uv.lock` (`uv sync --locked --no-dev`, with internal-mirror URLs rewritten to public PyPI). That is not part of this release — today's generated bundles use `requirements.txt` + pip.

When `app.enable_chat_proxy` is `true` (the default), the deployed app automatically clones and builds the Databricks [e2e-chatbot-app-next](https://github.com/databricks/app-templates/tree/main/e2e-chatbot-app-next) chat UI at startup. The Apps runtime has Node.js pre-installed, so no Node.js is needed on your development machine. Set `enable_chat_proxy: false` to deploy without the chat UI.

### Trace persistence on Apps requires `trace_location`

The default MLflow control-plane trace exporter does not work on Databricks Apps today: the artifact-storage host (`us-east-1.storage.cloud.databricks.com`) is unreachable from Apps containers and spans silently fail to upload. Watch for this line in `databricks apps logs`:

```
WARNING mlflow.tracing.export.mlflow_v3: Failed to send trace to MLflow backend:
HTTPSConnectionPool(host='us-east-1.storage.cloud.databricks.com', port=443):
... Connection refused
```

To capture traces, configure `app.trace_location` in your config so traces route through a SQL warehouse to UC OTEL tables (a path Apps CAN reach):

```yaml
app:
  name: my_app
  # ...
  trace_location:
    schema: *retail_schema                   # reference an existing SchemaModel anchor
    warehouse: "your-warehouse-id"           # or a *warehouse anchor reference
```

When `trace_location` is set, `generate-agent` wires up the SQL warehouse as an App resource (CAN_USE for the App SP) and adds `MLFLOW_TRACING_SQL_WAREHOUSE_ID` to the App's `env`. The OTEL trace tables themselves are auto-created by MLflow at first trace write — dao-ai does not emit per-table grants because the tables don't exist at deploy time. After deploy, grant the App SP schema-level privileges (one-time):

```bash
SP=$(databricks apps get <app-name> -p <profile> --output json | jq -r .service_principal_client_id)
databricks grants update catalog <catalog> -p <profile> \
  --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"USE_CATALOG\"]}]}"
databricks grants update schema <catalog>.<schema> -p <profile> \
  --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"USE_SCHEMA\",\"CREATE_TABLE\",\"MODIFY\",\"SELECT\"]}]}"
```

When `trace_location` is unset, `generate-agent` emits a `⚠` warning to alert you. Local notebook/CLI runs and Model Serving deploys are unaffected.

See `examples/01_getting_started/ai_gateway.yaml` for a drop-in example.

#### Linking the UC trace destination — run `dao-ai link-trace-destination` between deploy and run

MLflow requires the UC trace-destination link to be established on an experiment **before** that experiment receives any traces. On a re-deploy (or after a `trace_location` change), the experiment already has traces from prior runs, so the app's runtime attempt to link is rejected with `already contains traces` and every subsequent trace silently drops with `TABLE_DOES_NOT_EXIST`.

The fix is a standalone CLI verb that links from **your** machine (operator credentials, deterministic timing) instead of relying on the running app to link itself:

```bash
databricks bundle deploy --target dev -p <profile>
dao-ai link-trace-destination -c my_config.yaml -p <profile>
databricks bundle run <app-name> --target dev -p <profile>
# then restart to pick up the freshly-linked destination:
databricks apps restart <app-name> -p <profile>
```

The verb is idempotent — safe on every deploy — but load-bearing on re-deploys and after `trace_location` changes. `generate-agent` prints a one-line reminder in its "Next steps" when `trace_location` is configured.

See [Link Trace Destination](#link-trace-destination) for full flag reference and the migration playbook for moving traces between destinations.

#### Runtime trace-destination sync (`apply_runtime_trace_destination`)

`link-trace-destination` writes the trace-destination tag on the experiment record so that future traces route to the configured UC schema. That works when MLflow's runtime picks up the linkage from the experiment — but if the app also has `MLFLOW_TRACING_DESTINATION` env set (dao-ai's `generate-agent` sets it as `catalog.schema` for warehouse routing), MLflow parses that env value as the deprecated `UCSchemaLocation` and populates the `_MLFLOW_TRACE_USER_DESTINATION` ContextVar accordingly. The ContextVar SHADOWS MLflow's auto-resolver from experiment tags, so the exporter targets `mlflow_experiment_trace_otel_spans` (the un-prefixed default) which doesn't exist on the prefixed schema — and every span export fails with `TABLE_DOES_NOT_EXIST`.

To close that gap, the App startup path (`apps/handlers.py`) and the MCP-server startup path (`mcp/server.py`) both call `apply_runtime_trace_destination(config)` from `dao_ai.providers.databricks` right after `link_experiment_trace_location`. The helper:

- When `trace_location.table_prefix` is set: writes a `UnityCatalog(catalog, schema, table_prefix)` directly into `_MLFLOW_TRACE_USER_DESTINATION` so the exporter picks `<prefix>_otel_spans`.
- When `trace_location` is set but `table_prefix` is unset: **clears** the ContextVar so MLflow's own `_resolve_experiment_uc_location` reads the experiment-linked `UnityCatalog` (with the backend-computed experiment-id prefix) from the tracking store. Constructing `UnityCatalog(catalog, schema)` without a prefix raises at export time — clearing the ContextVar is the safe path.
- When `config.app.trace_location` is None: no-op. Traces use the MLflow control-plane store.

The helper is only invoked from the two container entrypoints that HAVE ambient OAuth (Apps + MCP server). The Model Serving entrypoint (`apps/model_serving.py`) intentionally makes no in-container calls to MLflow — deploy-time `agents.deploy()` sets `MLFLOW_EXPERIMENT_ID` + `MLFLOW_TRACING_DESTINATION` + `MLFLOW_TRACING_SQL_WAREHOUSE_ID` on the endpoint, and the container relies on MLflow's env-driven routing (see the header comment in `model_serving.py` for the rationale — trying `mlflow.set_experiment` in the MS container hits an OAuth-config crash on any container whose model wasn't logged with the experiment as a resource dependency).

#### `table_prefix` is permanent per experiment

Once an experiment has been linked to a UC trace destination with a specific `table_prefix`, MLflow rejects any attempt to change it — you'll see `already contains traces` on the re-link. To change catalog / schema / table_prefix, provision a fresh experiment:

```bash
dao-ai create-experiment --name /Shared/my-app/dao-ai-fresh -p <profile>
# Then reference the new experiment id under `app.experiment.id` in the config,
# or rev `app.name` so the auto-declared experiment path is distinct.
```

### Overwriting Existing Files

If the output directory already contains generated files, they are skipped by default. Use `--overwrite` to overwrite:

```bash
dao-ai generate-agent -c config/retail.yaml -o ./my-bundle --overwrite
```

### Using a Databricks Profile

If your config references workspace resources (Genie rooms, warehouses, etc.), specify a profile so they can be resolved during generation:

```bash
dao-ai generate-agent -c config/retail.yaml -o ./my-bundle --profile my-workspace
```

### Development Mode

Use `--development` to bundle a local build of dao-ai instead of pulling from PyPI. This is useful when testing unreleased dao-ai changes in a deployed app.

```bash
dao-ai generate-agent -c config/retail.yaml -o ./my-bundle --development
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

## Link Trace Destination

`dao-ai link-trace-destination` attaches an MLflow experiment to its Unity Catalog trace destination declared under `app.trace_location`. Run it as an explicit step **between** `databricks bundle deploy` and `databricks bundle run` — see the [background above](#linking-the-uc-trace-destination--run-dao-ai-link-trace-destination-between-deploy-and-run) for why the app's runtime attempt is unreliable.

### Basic usage

```bash
databricks bundle deploy --target dev -p <profile>
dao-ai link-trace-destination -c my_config.yaml -p <profile>
databricks bundle run <app-name> --target dev -p <profile>
databricks apps restart <app-name> -p <profile>    # required — see below
```

### Flags

| Flag | Purpose |
|---|---|
| `-c FILE`, `--config FILE` | Config file. Must set `app.trace_location`. |
| `-p PROFILE`, `--profile PROFILE` | Databricks profile for auth. |
| `--experiment-id ID` | Skip resolution and use this experiment id directly. |
| `--param KEY=VALUE` / `--var KEY=VALUE` | Config parameter overrides (repeatable). |

### Experiment resolution

Tries in order:

1. `--experiment-id` explicit override.
2. `config.app.experiment.resolved_id` when the config sets an explicit `experiment:` block.
3. Bundle-declared name lookup — tries **both** the plain `/Users/<user>/<app-name>` name AND the DABs `--target dev`-prefixed variant `/Users/<user>/[dev <sanitized-user>] <app-name>`, so the same command works for prod deploys and personal dev deploys.

If none of the above resolves, the CLI prints the candidates and exits 1.

### Two things that surprise operators

**1. You must restart the app after linking.** MLflow's OTEL exporter binds the trace destination at **process startup**. Running `link-trace-destination` while the app is up does not retroactively route in-flight traces to the new location — the running exporter is already bound to whatever destination was in effect when it started. Trigger `databricks apps restart <name>` (or any bundle re-deploy) so the app picks up the fresh linkage.

**2. A UC-linked experiment is permanently bound to that destination on Databricks.** Verified against a live Databricks workspace (MLflow 3.11):

- **Re-linking to the *same* destination** — safe (idempotent, no error).
- **`mlflow.tracing.unset_experiment_trace_location(...)`** — the OSS API exists (`tracing/enablement.py:115-163`), but the Databricks control plane explicitly rejects it:
  > `BAD_REQUEST: Unlinking an experiment from a Unity Catalog trace location is not allowed. Once linked, an experiment cannot be unlinked from its trace location.`
- **Changing `table_prefix` / `catalog` / `schema`** — impossible, since you can't un-link first. The `unset → set` swap that OSS MLflow supports does not work here.

There is no `force`, `replace`, or delete-and-recreate-linkage flag anywhere in the client or server API. The only recovery path is the fresh-experiment migration playbook below.

### Migration playbook — moving traces to a new UC destination

When you actually need to change `table_prefix`, `catalog`, `schema`, or want traces to land somewhere new, you can't mutate the existing experiment — create a fresh one and re-point the app:

```yaml
# 1. In your config, point at a new experiment. Two options:
#
#    (a) Change the experiment name explicitly:
app:
  experiment:
    name: /Users/me@databricks.com/my-app-v2       # new path

#    (b) Rename the app itself (auto-declared experiment path derives
#        from app.name — a rename gives you a fresh experiment):
app:
  name: my-app-v2                                  # was: my-app

# 2. (Optional) update trace_location. `table_prefix` is optional in
#    dao-ai — if omitted, MLflow uses the experiment id as the prefix,
#    which is fine (and often preferred) since a fresh experiment
#    already gives you a fresh table namespace. Only set an explicit
#    `table_prefix` when you want a human-readable name in the OTEL
#    Delta tables (e.g. for dashboarding).
  trace_location:
    schema: *my_schema
    warehouse: *my_warehouse
    # table_prefix: my_app_v2_traces               # optional
```

```bash
# 3. Deploy → link → run → restart.
dao-ai generate-agent -c my_config.yaml -o ./bundle --overwrite
cd ./bundle
databricks bundle deploy --target dev -p <profile>
dao-ai link-trace-destination -c ../my_config.yaml -p <profile>
databricks bundle run <new-app-name> --target dev -p <profile>
databricks apps restart <new-app-name> -p <profile>
```

The **old** experiment stays linked to its original destination — Databricks does not allow un-linking. If you no longer need the old data, delete the experiment outright via `mlflow.delete_experiment(<old-id>)` (or the workspace UI). That's a soft-delete; a hard purge is a workspace cleanup job. The OTEL Delta tables the old experiment wrote to are not affected by experiment deletion — drop them separately if you want the storage back.

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

### Generate Workflow Options

```bash
dao-ai generate-workflow -c config/my_config.yaml [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-c, --config FILE` | Path to the dao-ai configuration file (required) |
| `-d, --deploy` | Deploy the workflow bundle to Databricks |
| `-r, --run` | Run the `deploy_job` after deploying |
| `--destroy` | Destroy the deployed workflow bundle |
| `-o, --output-dir DIR` | Bundle staging dir (default: `$DAO_AI_BUNDLE_DIR/workflow/<app>` or `./.dao-ai/bundle/workflow/<app>`) |
| `--overwrite` | Overwrite copied-in files in the staging dir |
| `-p, --profile NAME` | Databricks CLI profile to use |
| `--cloud {azure,aws,gcp}` | Cloud provider (auto-detected if not specified) |
| `-t, --target NAME` | Bundle target name (auto-generated if not specified) |
| `--development` / `--no-development` | Ship the local dao-ai wheel vs pin PyPI (default: auto-detect) |
| `--dry-run` | Preview commands without executing |

With no action flag (`--deploy`/`--run`/`--destroy`), `generate-workflow`
stages the bundle only, for inspection or manual `databricks bundle deploy`.

### Generate Agent / Generate MCP Options

```bash
dao-ai generate-agent -c config/my_config.yaml [OPTIONS]
dao-ai generate-mcp   -c config/my_config.yaml [OPTIONS]
```

Both emit a Databricks App bundle and share the same options. With no action
flag they generate only; `--deploy`/`--run`/`--destroy` drive
`databricks bundle` from the generated dir.

| Option | Description |
|--------|-------------|
| `-c, --config FILE` | Path to the dao-ai configuration file (required) |
| `-o, --output-dir DIR` | Output dir (default: `$DAO_AI_BUNDLE_DIR/<kind>/<app>` or `./.dao-ai/bundle/<kind>/<app>`, where `<kind>` is `agent` or `mcp`) |
| `--overwrite` | Overwrite existing files in the output directory |
| `-d, --deploy` | After generating, run `databricks bundle deploy` (auto-links traces + grants SP when `app.trace_location` is set) |
| `-r, --run` | Start the app via `databricks bundle run <app>` |
| `--destroy` | Destroy the deployed app bundle |
| `--development` / `--no-development` | Bundle a local dao-ai wheel vs pin PyPI (default: auto-detect) |
| `--dry-run` | Preview commands without executing |
| `-p, --profile NAME` | Databricks profile for config loading and deploy |

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
dao-ai generate-workflow --deploy -c config/hardware_store.yaml --profile aws-prod

# Deploy same app to Azure
dao-ai generate-workflow --deploy -c config/hardware_store.yaml --profile azure-prod

# Deploy same app to GCP
dao-ai generate-workflow --deploy -c config/hardware_store.yaml --profile gcp-prod
```

### Development vs Production

```bash
# Deploy to development workspace
dao-ai generate-workflow --deploy -c config/my_app.yaml --profile aws-dev

# Deploy to production workspace
dao-ai generate-workflow --deploy -c config/my_app.yaml --profile aws-prod
```

### Full Deployment Pipeline

```bash
# Validate configuration
dao-ai validate -c config/my_app.yaml

# Generate workflow diagram
dao-ai graph -c config/my_app.yaml -o workflow.png

# Deploy and run
dao-ai generate-workflow --deploy --run -c config/my_app.yaml --profile aws-field-eng
```

---

## Navigation

- [← Previous: Examples](examples.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: Python API →](python-api.md)

