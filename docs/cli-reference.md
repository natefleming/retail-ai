# CLI Reference

## Shell Completion

`dao-ai` supports tab-completion of subcommands and flags via
[argcomplete](https://kislyuk.github.io/argcomplete/) (a core dependency —
nothing extra to install). Enable it once in your shell:

**bash** — add to `~/.bashrc`:

```bash
eval "$(register-python-argcomplete dao-ai)"
```

**zsh** — add to `~/.zshrc`:

```zsh
autoload -U bashcompinit && bashcompinit
eval "$(register-python-argcomplete dao-ai)"
```

Restart the shell (or `source` the rc file), then `dao-ai <TAB>` completes
subcommands and `dao-ai agent deploy --<TAB>` completes flags.

## Global Options

`-p/--profile` and `-v/--verbose` are accepted at any level (before or after the
subcommand). When `--profile` is set, dao-ai **clears the ambient `DATABRICKS_*`
environment variables** (`DATABRICKS_TOKEN`, `DATABRICKS_HOST`,
`DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`, `DATABRICKS_AUTH_TYPE`) for
the current process so the profile is authoritative. This prevents a stray token
or host in your shell or a `.env` file from silently overriding the profile and
targeting the wrong workspace. If you rely on env-var auth, omit `--profile`.

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

## Deploy

`dao-ai agent deploy` deploys the agent using its staged bundle (the default path;
it auto-generates the bundle first if nothing is staged). Pass `--mode model_serving`
to deploy directly to Databricks Model Serving (no bundle). All paths call
`AppConfig.create_agent()` + `deploy_agent()` in-process: for Model Serving it
registers the MLflow model and creates the serving endpoint (`agents.deploy`); for
Apps it uploads the config + source and drives the Apps REST API. Auto-links the UC
trace destination and auto-grants the runtime service principal the trace-write
permissions (gated on `app.manage_permissions`).

For the bundle-less SDK fast-path (no bundle written to disk), use `agent up --direct`
(apps/mcp only) — `--direct` is an `up`-only flag.

```bash
# Deploy to Model Serving (SDK path, no bundle)
dao-ai agent deploy -c config/my_config.yaml --mode model_serving --profile fevm

# Deploy as a Databricks App (bundle path, default)
dao-ai agent deploy -c config/my_config.yaml --mode apps --profile fevm

# Bring up as Apps via SDK directly (no bundle on disk — fast iteration)
dao-ai agent up -c config/my_config.yaml --mode apps --direct --profile fevm

# Deploy MCP server App
dao-ai agent deploy -c config/my_config.yaml --mode mcp --profile fevm

# Ship the local dao-ai wheel instead of the published PyPI package
dao-ai agent generate -c config/my_config.yaml --development --profile fevm
dao-ai agent deploy   -c config/my_config.yaml --profile fevm
```

Mode resolution: `--mode` flag wins; default is `apps`.

No extra install is required: a plain `pip install dao-ai` is enough to deploy to
any mode. Model Serving logs the MLflow model in-process, which touches
spark-connect — the core `databricks-connect` dependency supplies a
protobuf-5-compatible pyspark for this, so do **not** add a standalone `pyspark`
(pyspark 4.x needs protobuf ≥ 6.33 and collides with `databricks-ai-search`'s
`protobuf < 6` cap; see issue #211). The Databricks runtime provides its own Spark,
so this local stack never ships to the deployed endpoint.

**When to use which:**

- **`dao-ai agent deploy --mode model_serving`** — deploy to Model Serving (SDK, no bundle). Best for iterating on the serving endpoint.
- **`dao-ai agent deploy --mode apps` / `--mode mcp`** — deploy Apps or MCP bundle. Generates on first run if unstaged; ships staged bundle if already generated.
- **`dao-ai agent up --direct`** — bring up Apps/MCP via SDK without writing a bundle (fast iteration when you don't need an auditable bundle artifact). `--direct` is an `up`-only flag.
- **`dao-ai workflow`** — provision the full backing infra (schemas,
  Vector Search, Lakebase, Genie, UC functions) *and* deploy the agent, as a
  multi-task Databricks Job. The job's deploy step runs the same
  `create_agent`/`deploy_agent` code as the direct deploy paths.

## Bundle Generators: `agent`, `workflow`

The bundle generators are **verb-under-noun** commands — pick a noun for
what you're shipping, then a verb for the lifecycle step:

| Noun | What it ships |
|------|---------------|
| `dao-ai agent` | A Databricks App running the agent graph (default: `--mode apps`). Use `--mode mcp` to emit the MCP-server App instead, or `--mode model_serving` on `deploy` to go SDK-direct. |
| `dao-ai workflow` | A multi-task Databricks Job that provisions the backing infra (schemas, Vector Search, Lakebase, Genie, UC functions) *and* deploys the agent. |

Each noun takes the same five verbs:

```bash
dao-ai agent    up|generate|deploy|run|destroy  -c <cfg> [-p <profile>]
dao-ai workflow up|generate|deploy|run|destroy  -c <cfg> [-p <profile>]
```

**The one-command path — `up`:**

- **`up`** is the fast path to a live agent: it generates the bundle (if nothing
  is staged), deploys it, links the trace destination, then starts it —
  equivalent to `generate → deploy → run` in one command. This is what you want
  most of the time: `dao-ai agent up -c <cfg> -p fevm`. For `--mode model_serving`
  it registers the model and creates the endpoint (there is no separate `run`
  step — the endpoint serves once `READY`). Add `--direct` (apps/mcp only) to
  deploy via the SDK with no bundle written to disk.

**The granular lifecycle — `generate → deploy → run → destroy`:**

- **`generate`** stages a bundle to disk (`<base>/<noun>/<app>`, where `<base>`
  is `$DAO_AI_BUNDLE_DIR` or `./.dao-ai/bundle`, or `-o <dir>`) and does nothing
  else — inspect or hand-edit the staged files before shipping.
- **`deploy`** pushes the bundle. For `agent`/`mcp`, `deploy` is
  `databricks bundle deploy`; when a staged bundle exists it deploys it in place
  (preserving hand-edits), and when **nothing is staged it auto-generates
  first**, so `dao-ai agent deploy -c <cfg> -p fevm` works from a clean tree. For
  `workflow`, `deploy` acts on the **already-staged** bundle and does **not**
  auto-generate — run `dao-ai workflow generate` (or `workflow up`) first; it
  errors if nothing is staged. Use `--mode model_serving` on `agent deploy` to go
  SDK-direct to an endpoint.
- **`run`** starts the deployed bundle and **does not auto-generate or deploy** —
  it errors if nothing is deployed. For `agent`/`mcp`, `run` is
  `databricks bundle run <app>`; for `workflow`, `run` is
  `databricks bundle run deploy_job` (the provisioning job).
- **`destroy`** tears the deployed bundle down.

This is the payoff: a deploy that failed on a transient error can be retried
with just `dao-ai agent deploy -c <cfg> -p fevm` — no regeneration. To deploy
*and* start in one shot, use `dao-ai agent up` instead.

**Edit safety.** Re-running `generate` into the **default** staging dir refuses
to wipe it once it has local hand-edits (detected via the `.dao-ai-generated`
marker file's mtime), unless you pass `--overwrite`. The error points you at
`<noun> deploy` to ship the edits instead. A `-o <dir>` is never auto-wiped.
The source-selection flags `--overwrite`, `--development`, and `--no-development`
are on `up`, `generate`, and `deploy` (for `agent`/`mcp`, `deploy` can
auto-generate; for `workflow` it acts on the already-staged bundle); `run` and
`destroy` act on already-built artifacts and don't carry them.

> **Migration:** the flat commands `generate-agent`, `generate-mcp`, and
> `generate-workflow` have been removed. Use `dao-ai agent generate`,
> `dao-ai agent generate --mode mcp`, and `dao-ai workflow generate` instead.
> See [Migration reference](#migration-from-pre-v2-cli) below.

## Workflow: Provision and Deploy

The `dao-ai workflow` command deploys your agent to Databricks (under the hood it drives a Databricks Asset Bundle) and supports multi-cloud deployments with automatic cloud detection.

### Basic Deployment

```bash
# Provision infra + deploy + run the deploy_job in one command
dao-ai workflow up -c config/my_config.yaml

# With parameter overrides
dao-ai workflow up -c config/my_config.yaml --param catalog=prod_catalog --param schema=prod_schema

# Re-deploy the already-staged bundle without regenerating (e.g. retry a transient failure)
dao-ai workflow deploy -c config/my_config.yaml
```

`--param` (and the `--var` alias) values are baked into the staged config, and are **also** forwarded to the underlying `databricks bundle ...` invocation as `--var` **only for names the generated `databricks.yaml` actually declares as bundle variables** — so Databricks Asset Bundles' own `${var.NAME}` substitution sees the same values when the names overlap, without failing on a dao-ai-only parameter the bundle doesn't declare.

### Multi-Cloud Deployment

The CLI automatically detects the cloud provider from your Databricks workspace and selects the appropriate configuration (node types, etc.):

```bash
# Deploy to AWS workspace
dao-ai workflow up -c config/my_config.yaml --profile aws-field-eng

# Deploy to Azure workspace
dao-ai workflow up -c config/my_config.yaml --profile azure-retail

# Deploy to GCP workspace
dao-ai workflow up -c config/my_config.yaml --profile gcp-analytics
```

### Granular Lifecycle

```bash
# Stage the bundle only (inspect / hand-edit before shipping)
dao-ai workflow generate -c config/my_config.yaml --profile aws-field-eng

# Deploy the staged bundle (run `generate` or `up` first — this does not auto-generate)
dao-ai workflow deploy -c config/my_config.yaml --profile aws-field-eng

# Run the deploy_job on an already-deployed bundle (databricks bundle run deploy_job)
dao-ai workflow run -c config/my_config.yaml --profile aws-field-eng
```

### Explicit Cloud Override

Cloud is auto-detected from the workspace URL. If detection can't determine it,
the command stops and asks you to pass `--cloud` explicitly:

```bash
dao-ai workflow up -c config/my_config.yaml --cloud aws
```

### Dry Run

Preview commands without executing:

```bash
dao-ai workflow up -c config/my_config.yaml --profile aws-field-eng --dry-run
```

## Agent / MCP Bundle

Generate a complete, deployable Databricks Apps bundle directory from a dao-ai config file. This is distinct from the `bundle` command -- while `bundle` wraps `databricks bundle deploy/run/destroy`, `dao-ai agent generate` **creates** the bundle project itself.

When the source config uses `${param.NAME}` / `${var.NAME}` parameters or `${workspace.*}` references, the generated bundle writes the **resolved** config (all references substituted to literal values, `parameters:` block dropped) so the deployed app does not need the original `--param` flags or a runtime workspace lookup.

### Basic Usage

```bash
dao-ai agent generate -c config/retail.yaml -o ./my-bundle

# With parameter overrides baked into the generated bundle
dao-ai agent generate -c config/retail.yaml -o ./my-bundle --param catalog=prod_catalog

# Generate, deploy, and start the app in one command
dao-ai agent up -c config/retail.yaml -p fevm

# Ship the already-staged bundle without regenerating (e.g. after hand-editing, or retrying a transient deploy failure)
dao-ai agent deploy -c config/retail.yaml -p fevm

# Deploy the staged bundle, then start it
dao-ai agent deploy -c config/retail.yaml -p fevm
dao-ai agent run    -c config/retail.yaml -p fevm
```

MCP server bundles use `dao-ai agent generate --mode mcp` (not a separate noun). Use `dao-ai agent run` to `databricks bundle run <app>` an already-deployed bundle, and `dao-ai agent destroy` to tear it down. See [Bundle Generators](#bundle-generators-agent-workflow) for the full lifecycle.

### Output location

All three generators (`agent`, `mcp`, `workflow`)
resolve where to write the bundle in this order:

1. `-s/--staging-dir <dir>` — used verbatim.
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

### Dependency install: `pyproject.toml` + portable `uv.lock`

`dao-ai agent generate` writes a `pyproject.toml` and a portable `uv.lock` to the bundle (no `requirements.txt` — its presence would take precedence and force the pip path). The Databricks Apps build phase runs `uv sync --locked --no-dev` from them. Published mode (`--no-development`) pins `dao-ai[<extras>]==<version>` for reproducible redeploys; `--development` redirects dao-ai to the bundled local wheel via `[tool.uv.sources]`. `uv lock` records the full closure, and any internal-mirror host (`pypi-proxy.dev.databricks.com`) is rewritten to the public CDN so the lock resolves from Apps containers.

> **Pre-publish note:** published-mode lock generation resolves `dao-ai==<version>` from PyPI, so it fails with an actionable error until that version is published (release-time / CI). For local/pre-release iteration, generate with `--development` (locks against the bundled wheel — works anytime).

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

When `trace_location` is set, `agent generate` wires up the SQL warehouse as an App resource (CAN_USE for the App SP) and adds `MLFLOW_TRACING_SQL_WAREHOUSE_ID` to the App's `env`. The OTEL trace tables themselves are auto-created by MLflow at first trace write — dao-ai does not emit per-table grants because the tables don't exist at deploy time. After deploy, grant the App SP schema-level privileges (one-time):

```bash
SP=$(databricks apps get <app-name> -p <profile> --output json | jq -r .service_principal_client_id)
databricks grants update catalog <catalog> -p <profile> \
  --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"USE_CATALOG\"]}]}"
databricks grants update schema <catalog>.<schema> -p <profile> \
  --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"USE_SCHEMA\",\"CREATE_TABLE\",\"MODIFY\",\"SELECT\"]}]}"
```

When `trace_location` is unset, `agent generate` emits a `⚠` warning to alert you. Local notebook/CLI runs and Model Serving deploys are unaffected.

See `examples/01_getting_started/ai_gateway.yaml` for a drop-in example.

#### Linking the UC trace destination — run `dao-ai trace link` between deploy and run

MLflow requires the UC trace-destination link to be established on an experiment **before** that experiment receives any traces. On a re-deploy (or after a `trace_location` change), the experiment already has traces from prior runs, so the app's runtime attempt to link is rejected with `already contains traces` and every subsequent trace silently drops with `TABLE_DOES_NOT_EXIST`.

The fix is a standalone CLI verb that links from **your** machine (operator credentials, deterministic timing) instead of relying on the running app to link itself:

```bash
databricks bundle deploy --target dev -p <profile>
dao-ai trace link -c my_config.yaml -p <profile>
databricks bundle run <app-name> --target dev -p <profile>
# then restart to pick up the freshly-linked destination:
databricks apps restart <app-name> -p <profile>
```

The verb is idempotent — safe on every deploy — but load-bearing on re-deploys and after `trace_location` changes. `agent generate` prints a one-line reminder in its "Next steps" when `trace_location` is configured.

See [Trace Commands](#trace-commands) for full flag reference and the migration playbook for moving traces between destinations.

#### Runtime trace-destination sync (`apply_runtime_trace_destination`)

`dao-ai trace link` writes the trace-destination tag on the experiment record so that future traces route to the configured UC schema. That works when MLflow's runtime picks up the linkage from the experiment — but if the app also has `MLFLOW_TRACING_DESTINATION` env set (dao-ai's `agent generate` sets it as `catalog.schema` for warehouse routing), MLflow parses that env value as the deprecated `UCSchemaLocation` and populates the `_MLFLOW_TRACE_USER_DESTINATION` ContextVar accordingly. The ContextVar SHADOWS MLflow's auto-resolver from experiment tags, so the exporter targets `mlflow_experiment_trace_otel_spans` (the un-prefixed default) which doesn't exist on the prefixed schema — and every span export fails with `TABLE_DOES_NOT_EXIST`.

To close that gap, the App startup path (`apps/handlers.py`) and the MCP-server startup path (`mcp/server.py`) both call `apply_runtime_trace_destination(config)` from `dao_ai.providers.databricks` right after `link_experiment_trace_location`. The helper:

- When `trace_location.table_prefix` is set: writes a `UnityCatalog(catalog, schema, table_prefix)` directly into `_MLFLOW_TRACE_USER_DESTINATION` so the exporter picks `<prefix>_otel_spans`.
- When `trace_location` is set but `table_prefix` is unset: **clears** the ContextVar so MLflow's own `_resolve_experiment_uc_location` reads the experiment-linked `UnityCatalog` (with the backend-computed experiment-id prefix) from the tracking store. Constructing `UnityCatalog(catalog, schema)` without a prefix raises at export time — clearing the ContextVar is the safe path.
- When `config.app.trace_location` is None: no-op. Traces use the MLflow control-plane store.

The helper is only invoked from the two container entrypoints that HAVE ambient OAuth (Apps + MCP server). The Model Serving entrypoint (`apps/model_serving.py`) intentionally makes no in-container calls to MLflow — deploy-time `agents.deploy()` sets `MLFLOW_EXPERIMENT_ID` + `MLFLOW_TRACING_DESTINATION` + `MLFLOW_TRACING_SQL_WAREHOUSE_ID` on the endpoint, and the container relies on MLflow's env-driven routing (see the header comment in `model_serving.py` for the rationale — trying `mlflow.set_experiment` in the MS container hits an OAuth-config crash on any container whose model wasn't logged with the experiment as a resource dependency).

#### `table_prefix` is permanent per experiment

Once an experiment has been linked to a UC trace destination with a specific `table_prefix`, MLflow rejects any attempt to change it — you'll see `already contains traces` on the re-link. To change catalog / schema / table_prefix, provision a fresh experiment:

```bash
dao-ai trace create --name /Shared/my-app/dao-ai-fresh -p <profile>
# Then reference the new experiment id under `app.experiment.id` in the config,
# or rev `app.name` so the auto-declared experiment path is distinct.
```

### Overwriting Existing Files

If the output directory already contains generated files, they are skipped by default. Use `--overwrite` to overwrite:

```bash
dao-ai agent generate -c config/retail.yaml -o ./my-bundle --overwrite
```

Re-running `generate` into the **default** staging dir (no `-o`) refuses to wipe it once it has local hand-edits (detected via the `.dao-ai-generated` marker's mtime) unless you pass `--overwrite`; the error points you at `dao-ai agent deploy` to ship the edits as-is. A `-o <dir>` is never auto-wiped. `--overwrite` is only valid on `generate`.

### Using a Databricks Profile

If your config references workspace resources (Genie rooms, warehouses, etc.), specify a profile so they can be resolved during generation:

```bash
dao-ai agent generate -c config/retail.yaml -o ./my-bundle --profile my-workspace
```

### Development Mode

Use `--development` to bundle a local build of dao-ai instead of pulling from PyPI. This is useful when testing unreleased dao-ai changes in a deployed app.

```bash
dao-ai agent generate -c config/retail.yaml -o ./my-bundle --development
```

Development mode changes the generated bundle in several ways:

- **Local wheel**: Copies the dao-ai wheel from `dist/` into the bundle. If no wheel exists, one is built automatically via `uv build --wheel`.
- **Path dependency**: The generated `pyproject.toml` uses a `[tool.uv.sources]` path dependency pointing at the local wheel instead of pinning a PyPI version.
- **No artifacts block**: The `databricks.yaml` omits the `artifacts` section so the wheel uploads as a regular source file rather than being intercepted by the artifact system.
- **Adjusted .gitignore**: The `dist/` directory is not ignored, since the wheel must be included in the bundle.

### Next Steps

After generating the bundle, the command prints the next steps. You can either drive Databricks directly, or use the `deploy`/`run` verbs (which act on the staged dir without regenerating):

```bash
# Option A — dao-ai verbs (deploy then start; or use `agent up` to do both at once)
dao-ai agent deploy -c config/retail.yaml -p <profile>
dao-ai agent run    -c config/retail.yaml -p <profile>

# Option B — drive databricks bundle directly
cd ./my-bundle
uv sync
databricks bundle deploy --target dev
databricks bundle run <app-name> --target dev
```

## Trace Commands

The `dao-ai trace` group manages MLflow experiments and UC trace destinations.

### `dao-ai trace link`

`dao-ai trace link` attaches an MLflow experiment to its Unity Catalog trace destination declared under `app.trace_location`. Run it as an explicit step **between** `databricks bundle deploy` and `databricks bundle run` — see the [background above](#linking-the-uc-trace-destination--run-dao-ai-trace-link-between-deploy-and-run) for why the app's runtime attempt is unreliable.

```bash
databricks bundle deploy --target dev -p <profile>
dao-ai trace link -c my_config.yaml -p <profile>
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

**1. You must restart the app after linking.** MLflow's OTEL exporter binds the trace destination at **process startup**. Running `dao-ai trace link` while the app is up does not retroactively route in-flight traces to the new location — the running exporter is already bound to whatever destination was in effect when it started. Trigger `databricks apps restart <name>` (or any bundle re-deploy) so the app picks up the fresh linkage.

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
dao-ai agent generate -c my_config.yaml -o ./bundle --overwrite
cd ./bundle
databricks bundle deploy --target dev -p <profile>
dao-ai trace link -c ../my_config.yaml -p <profile>
databricks bundle run <new-app-name> --target dev -p <profile>
databricks apps restart <new-app-name> -p <profile>
```

The **old** experiment stays linked to its original destination — Databricks does not allow un-linking. If you no longer need the old data, delete the experiment outright via `mlflow.delete_experiment(<old-id>)` (or the workspace UI). That's a soft-delete; a hard purge is a workspace cleanup job. The OTEL Delta tables the old experiment wrote to are not affected by experiment deletion — drop them separately if you want the storage back.

## Monitor

`dao-ai monitor` groups production observability for the deployed agent.

### `dao-ai monitor scorers enable|status|disable`

Register, inspect, or stop MLflow monitoring scorers that continuously evaluate
production traces for quality, safety, and guideline compliance. Requires
`app.monitoring` in the YAML config.

```bash
dao-ai monitor scorers enable  -c config/model_config.yaml   # register + start
dao-ai monitor scorers status  -c config/model_config.yaml   # list active scorers
dao-ai monitor scorers disable -c config/model_config.yaml   # stop all scorers
```

### `dao-ai monitor logs`

Fetch or stream runtime logs for the deployed agent, to stdout. Provide either
`-c/--config` (derives the app/endpoint name from the YAML) or `--name` (an
explicit name, used literally) — the two are mutually exclusive.

```bash
# Databricks Apps (default mode) — last 200 lines, then last 500
dao-ai monitor logs -c config/model_config.yaml
dao-ai monitor logs -c config/model_config.yaml --lines 500

# Stream continuously until Ctrl-C (apps only)
dao-ai monitor logs -c config/model_config.yaml --follow

# Model Serving snapshot (no streaming)
dao-ai monitor logs -c config/model_config.yaml -m model_serving

# Explicit name instead of a config file
dao-ai monitor logs --name my-app -p fevm
```

**Capability matrix** — the two deployment targets expose logs differently:

| Capability | `-m apps` (default) | `-m model_serving` |
|---|---|---|
| Snapshot (last N via `--lines`) | ✅ | ✅ |
| Streaming (`--follow`) | ✅ | ❌ (snapshot only — `--follow` is rejected) |
| Mechanism | `databricks apps logs` (CLI; requires the `databricks` CLI ≥ 1.3.0 on `PATH`) | Databricks SDK `serving_endpoints.logs` |

The Apps path shells out to the `databricks` CLI because the Databricks Python
SDK exposes no Apps-logs API; the CLI streams the app's `logz/stream` websocket.
`--lines 0` fetches all buffered lines (apps only).

## Interactive Chat

Start an interactive chat session with your agent:

```bash
dao-ai chat -c config/my_config.yaml

# With parameter overrides
dao-ai chat -c config/my_config.yaml --param catalog=nfleming --param module_id=09
```

## MCP Utilities

Inspect and test MCP (Model Context Protocol) servers and tools, grouped under
the `dao-ai mcp` noun. There are two distinct surfaces, told apart by their
flags:

- `-c/--config` → the MCP tools an agent **config** declares (what your agent sees)
- `--url`/`--app` → a **live** MCP server (what a running server exposes)

> To **deploy** a dao-ai agent as an MCP server, use `agent --mode mcp`.
> Deployment is intentionally not part of this noun.

| Verb | Purpose |
|---|---|
| `dao-ai mcp tools`   | List the MCP tools an agent config declares (with filter status) |
| `dao-ai mcp inspect` | Connect to a live MCP server and show its health + available tools |
| `dao-ai mcp call`    | Invoke a single tool on a live MCP server and print the result |

### `mcp tools` — inspect config-declared tools

List all MCP tools declared in a config with full descriptions and schemas:

```bash
dao-ai mcp tools -c config/my_config.yaml

# With parameter overrides
dao-ai mcp tools -c config/my_config.yaml --param catalog=main
```

Use `--apply-filters` to see only the tools that will actually be loaded
(respecting `include_tools` and `exclude_tools` configuration):

```bash
dao-ai mcp tools -c config/my_config.yaml --apply-filters
```

### `mcp inspect` — introspect a live server

Connect to a running MCP server and show its health (best-effort `/healthz`)
plus the tools it exposes. Target any MCP server with `--url`, or a Databricks
App (e.g. a dao-ai agent deployed via `agent --mode mcp`) with `--app`:

```bash
# A deployed dao-ai MCP App, resolved by name via the SDK
dao-ai mcp inspect --app my-mcp-app -p fevm

# Any MCP server URL
dao-ai mcp inspect --url https://<host>/api/2.0/mcp/sql -p fevm
```

### `mcp call` — smoke-test a single tool

Invoke one tool on a live MCP server and print its result — an end-to-end smoke
test of a deployed server. Arguments are passed as a JSON object via `--args`:

```bash
dao-ai mcp call ask --app my-mcp-app --args '{"input": "hello"}' -p fevm

dao-ai mcp call execute_sql \
  --url https://<host>/api/2.0/mcp/sql \
  --args '{"query": "SELECT 1"}' -p fevm
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
dao-ai parameters -c config/my_config.yaml         # 'list' is the default action

# Explicit (equivalent) + overrides to see how they resolve
dao-ai parameters list -c dao_ai.yaml --param module_id=09

# Print ONE parameter's resolved value as a bare line (script-friendly)
CATALOG=$(dao-ai parameters get catalog -c config/my_config.yaml)
```

The action word is optional — `dao-ai parameters -c <file>` lists. `dao-ai vars` is kept as an alias, and `--var` works alongside `--param`.

Sample output:

```
NAME       REQUIRED  PROVIDED  DEFAULT  RESOLVED  SOURCE    DESCRIPTION
----------------------------------------------------------------------
catalog    no        no        main     main      default   Unity Catalog catalog name
module_id  yes       no        -        09        --param   Workshop module identifier
genie_id   no        yes       -        -         provided  Genie space id (provisioned at run time)
```

Source values: `--param`, `env`, `default`, `inline-default`, `provided`, `MISSING`. A `provided: true` parameter (see the [configuration reference](configuration-reference.md)) reports source `provided` when unsupplied — its value is furnished at run time (e.g. by a workflow provisioning task), so it is **not** flagged `MISSING` or required.

`get <name>` prints the bare resolved value to stdout and exits non-zero (with a targeted message) when the name is undeclared, required-but-unset, or a `provided` param with no run-time value yet.

Exit code for `list` is 1 if any required parameter is `MISSING`, 0 otherwise. This makes it useful in CI pipelines to verify all overrides are wired up before deploying.

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

### Workflow Options (`dao-ai workflow up|generate|deploy|run|destroy`)

```bash
dao-ai workflow up       -c config/my_config.yaml [OPTIONS]
dao-ai workflow generate -c config/my_config.yaml [OPTIONS]
dao-ai workflow deploy   -c config/my_config.yaml [OPTIONS]
dao-ai workflow run      -c config/my_config.yaml [OPTIONS]
dao-ai workflow destroy  -c config/my_config.yaml [OPTIONS]
```

`up` generates (if needed) → deploys → runs in one command. `generate` stages
the bundle only; `deploy` pushes the already-staged bundle (unlike `agent deploy`,
`workflow deploy` does **not** auto-generate — run `generate` or `up` first);
`run`/`destroy` act on already-built artifacts. For `workflow`, `run` is
`databricks bundle run deploy_job` (the provisioning job).

| Option | Description | Verbs |
|--------|-------------|-------|
| `-c, --config FILE` | Path to the dao-ai configuration file (required) | all |
| `-s, --staging-dir DIR` | Bundle staging dir (default: `$DAO_AI_BUNDLE_DIR/workflow/<app>` or `./.dao-ai/bundle/workflow/<app>`) | all |
| `-p, --profile NAME` | Databricks CLI profile to use | all |
| `--param KEY=VALUE` / `--var KEY=VALUE` | Config parameter overrides (repeatable) | all |
| `--cloud {azure,aws,gcp}` | Cloud provider (auto-detected from the workspace URL; required only if detection fails) | all |
| `-t, --target NAME` | Bundle target name (auto-generated if not specified) | all |
| `--mode {apps,mcp}` | Serving mode selector; also `model_serving` on `up`/`deploy` | all |
| `--dry-run` | Preview commands without executing | all |
| `--overwrite` | Overwrite copied-in files in the staging dir | `up`, `generate`, `deploy` |
| `--development` / `--no-development` | Ship the local dao-ai wheel vs pin PyPI (default: auto-detect) | `up`, `generate`, `deploy` |
| `--direct` | Deploy via SDK directly, no bundle on disk (apps/mcp) | `up` |

The flat `generate-workflow` command and the one-shot `generate --deploy/--run`
flags have been removed — use `dao-ai workflow up` (or `generate` → `deploy` → `run`).

### Agent Options (`dao-ai agent up|generate|deploy|run|destroy`)

```bash
dao-ai agent up       -c config/my_config.yaml [OPTIONS]
dao-ai agent generate -c config/my_config.yaml [OPTIONS]
dao-ai agent deploy   -c config/my_config.yaml [OPTIONS]
dao-ai agent run      -c config/my_config.yaml [OPTIONS]
dao-ai agent destroy  -c config/my_config.yaml [OPTIONS]
# Use --mode mcp to build/deploy the MCP-server bundle instead of the chat-agent bundle
```

`up` generates (if needed) → deploys → runs in one command. `generate` stages
the bundle only; `deploy` pushes it (auto-generating if nothing is staged);
`run`/`destroy` act on already-built artifacts. For `agent`, `run` is
`databricks bundle run <app>`.

| Option | Description | Verbs |
|--------|-------------|-------|
| `-c, --config FILE` | Path to the dao-ai configuration file (required) | all |
| `-s, --staging-dir DIR` | Bundle staging dir (default: `$DAO_AI_BUNDLE_DIR/<kind>/<app>` or `./.dao-ai/bundle/<kind>/<app>`, where `<kind>` is `agent` or `mcp`) | all |
| `-p, --profile NAME` | Databricks profile for config loading and deploy | all |
| `--param KEY=VALUE` / `--var KEY=VALUE` | Config parameter overrides (repeatable) | all |
| `--mode {apps,mcp,model_serving}` | Serving target (default: `apps`; `ms`/`model-serving` accepted as aliases). `run`/`destroy` accept `apps`/`mcp` only. | all |
| `--dry-run` | Preview commands without executing | all |
| `--direct` | Deploy via SDK directly, no bundle on disk (apps/mcp) | `up` |
| `--overwrite` | Overwrite existing files in the output directory | `up`, `generate`, `deploy` |
| `--development` / `--no-development` | Bundle a local dao-ai wheel vs pin PyPI (default: auto-detect) | `up`, `generate`, `deploy` |

The flat `generate-agent` / `generate-mcp` commands and the one-shot
`generate --deploy/--run` flags have been removed — use `dao-ai agent up` (or
`generate` → `deploy` → `run`), with `--mode mcp` for the MCP-server bundle.

### Chat Options

```bash
dao-ai chat -c config/my_config.yaml [OPTIONS]
```

Starts an interactive REPL session where you can chat with your agent locally.

### MCP Utilities Options

```bash
dao-ai mcp tools   -c config/my_config.yaml [OPTIONS]
dao-ai mcp inspect (--url URL | --app NAME) [OPTIONS]
dao-ai mcp call    TOOL (--url URL | --app NAME) [--args JSON] [OPTIONS]
```

**`mcp tools`**

| Option | Description |
|--------|-------------|
| `-c, --config FILE` | Path to configuration file (default: `./config/model_config.yaml`) |
| `--apply-filters` | Only show tools that pass include/exclude filters (hide excluded tools) |

Lists all MCP tools declared in a config with full descriptions and readable parameter schemas. Supports filtering to show only included tools.

**`mcp inspect`** / **`mcp call`**

| Option | Description |
|--------|-------------|
| `--url URL` | Direct MCP server URL (e.g. `https://<host>/.../mcp`). Mutually exclusive with `--app`. |
| `--app NAME` | Databricks App name; its `/mcp` endpoint is resolved via the SDK. Mutually exclusive with `--url`. |
| `--args JSON` | (`call` only) JSON object of tool arguments (default: `{}`). |

`inspect` and `call` connect to a **live** MCP server and require valid auth (a `-p/--profile` or ambient credentials).

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
dao-ai workflow up -c config/hardware_store.yaml --profile aws-prod

# Deploy same app to Azure
dao-ai workflow up -c config/hardware_store.yaml --profile azure-prod

# Deploy same app to GCP
dao-ai workflow up -c config/hardware_store.yaml --profile gcp-prod
```

### Development vs Production

```bash
# Deploy to development workspace
dao-ai workflow up -c config/my_app.yaml --profile aws-dev

# Deploy to production workspace
dao-ai workflow up -c config/my_app.yaml --profile aws-prod
```

### Full Deployment Pipeline

```bash
# Validate configuration
dao-ai validate -c config/my_app.yaml

# Generate workflow diagram
dao-ai graph -c config/my_app.yaml -o workflow.png

# Provision infra, deploy, and run
dao-ai workflow up -c config/my_app.yaml --profile aws-field-eng
```

---

## Migration from pre-v2 CLI

The deploy-model v2 release removed several commands and renamed others. Use this table to update scripts and docs.

| Old command (removed) | New command |
|---|---|
| `dao-ai deploy -c ... --target model_serving` | `dao-ai agent deploy -c ... --mode model_serving` |
| `dao-ai deploy -c ... --target apps` | `dao-ai agent deploy -c ... --mode apps` |
| `dao-ai deploy -c ... --target both` | Run `dao-ai agent deploy --mode model_serving` then `dao-ai agent deploy --mode apps` |
| `dao-ai generate-agent ...` | `dao-ai agent generate ...` |
| `dao-ai generate-mcp ...` | `dao-ai agent generate --mode mcp ...` |
| `dao-ai generate-workflow ...` | `dao-ai workflow generate ...` |
| `dao-ai mcp generate\|deploy\|run\|destroy` | `dao-ai agent generate\|deploy\|run\|destroy --mode mcp` |
| `dao-ai agent generate --deploy --run ...` | `dao-ai agent up ...` (one command: generate → deploy → run) |
| `dao-ai agent generate --deploy ...` | `dao-ai agent up ...`, or `generate` then `deploy` for a staged/hand-editable bundle |
| `dao-ai agent deploy --run ...` | `dao-ai agent up ...`, or `deploy` then `run` |
| `dao-ai workflow generate --deploy --run ...` | `dao-ai workflow up ...` |
| `dao-ai create-experiment ...` | `dao-ai trace create ...` |
| `dao-ai link-trace-destination ...` | `dao-ai trace link ...` |
| `dao-ai grant-trace-permissions ...` | `dao-ai trace grant ...` |
| `dao-ai list-mcp-tools ...` | `dao-ai mcp tools ...` |
| `--deployment-target <mode>` flag | `--mode <mode>` flag |
| `--deploy` / `--run` one-shot flags on `generate` | Removed — use the `up` verb |
| `app.deployment_target:` config field | Removed — serving mode is chosen at deploy time via `--mode` (default `apps`) |
| `DeploymentTarget` enum (Python API) | Renamed `ServingMode` — `from dao_ai.config import ServingMode`; `ServingMode.APPS` / `.MCP` / `.MODEL_SERVING` |
| `DeploymentTarget.BOTH` | Removed — deploy each mode separately |

---

## Navigation

- [← Previous: Examples](examples.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: Python API →](python-api.md)

