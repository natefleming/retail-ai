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
subcommands and `dao-ai agent sync --<TAB>` completes flags.

## Global Options

`-p/--profile` and `-v/--verbose` are accepted at any level (before or after the
subcommand). When `--profile` is set, dao-ai **clears the ambient `DATABRICKS_*`
environment variables** (`DATABRICKS_TOKEN`, `DATABRICKS_HOST`,
`DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`, `DATABRICKS_AUTH_TYPE`) for
the current process so the profile is authoritative. This prevents a stray token
or host in your shell or a `.env` file from silently overriding the profile and
targeting the wrong workspace. If you rely on env-var auth, omit `--profile`.

## Config Sources: local, URL, or git

Every command that takes `-c/--config` accepts a local path, an `http(s)` URL, or
a **git locator** — so a project that isn't on your machine runs like one that is.

```bash
# Local path
dao-ai validate -c config/my_config.yaml

# Git locator: repo, optional @ref, optional #path-in-repo
dao-ai validate -c 'git+https://github.com/org/repo@v1.0#examples/retail/agent.yaml'

# gh: shorthand
dao-ai agent up -c 'gh:org/repo@main#examples/retail/agent.yaml' -p my-profile

# Split spelling — handy when a repo ships several config variants
dao-ai agent up --from 'gh:org/repo@v1.0' -c examples/retail/agent.yaml -p my-profile

# SSH remote (auth via ssh-agent)
dao-ai validate -c 'git+ssh://git@github.com/org/private@v1#agent.yaml'
```

**Quote the locator.** `#` starts a comment in every common shell, so an unquoted
locator loses its in-repo path.

Unlike a URL — which fetches a single YAML and therefore *rejects* a config
declaring relative `ddl` / `data` / `code_paths` — a git locator brings the whole
project tree, so colocated assets, `src/`, `skills/`, and `resources/` all resolve
exactly as they do locally.

`@ref` may be a branch, tag, or full 40-character commit SHA; omit it for the
remote's default branch. `#path` may name a file or a directory; omit it (or point
at a directory) and dao-ai discovers the config, preferring `dao-ai.yaml` and
erroring with the candidates listed if the choice is ambiguous.

| Flag | Purpose |
|---|---|
| `--from REPO` | Repository to load from; `-c` is then a repo-relative path |
| `--refresh` | Re-fetch even if the ref is already cached |

**Trust.** A git locator runs the repository's code — a config can ship Python via
`code_paths` / `src/` and inline tool code — exactly as `git clone` followed by
`dao-ai agent up` would. The resolved commit SHA is reported on every load. Pin a
tag or SHA for repositories you do not control.

**Caching.** Checkouts are keyed by commit under `$DAO_AI_GIT_CACHE`, else
`$XDG_CACHE_HOME/dao-ai/git`, else `~/.cache/dao-ai/git`. A full SHA is immutable
and never re-fetched. A branch or tag is re-resolved via `git ls-remote` on each
run and re-fetched only when it moved, so `up` on a branch always deploys its
current HEAD; if the remote is unreachable, the newest cached checkout is used with
a warning.

```bash
dao-ai cache dir                      # where checkouts live, and how much space
dao-ai cache clear                    # remove all of them
dao-ai cache clear --repo gh:org/repo # remove just one repository's
```

**Private repositories.** Auth is delegated to `git`, so ssh-agent and credential
helpers work with no dao-ai configuration. For headless use (a notebook, CI) set
`DAO_AI_GIT_TOKEN` or `GITHUB_TOKEN`; it is handed to git through a credential
helper and never written to disk, never placed in a remote URL, and never in a
command line.

Requires `git` on `PATH`. Resolution is client-side only — the generated bundle is
self-contained, so nothing needs `git` at deploy or run time.

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

Deploying an agent follows one lifecycle — **`build → sync → start`** — whether you
run it as three explicit steps or let a single command do all three. Start with the
one-command path and reach for the granular verbs only when you need them.

All deploy paths call `AppConfig.create_agent()` + `deploy_agent()` in-process: for
Model Serving it registers the MLflow model and creates the serving endpoint
(`agents.deploy`); for Apps it uploads the config + source and drives the Apps REST
API. Every path auto-links the UC trace destination and auto-grants the runtime
service principal the trace-write permissions (gated on `app.manage_permissions`).

### Start here: `up` — build, sync, and start in one command

`dao-ai agent up` is the fast path to a live agent. It **builds** the bundle (if
nothing is staged), **syncs** it to the workspace, links the trace destination, then
**starts** it — the whole `build → sync → start` lifecycle in one idempotent command.
This is what you want most of the time.

```bash
# Bring up a Databricks App (default mode) — build → sync → start
dao-ai agent up -c config/my_config.yaml --profile fevm

# Bring up the MCP-server App
dao-ai agent up -c config/my_config.yaml --mode mcp --profile fevm

# Bring up on Model Serving (builds a thin deploy-agent Job, runs it to
# register the model + create the endpoint)
dao-ai agent up -c config/my_config.yaml --mode model_serving --profile fevm
```

`up` is safe to re-run: an unchanged config **skips the build** (config
checksum) and the sync is **convergent**, so re-running never duplicates the
bundle. The `start` step always executes — an app restarts, a model_serving job
re-runs and registers a new model version — which is `start` doing its job.

#### The `--direct` option — skip the bundle on disk

Add `--direct` to `up` to go straight through the SDK **without writing a bundle to
disk**. There is no staged artifact to inspect or hand-edit — dao-ai calls
`create_agent`/`deploy_agent` directly. It works for **all three modes** (`apps`,
`mcp`, `model_serving`) and inherently syncs and starts. Use it for fast iteration
when you don't need an auditable bundle artifact. `--direct` is an **`up`-only** flag
(it has no meaning on `build`/`sync`/`start`, which are defined by the bundle they act
on).

```bash
# Bring up as an App via the SDK directly — no bundle on disk (fast iteration)
dao-ai agent up -c config/my_config.yaml --mode apps --direct --profile fevm
```

### The granular lifecycle: `build → sync → start`

When you want to inspect or hand-edit the bundle before it ships — or run the
CI-style **build once, sync once, start N times** flow — drive the three steps
yourself, in order:

```bash
# 1. build — stage the bundle to disk (inspect / hand-edit before shipping)
dao-ai agent build -c config/my_config.yaml --profile fevm

# 2. sync — push the staged bundle to the workspace (does NOT start it)
dao-ai agent sync -c config/my_config.yaml --profile fevm

# 3. start — make the synced bundle live (no re-sync; starts/restarts the app)
dao-ai agent start -c config/my_config.yaml --profile fevm
```

- **`build`** stages the bundle and does nothing else. `sync`/`start`/`down`
  require it first (they never build) — or use `up`, the one command that builds
  for you.
- **`sync`** pushes to the workspace but **does not start** the app (it runs
  `databricks bundle deploy`) and **does not build** — it errors if nothing is
  staged. A `sync` that failed on a transient error is safe to retry on its own —
  no rebuild.
- **`start`** makes the synced bundle live (`databricks bundle run <app>`), and
  **does not re-sync or rebuild** — it errors if nothing is synced. Re-run it any
  time to restart an app or re-execute a model_serving/workflow job.

> If `app.trace_location` is set, run `dao-ai trace link` **between `sync` and
> `start`** — otherwise traces silently drop (`TABLE_DOES_NOT_EXIST`) on re-deploys.
> See [Linking the UC trace destination](#linking-the-uc-trace-destination-run-dao-ai-trace-link-between-deploy-and-run).
> The one-command `up` path does this linking for you.

To ship the **local dao-ai wheel** instead of the published PyPI package, add
`--development` on `build` (or on `up`, which builds):

```bash
dao-ai agent build -c config/my_config.yaml --development --profile fevm
dao-ai agent sync  -c config/my_config.yaml --profile fevm
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

- **`dao-ai agent up`** — the one-command path (`build → sync → start`). Reach for
  this first for any mode: `apps` (default), `mcp`, or `model_serving`.
- **`dao-ai agent up --direct`** — same, but SDK-direct with no bundle on disk. Best
  for fast iteration when you don't need an auditable bundle artifact.
- **`dao-ai agent build → sync → start`** — the granular flow. Use it to inspect or
  hand-edit the staged bundle before shipping, or for the CI pattern *build once,
  sync once, start N times*.
- **`dao-ai workflow`** — provision the full backing infra (schemas,
  Vector Search, Lakebase, Genie, UC functions) *and* deploy the agent, as a
  multi-task Databricks Job. The job's deploy step runs the same
  `create_agent`/`deploy_agent` code as the direct deploy paths.

## Bundle Generators: `agent`, `workflow`

The bundle generators are **verb-under-noun** commands — pick a noun for
what you're shipping, then a verb for the lifecycle step:

| Noun | What it ships |
|------|---------------|
| `dao-ai agent` | A Databricks App running the agent graph (default: `--mode apps`). Use `--mode mcp` to emit the MCP-server App instead, or `--mode model_serving` on `sync` to go SDK-direct. |
| `dao-ai workflow` | A multi-task Databricks Job that provisions the backing infra (schemas, Vector Search, Lakebase, Genie, UC functions) *and* deploys the agent. |

Each noun takes the same five verbs:

```bash
dao-ai agent    up|build|sync|start|down  -c <cfg> [-p <profile>]
dao-ai workflow up|build|sync|start|down  -c <cfg> [-p <profile>]
```

**The mental model — plain-language lifecycle:**

| Verb | What it does | Databricks CLI underneath |
|---|---|---|
| **`build`** | build the bundle artifact (stage to disk) | — (writes files) |
| **`sync`** | push the bundle to the workspace — does *not* start it | `databricks bundle deploy` |
| **`start`** | make it live — *no re-sync* | `databricks bundle run <resource>` |
| **`up`** | all-in-one: build-if-needed → sync → start (idempotent) | — (orchestrates the three) |
| **`down`** | tear the deployment down (never your data) | `databricks bundle destroy` (+ delete the serving endpoint for model_serving) |

> The verbs are dao-ai's plain-language names for the DAB lifecycle; the
> "underneath" column is the `databricks bundle` subcommand each one runs.
> Note `sync` runs `bundle deploy` — the word "deploy" is the *Databricks CLI*
> subcommand, not the dao-ai verb; in dao-ai, syncing does **not** start the app.

**The one-command path — `up`:**

- **`up`** is the fast path to a live agent: it builds the bundle (if nothing is
  staged), syncs it, links the trace destination, then starts it — equivalent to
  `build → sync → start` in one command. This is what you want most of the time:
  `dao-ai agent up -c <cfg> -p fevm`. For `--mode model_serving` it builds a thin
  deploy-agent Job bundle, syncs it, then starts it (`bundle run deploy_job`) —
  which registers the model and deploys the endpoint (the endpoint serves once
  `READY`). Add `--direct` (apps/mcp/model_serving) to go via the SDK with no
  bundle written to disk.
- **`up` is safe to re-run.** On the artifact-and-sync axes it is idempotent: an
  unchanged config **skips the build** (config checksum — see *The staging dir is
  ephemeral build output* below) and the sync is **convergent**, so re-running
  `up` never duplicates the bundle. The `start` step, by contrast, always
  *executes*: an app restarts, and
  a workflow/model_serving job re-runs (a model_serving `start` registers a new
  model version each time). That is the start step doing its job, not an
  artifact/sync concern.

**The granular lifecycle — `build → sync → start → down`:**

- **`build`** stages a bundle to disk (`<base>/<noun>/<app>`, where `<base>` is
  `$DAO_AI_BUNDLE_DIR` or `./.dao-ai/bundle`, or `-s <dir>`) and does nothing
  else — inspect or hand-edit the staged files before shipping.
- **`sync`** pushes the **already-built** bundle to the workspace but **does not
  start it**, and **does not build** — run `build` (or `up`) first; it errors
  with the exact next command if nothing is staged. This is uniform across every
  noun and mode (`agent`/`workflow` × `apps`/`mcp`/`model_serving`) — a primitive
  acts on prepared state, it never provisions its own prerequisites. For
  `agent`/`mcp`, `sync` runs `databricks bundle deploy` — it creates/updates the
  App resource and uploads its source; a staged bundle is synced in place, and
  on config drift it warns and deploys as-is rather than rebuilding (run `build`
  or `up` to pick up the change). Use `--mode model_serving` on `agent sync` to sync the
  deploy-agent Job bundle.
- **`start`** makes the synced bundle live and **does not re-sync, build, or
  push** — it errors if nothing is synced. For `agent`/`mcp`, `start` runs
  `databricks bundle run <app>` (starts/restarts the app — a DABs App is not
  serving until `bundle run`); for `workflow` and `model_serving`, `start` runs
  `databricks bundle run deploy_job` (executes the job — for model_serving that
  registers the model and deploys the endpoint). `start` is the verb for the
  manual/CI flow — **build once, sync once, start N times** — and for restarting
  an app or re-executing a job without re-syncing.
- **`down`** tears the deployment down — **it removes the deployment, never your
  data.** For `agent`/`mcp` it runs `databricks bundle destroy`, deleting the App.
  For `agent --mode model_serving` it runs `bundle destroy` (removing the
  deploy_job) **and** deletes the serving endpoint — the endpoint is created by
  the deploy-agent job, not the DAB, so `bundle destroy` alone would leave it
  running and billing; the registered UC model + versions are **kept** (a
  reusable artifact). For `workflow`, `down` removes the provisioning job **and**
  the agent it deployed — the App (apps/mcp) or serving endpoint (model_serving)
  the `deploy_agent` step created imperatively, which `bundle destroy` alone
  would orphan. It does **not** delete the *data* infrastructure that job
  provisioned (Vector Search indexes, Lakebase, Genie spaces, UC
  schemas/functions); tear those down yourself if you no longer need them.

This is the payoff: a sync that failed on a transient error can be retried with
just `dao-ai agent sync -c <cfg> -p fevm` — no rebuild. To build, sync, *and*
start in one shot, use `dao-ai agent up` instead.

**The staging dir is ephemeral build output.** Everything in the **default**
staging dir (`<base>/<noun>/<app>`) is either generated from your config or
copied from the config directory (custom `code_paths`, `src/` packages,
`resources/` overlays, the rendered config), so `build`/`up` regenerate it in
place — a default dir is wiped and rebuilt on each run, and on config drift `up`
rebuilds it automatically. **Don't hand-edit generated files** — to add your own
Databricks resources (Jobs, Pipelines, …) to the bundle, drop a `*.yml` in a
`resources/` directory next to your config (auto-shipped, like `src/` for code),
or list files explicitly via `app.resource_paths: [path/to/jobs.yml, …]`; each is
copied into the bundle's `resources/` directory where DABs'
`include: [resources/*.yml]` merges it at deploy, with no generated file touched.
Your own `src/`/`code_paths` are always preserved (copied once, never
overwritten). A `-s <dir>` you supply is treated as your territory: it is never
auto-wiped, and its files follow the writer's per-file overwrite rules. The
source-selection flags `--overwrite`, `--development`, and `--no-development`
take effect on the verbs that build — `up` and `build` (`--overwrite` also
re-copies the user-owned artifacts for a full clean slate). `sync`/`start`/`down`
act on already-built artifacts and never build, so these flags don't apply
there.

> **Migration:** the flat commands `generate-agent`, `generate-mcp`, and
> `generate-workflow` have been removed. Use `dao-ai agent build`,
> `dao-ai agent build --mode mcp`, and `dao-ai workflow build` instead.
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
dao-ai workflow sync -c config/my_config.yaml
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
dao-ai workflow build -c config/my_config.yaml --profile aws-field-eng

# Sync the staged bundle to the workspace (run `build` or `up` first — this does not auto-build)
dao-ai workflow sync -c config/my_config.yaml --profile aws-field-eng

# Start the deploy_job on an already-synced bundle (databricks bundle run deploy_job)
dao-ai workflow start -c config/my_config.yaml --profile aws-field-eng
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

Generate a complete, deployable Databricks Apps bundle directory from a dao-ai config file. This is distinct from the `bundle` command -- while `bundle` wraps `databricks bundle deploy/run/destroy`, `dao-ai agent build` **creates** the bundle project itself.

When the source config uses `${param.NAME}` / `${var.NAME}` parameters or `${workspace.*}` references, the generated bundle writes the **resolved** config (all references substituted to literal values, `parameters:` block dropped) so the deployed app does not need the original `--param` flags or a runtime workspace lookup.

### Basic Usage

```bash
dao-ai agent build -c config/retail.yaml -s ./my-bundle

# With parameter overrides baked into the generated bundle
dao-ai agent build -c config/retail.yaml -s ./my-bundle --param catalog=prod_catalog

# Generate, deploy, and start the app in one command
dao-ai agent up -c config/retail.yaml -p fevm

# Ship the already-staged bundle without regenerating (e.g. after hand-editing, or retrying a transient deploy failure)
dao-ai agent sync -c config/retail.yaml -p fevm

# Deploy the staged bundle, then start it
dao-ai agent sync -c config/retail.yaml -p fevm
dao-ai agent start    -c config/retail.yaml -p fevm
```

MCP server bundles use `dao-ai agent build --mode mcp` (not a separate noun). Use `dao-ai agent start` to `databricks bundle run <app>` an already-deployed bundle, and `dao-ai agent down` to tear it down. See [Bundle Generators](#bundle-generators-agent-workflow) for the full lifecycle.

> **One config serves one mode at a time.** `apps` and `mcp` from the same config deploy to the **same** Databricks App resource (named from `app.name`), differing only in the runtime command — so `agent up --mode mcp` after `agent up --mode apps` **replaces** the chat App with the MCP server (it does not create a second app). The staging dirs are separate (`agent/<app>/apps` vs `.../mcp`) so you can build/inspect both, but a workspace has one app per `app.name`. To run a chat App **and** an MCP server from the same agent simultaneously, give them distinct `app.name` values (two configs). This mirrors `model_serving`, which keys its endpoint off `app.name`/`endpoint_name`.

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
| `resources/app.yml` | The App + experiment resource block (owned by dao-ai) |
| `resources/<your>.yml` | Your own resource overlays (see below) |

### Extending the bundle with your own resources

The staging dir is regenerated on every `build`, so **don't hand-edit generated
files**. To add your own Databricks Asset Bundle resources (Jobs, Pipelines, …)
alongside the generated App, you have two options (they compose):

**Convention** — drop `*.yml` files in a `resources/` directory next to your
config. They're auto-shipped with no declaration, exactly like `src/` packages
are for code:

```
my-app/
  dao_ai.yaml
  resources/
    nightly_job.yml     # auto-discovered and staged
```

**Explicit** — list files anywhere (relative to the config dir) via
`app.resource_paths`:

```yaml
app:
  name: my_app
  resource_paths:
    - overlays/nightly_job.yml   # relative to your config file's directory
```

Either way, each file is copied into the bundle's `resources/` directory, where
the generated `databricks.yaml`'s `include: [resources/*.yml]` merges it at
deploy — so your resources ship without touching a single generated file.
Overlays are user-owned: copied once and never overwritten by a rebuild (pass
`--overwrite` to re-copy). File basenames must be unique and may not be `app.yml`
(reserved for the generated App block). This works identically on the `agent`,
`mcp`, and `workflow` nouns.

### Dependency install: `pyproject.toml` + portable `uv.lock`

`dao-ai agent build` writes a `pyproject.toml` and a portable `uv.lock` to the bundle (no `requirements.txt` — its presence would take precedence and force the pip path). The Databricks Apps build phase runs `uv sync --locked --no-dev` from them. Published mode (`--no-development`) pins `dao-ai[<extras>]==<version>` for reproducible redeploys; `--development` redirects dao-ai to the bundled local wheel via `[tool.uv.sources]`. `uv lock` records the full closure, and any internal-mirror host (`pypi-proxy.dev.databricks.com`) is rewritten to the public CDN so the lock resolves from Apps containers.

> **Pre-publish note:** published-mode lock generation resolves `dao-ai==<version>` from PyPI, so it fails with an actionable error until that version is published (release-time / CI). For local/pre-release iteration, generate with `--development` (locks against the bundled wheel — works anytime).

### Upgrading dao-ai in an existing bundle

When you `pip`/`uv` upgrade dao-ai and want an already-built bundle on the new
version, the right move differs by surface — because the artifacts differ:

- **Apps / MCP bundles** — the deployed app installs dao-ai as a *dependency*
  (`pyproject.toml` + `uv.lock`), and new runtime behavior ships inside that
  wheel. The low-risk default is a **version bump, not a regenerate**: update the
  pin and re-lock, then redeploy.

  ```bash
  # in the staged bundle dir (or edit pyproject.toml's dao-ai==<ver> then):
  uv lock --upgrade-package dao-ai
  dao-ai agent sync -c <config> -p <profile>     # redeploy the same bundle
  ```

  Only run a full `dao-ai agent build --overwrite` when you want to adopt a new
  bundle **shape** (a dao-ai release that changed the generated `databricks.yaml`
  / `resources/` layout). Because the staging dir is ephemeral and your resources
  live in the config's `resources/` dir (or `app.resource_paths`), a rebuild is
  safe — nothing you authored is lost.

- **Workflow bundles** — the provisioning notebooks (`01`–`08`) ship *inside the
  dao-ai wheel* and are materialized into the bundle at build time, so a stale
  bundle would run *old* notebooks against a *new* runtime. **Always regenerate**
  after an upgrade:

  ```bash
  dao-ai workflow build -c <config>              # re-materializes 01–08 from the new wheel
  dao-ai workflow up   -c <config> -p <profile>  # or build + deploy + run in one step
  ```

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

When `trace_location` is set, `agent build` wires up the SQL warehouse as an App resource (CAN_USE for the App SP) and adds `MLFLOW_TRACING_SQL_WAREHOUSE_ID` to the App's `env`. The OTEL trace tables themselves are auto-created by MLflow at first trace write — dao-ai does not emit per-table grants because the tables don't exist at deploy time. After deploy, grant the App SP schema-level privileges (one-time):

```bash
SP=$(databricks apps get <app-name> -p <profile> --output json | jq -r .service_principal_client_id)
databricks grants update catalog <catalog> -p <profile> \
  --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"USE_CATALOG\"]}]}"
databricks grants update schema <catalog>.<schema> -p <profile> \
  --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"USE_SCHEMA\",\"CREATE_TABLE\",\"MODIFY\",\"SELECT\"]}]}"
```

When `trace_location` is unset, `agent build` emits a `⚠` warning to alert you. Local notebook/CLI runs and Model Serving deploys are unaffected.

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

The verb is idempotent — safe on every deploy — but load-bearing on re-deploys and after `trace_location` changes. `agent build` prints a one-line reminder in its "Next steps" when `trace_location` is configured.

See [Trace Commands](#trace-commands) for full flag reference and the migration playbook for moving traces between destinations.

#### Runtime trace-destination sync (`apply_runtime_trace_destination`)

`dao-ai trace link` writes the trace-destination tag on the experiment record so that future traces route to the configured UC schema. That works when MLflow's runtime picks up the linkage from the experiment — but if the app also has `MLFLOW_TRACING_DESTINATION` env set (dao-ai's `agent build` sets it as `catalog.schema` for warehouse routing), MLflow parses that env value as the deprecated `UCSchemaLocation` and populates the `_MLFLOW_TRACE_USER_DESTINATION` ContextVar accordingly. The ContextVar SHADOWS MLflow's auto-resolver from experiment tags, so the exporter targets `mlflow_experiment_trace_otel_spans` (the un-prefixed default) which doesn't exist on the prefixed schema — and every span export fails with `TABLE_DOES_NOT_EXIST`.

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

### Regenerating and Overwriting

The **default** staging dir (no `-s`) is ephemeral build output: `build`/`up`
regenerate it in place on every run (a default dir is wiped and rebuilt), so
there is nothing to hand-edit and nothing to lose. Add your own bundle resources
via a colocated `resources/` dir or `app.resource_paths` (see [Extending the bundle](#extending-the-bundle-with-your-own-resources)) rather than editing generated files.

For a **user-supplied** `-s <dir>`, existing generated files are skipped by
default; use `--overwrite` to rewrite them (and re-copy user-owned artifacts).
A `-s <dir>` is never auto-wiped. `--overwrite` is only valid on `build`/`up`.

```bash
dao-ai agent build -c config/retail.yaml -s ./my-bundle --overwrite
```

### Using a Databricks Profile

If your config references workspace resources (Genie rooms, warehouses, etc.), specify a profile so they can be resolved during generation:

```bash
dao-ai agent build -c config/retail.yaml -s ./my-bundle --profile my-workspace
```

### Development Mode

Use `--development` to bundle a local build of dao-ai instead of pulling from PyPI. This is useful when testing unreleased dao-ai changes in a deployed app.

```bash
dao-ai agent build -c config/retail.yaml -s ./my-bundle --development
```

Development mode changes the generated bundle in several ways:

- **Local wheel**: Copies the dao-ai wheel from `dist/` into the bundle. If no wheel exists, one is built automatically via `uv build --wheel`.
- **Path dependency**: The generated `pyproject.toml` uses a `[tool.uv.sources]` path dependency pointing at the local wheel instead of pinning a PyPI version.
- **No artifacts block**: The `databricks.yaml` omits the `artifacts` section so the wheel uploads as a regular source file rather than being intercepted by the artifact system.
- **Adjusted .gitignore**: The `dist/` directory is not ignored, since the wheel must be included in the bundle.

### Next Steps

After building the bundle, the command prints the next steps. You can either drive Databricks directly, or use the `sync`/`start` verbs (which act on the staged dir without rebuilding):

```bash
# Option A — dao-ai verbs (deploy then start; or use `agent up` to do both at once)
dao-ai agent sync -c config/retail.yaml -p <profile>
dao-ai agent start    -c config/retail.yaml -p <profile>

# Option B — drive databricks bundle directly
cd ./my-bundle
uv sync
databricks bundle deploy --target dev
databricks bundle run <app-name> --target dev
```

## Trace Commands

The `dao-ai trace` group manages MLflow experiments and UC trace destinations.

### `dao-ai trace link`

`dao-ai trace link` attaches an MLflow experiment to its Unity Catalog trace destination declared under `app.trace_location`. Run it as an explicit step **between** `databricks bundle deploy` and `databricks bundle run` — see the [background above](#linking-the-uc-trace-destination-run-dao-ai-trace-link-between-deploy-and-run) for why the app's runtime attempt is unreliable.

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
dao-ai agent build -c my_config.yaml -s ./bundle --overwrite
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

### Workflow Options (`dao-ai workflow up|build|sync|start|down`)

```bash
dao-ai workflow up    -c config/my_config.yaml [OPTIONS]
dao-ai workflow build -c config/my_config.yaml [OPTIONS]
dao-ai workflow sync  -c config/my_config.yaml [OPTIONS]
dao-ai workflow start -c config/my_config.yaml [OPTIONS]
dao-ai workflow down  -c config/my_config.yaml [OPTIONS]
```

`up` builds (if needed) → syncs → starts in one command. `build` stages the
bundle only; `sync`/`start`/`down` act on the already-built bundle and never
build — run `build` (or `up`) first, else they error with the next command. For
`workflow`, `start` is `databricks bundle run deploy_job` (the provisioning job).

| Option | Description | Verbs |
|--------|-------------|-------|
| `-c, --config FILE` | Path to the dao-ai configuration file (required) | all |
| `-s, --staging-dir DIR` | Bundle staging dir (default: `$DAO_AI_BUNDLE_DIR/workflow/<app>` or `./.dao-ai/bundle/workflow/<app>`) | all |
| `-p, --profile NAME` | Databricks CLI profile to use | all |
| `--param KEY=VALUE` / `--var KEY=VALUE` | Config parameter overrides (repeatable) | all |
| `--cloud {azure,aws,gcp}` | Cloud provider (auto-detected from the workspace URL; required only if detection fails) | all |
| `-t, --target NAME` | Bundle target name (auto-generated if not specified) | all |
| `--mode {apps,mcp,model_serving}` | Serving mode selector (default: `apps`; `ms`/`model-serving` accepted as aliases). Forwarded to the deploy-agent job step as a runtime var. | all |
| `--dry-run` | Preview commands without executing | all |
| `--overwrite` | Overwrite copied-in files in the staging dir | `up`, `build` |
| `--development` / `--no-development` | Ship the local dao-ai wheel vs pin PyPI (default: auto-detect) | `up`, `build` |
| `--direct` | Go via SDK directly, no bundle on disk (apps/mcp) | `up` |

The flat `generate-workflow` command and the one-shot `generate --deploy/--run`
flags have been removed — use `dao-ai workflow up` (or `build` → `sync` → `start`).

### Agent Options (`dao-ai agent up|build|sync|start|down`)

```bash
dao-ai agent up    -c config/my_config.yaml [OPTIONS]
dao-ai agent build -c config/my_config.yaml [OPTIONS]
dao-ai agent sync  -c config/my_config.yaml [OPTIONS]
dao-ai agent start -c config/my_config.yaml [OPTIONS]
dao-ai agent down  -c config/my_config.yaml [OPTIONS]
# Use --mode mcp to build the MCP-server bundle instead of the chat-agent bundle
```

`up` builds (if needed) → syncs → starts in one command. `build` stages the
bundle only; `sync`/`start`/`down` act on the already-built bundle and never
build — run `build` (or `up`) first, else they error with the next command. For
`agent`, `start` is `databricks bundle run <app>`.

| Option | Description | Verbs |
|--------|-------------|-------|
| `-c, --config FILE` | Path to the dao-ai configuration file (required) | all |
| `-s, --staging-dir DIR` | Bundle staging dir (default: `$DAO_AI_BUNDLE_DIR/<kind>/<app>` or `./.dao-ai/bundle/<kind>/<app>`, where `<kind>` is `agent` or `mcp`) | all |
| `-p, --profile NAME` | Databricks profile for config loading and sync | all |
| `--param KEY=VALUE` / `--var KEY=VALUE` | Config parameter overrides (repeatable) | all |
| `--mode {apps,mcp,model_serving}` | Serving target (default: `apps`; `ms`/`model-serving` accepted as aliases) | all |
| `--dry-run` | Preview commands without executing | all |
| `--direct` | Go via SDK directly, no bundle on disk (all modes) | `up` |
| `--overwrite` | Overwrite existing files in the output directory | `up`, `build` |
| `--development` / `--no-development` | Bundle a local dao-ai wheel vs pin PyPI (default: auto-detect) | `up`, `build` |

The flat `generate-agent` / `generate-mcp` commands and the one-shot
`generate --deploy/--run` flags have been removed — use `dao-ai agent up` (or
`build` → `sync` → `start`), with `--mode mcp` for the MCP-server bundle.

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
| `dao-ai agent generate ...` | `dao-ai agent build ...` |
| `dao-ai agent deploy ...` | `dao-ai agent sync ...` |
| `dao-ai agent run ...` | `dao-ai agent start ...` |
| `dao-ai agent destroy ...` | `dao-ai agent down ...` |
| `dao-ai workflow generate\|deploy\|run\|destroy ...` | `dao-ai workflow build\|sync\|start\|down ...` |
| `dao-ai deploy -c ... --target model_serving` | `dao-ai agent sync -c ... --mode model_serving` |
| `dao-ai deploy -c ... --target apps` | `dao-ai agent sync -c ... --mode apps` |
| `dao-ai deploy -c ... --target both` | Run `dao-ai agent sync --mode model_serving` then `dao-ai agent sync --mode apps` |
| `dao-ai generate-agent ...` | `dao-ai agent build ...` |
| `dao-ai generate-mcp ...` | `dao-ai agent build --mode mcp ...` |
| `dao-ai generate-workflow ...` | `dao-ai workflow build ...` |
| `dao-ai mcp generate\|deploy\|run\|destroy` | `dao-ai agent build\|sync\|start\|down --mode mcp` |
| `dao-ai agent generate --deploy --run ...` | `dao-ai agent up ...` (one command: build → sync → start) |
| `dao-ai agent generate --deploy ...` | `dao-ai agent up ...`, or `build` then `sync` for a staged/hand-editable bundle |
| `dao-ai agent deploy --run ...` | `dao-ai agent up ...`, or `sync` then `start` |
| `dao-ai workflow generate --deploy --run ...` | `dao-ai workflow up ...` |
| `dao-ai create-experiment ...` | `dao-ai trace create ...` |
| `dao-ai link-trace-destination ...` | `dao-ai trace link ...` |
| `dao-ai grant-trace-permissions ...` | `dao-ai trace grant ...` |
| `dao-ai list-mcp-tools ...` | `dao-ai mcp tools ...` |
| `--deployment-target <mode>` flag | `--mode <mode>` flag |
| `--deploy` / `--run` one-shot flags on `generate` | Removed — use the `up` verb |
| `app.deployment_target:` config field | Removed — serving mode is chosen at sync time via `--mode` (default `apps`) |
| `DeploymentTarget` enum (Python API) | Renamed `ServingMode` — `from dao_ai.config import ServingMode`; `ServingMode.APPS` / `.MCP` / `.MODEL_SERVING` |
| `DeploymentTarget.BOTH` | Removed — sync each mode separately |

---

## Navigation

- [← Previous: Examples](examples.md)
- [↑ Back to Documentation Index](https://github.com/natefleming/dao-ai/blob/main/README.md#-documentation)
- [Next: Python API →](python-api.md)

