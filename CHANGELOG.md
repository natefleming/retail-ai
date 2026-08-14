# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING — `--mode mcp` is replaced by `--as-mcp`, a protocol modifier on `--mode apps`.** MCP is a *wire protocol*, not a hosting platform, so it no longer sits on the `--mode` axis. `ServingMode` mixed the two vocabulary levels: `model_serving` and `apps` name Databricks platforms while `mcp` named a protocol served **by** the Apps platform — and the code showed it, since `deploy_mcp_agent` and `deploy_apps_agent` both called the same `_deploy_app()` and the MCP bundle writer delegates to the Apps bundle helpers. `--mode` is now strictly the platform axis (`model_serving | apps`), keeping it distinct from `--target` (the DAB *environment*) and from `--as-mcp` (the protocol). Migration: `dao-ai agent <verb> --mode mcp` → `dao-ai agent <verb> --as-mcp` (the old spelling is an argparse "invalid choice"); `ServingMode.MCP` is removed → `config.deploy_agent(mode=ServingMode.APPS, as_mcp=True)`. `--as-mcp` is accepted on every verb of **both** the `agent` and `workflow` nouns, plus `monitor logs` and `trace link`/`grant`. The `mcp` utilities noun (`tools`/`inspect`/`call`) is unaffected.
- **BREAKING — MCP servers now deploy as `mcp-<app>` instead of `<app>`.** Previously `--mode apps` and `--mode mcp` derived the *same* Databricks App name, so deploying one silently **replaced** the other (their staging dirs were separated, but the App resource was not); the docs worked around this by telling you to hand-name your app `mcp-*`. dao-ai now applies the `mcp-` prefix automatically to the App name, the DAB bundle name, and the default MLflow experiment, so a chat App and an MCP server built from one config coexist with separate service principals and experiments. The prefix is idempotent — an `app.name` already starting with `mcp-` is not doubled. The prefix is also what Multi-Agent Supervisor pattern-matches when auto-discovering MCP-hosted Apps. The exposed MCP *tool* name still comes from the unprefixed `app.name`. Existing MCP deployments are re-created under the new name; remove the old App if you don't want it lingering.
- **Agent staging dirs now mirror the platform/protocol axes.** `agent/<app>/{apps,mcp,ms}` becomes `agent/<app>/apps/chat`, `agent/<app>/apps/mcp`, and `agent/<app>/model_serving` — the protocol nests *under* the platform instead of sitting beside it, and `model_serving` is spelled out rather than abbreviated `ms`. `workflow/<app>` is unchanged (its bundle is mode-agnostic). Isolation guarantees are unchanged. Previously staged dirs are orphaned and simply rebuild on the next `build`/`up`; delete `.dao-ai/bundle` to reclaim the space.
- **`AppModel.app_resource_name` is joined by `dao_ai.config.app_name_for(name, as_mcp=...)`** as the single source of truth for the deployed App name, replacing three inline copies of the lowercase/hyphenate normalization. The property remains for the chat-protocol name.

- **BREAKING — Renamed `AppConfig.deploy_agent(target=...)` → `deploy_agent(mode=...)`** (and the same on `ServiceProvider`/`DatabricksProvider.deploy_agent`). The parameter is a `ServingMode` (`MODEL_SERVING`/`APPS`) and the CLI already exposes it as `--mode`; the method parameter now matches the type and the flag. Update any programmatic callers using the `target=` keyword: `config.deploy_agent(mode=ServingMode.APPS)`. Unrelated `target` names (DAB bundle targets, Vector Search `target_qps`, handoff `targets`) are unchanged.
- **BREAKING — Removed `PromptModel.schema` and `PromptModel.tags`** (and the derived `full_name` property). These were vestiges of the removed MLflow Prompt Registry — a UC-qualified name and metadata tags only mattered for a *registered* prompt, and inline prompts never register. `PromptModel` now carries only `name`, `description`, and `template` (plus the `jinja_template` property). Remove `schema:`/`tags:` keys from any `prompts:` block in your config; `make schema` regenerates the JSON schema accordingly.

### Fixed

- **The caller's OBO bearer token is no longer echoed back in responses or error text (security).** `dao_ai.apps.handlers._inject_headers_into_request` injects the *entire* inbound header map onto the runtime `Context` so tools can act as the calling user — including the live bearer in `x-forwarded-access-token`. Two outbound paths then serialized that context back toward the caller: `custom_outputs.configurable.headers` on every successful `predict`/`predict_stream`, and — worse — the "copy-paste this config" JSON block inside the user-facing `ValueError` raised by the `user_id`/`thread_id`/custom-field validation middleware, which lands in assistant message text and from there in the MLflow trace and any saved transcript. Both now route `configurable` extras through one shared filter, `dao_ai.state.context_configurable_fields`, which drops the declared `headers`/`user_id`/`thread_id` fields plus any extra whose *name* is credential-shaped (`authorization`, `api_key`, `session_cookie`, …) — `Context` is `extra="allow"` by design, so a caller can otherwise smuggle a secret in through the extras channel. Benign extras such as `store_num` still round-trip, so the block remains usable as the next request's `custom_inputs`. Headers are untouched **inbound**: OBO authentication and the `x-forwarded-user` → `user_id` fallback both still work. Nothing in-repo or in the chat UI read `custom_outputs.configurable.headers`, so this is behavioral only — no API contract changes and the config schema is unchanged. The name test lives in `dao_ai.diagnostics.is_secret_field_name`, which matches whole words rather than substrings, so `api_key`/`apiKey`/`x-forwarded-access-token` are filtered while `session_id` and `monkey_wrench` still round-trip; the substring matcher (`is_secret_key`) stays for env-var dumps, where over-redacting is free. A required custom field the caller *did* supply but whose name is credential-shaped is no longer replaced with its `example_value` in the copy-paste block — that told callers to paste `sk-proj-xxxxxxxxxxxxx` over a key they had already sent correctly — and is marked "keep the value you already sent" instead. The same bearer also reached the **MLflow trace**, because `_inject_headers_into_request` mutates the request before the `@mlflow.trace`-decorated `apredict`, so the span serialized it as an input attribute; `dao_ai._tracing.install_trace_redaction` now registers an MLflow span processor in the Apps, Model Serving and MCP entry points that replaces credential-shaped entries with `<redacted>` in span inputs and outputs, leaving the rest of the payload (including benign headers such as `x-forwarded-user`) intact for debugging.
- **Portable Apps bundle lock now rewrites every internal mirror host, not just `.dev.`.** `dao_ai._locking` hardcoded `pypi-proxy.dev.databricks.com`, so workspaces whose mirror is `pypi-proxy.cloud.databricks.com` (or a regional `*.cloud.` variant) shipped a `uv.lock` with unreachable URLs and the Apps build failed at `uv sync` with 404s (e.g. `pyarrow`, `scikit-learn`). The rewrite now matches any `pypi-proxy*.databricks.com` host, with an independent survivor guard so a future regression still fails loudly rather than shipping an unresolvable lock.
- **Sync `predict()` / `predict_stream()` no longer crash in a running event loop (#272).** The `LanggraphResponsesAgent`/`BackgroundResponsesAgent`/`LanggraphChatModel` sync wrappers previously called bare `asyncio.run()` (or `get_event_loop().run_until_complete()`), which raised `RuntimeError: asyncio.run() cannot be called from a running event loop` inside notebooks/IPython — requiring a manual `nest_asyncio.apply()`. They now drive the async graph on a shared, lazily-created background event loop (`dao_ai._async.run_sync`/`iter_sync`) that works from any calling context (notebook, Model Serving worker, plain sync) without patching the caller's loop — so it is also uvloop-safe on Databricks Apps (which uses `apredict`/`apredict_stream` directly and never the sync bridge). Caller contextvars (MLflow active-span, request state) propagate across the thread hop and stay isolated per concurrent caller.

- **BREAKING — Renamed the bundle lifecycle verbs to plain-language names** on the `agent` and `workflow` nouns: `generate → build`, `deploy → sync`, `run → start`, `destroy → down` (`up` is unchanged). The DAB names misled newcomers — mainstream tooling makes "deploy" mean "make it live", but the dao-ai `deploy` verb only *synced* the bundle to the workspace and a separate `run` started it. The new names say what each step does: **`build`** the bundle artifact, **`sync`** it to the workspace (does not start it), **`start`** it (make the app live / run the job), **`up`** for all-in-one. The old verbs are **removed** — `dao-ai agent deploy` / `run` / `generate` / `destroy` (and the `workflow` equivalents) now exit with an argparse "invalid choice" error. Update scripts; see the migration table in `docs/cli-reference.md`. The underlying `databricks bundle deploy`/`run`/`destroy` subcommands, the `deploy_job` resource, and the `up` orchestration are unchanged; the `mcp` noun's `tools`/`inspect`/`call` verbs are unaffected. No aliases kept.
- **BREAKING — Renamed the bundle staging-dir flag `-o/--output-dir` → `-s/--staging-dir`** on every bundle verb (`build`/`up`/`sync`/`start`/`down` across the `agent`, `workflow`, and `mcp` nouns). The flag never wrote a one-shot output — it names a persistent, reused staging directory (`sync`/`start`/`down` act on the same dir without rebuilding), and every other reference to it already used "bundle"/"staging" vocabulary (`DAO_AI_BUNDLE_DIR`, `_resolve_bundle_dir`). `-o` is now free for genuine output flags (`graph -o FILE`, `trace create -o FORMAT`). No alias is kept — update any scripts using `-o`/`--output-dir` on bundle verbs. The `DAO_AI_BUNDLE_DIR` env var and default `<base>/<kind>/<app>` layout are unchanged.

### Fixed

- **`agent down --mode model_serving` no longer orphans the serving endpoint.** The model_serving bundle is a Lakeflow Job that manages only the `deploy_job`; the serving endpoint is created imperatively by the deploy-agent notebook, so `bundle destroy` alone removed the job and left the endpoint running (and billing). `down` now also deletes the endpoint (`serving_endpoints.delete`, best-effort — a missing endpoint is not an error), keeping the registered UC model + versions as a reusable artifact. Establishes the contract that **`down` removes the deployment, never your data**: `workflow down` likewise removes only the provisioning job and leaves the provisioned infrastructure (Vector Search, Lakebase, Genie, UC schemas) intact.

## [0.2.4] - 2026-07-23

### Changed

- **Apps/MCP deploy switched from `requirements.txt` to `pyproject.toml` + a portable `uv.lock`.** `generate-agent`, `generate-mcp`, and the direct `deploy_apps_agent` path now emit `pyproject.toml` + `uv.lock` (no `requirements.txt` — its presence would take precedence and force the slower pip path). Databricks Apps' build phase runs `uv sync --locked --no-dev`. `dao_ai._locking.generate_bundle_lock` runs `uv lock` then rewrites any internal-mirror host (`pypi-proxy.dev.databricks.com`) to the public CDN so the lock resolves from Apps containers, and guards that no mirror URL survives. The exact-version pin and optional-feature extras are preserved in the generated pyproject: `dao-ai[<extras>]==<version>` (published) or a `[tool.uv.sources]` local-wheel redirect (`--development`). **Published-mode lock generation resolves `dao-ai==<version>` from PyPI, so it runs at release time (in CI, after publish); pre-release, use `--development`.** New `make` targets `lock` / `check-lock` / `lock-local` manage the repo's own `uv.lock` (frozen everyday sync, public-CDN re-lock, on-corp-network re-lock-and-rewrite). Workflows are unchanged (serverless env `${var.dao_ai_dep}` + `%uv pip install` notebook bootstrap).

### Fixed

- **Generated deploy bundles now pin the exact dao-ai version in published mode.** `dao-ai generate-agent` / `generate-mcp` previously wrote an *unbounded* `dao-ai` (and a `>=` floor) into the App bundle, so a redeploy months later silently resolved whatever was newest on PyPI — non-reproducible, and an unintended upgrade on every rebuild. Published mode now emits `dao-ai[extras]==<version>` in the generated `pyproject.toml` (threaded into `uv.lock`), matching the Model Serving deploy path (`create_agent` already pinned `==`). `generate-workflow`'s `dao_ai_dep` bundle variable is likewise pinned (CLI override and the `databricks.yaml` default). `--development` is unchanged — it references the bundled local wheel by path.
- **`config.app.pip_requirements` is honored on the Apps/MCP uv path again.** The switch to `pyproject.toml` + `uv.lock` had dropped the user's extra pip packages (they previously appended to the now-removed `requirements.txt`). They are now folded into the generated pyproject `dependencies` array so `uv lock` captures them, across `generate-agent`, `generate-mcp`, and the direct `deploy_apps_agent` path. (`config.app.code_paths` remains a Model Serving concern — for Apps/MCP, custom modules live under the bundle's `src/`.)

### Removed

- **BREAKING — Removed MLflow Prompt Registry load/store and GEPA prompt optimization.** The Databricks/MLflow prompt registry remained in Beta and is not reaching GA, so dao-ai no longer loads prompts from or registers prompts to it. `PromptModel` stays a first-class config object so prompts remain reusable YAML anchors/aliases, but it now carries its template **inline** — there is no registry round-trip.
  - `PromptModel`: renamed `default_template` → `template` (now required); removed the `alias`, `version`, and `auto_register` fields, the `uri` property, the `as_prompt()` method, and the alias/version mutual-exclusion validator. `.template` returns the inline text directly; `.jinja_template` still normalizes MLflow judge variables to double-brace form.
  - Removed `DatabricksProvider.get_prompt` and `DatabricksProvider._register_default_template` (the only callers of `mlflow.genai.load_prompt` / `register_prompt` / `set_prompt_alias`). The UC **model** registry and Model Serving deployment path are unchanged.
  - Removed the prompt-version trace-linking plumbing: `make_prompt(prompt_model=...)` param, `_cached_prompt_versions` / `get_cached_prompt_versions`, `create_responses_agent(prompt_versions=...)`, and `LanggraphResponsesAgent._prompt_versions`.
  - Removed GEPA prompt optimization entirely: `dao_ai.optimization`, `PromptOptimizationModel`, `OptimizationsModel.prompt_optimizations`, `notebooks/10_optimize_prompts.py`, `config/examples/11_prompt_engineering/prompt_optimization.yaml`, and the `gepa` dependency. `OptimizationsModel.training_datasets` (offline eval) and `cache_threshold_optimizations` (Genie cache tuning) are retained.

### Changed

- **CLI: unified the three DAB generators into a consistent `generate-*` family.** All three now generate a Databricks Asset Bundle and share `--deploy`/`--run`/`--destroy`/`--dry-run` action flags, a common deploy driver, and a common bundle-staging namespace.
  - **BREAKING — renamed `dao-ai generate-bundle` → `dao-ai generate-agent`** (emits the agent's Databricks App bundle) and **`dao-ai pipeline` → `dao-ai generate-workflow`** (emits the provisioning multi-task Job / Workflow — not a Lakeflow Declarative Pipeline). The old names are **removed** — invoking `generate-bundle` or `pipeline` now exits with an `invalid choice` error. Update scripts to the new verbs. Sibling `generate-mcp` is unchanged.
  - **`generate-agent` / `generate-mcp` gained `--deploy`/`--run`/`--destroy`.** They previously only wrote a bundle. When `--deploy` is passed the shared driver runs `databricks bundle deploy`, then — when `app.trace_location` is set — **automatically links the MLflow experiment trace destination and grants the App SP** the experiment + UC OTEL table privileges (the steps previously documented as manual follow-ups; silent trace loss otherwise), then `--run` starts the app. No action flags → generate-only, as before.
  - **`generate-workflow` is wheel-only** and stages a self-contained bundle: step notebooks ship as package data; `databricks.yaml` is built programmatically (shared `dump_bundle_yaml`) — no source checkout and no repo-root template. There is no pinned `requirements.txt`: dao-ai (+ its transitive deps) installs via the serverless job environment's `dao_ai_dep` dependency — the bundled wheel (`--development`) or the `dao-ai` PyPI package.
  - **Unified bundle location** `./.dao-ai/bundle/{workflow,agent,mcp}/<app-name>` (per-app, so deploying multiple configs never collides on DAB state). Override the base dir with the `DAO_AI_BUNDLE_DIR` env var (the `{kind}/<app>` structure is appended underneath — e.g. `DAO_AI_BUNDLE_DIR=~/.dao-ai/bundle` for a central location), or a specific bundle dir with `-o/--output-dir` (highest precedence). New `--overwrite` on `generate-workflow`.
    - Each generated `databricks.yaml` carries an explicit `sync.include` for its own source. Databricks bundle sync honors `.gitignore`, and the default `.dao-ai/` base is gitignored — without the include the deployed App would fail with "no files found".
  - **Post-deploy links.** All four deploy paths now print a link on success: `generate-agent`/`generate-mcp` and `dao-ai deploy -t apps` print the **App URL**; `generate-workflow --deploy` prints the **Job URL**; `dao-ai deploy` (Model Serving) prints the **serving-endpoint URL**. Best-effort — a lookup failure never fails the deploy.
  - **Contradictory action flags rejected uniformly.** `--destroy` combined with `--deploy` or `--run` now exits with a clean error on all three generate-* commands. Previously the two code paths diverged silently — `generate-workflow` deployed then immediately destroyed, while `generate-agent`/`generate-mcp` destroyed only.
- **Demo notebooks renumbered to clear the packaged pipeline range.** The wheel-only move relocated the 8 provisioning notebooks into the package (`src/dao_ai/pipeline/notebooks/01`–`08`). The standalone demo notebooks left in the repo-root `notebooks/` dir were renumbered off the now-reserved `01`–`08` range into a contiguous `10`–`16` sequence: `08_run_examples`→`10`, `09_evaluate_inferences`→`11`, `11_optimize_context_aware_genie_cache`→`12`, `12_genie_cache_service`→`13`, `13_enable_monitoring`→`14`, `14_background_agents_demo`→`15`, `15_feedback_demo`→`16` (`99_scratchpad` unchanged). Doc/test/config references updated.
- **CLI: `dao-ai deploy` brought in line with the `generate-*` family.** `deploy` remains the direct, bundle-free deploy path (calls `create_agent`/`deploy_agent` in-process — Model Serving via `agents.deploy`, or Apps via the Apps REST API — the same code the `generate-workflow` job's deploy step runs). It now also prints a link to the deployed endpoint/app on success, and is documented in `docs/cli-reference.md` alongside the generate-* commands with guidance on when to use each.
- **`InferenceEndpointModel.temperature` and `InferenceEndpointModel.max_tokens` now default to `None` instead of `0.1` and `8192`.** When a user does not set these fields in YAML, dao-ai omits them from the outbound request payload entirely, so each serving endpoint uses its own default. This unblocks reasoning-mode endpoints (e.g. Anthropic Claude Sonnet 5 via `databricks-claude-sonnet-5`) which reject any request that carries a `temperature` parameter regardless of value. The change is fully model-agnostic — no model names, families, or capability flags are hardcoded. `databricks_langchain.ChatDatabricks._prepare_inputs` already gates both fields on `is not None`, so `None` values are dropped before hitting the wire. Users who explicitly set `temperature: X` or `max_tokens: N` see no change in behavior; only the *absence-of-config* case shifts from "send dao-ai's default" to "let the endpoint decide."
- **BREAKING — `dao-ai generate-mcp` now emits an agent-as-tool MCP server, not a per-tool fan-out.** The generated Databricks App registers a single MCP tool that delegates to the whole dao-ai agent graph (`AppConfig.as_responses_agent()`); the tool's name is a slugified `config.app.name` and its description is `config.app.description`. Customers integrating dao-ai with external agent frameworks (LangGraph, ADK, OpenAI Agents SDK, MAS, IDE assistants) get one high-level capability instead of duplicating individual tools that already have first-class Databricks MCP or UC surfaces. OBO forwarding is preserved: `x-forwarded-access-token` captured by `RequestContextMiddleware` flows into the agent's `Context` via `custom_inputs.configurable.headers`, so downstream Genie / Vector Search / UC function calls run as the caller. No backwards-compatibility shims.
  - Removed: `dao_ai.mcp.service`, `dao_ai.mcp.adapters/` (entire directory including `genie` and `vector_search` adapters), `AppModel.mcp_only` field, `config/examples/15_complete_applications/sporting_goods_store_mcp.yaml`.
  - Added: `dao_ai.mcp.agent_tool.register_agent_as_tool` — the single-tool registration surface.
  - Config requirement: `app.name` is now mandatory for `generate-mcp` (used as both the deployed App name and the MCP tool name). `app.description` is strongly recommended.
  - `dao_ai.mcp.config.load_app_config` gains an `initialize: bool = True` param — runtime boot uses full initialization; bundle generation and unit tests pass `initialize=False`.
  - `docs/mcp_server.md` rewritten end-to-end for the agent-as-tool model.
  - **Structured MCP response.** The tool now returns a `CallToolResult` with `content[0].text` (plain text for legacy clients), `structuredContent` matching an `AgentInvocationResult` schema (`final_message`, `trace_id`, `confidence`), and `_meta.databricks.*` observability fields (`trace_id`, `experiment_id`, `model`, `latency_ms`, `request_id`, `obo_present`). Agent-side failures surface as `isError: true` so the caller LLM still receives the error text; JSON-RPC errors are reserved for input-schema violations. Skipped for now: progress notifications, sampling, elicitation, `resource_link` — all deferred to follow-up PRs.
  - **MLflow experiment provisioning parity with `generate-bundle`.** `generate-mcp` now emits an MLflow experiment resource in the DAB (either from `config.app.experiment` or auto-declared under `/Users/${workspace.current_user.userName}/<app-name>`), binds it as an App resource, and injects `MLFLOW_EXPERIMENT_ID: valueFrom: experiment` into the runtime `app.yaml` (camelCase — Apps consumes the file directly, DABs isn't in the loop). When `config.app.trace_location` is set, `MLFLOW_TRACING_SQL_WAREHOUSE_ID` is also injected and the trace warehouse is added to the App's resource list. This closes the gap where server-side `trace_location` was dead code and OTEL Delta tables were never materialized on the server side.
  - **OBO passthrough diagnostic.** The MCP server logs a stable SHA-256 token fingerprint + decoded JWT `sub` claim (best-effort, no signature verification) on every tool invocation, so operators can verify OBO propagation end-to-end without deploying a diagnostic tool. The raw token itself is never logged. See `docs/mcp_server.md` "OBO passthrough" section.
  - **Server-side trace root is now `dao_ai_apredict`.** Before Change 1 landed, the MCP server booted with no `MLFLOW_EXPERIMENT_ID` env, so `mlflow.set_experiment(...)` was skipped, `LanggraphResponsesAgent.apredict`'s `@mlflow.trace` decorator had no experiment to write to, and its span was dropped — trees rooted at `LangGraph`. With experiment provisioning wired up, the apredict span exports correctly and now serves as the trace root, giving readers a clear "this was an agent invocation" outer node. MCP-specific request context (`request_id`, `obo_present`) lives on the response's `_meta` block rather than on a synthetic span.
  - **Known limitation — `_meta` stripped by LangChain MCP adapter.** External MCP clients that consume the raw MCP protocol (Claude Desktop, MAS, Cursor, direct `curl`) see the full `_meta.databricks.*` block on every tool result. But when a *dao-ai consumer* calls the MCP server via `type: mcp`, `langchain-mcp-adapters` wraps the `CallToolResult` into a LangChain `ToolMessage` that only preserves `content` + `name` + `status` + `tool_call_id` — `_meta` and `structuredContent` are dropped. The observability chaining win (jump from consumer trace to server trace via `_meta.databricks.trace_id`) is invisible to nested dao-ai consumers. Follow-up: extend `dao_ai.tools.mcp`'s tool wrapper to lift `_meta` into `ToolMessage.additional_kwargs`, or contribute upstream. Tracked separately.

- **BREAKING — Renamed `long_running` → `background` throughout the API.** Aligns dao-ai with OpenAI's "background" terminology for async/fire-and-forget Responses. No backwards-compatibility shims. Specifically:
  - **Pydantic class.** `LongRunningModel` → `BackgroundModel`. Field `default_background` → `default_enabled` (removes the `BackgroundModel.default_background` redundancy).
  - **Config field.** `AppModel.long_running` → `AppModel.background`; YAML key `app.long_running:` → `app.background:`.
  - **Wrapper class + store.** `LongRunningResponsesAgent` → `BackgroundResponsesAgent`; `LongRunningStore` → `BackgroundStore`.
  - **Module path.** `dao_ai.long_running` → `dao_ai.background`. Imports like `from dao_ai.long_running import LongRunningResponsesAgent` must become `from dao_ai.background import BackgroundResponsesAgent`.
  - **Wire-protocol envelope key.** `custom_outputs["long_running"]` → `custom_outputs["background"]`. Any client reading `response.custom_outputs.long_running.status` must update to `.background.status`.
  - **Example configs renamed.** `config/examples/19_long_running_agents/` → `19_background_agents/`; `config/examples/20_a2a_protocol/a2a_long_running.yaml` → `a2a_background.yaml`.
  - **Docs renamed.** `docs/long_running_agents.md` → `docs/background_agents.md`.
  - **Demo notebook renamed.** `notebooks/14_long_running_agents_demo.py` → `14_background_agents_demo.py`.
  - The external request field `ResponsesAgentRequest.background: bool` (from MLflow SDK) is unchanged — dao-ai now mirrors that name end-to-end.

### Added

- **Run agents and workflows straight from a git repository.** `--config` (and `AppConfig.from_git`) now accepts a git locator, so a dao-ai project that isn't on your machine runs like one that is:

  ```bash
  dao-ai agent up -c 'git+https://github.com/org/repo@v1.0#examples/retail/agent.yaml' -p my-profile
  dao-ai agent up --from 'gh:org/repo@v1.0' -c examples/retail/agent.yaml -p my-profile
  ```

  The recently-added URL support only ever worked for a *single self-contained* YAML: with no directory behind a URL there is nothing to anchor relative paths to, so a config declaring `ddl: data/products.sql` was rejected outright — which excluded most real projects. A git locator materializes the whole tree into a local cache and hands the loader a real path inside it, so every colocated-asset convention (`ddl`/`data`, `code_paths`, `src/`, `skills/`, `resources/`) resolves exactly as it does locally. No new dependency-declaration concept was needed: they all already anchor on the config's own directory.

  - **Grammar** follows pip/uv: `git+<scheme>://<host>/<owner>/<repo>[@<ref>][#<in-repo-path>]`, plus a `gh:owner/repo` shorthand. `@ref` takes a branch, tag, or full 40-character SHA; `#path` may name a file or a directory. Both are optional — omit the path (or point at a directory) and the config is discovered, preferring `dao-ai.yaml` and erroring with the candidates listed when ambiguous. Refs containing `/` (`@feature/foo`) work. Quote the locator: `#` starts a shell comment. In `--from` the `git+` prefix is **optional** — a browser URL (`https://github.com/org/repo@v1`) or an `scp`-style SSH reference (`git@github.com:org/repo.git`) is accepted, since `--from` can only mean a repository. In `--config` it is required, because there a plain `https://` URL means "fetch one YAML" and `git+https://` means "clone the tree".
  - **Caching** is per commit under `~/.dao-ai/git` (override: `$DAO_AI_GIT_CACHE`). A full SHA is immutable and never re-fetched; a branch or tag is re-resolved with `git ls-remote` each run and re-fetched only when it moved, so `up` on a branch always deploys current HEAD. An unreachable remote falls back to the newest cached checkout with a warning. New `dao-ai cache dir` / `cache clear [--repo LOCATOR | --bundles]` manage the space.
  - **Staging follows the config.** A local config still stages into the project-local `./.dao-ai/bundle` (unchanged). A git locator has no project directory to sit beside, so it stages into `~/.dao-ai/bundle/<repo>-<digest>/` — keyed by repository plus in-repo config path, so the same locator reuses one staging dir (and its idempotent-skip) regardless of the directory you run it from, and two projects that happen to name their app the same thing cannot collide. Previously a locator staged CWD-relative, so the same locator run from two directories built twice and neither saw the other's work. `$DAO_AI_BUNDLE_DIR` and `-s/--staging-dir` override as before. `~/.dao-ai` is deliberately not under `$XDG_CACHE_HOME`: a checkout is not disposable the way a cache is, and the name mirrors the project-local `.dao-ai/`.
  - **Auth** delegates to `git`, so ssh-agent and credential helpers work untouched. For headless use (notebook, CI) `DAO_AI_GIT_TOKEN` / `GITHUB_TOKEN` is passed via an inline credential helper — never interpolated into the remote URL (which would persist it in the cache's `.git/config`), never in argv, and redacted from errors. A locator carrying inline credentials is rejected with a pointer to the env var.
  - **Trust:** a git locator runs the repository's code, exactly as `git clone` + `dao-ai agent up` would. The resolved SHA is reported on every load. Pin a tag or SHA for repos you don't control.
  - Requires `git` on `PATH`. Resolution is client-side only — the generated bundle stays self-contained, so nothing needs `git` at deploy or run time.

  Alongside this, `AppConfig.from_*` was refactored onto a `ConfigSource` abstraction (`dao_ai.sources`): `FileSource` / `UrlSource` / `GitSource` each answer where the bytes come from and whether a local tree backs them, and `from_file` delegates to a single `from_source` implementation. Every `from_*` method now accepts either a spec string or an explicitly-constructed source (`AppConfig.from_git(GitSource(spec, token=..., refresh=True))`), differing only in the source type it pins — `from_file`/`from_source` are lenient, `from_url`/`from_git` validate. Behavior for existing local and URL configs is unchanged.

- **First-class `type: sql` tool + parameterized SQL statements against warehouse *and* Lakebase.** The SQL statement tool now accepts optional, typed bound parameters and can target either a SQL warehouse (`WarehouseModel`) or a Lakebase / Postgres database (`DatabaseModel`), dispatching on backend type. `dao_ai.tools.sql.create_execute_statement_tool` gains a `params: list[StatementParam]` argument and two public backend factories — `create_warehouse_statement_tool` / `create_lakebase_statement_tool`. Each `StatementParam` declares a `source` of `llm` (surfaced in the tool schema the model sees) or `context` (bound server-side from the runtime `Context`, hidden from the model); values are bound **natively** (`:name` for warehouse, `%(name)s` for Lakebase) — never string-interpolated, so both backends are injection-safe. A new first-class `SqlToolModel` (`type: sql`, with mutually-exclusive `warehouse:` / `database:` fields) is added to the `AnyTool` union alongside `genie` / `lakebase_search`, so the tool is declaratively configurable with typed fields; it inherits `human_in_the_loop` / `audit` / `call_limit` from the base tool model, so a mutating statement can be gated behind human approval with a signed audit receipt. OBO follows the existing `workspace_client_from(context)` pattern (warehouse OBO works on Model Serving and Apps); Lakebase Model-Serving OBO is honored today and a documented seam makes Apps OBO a one-line change once the shared pool threads request context. `params=None` preserves the legacy zero-argument tool. Schema regenerated. Examples added to `config/examples/14_basic_tools/sql_tool_example.yaml` (warehouse, parameterized) and `config/examples/15_complete_applications/hardware_store_lakebase.yaml` (Lakebase; mixed LLM+context params; HITL+audit-guarded mutation).

- **Diagnostic visibility for cache auth-mode resolution.** Two new structured log lines surface the root cause of the most common Postgres semantic-cache permission failure — operators configuring `DatabaseModel.client_id` but having the secret resolution silently fail back to ambient (App SP) auth. (1) `PostgresContextAwareGenieService._setup` now emits an INFO line `dao_ai.cache.auth.mode | mode=service_principal\|ambient | sp_client_id=...` at boot so operators can see at a glance whether the cache is connecting as the stable configured SP or as the App's auto-injected SP. (2) `IsDatabricksResource.workspace_client` now emits a WARNING `dao_ai.auth.client_id.unresolved` when `client_id` is configured in YAML but `value_of()` returns `None` (silent fallback) — naming the resource class + the scope/key the operator likely needs to grant the App SP `READ` on. Closes [#107](https://github.com/natefleming/dao-ai/issues/107) — the previously-suspected "Lakebase orphaned cache tables" failure mode was actually misconfigured `DatabaseModel.client_id`/`client_secret`; the existing dao-ai auth chain in `IsDatabricksResource.workspace_client:416-490` already prefers configured SP creds over ambient, and `dao-ai generate-mcp` already auto-emits the corresponding `secret` App-resource bindings so the App SP can read the scope at deploy time. End-to-end verified on FEVM: with `client_id`/`client_secret`/`workspace_host` restored to the `DatabaseModel` referencing `retail_consumer_goods` scope, all Apps connect as the stable cache SP `ad1118d0-...` (visible in `pg_stat_activity.usename`), and multi-tenant cache works without any operator-side `GRANT` statements.


- **MCP server for dao-ai tools, with one-command `dao-ai generate-mcp` deploy.** Any dao-ai config that defines `create_genie_toolkit` or `create_vector_search_tool` factories can now be exposed as a Model Context Protocol (MCP) server hosted on Databricks Apps — no agent runtime, no LangGraph, just the tools served over [Streamable HTTP](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports). Any MCP client (Claude Desktop, Cursor, agent platforms) can connect to a deployed app and call the tools natively.

  Highlights:
  - **New CLI subcommand `dao-ai generate-mcp`.** Mirrors `dao-ai generate-bundle` — reads an AppConfig, walks `config.tools`, and emits a deploy-ready Databricks Apps bundle (`databricks.yml` with `bundle.engine: direct`, `app.yaml`, `pyproject.toml`, `requirements.txt`, README) that runs the MCP server on Databricks Apps. Skips `agents:` / `app:` blocks entirely — the MCP server is tool-only. Supports `--development` (bundles a local dao-ai wheel + switches the generated pyproject to install from it; needed until 0.1.89+ is on PyPI).
  - **Pluggable adapter registry.** Tools are NOT hardcoded in the MCP server. Each dao-ai tool factory is exposed via its own adapter in `dao_ai/mcp/adapters/` that calls `register_adapter(McpAdapter(factory_name=..., register=...))` at import time. Two adapters ship: `genie` (rebuilds the `Genie → GenieService → PostgresContextAwareGenieService → LRUCacheService` chain and registers `<name>` + `<name>_feedback` MCP tools per `create_genie_toolkit` entry) and `vector_search` (invokes `create_vector_search_tool` and wraps its `StructuredTool` — inheriting hybrid + query decomposition + RRF + reranking + verifier). A new factory needs only a new adapter module — no edits to the MCP server core.
  - **Names and descriptions come from dao-ai.** The MCP server lifts each tool's `name` / `description` from the configured factory `args` (or from the LangChain tool object the factory returns), so what the LLM sees as a tool is what dao-ai's own factories advertise. No MCP-side string prefixes/suffixes.
  - **Streamable HTTP, stateless, single `/mcp/` endpoint.** `FastMCP(stateless_http=True, json_response=True, streamable_http_path="/")` mounted at `/mcp` on a FastAPI app, with the session manager driven by FastAPI's lifespan. Scales horizontally across Databricks Apps replicas with no sticky sessions. Blocking dao-ai calls run inside `asyncio.to_thread` so the event loop stays unblocked under concurrency.
  - **Rich `_meta` on every response.** `ask_<name>` returns SQL + result preview + `_meta.{cache_hit, served_by, latency_ms, message_id, cache_entry_id, conversation_id, trace_id, space_id, tool_name}`. `submit_feedback` and `<vs_tool>` return their own `_meta`. Honors client `_meta` on `tools/call` for `dao-ai/conversation_id` (multi-turn) and `dao-ai/disable_cache` (bypass).
  - **Auth.** Default is App SP via Databricks Apps' auto-injected `DATABRICKS_CLIENT_ID`/`DATABRICKS_CLIENT_SECRET`. The `x-forwarded-access-token` header is captured per-request and surfaced as `obo_present` in every loguru log line — OBO is wired through dao-ai's existing `IsDatabricksResource.on_behalf_of_user` flags (set per-resource for selective OBO).
  - **New `[mcp]` extra.** `pip install 'dao-ai[mcp]'` brings in `fastapi` + `uvicorn[standard]`. The core MCP SDK was already a dependency.
  - **New console script `dao-ai-mcp-server`.** Invoked by the generated `app.yaml` as `["uv", "run", "dao-ai-mcp-server"]`. Reads its config from `$DAO_AI_MCP_CONFIG_PATH` (the rendered dao-ai.yaml co-located in the bundle).
  - **Default Databricks App name is `mcp-dao-ai`** (overridable via `config.app.name`). The `mcp-` prefix is a discovery signal for Databricks Multi-Agent Supervisor (MAS), which pattern-matches it when enumerating MCP-hosted Apps across an account.
  - **Postgres semantic cache: multi-tenant guidance in docs.** The cache table is designed for multi-tenant use — rows are isolated by `genie_space_id`, so multiple Genie spaces and multiple MCP/agent deployments sharing one Lakebase project share one cache table (`genie_context_aware_cache` / `genie_prompt_history`). Verified on FEVM that two distinct Apps with distinct SPs (`mcp-sporting-goods` and `mcp-hardware-store`) both write to and read from the shared tables once the project owner has run a one-time `GRANT ... TO PUBLIC` on the table + sequence. Troubleshooting docs explain the exact GRANT statements and call out the underlying schema asymmetry (the cache tables' `SERIAL` PK introduces an implicit sequence with its own ACL; LangGraph's checkpointer sidesteps this by using composite text PKs — tracked as a follow-up improvement to migrate the cache tables to the same pattern).
  - **Example config.** `config/examples/15_complete_applications/sporting_goods_store_mcp.yaml` — a slim MCP-only flavor of the sporting-goods config; 2 Genie toolkits + 1 instructed VS retriever, no agents/app blocks. End-to-end tested against the FEVM workspace: LRU `cache_hit` progression demonstrated (`false → true(lru) → false-after-feedback-invalidates`), instructed retrieval pipeline confirmed firing (decomposition into 3 subqueries + RRF + fallback to standard search), App SP auth across Genie / VS index / Lakebase / SQL warehouse / serving endpoints.
  - **Docs.** New `docs/mcp_server.md` covers the wire shape, adapter registry, generator UX, deploy flow, troubleshooting, and the `_meta` contract.
  - **8 new unit tests** covering config loading, adapter dispatch, tool registration, and bundle generation.
  - **Schema unchanged.** No `config.py` Pydantic models were modified — `schemas/model_config_schema.json` does not need regeneration.

  This pairs with but is independent of the existing `dao_ai.apps.mcp` module, which provides primitives for *consuming* external MCP servers from a dao-ai agent (client-side). The new `dao_ai.mcp` package is the *server* side.

- **Google A2A (Agent2Agent) protocol support on Databricks Apps deployments.** Every dao-ai agent deployed to Databricks Apps now automatically exposes a fully [A2A v0.3](https://a2a-protocol.org)-compliant endpoint alongside the existing OpenAI Responses contract. Two new routes are mounted on the same FastAPI app:
  - `GET  /.well-known/agent-card.json` — public Agent Card discovery.
  - `POST /a2a`                          — A2A JSON-RPC 2.0 (`message/send`, `message/stream` over SSE, `tasks/get`, `tasks/list`, `tasks/cancel`, `tasks/subscribe`).

  Both protocols share one compiled LangGraph and one checkpointer, so conversations stay consistent across contracts. No config change is required to enable A2A — set `app.deployment_target: apps` and you're done. Opt out via `app.a2a.enabled: false`.

  Highlights:
  - **Agent Card auto-derivation.** Skills default to one `AgentSkill` per entry in `app.agents`. The `bearer` security scheme description is conditioned on `app.a2a.on_behalf_of_user` to advertise OBO support. The card `url` derives from `$DATABRICKS_APP_URL` at startup. Override any field via `app.a2a.skills` / `app.a2a.security_schemes` / `app.a2a.server_url`.
  - **HITL parity.** A LangGraph `interrupt()` raises a terminal `TaskStatusUpdateEvent(state=INPUT_REQUIRED)` with the interrupt payload as a `DataPart`. Clients resume by sending another `message/send` for the same `taskId`+`contextId` carrying either a structured `DataPart {"decisions": [...]}` (machine-to-machine) or a free-text `TextPart` (the existing dao-ai `handle_interrupt_response()` LLM parser handles it).
  - **OBO parity.** The Databricks Apps proxy's `x-forwarded-access-token` header is captured by a2a-sdk's `DefaultCallContextBuilder` and injected into `configurable.headers` so OBO tools see the end-user's token unchanged.
  - **Task persistence.** `app.a2a.task_store` is an `A2ATaskStoreModel { database, table }` — independent of `app.long_running`. No `database` → in-memory; set `database` → Lakebase persistence in the configured `table`. Share a connection pool with `LongRunningStore` and the LangGraph Postgres checkpointer by pointing all three at the same `DatabaseModel`; `AsyncPostgresPoolManager` dedupes by connection-string value.
  - **Typed security schemes.** `A2AModel.security_schemes` is typed against a2a-sdk's `SecurityScheme` discriminated union, so malformed schemes fail at config-load time. dao-ai ships ready-made constants and factories in `dao_ai.apps.a2a.security` (`BEARER_DATABRICKS_PAT`, `BEARER_DATABRICKS_M2M`, `BEARER_DATABRICKS_OBO`, `api_key_header`, `oauth2_databricks_authorization_code`, `oauth2_databricks_client_credentials`, `oauth2_databricks_obo`, `openid_connect_databricks`). YAML users can author the equivalents inline using `${workspace.host}` substitution.
  - **Auto-derived OBO advertisement.** `A2AModel.on_behalf_of_user` is now three-state (`Optional[bool]`, default `None`). With `None`, dao-ai scans the config for any `IsDatabricksResource` with `on_behalf_of_user: true` and auto-advertises OBO on the Agent Card. When effective OBO is True, the Agent Card emits **both** an `oauth2` scheme (the declarative authorization-code flow with `user_impersonation` scope, URLs from `$DATABRICKS_HOST`) and a `bearer` scheme (the wire shape); A2A clients pick the one their auth machinery supports. Users can pin `a2a.on_behalf_of_user: true|false` to override.
  - **Shared HITL helper.** Both the Responses path (`LanggraphResponsesAgent.apredict` / `apredict_stream`) and the A2A executor delegate to a new `dao_ai.hitl.decide_graph_turn`, eliminating ~150 lines of duplicated HITL decision logic.
  - **New config models.** `A2AModel`, `A2ASkillModel`, `A2ATaskStoreModel`. `A2AModel.on_behalf_of_user: bool` replaces the previously-proposed (unshipped) `AppModel.on_behalf_of_user` advisory flag — A2A's Agent Card was its only consumer. Per-resource `on_behalf_of_user` fields elsewhere (AgentModel, GenieRoomModel, VectorStoreModel, etc.) are unchanged. Schema regenerated.
  - **New `app.a2a` field on AppModel** (`Optional[A2AModel]`). Default `None` is treated as `A2AModel()` (enabled with sensible defaults).
  - **Worked examples.** `config/examples/20_a2a_protocol/a2a_minimal.yaml` (deploy-ready, dependency-free), `a2a_long_running.yaml` (Lakebase-persistent task store), and `a2a_hitl_obo.yaml` (HITL + OBO over A2A). `examples/a2a/client.py` is an end-to-end Python client that exercises the agent card, message/send, message/stream, and HITL resume flows.
  - **Docs.** New `docs/a2a_protocol.md` covers the wire-shape mappings, configuration surface, HITL/OBO semantics, task-store configuration, security-scheme recipes (Python + YAML), and spec-compliance scope.

  Model Serving deployments are unchanged — A2A is Apps-only because the MLflow Model Serving runtime cannot mount arbitrary FastAPI routes.

## [0.1.76]

### Fixed
- **Schema regeneration restored alias names** so existing example configs validate cleanly against the JSON schema again. 0.1.75 set `model_json_schema(by_alias=False)` to make `resources.models:` canonical, but that change cascaded to *every* field with a Pydantic alias — most importantly `TableModel.schema_model` / `VolumeModel.schema_model` / etc., which are Python-side aliased to `schema:` because `schema` collides with `BaseModel.schema()`. The result: yaml-language-server lit up every `schema:` key in every config as "additional property not allowed", even though runtime parsing was unaffected. The fix:
  - Reverted the `Makefile` schema target to default `model_json_schema()` (`by_alias=True`), so aliased fields like `schema_model→schema` emit their alias as the canonical schema property.
  - Switched `ResourcesModel.models` from `Field(alias="llms")` to `Field(validation_alias=AliasChoices("models", "llms"))` so the *rename* keeps `models` as canonical in the schema while still accepting `llms` as input. This is the right Pydantic tool for "rename with input-only back-compat".
  - Removed the now-unneeded `populate_by_name=True` from `ResourcesModel.model_config` (AliasChoices handles both keys directly).
- Schema regenerated; `config/examples/15_complete_applications/brick_store.yaml` and other in-repo examples that use `schema:` and `models:` now validate cleanly. All 11 alias regression tests still pass.

## [0.1.75]

### Changed
- **Renamed `resources.llms` → `resources.models` and `LLMModel` → `InferenceEndpointModel`** for accuracy. The same class is reused for chat LLMs, embedding endpoints, judge / extraction / reflection / query models, and custom agent endpoints — anything reachable via `/serving-endpoints/<name>/invocations`. The previous names were specific to chat LLMs and actively misled new readers. Both renames are **fully backward compatible**:
  - The Pydantic field carries `alias="llms"` on `ResourcesModel.models` and `ResourcesModel.model_config` sets `populate_by_name=True`, so existing YAML configs with `resources.llms:` keep parsing unchanged.
  - The class has a module-level alias (`LLMModel = InferenceEndpointModel`), so `from dao_ai.config import LLMModel` keeps working and `isinstance(x, LLMModel)` continues to return `True`.
  - Reading the legacy `ResourcesModel.llms` attribute now returns `self.models` and emits a `DeprecationWarning`.
  - The generated JSON schema documents `models:` as the canonical key (via `model_json_schema(by_alias=False)`). IDE schema linting (yaml-language-server) will flag `llms:` as unknown after this release; the runtime still accepts it. Bulk-migrate your configs at your own pace, or let dao-ai-builder rewrite them on the next save.
  - All shipped example configs under `config/examples/**` were migrated to the new key. The 27 existing test files were left as-is — they still pass via the alias, which is the regression guard.
  - The legacy names will be removed in a future major release.

## [0.1.74]

### Changed
- **CLI: renamed `dao-ai bundle` to `dao-ai pipeline`** to disambiguate from the underlying Databricks Asset Bundle (DAB) feature it wraps. The old verb is no longer recognized — running `dao-ai bundle` now exits with `argparse: invalid choice: 'bundle'`. Help, examples, and documentation have been updated accordingly. The separate `dao-ai generate-bundle` subcommand keeps its name since it generates a literal Databricks Asset Bundle artifact.

### Added
- **Best-of-N + LLM-as-judge wrapper (`LLMModel.best_of_n`)**: opt-in primitive that fans out N parallel candidate generations on any configured `LLMModel` and asks a judge model to score them. The wrapper returns `candidates[argmax(scores)]` verbatim — never synthesised. Works as a drop-in `BaseChatModel`: every existing call site (agent nodes via `as_chat_model()`, OBO retrievers via `_get_cached_llm()`, structured-output classifiers like the verifier/router/reranker via a dedicated `with_structured_output` runnable) picks up the wrapping when `best_of_n: { n: 8, judge: <ref-or-name> }` is added to an LLM block. Diversity floor enforced (`max(LLMModel.temperature, 0.7)` unless overridden via `temperature_override`). Disables streaming when set. MLflow trace shape per invocation: one `best_of_n` parent CHAIN span (records `n`, `selected_index`, `scores`, `judge_reasoning` as attributes) + N `candidate_*` LLM child spans + one `judge` LLM child span. Hard cap `n ≤ 16`. Default `n = 8`. No shipped example enables it; opt in by adding the field to any `LLMModel` block.
- **Vector Search endpoint `target_qps` (Public Preview)**: New optional field on `VectorSearchEndpoint` that provisions target queries-per-second on STANDARD endpoints. Scales endpoint compute linearly (cost scales linearly too). Honored at endpoint-creation time only — if the endpoint already exists, the configured value is logged at debug level but not reconciled. Validated at config-load time: setting `target_qps` on an `OPTIMIZED_STORAGE` endpoint raises a `ValidationError`. Bumps `databricks-vectorsearch>=0.67` (which introduced the underlying `min_qps` SDK kwarg) and `databricks-sdk[openai]>=0.106.0`. See `docs/configuration-reference.md` for usage and constraints.
- **Swarm handoff constraints (`requires`)**: New optional field on the agent model declaring prerequisite agents that must have run before this agent can be reached via handoff. Enforced inside swarm handoff tools — when prereqs are unmet, the tool returns a refusal `ToolMessage` naming the missing agents and `active_agent` stays unchanged so the LLM can self-correct. Validated at config-build time (unknown agents, self-reference, cycles in the `requires` DAG, and deterministic handoffs to constrained targets are all rejected). Swarm-only in this release; supervisor adoption is planned. See `docs/architecture.md` → "Swarm Pattern" → "Handoff constraints" for details.

- **MCP Tool Filtering**: Control which tools are loaded from MCP servers
  - `include_tools`: Optional allowlist with glob pattern support (e.g., `["query_*", "list_*"]`)
  - `exclude_tools`: Optional denylist with glob pattern support (e.g., `["drop_*", "delete_*"]`)
  - Precedence: exclude always overrides include for maximum security
  - Pattern syntax: `*` (any chars), `?` (single char), `[abc]` (char set), `[!abc]` (negation)
  - Use cases: Security (block dangerous operations), performance (reduce context), access control
  - New example config: `config/examples/02_mcp/filtered_mcp.yaml` with 6 filtering strategies
  - Comprehensive documentation in configuration reference and MCP README

- **CLI: list-mcp-tools Command**: Discover and inspect MCP tools from configuration
  - Lists all available tools from configured MCP servers with full details
  - Shows tool descriptions (no truncation), parameters, types, and requirements
  - Pretty-printed schemas in readable format (53% more compact than JSON)
  - Filter statistics: total available, included, and excluded tool counts
  - `--apply-filters` flag: Show only tools that will be loaded (respects include/exclude)
  - Aggregated output: Collects all data before display (no logging interference)
  - Detailed exclusion reasons: Shows why tools are filtered out
  - Use cases: Discovery, debugging, validation, planning, documentation

- **AnyVariable Support for Additional Fields**: More configuration flexibility
  - `SchemaModel.catalog_name` and `SchemaModel.schema_name` now support AnyVariable
  - `DatabricksAppModel.url` now supports AnyVariable
  - Allows environment variables, Databricks secrets, and fallback chains
  - Benefits: Environment flexibility, security, portability, backwards compatible
  - Examples: `{env: CATALOG_NAME}`, `{scope: secrets, secret: url}`, composite fallbacks

### Changed
- **Refactored Dynamic Prompt Creation**: Simplified and improved `prompts.py`
  - Consolidated redundant prompt creation logic into single `make_prompt()` function
  - Removed unused `create_prompt_middleware()` function (dead code)
  - Cleaner context field handling with generic loop over all context attributes
  - More maintainable codebase with reduced duplication

## [0.1.0] - 2025-12-19

### Added
- **DSPy-Style Assertion Middleware**: New middleware for output validation and refinement
  - `AssertMiddleware`: Hard constraints with retry - enforces requirements or fails after max attempts
  - `SuggestMiddleware`: Soft constraints with optional single retry - provides feedback without blocking
  - `RefineMiddleware`: Iterative improvement - runs multiple iterations to optimize output quality
  - Multiple constraint types: `FunctionConstraint`, `LLMConstraint`, `KeywordConstraint`, `LengthConstraint`
  - Factory functions: `create_assert_middleware()`, `create_suggest_middleware()`, `create_refine_middleware()`

- **Conversation Summarization**: Automatic summarization of long chat histories
  - `LoggingSummarizationMiddleware`: Extends LangChain's `SummarizationMiddleware` with detailed logging
  - Configurable via `chat_history` in YAML with `max_tokens`, `max_tokens_before_summary`, `max_messages_before_summary`
  - Logs original and summarized message/token counts for observability
  - New example config: `config/examples/04_memory/conversation_summarization.yaml`

- **GEPA-Based Prompt Optimization**: Replaced MLflow optimizer with GEPA (Generative Evolution of Prompts and Agents)
  - `optimize_prompt()` function using DSPy's evolutionary optimization
  - `DAOAgentAdapter` bridges DAO ResponsesAgent with GEPA optimizer
  - Automatic prompt registration with comprehensive tags
  - Reflective dataset generation for self-improvement

- **Structured Input/Output Format**: New `configurable` and `session` structure
  - `configurable`: Static configuration (thread_id, conversation_id, user_id, store_num)
  - `session`: Accumulated runtime state (Genie conversation IDs, cache hits, follow-up questions)
  - Backward compatible with legacy flat `custom_inputs` format

- **conversation_id/thread_id Interchangeability**: Databricks-friendly naming
  - Input accepts either `thread_id` or `conversation_id` (conversation_id takes precedence)
  - Output includes both in `configurable` section with synchronized values
  - Auto-generation of UUID if neither is provided

- **In-Memory Memory Configuration**: Added to Genie example config
  - Simplified setup for development and testing

### Changed
- **ChatHistoryModel Refinements**:
  - Removed unused `max_summary_tokens` attribute
  - Updated `max_tokens` default from 256 to 2048
  - Added `gt=0` validation for numeric fields
  - Improved docstrings

- **CLI Thread ID Handling**:
  - `--thread-id` now defaults to auto-generated UUID instead of "1"
  - YAML configs no longer require hardcoded thread_id values

- **Orchestration Package Refactoring**:
  - Created `orchestration` package with `supervisor` and `swarm` submodules
  - Shared code consolidated in `orchestration/__init__.py`
  - Improved code organization and maintainability

### Removed
- MLflow `GepaPromptOptimizer` wrapper (replaced with direct GEPA integration)
- `backend` and `scorer_model` fields from `PromptOptimizationModel`
- Hardcoded `thread_id: "1"` from all example configurations

### Fixed
- Handoff issues in supervisor pattern with `Command.PARENT` graph reference
- Pydantic serialization warnings suppressed for Context serialization
- StopIteration error in Genie tests (upgraded databricks-ai-bridge to 0.11.0)
- Message validation middleware now properly terminates with `@hook_config(can_jump_to=["end"])`

### Dependencies
- Added `dspy>=2.6.27` for assertion middleware patterns
- Added `gepa` for prompt optimization
- Updated `databricks-ai-bridge` to 0.11.0

## [0.0.1] - 2025-06-19

### Added
- Initial release of DAO AI multi-agent orchestration framework
- Support for Databricks Vector Search integration
- LangGraph-based workflow orchestration
- YAML-based configuration system
- Multi-agent supervisor and swarm patterns
- Unity Catalog integration
- MLflow model packaging and deployment
- Command-line interface (CLI)
- Python API for programmatic access
- Built-in guardrails and evaluation capabilities
- Retail reference implementation

### Features
- **Multi-Modal Interface**: CLI commands and Python API
- **Agent Lifecycle Management**: Create, deploy, and monitor agents
- **Vector Search Integration**: Built-in Databricks Vector Search support
- **Configuration-Driven**: YAML-based configuration with validation
- **MLflow Integration**: Automatic model packaging and deployment
- **Monitoring & Evaluation**: Built-in assessment capabilities

[Unreleased]: https://github.com/natefleming/dao-ai/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/natefleming/dao-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/natefleming/dao-ai/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/natefleming/dao-ai/releases/tag/v0.0.1
