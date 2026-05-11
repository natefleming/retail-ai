# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/natefleming/dao-ai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/natefleming/dao-ai/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/natefleming/dao-ai/releases/tag/v0.0.1
