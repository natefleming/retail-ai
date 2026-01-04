# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
