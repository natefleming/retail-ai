# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
