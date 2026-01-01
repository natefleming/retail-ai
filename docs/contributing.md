# Contributing to DAO

Thank you for your interest in contributing to DAO! This guide will help you get started.

## Project Structure

```
dao-ai/
├── src/dao_ai/
│   ├── config.py          # Pydantic configuration models
│   ├── graph.py           # LangGraph workflow builder
│   ├── nodes.py           # Agent node factories
│   ├── state.py           # State management
│   ├── optimization.py    # GEPA-based prompt optimization
│   ├── tools/             # Tool implementations
│   │   ├── genie.py       # Genie tool with caching
│   │   ├── mcp.py         # MCP integrations
│   │   ├── vector_search.py
│   │   └── ...
│   ├── middleware/        # Agent middleware
│   │   ├── assertions.py  # Assert, Suggest, Refine middleware
│   │   ├── summarization.py # Conversation summarization
│   │   ├── guardrails.py  # Content filtering and safety
│   │   └── ...
│   ├── orchestration/     # Multi-agent orchestration
│   │   ├── supervisor.py  # Supervisor pattern
│   │   ├── swarm.py       # Swarm pattern
│   │   └── ...
│   ├── genie/
│   │   └── cache/         # LRU and Semantic cache
│   ├── memory/            # Checkpointer and store
│   └── hooks/             # Lifecycle hooks
├── config/
│   ├── examples/          # Example configurations
│   └── hardware_store/    # Reference implementation
├── tests/                 # Test suite
└── schemas/               # JSON schemas for validation
```

## Development Setup

### Prerequisites

- Python 3.11 or newer
- Git
- Access to a Databricks workspace (for integration tests)

### Installation

1. Fork and clone the repository:

```bash
git clone https://github.com/your-username/dao-ai.git
cd dao-ai
```

2. Create a virtual environment:

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using standard Python
python3 -m venv .venv
source .venv/bin/activate
```

3. Install development dependencies:

```bash
make install
```

## Contributing Guidelines

### 1. Fork the Repository

Create a fork of the repository on GitHub and clone it locally.

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-new-tool`
- `bugfix/fix-cache-issue`
- `docs/update-readme`

### 3. Make Your Changes

- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 4. Run Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_config.py

# Run with coverage
pytest --cov=dao_ai tests/
```

### 5. Format Code

```bash
# Format code with black and isort
make format

# Check linting
make lint
```

### 6. Update Documentation

- Update README.md if adding new features
- Add docstrings to new functions/classes
- Update configuration examples if needed
- Add entries to CHANGELOG.md

### 7. Submit a Pull Request

1. Push your branch to your fork
2. Open a pull request against the main repository
3. Describe your changes clearly
4. Reference any related issues

## Code Style

### Python Style Guide

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Write descriptive docstrings (Google style)
- Keep functions focused and small

Example:

```python
def create_agent_tool(
    config: AgentConfig,
    workspace_client: WorkspaceClient,
) -> BaseTool:
    """
    Create a tool that calls another agent endpoint.

    Args:
        config: Agent configuration
        workspace_client: Databricks workspace client

    Returns:
        Configured tool instance

    Raises:
        ValueError: If configuration is invalid
    """
    # Implementation
    pass
```

### YAML Configuration Style

- Use 2 spaces for indentation
- Keep configurations readable and well-commented
- Use anchors and references to avoid repetition

Example:

```yaml
resources:
  llms:
    default_llm: &default_llm
      name: databricks-meta-llama-3-3-70b-instruct
      temperature: 0.7

agents:
  my_agent:
    name: my_agent
    model: *default_llm  # Reference the anchor
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure (e.g., `tests/dao_ai/test_config.py`)
- Use pytest fixtures for common setup
- Mock external services (Databricks APIs, databases)

Example:

```python
import pytest
from dao_ai.config import AppConfig

def test_load_config(tmp_path):
    """Test configuration loading from file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
    app:
      name: test_agent
    """)
    
    config = AppConfig.from_file(str(config_file))
    assert config.app.name == "test_agent"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_config.py::test_load_config

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=dao_ai --cov-report=html tests/
```

## Adding New Features

### Adding a New Tool

1. Create tool module in `src/dao_ai/tools/`
2. Implement tool following LangChain patterns
3. Add factory function if needed
4. Add tests in `tests/dao_ai/tools/`
5. Add example configuration in appropriate `config/examples/` category
6. Update documentation

### Adding New Middleware

1. Create middleware in `src/dao_ai/middleware/`
2. Inherit from `BaseMiddleware`
3. Implement required methods
4. Add tests
5. Add configuration example
6. Document usage

### Adding New Orchestration Pattern

1. Create module in `src/dao_ai/orchestration/`
2. Implement pattern using LangGraph
3. Add configuration model in `src/dao_ai/config.py`
4. Add tests
5. Add example configuration
6. Document pattern

## Documentation

### Updating Documentation

- Main docs are in `docs/` directory
- Update relevant sections when adding features
- Include code examples
- Add diagrams if helpful (ASCII art or images)

### Adding Examples

1. Choose the appropriate category in `config/examples/` based on **primary feature** demonstrated:
   - `01_getting_started/` - Foundation concepts for beginners
   - `02_tools/` - Tool integrations (Genie, Vector Search, Slack, MCP, etc.)
   - `04_genie/` - Performance optimization strategies
   - `05_memory/` - State management and persistence
   - `06_on_behalf_of_user/` - User-level authentication and access control
   - `07_human_in_the_loop/` - Approval workflows
   - `08_guardrails/` - Safety and validation
   - `09_structured_output/` - Enforce JSON schemas
   - `10_agent_integrations/` - External agent platforms
   - `11_prompt_engineering/` - Prompt management and optimization
   - `12_middleware/` - Validation, logging, monitoring
   - `13_orchestration/` - Multi-agent coordination patterns
   - `14_basic_tools/` - Simple tool patterns
   - `15_complete_applications/` - Full-featured, production-ready applications

2. **Use descriptive file names**: `tool_name_variant.yaml` (e.g., `slack_with_threads.yaml`)

3. Create your example config in the chosen category directory

4. Add entry to `docs/examples.md` in the appropriate category table

5. Test the example thoroughly:
   ```bash
   dao-ai validate -c config/examples/0X_category/your_example.yaml
   dao-ai chat -c config/examples/0X_category/your_example.yaml
   ```

6. Add inline comments explaining key concepts and design decisions

7. Update the category README.md with prerequisites and usage notes

## Release Process

Maintainers will handle releases, but here's the process:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with changes
3. Create git tag: `git tag v0.x.0`
4. Push tag: `git push origin v0.x.0`
5. GitHub Actions will build and publish

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Features**: Open a GitHub Issue with [Feature Request] prefix
- **Security**: See SECURITY.md (if available) or email maintainers

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DAO! 🎉

---

## Navigation

- [← Previous: FAQ](faq.md)
- [↑ Back to Documentation Index](../README.md#-documentation)

