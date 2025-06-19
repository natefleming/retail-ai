# Contributing to DAO AI

Thank you for your interest in contributing to the Multi-Agent AI Orchestration Framework! This guide will help you get started with contributing new industry use cases and improvements.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate
   make install
   ```

## Branching Strategy

- **`main`**: Production-ready code
- **Feature branches**: Use descriptive names like `feature/healthcare-agents` or `feature/finance-use-case`
- **Bug fixes**: Use `fix/` prefix like `fix/agent-routing-issue`

### Branch Naming Convention
```
feature/[industry-name]-[brief-description]
fix/[brief-description]
docs/[brief-description]
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, descriptive commits
3. Test your changes thoroughly
4. Submit a pull request with:
   - Clear title describing the change
   - Description of what was added/changed
   - Any breaking changes noted
   - Example usage if applicable

## Adding New Industry Use Cases

To add a new industry vertical (e.g., healthcare, finance, manufacturing), create corresponding directories and modules:

> **Reference Example**: Look at the existing `retail` structure in this repository as a working example of how to organize an industry use case.

### Directory Structure
When adding a new industry called `[industry-example]`, create:

> **Note**: All directories are optional - create only what you need for your specific use case.

```
src/[industry-example]/             # Core industry-specific code (optional)
├── __init__.py
├── tools.py                         # Industry-specific tools
└── hooks.py                         # Industry-specific hooks

config/[industry-example]/          # Industry configuration directory (optional)
├── model_config.yaml               # Main model configuration


examples/[industry-example]/        # Industry examples and demos (optional)
├── README.md                        # Industry-specific documentation
└── sample_queries.yaml             # Example queries/scenarios

data/[industry-example]/           # Sample industry data (if applicable)
└── sample_data.parquet

tests/[industry-example]/          # Industry-specific tests (optional)
└── test_[industry-example]_agents.py
```

### Example: Adding Healthcare Use Case

For a healthcare industry use case:

1. **Create core modules:**
   ```
   src/healthcare/
   ├── __init__.py
   ├── tools.py           # FHIR tools, medical database tools
   └── hooks.py           # Healthcare-specific hooks
   ```

2. **Add configuration directory:**
   ```
   config/healthcare/
   └── model_config.yaml        # Main configuration
   ```

3. **Create examples:**
   ```
   examples/healthcare/
   ├── README.md
   └── patient_diagnosis_flow.yaml
   ```

4. **Add sample data:**
   ```
   data/healthcare/
   └── synthetic_patient_data.parquet
   ```

### Key Guidelines

- **Naming consistency**: Directory names should match your industry domain (e.g., `healthcare`, `finance`, `manufacturing`)
- **Self-contained**: Each industry should be independent with its own src/, config/, examples/, and tests/
- **Configuration organization**: Use the config/[industry]/ directory structure to organize multiple config files
- **Documentation**: Include clear README files with usage examples in each industry's examples/ directory
- **Testing**: Add comprehensive tests for new functionality in tests/[industry]/

## Code Style

- Follow existing code patterns and structure
- Use descriptive variable and function names
- Add docstrings to all public functions and classes
- Run `make format` before committing

## Testing

- Add tests for new functionality in the appropriate `tests/[industry-name]/` directory
- Run tests with `make test`
- Ensure all tests pass before submitting PR

## Questions?

- Open an issue for questions about contributing
- Check existing issues and PRs to avoid duplicates
- For major changes, consider opening an issue first to discuss the approach

Thank you for helping make DAO AI more versatile across industries!
