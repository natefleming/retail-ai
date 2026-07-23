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

## Dependencies & the lock file

`make install` runs `uv sync --frozen` — it installs the exact versions pinned
in the committed `uv.lock` without re-resolving. Keep it that way:

- **Do NOT run a bare `uv sync` or `uv lock`.** A bare command re-resolves against
  whatever package index your environment is configured for. Behind Databricks'
  internal PyPI mirror, that rewrites `uv.lock` with `pypi-proxy.dev.databricks.com`
  URLs, which are unreachable outside Databricks — this breaks customer installs
  and the Databricks Apps `uv sync --frozen` build path, and (the repo is public)
  leaks the internal hostname.
- **To change dependencies:** edit `pyproject.toml`, then regenerate the lock:
  - On the **corp network** (mirror reachable, `pypi.org` blocked): `make lock-local`
    — re-resolves via the mirror, rewrites the recorded host to the public CDN
    (safe: the mirror is a transparent passthrough), and runs `make check-lock`.
    This is the everyday path; no off-network trip needed.
  - Where **`pypi.org` is reachable** (Databricks sandbox / CI / release): `make lock`
    — re-resolves directly against public PyPI. The authoritative path; use it when
    you want zero mirror influence.
  - Then `make install` to sync your env, and commit `pyproject.toml` + `uv.lock`
    together. `lock-local` only works for public packages — a mirror-only dependency
    would produce a public-CDN URL that doesn't exist, caught by a later
    `uv sync --frozen`.
- **`make check-lock`** fails if `uv.lock` references the mirror host; it runs as
  part of `make check` and in the publish CI workflow.
- **`--development` builds are unaffected.** `dao-ai generate-agent/generate-mcp
  --development` build a local wheel via `uv build --wheel`, which doesn't read or
  modify `uv.lock`. Frozen installs work even on a mirror-bound laptop because they
  fetch pinned artifacts from the public CDN (`files.pythonhosted.org`) rather than
  querying the index API.

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

examples/15_complete_applications/[use-case]/   # Self-contained use-case dir
├── README.md                        # Use-case documentation
├── [use-case].yaml                  # dao-ai config (one or more variants)
├── functions/                       # UC SQL function DDL, colocated
│   └── *.sql                        # referenced as `ddl: functions/x.sql`
└── data/                            # Dataset DDL + seed data, colocated
    └── *.sql, *.parquet             # referenced as `data: data/x.parquet`

tests/[industry-example]/          # Industry-specific tests (optional)
└── test_[industry-example]_agents.py
```

Assets (`functions/`, `data/`) live **inside** the use-case directory next to the
config and are referenced with **config-relative paths** (bare `functions/x.sql`,
not `../functions/...`). They resolve against the config file's own directory, so
a use-case dir is self-contained and portable.

### Example: Adding Healthcare Use Case

For a healthcare industry use case:

1. **Create core modules:**
   ```
   src/healthcare/
   ├── __init__.py
   ├── tools.py           # FHIR tools, medical database tools
   └── hooks.py           # Healthcare-specific hooks
   ```

2. **Add a self-contained use-case directory with colocated assets:**
   ```
   examples/15_complete_applications/healthcare/
   ├── README.md
   ├── healthcare.yaml                      # dao-ai config
   ├── functions/
   │   └── lookup_patient.sql               # ddl: functions/lookup_patient.sql
   └── data/
       └── synthetic_patient_data.parquet   # data: data/synthetic_patient_data.parquet
   ```

### Key Guidelines

- **Naming consistency**: Directory names should match your use-case domain (e.g., `healthcare`, `finance`, `manufacturing`)
- **Self-contained**: Each use case is a single directory holding its config variant(s) and its colocated `functions/` and `data/`
- **Config-relative paths**: Reference assets with bare `functions/x.sql` / `data/x.parquet` so they resolve against the config's own directory
- **Documentation**: Include a clear README with usage examples in the use-case directory
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
