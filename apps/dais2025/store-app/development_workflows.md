# Development Workflows Guide

Quick reference for getting started with each component of the RCT Store Operations Demo.

## 🚀 Environment Setup (Do This First!)

```bash
# 1. Create environment file (project root)
make create-env-local           # Creates .env.local from .env.example
# Edit .env.local with your Databricks credentials

# 2. Load environment variables (project root)
source load-env.sh             # Single command - loads all variables

# Now you can work in any component directory with authentication ready!
```

## 📦 Retail AI Package

### Quick Start
```bash
cd retail_ai
make install              # Install dependencies with UV
make test                # Run tests
make build               # Build package

```

**Deploy to Databricks Workspace:**
```bash
make deploy ENV=dev      # Build and deploy using env/dev.env
```


---

## 🏪 Main Store App (`streamlit_store_app/`)

**Primary Streamlit application**

```bash
cd streamlit_store_app
make install           # Install dependencies
make start ENV=dev     # Start Streamlit app with development environment
```

**Deploy to Databricks Apps:**
```bash
make build-deploy ENV=dev    # Build and deploy using env/dev.env
```

---

## 🎨 TailAdmin Store App (`streamlit_store_app_tailadmin/`)

**Modern UI version with TailAdmin template**

```bash
cd streamlit_store_app_tailadmin
make install           # Install dependencies  
make start ENV=dev     # Start TailAdmin app with development environment
```

**Deploy to Databricks Apps:**
```bash
make build-deploy ENV=dev    # Build and deploy using env/dev.env
```

---

## 📚 Documentation (`docs/`)

**MkDocs documentation site**

```bash
cd docs
make install    # Install MkDocs dependencies
make serve      # Start local docs server
```

**Deploy documentation:**
```bash
make deploy     # Deploy to GitHub Pages
```

---

## 🚀 Quick Development Commands

| Component | Start Development | Run Tests | Deploy |
|-----------|-------------------|-----------|---------|
| `retail_ai` | `make install` | `make test` | `make deploy ENV=dev` |
| `streamlit_store_app` | `make start ENV=dev` | `make test` | `make build-deploy ENV=dev` |
| `streamlit_store_app_tailadmin` | `make start ENV=dev` | `make test` | `make build-deploy ENV=dev` |
| `docs` | `make serve` | N/A | `make deploy` |

---

## 💡 Tips

- Use `make help` in any directory to see all available commands
- **Both Streamlit apps require ENV parameter**: `make start ENV=dev/staging/prod/dais`
- Use `ENV=dev/staging/prod/dais` parameter for different environments
- Use `make build-fast` for faster builds (skips tests)
- Use `make deploy-config ENV=dev` to check deployment settings
- Streamlit apps validate Databricks configuration before starting

---

## 🔗 Next Steps

- See component-specific READMEs for detailed information
- Check `docs/` for comprehensive architecture and setup guides
- Review `env/` folders for environment configuration examples