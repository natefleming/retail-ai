# TailAdmin Store App Makefile
# Commands for running, testing, and managing the TailAdmin Streamlit application

.PHONY: help install start start-debug test clean build build-fast format lint check-deps add remove upgrade upgrade-pkg export-requirements load-env check-deploy-vars sync-app deploy-config deploy build-deploy

# Variables
UV := uv
APP_FILE := app.py

# Databricks deployment configuration
DATABRICKS_APP_NAME := $(DATABRICKS_APP_NAME)
LOCAL_APP_PATH := $(LOCAL_APP_PATH)
REMOTE_APP_PATH := $(REMOTE_APP_PATH)

# Default target
help:
	@echo "🎨 TailAdmin Store App - Available Commands"
	@echo ""
	@echo "🛠️  Development Commands:"
	@echo "  install      - Install all dependencies (including dev)"
	@echo "  start        - Start TailAdmin Streamlit app"
	@echo "  start-debug  - Start Streamlit app in debug mode"
	@echo "  test         - Run tests"
	@echo "  format       - Format code with ruff"
	@echo "  lint         - Lint code with ruff"
	@echo "  clean        - Clean build artifacts and cache"
	@echo ""
	@echo "📦 Build & Dependencies:"
	@echo "  build        - Build application for deployment (with tests)"
	@echo "  build-fast   - Build application for deployment (skip tests)"
	@echo "  check-deps   - Check dependency security and updates"
	@echo "  add          - Add a new dependency (usage: make add PKG=package-name)"
	@echo "  remove       - Remove a dependency (usage: make remove PKG=package-name)"
	@echo "  upgrade      - Upgrade all dependencies"
	@echo "  upgrade-pkg  - Upgrade specific dependency (usage: make upgrade-pkg PKG=package-name)"
	@echo "  export-requirements - Export requirements.txt for compatibility"
	@echo ""
	@echo "🌍 Environment Management:"
	@echo "  load-env     - Load environment variables from env/\$$(ENV).env and project root .env.local"
	@echo "  check-deploy-vars - Check deployment environment variables"
	@echo "  deploy-config    - Show current deployment configuration"
	@echo ""
	@echo "🚀 Deployment:"
	@echo "  sync-app     - Sync files from local to remote path"
	@echo "  deploy       - Deploy to Databricks Apps (without building)"
	@echo "  build-deploy - Build and deploy to Databricks Apps (full workflow)"
	@echo ""
	@echo "💡 Usage Examples:"
	@echo "  # RECOMMENDED - Combined environment loading:"
	@echo "  source <(make load-env ENV=dev) && make build-deploy"
	@echo ""
	@echo "  # Direct deployment with ENV parameter:"
	@echo "  make build-deploy ENV=dev      # Build and deploy using env/dev.env"
	@echo "  make deploy ENV=staging        # Deploy only using env/staging.env"
	@echo "  make deploy-config ENV=prod    # Show config for env/prod.env"

# Install all dependencies (including dev)
install:
	@echo "📦 Installing all dependencies..."
	$(UV) sync

# Start TailAdmin Streamlit app
start: install
	@echo "🎨 Starting TailAdmin Streamlit app..."
	@# Check if ENV parameter is provided
	@if [ -z "$(ENV)" ]; then \
		echo "❌ Error: ENV parameter is required"; \
		echo "💡 Usage: make start ENV=dev"; \
		echo "💡 Available environments: dev, staging, prod, dais"; \
		exit 1; \
	fi
	@# Check if environment file exists
	@if [ ! -f "env/$(ENV).env" ]; then \
		echo "❌ Error: env/$(ENV).env not found"; \
		echo "💡 Create env/$(ENV).env with DATABRICKS_WAREHOUSE_ID"; \
		exit 1; \
	fi
	@# Check if .env.local exists
	@if [ ! -f "../.env.local" ]; then \
		echo "❌ Error: .env.local not found in project root"; \
		echo "💡 Create .env.local with DATABRICKS_HOST and DATABRICKS_TOKEN"; \
		exit 1; \
	fi
	@# Load and validate environment variables, then start app
	@( \
		echo "📋 Loading environment from env/$(ENV).env..."; \
		set -a; source env/$(ENV).env; set +a; \
		echo "📋 Loading authentication from project root .env.local..."; \
		set -a; source ../.env.local; set +a; \
		if [ -z "$$DATABRICKS_HOST" ]; then \
			echo "❌ Error: DATABRICKS_HOST not found in .env.local"; \
			echo "💡 Add DATABRICKS_HOST=your-databricks-workspace-url to .env.local"; \
			exit 1; \
		fi; \
		if [ -z "$$DATABRICKS_TOKEN" ]; then \
			echo "❌ Error: DATABRICKS_TOKEN not found in .env.local"; \
			echo "💡 Add DATABRICKS_TOKEN=your-databricks-token to .env.local"; \
			exit 1; \
		fi; \
		if [ -z "$$DATABRICKS_WAREHOUSE_ID" ]; then \
			echo "❌ Error: DATABRICKS_WAREHOUSE_ID not found in env/$(ENV).env"; \
			echo "💡 Add DATABRICKS_WAREHOUSE_ID=your-warehouse-id to env/$(ENV).env"; \
			exit 1; \
		fi; \
		echo "✅ Environment variables loaded successfully"; \
		echo "  🌍 Environment: $(ENV)"; \
		echo "  🏢 Databricks Host: $$DATABRICKS_HOST"; \
		echo "  🏭 Warehouse ID: $$DATABRICKS_WAREHOUSE_ID"; \
		echo "🚀 Starting Streamlit app with Databricks configuration..."; \
		env DATABRICKS_HOST="$$DATABRICKS_HOST" \
		    DATABRICKS_TOKEN="$$DATABRICKS_TOKEN" \
		    DATABRICKS_WAREHOUSE_ID="$$DATABRICKS_WAREHOUSE_ID" \
		    $(UV) run streamlit run $(APP_FILE); \
	)

# Start Streamlit app in debug mode
start-debug: install
	@echo "🐛 Starting TailAdmin Streamlit app in debug mode..."
	$(UV) run streamlit run $(APP_FILE) --logger.level=debug

# Run tests
test: install
	@echo "🧪 Running tests..."
	$(UV) run pytest tests/ -v

# Format code
format:
	@echo "🎨 Formatting code..."
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

# Lint code
lint:
	@echo "🔍 Linting code..."
	$(UV) run ruff check .

# Clean build artifacts and cache
clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".ruff_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .streamlit/ 2>/dev/null || true
	@echo "✅ Artifacts cleaned"

# Build for deployment (with tests)
build: install format lint test
	@echo "🏗️  Building TailAdmin application..."
	@# Validate configuration
	@if [ ! -f pyproject.toml ]; then \
		echo "❌ pyproject.toml not found"; \
		exit 1; \
	fi
	@echo "✅ TailAdmin application ready for deployment"

# Build without running tests (faster for development)
build-fast: install format lint
	@echo "🏗️  Building TailAdmin application (skipping tests)..."
	@# Validate configuration
	@if [ ! -f pyproject.toml ]; then \
		echo "❌ pyproject.toml not found"; \
		exit 1; \
	fi
	@echo "✅ TailAdmin application ready for deployment"

# Check dependency security and updates
check-deps:
	@echo "🔍 Checking dependencies..."
	$(UV) pip list --outdated || true
	@echo "💡 Run 'uv sync --upgrade' to update dependencies"

# Add a new dependency
add:
	@echo "📦 Adding dependency..."
	@if [ -z "$(PKG)" ]; then \
		echo "❌ Usage: make add PKG=package-name"; \
		exit 1; \
	fi
	$(UV) add $(PKG)

# Remove a dependency  
remove:
	@echo "🗑️  Removing dependency..."
	@if [ -z "$(PKG)" ]; then \
		echo "❌ Usage: make remove PKG=package-name"; \
		exit 1; \
	fi
	$(UV) remove $(PKG)

# Upgrade all dependencies
upgrade:
	@echo "⬆️  Upgrading all dependencies..."
	$(UV) sync --upgrade

# Upgrade specific dependency
upgrade-pkg:
	@echo "⬆️  Upgrading specific dependency..."
	@if [ -z "$(PKG)" ]; then \
		echo "❌ Usage: make upgrade-pkg PKG=package-name"; \
		exit 1; \
	fi
	$(UV) sync --upgrade-package $(PKG)

# Export requirements.txt for compatibility (if needed)
export-requirements:
	@echo "📝 Exporting requirements.txt for compatibility..."
	$(UV) export --format requirements-txt --output-file requirements.txt --no-dev
	@echo "✅ requirements.txt exported"

# Sync files from local to remote path
sync-app: 
	@echo "🔄 Starting app sync..."
	@# Create a shell script that loads all environment variables and executes databricks sync
	@( \
		if [ -n "$(ENV)" ] && [ -f "env/$(ENV).env" ]; then \
			echo "📋 Loading environment from env/$(ENV).env..."; \
			set -a; source env/$(ENV).env; set +a; \
		fi; \
		if [ -f "../.env.local" ]; then \
			echo "📋 Loading authentication from project root .env.local..."; \
			set -a; source ../.env.local; set +a; \
		fi; \
		if [ -z "$$DATABRICKS_HOST" ]; then \
			echo "❌ Error: DATABRICKS_HOST environment variable is required"; \
			echo "💡 Set in project root .env.local"; \
			exit 1; \
		fi; \
		if [ -z "$$DATABRICKS_TOKEN" ]; then \
			echo "❌ Error: DATABRICKS_TOKEN environment variable is required"; \
			echo "💡 Set in project root .env.local"; \
			exit 1; \
		fi; \
		if [ -z "$$LOCAL_APP_PATH" ]; then \
			echo "❌ Error: LOCAL_APP_PATH environment variable is required"; \
			echo "💡 Set in env/$(ENV).env"; \
			exit 1; \
		fi; \
		if [ -z "$$REMOTE_APP_PATH" ]; then \
			echo "❌ Error: REMOTE_APP_PATH environment variable is required"; \
			echo "💡 Set in env/$(ENV).env"; \
			exit 1; \
		fi; \
		echo "🚀 Syncing app from $$LOCAL_APP_PATH to $$REMOTE_APP_PATH..."; \
		databricks sync "$$LOCAL_APP_PATH" "$$REMOTE_APP_PATH"; \
		echo "✅ App synced successfully"; \
	)

# Deploy to Databricks Apps (without building)
deploy: sync-app
	@echo "✅ Deployment complete!"
	@echo ""
	@echo "📱 Your TailAdmin app should be available in Databricks Apps"
	@echo "💡 Monitor deployment status in Databricks console"

# Complete build and deployment workflow
build-deploy: build deploy
	@echo "✅ Build and deployment workflow complete!" 