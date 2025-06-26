# Documentation Makefile
# Commands for building, serving, and deploying documentation

.PHONY: help install serve build deploy clean setup check

# Default target
help:
	@echo "📚 Documentation Commands:"
	@echo "  install   - Install documentation dependencies"
	@echo "  setup     - Initial setup for documentation"
	@echo "  serve     - Serve documentation locally (http://localhost:8000)"
	@echo "  build     - Build documentation for production"
	@echo "  deploy    - Deploy documentation to GitHub Pages"
	@echo "  clean     - Clean documentation build artifacts"
	@echo "  check     - Validate documentation links and structure"

# Install documentation dependencies
install:
	@echo "📦 Installing documentation dependencies..."
	pip install -r ../requirements-docs.txt

# Setup documentation environment
setup: install
	@echo "🔧 Setting up documentation environment..."
	@if [ ! -f mkdocs.yml ]; then \
		echo "❌ mkdocs.yml not found in docs directory"; \
		exit 1; \
	fi

# Serve documentation locally
serve: setup
	@echo "🚀 Serving documentation locally..."
	mkdocs serve -f mkdocs.yml

# Build documentation
build: setup
	@echo "🏗️  Building documentation..."
	mkdocs build -f mkdocs.yml

# Deploy to GitHub Pages
deploy: setup
	@echo "🚀 Deploying documentation to GitHub Pages..."
	mkdocs gh-deploy -f mkdocs.yml

# Clean build artifacts
clean:
	@echo "🧹 Cleaning documentation build artifacts..."
	rm -rf site/
	@echo "✅ Documentation artifacts cleaned"

# Check documentation
check: setup
	@echo "🔍 Checking documentation..."
	@# Check for broken links (requires linkchecker)
	@if command -v linkchecker >/dev/null 2>&1; then \
		mkdocs build -f mkdocs.yml; \
		linkchecker site/; \
	else \
		echo "💡 Install linkchecker for link validation: pip install linkchecker"; \
	fi
	@echo "✅ Documentation check complete" 