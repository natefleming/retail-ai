#!/bin/bash

# Documentation setup script for Retail AI
# This script sets up the MkDocs documentation system

set -e

echo "🚀 Setting up Retail AI Documentation..."
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Install documentation dependencies
echo "📦 Installing documentation dependencies..."
if [ -f "requirements-docs.txt" ]; then
    pip install -r requirements-docs.txt
    echo "✅ Documentation dependencies installed"
else
    echo "❌ requirements-docs.txt not found"
    exit 1
fi

# Create missing directories
echo "📁 Creating documentation directories..."
mkdir -p docs/{getting-started,architecture,tools,guides,development,api,examples}
echo "✅ Documentation directories created"

# Test MkDocs build
echo "🔨 Testing documentation build..."
if mkdocs build --quiet; then
    echo "✅ Documentation builds successfully"
else
    echo "❌ Documentation build failed"
    exit 1
fi

# Clean up build artifacts
rm -rf site/

echo ""
echo "🎉 Documentation setup complete!"
echo ""
echo "Next steps:"
echo "  1. Serve locally:    mkdocs serve"
echo "  2. Build for prod:   mkdocs build"
echo "  3. Deploy to GitHub: mkdocs gh-deploy"
echo ""
echo "Or use the Makefile commands:"
echo "  make docs-serve"
echo "  make docs-build"
echo "  make docs-deploy"
echo ""
echo "Documentation will be available at: http://localhost:8000" 