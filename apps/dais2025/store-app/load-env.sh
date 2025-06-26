#!/bin/bash
# load-env.sh - Simple script to load environment variables from .env.local
# Usage: source load-env.sh

# Check if .env.local exists
if [ ! -f .env.local ]; then
    echo "❌ .env.local not found in current directory"
    if [ -f .env.example ]; then
        echo "💡 Found .env.example template. Run 'make create-env-local' to create .env.local"
        echo "   Or manually: cp .env.example .env.local"
    else
        echo "💡 Run 'make create-env-local' to create .env.local, or check you're in the project root"
    fi
    return 1 2>/dev/null || exit 1
fi

# Load environment variables
echo "🚀 Loading environment variables from .env.local..."
set -a  # automatically export all variables
source .env.local
set +a  # turn off automatic export

# Show what was loaded (with redacted sensitive values)
echo "✅ Environment variables loaded!"
echo ""
echo "📋 Loaded variables:"
cat .env.local | grep -v '^#' | grep -v '^$' | sed -E 's/(TOKEN|PASSWORD|SECRET|KEY)=.*/\1=[REDACTED]/' | sed 's/^/  /'
echo ""
echo "💡 Environment is ready for use!" 