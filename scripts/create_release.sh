#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Create a GitHub release for dao-ai based on the version in pyproject.toml.

Steps performed:
  1. Extract version from pyproject.toml
  2. Create a release/v{version} branch from current HEAD
  3. Push the branch to origin
  4. Create a GitHub release with auto-generated notes

Options:
  -h, --help      Show this help message and exit
  -n, --dry-run   Print what would be done without making any changes
EOF
}

DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v gh &>/dev/null; then
    echo "Error: gh CLI is not installed. Install it from https://cli.github.com/" >&2
    exit 1
fi

if ! gh auth status &>/dev/null; then
    echo "Error: gh CLI is not authenticated. Run 'gh auth login' first." >&2
    exit 1
fi

VERSION=$(grep '^version' pyproject.toml | head -1 | sed 's/version *= *"\(.*\)"/\1/')

if [[ -z "$VERSION" ]]; then
    echo "Error: Could not extract version from pyproject.toml" >&2
    exit 1
fi

TAG="v${VERSION}"
BRANCH="release/${TAG}"

echo "Detected version: ${VERSION}"
echo "Tag: ${TAG}"
echo "Branch: ${BRANCH}"

if git rev-parse "$TAG" &>/dev/null; then
    echo "Error: Tag ${TAG} already exists." >&2
    exit 1
fi

if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    echo "Error: Branch ${BRANCH} already exists locally." >&2
    exit 1
fi

if $DRY_RUN; then
    echo "[dry-run] Would create branch ${BRANCH}"
    echo "[dry-run] Would push branch ${BRANCH} to origin"
    echo "[dry-run] Would create GitHub release ${TAG} targeting ${BRANCH}"
    exit 0
fi

echo "Creating branch ${BRANCH}..."
git checkout -b "$BRANCH"

echo "Pushing branch ${BRANCH} to origin..."
git push -u origin "$BRANCH"

echo "Creating GitHub release ${TAG}..."
gh release create "$TAG" \
    --title "$TAG" \
    --generate-notes \
    --target "$BRANCH"

echo "Release ${TAG} created successfully."
echo "Branch: ${BRANCH}"
echo "URL: $(gh release view "$TAG" --json url --jq .url)"
