TOP_DIR := .
SRC_DIR := $(TOP_DIR)/src
TEST_DIR := $(TOP_DIR)/tests
DIST_DIR := $(TOP_DIR)/dist
LIB_NAME := dao_ai
LIB_VERSION := $(shell grep -m 1 version pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)
LIB := $(LIB_NAME)-$(LIB_VERSION)-py3-none-any.whl
TARGET := $(DIST_DIR)/$(LIB)

ifeq ($(OS),Windows_NT)
    PYTHON := py.exe
else
    PYTHON := python3
endif

UV := uv
# Everyday sync installs the COMMITTED lock verbatim (no re-resolution). This is
# what keeps the lock clean: a bare `uv sync` re-resolves against the ambient
# index and, behind the internal mirror, rewrites uv.lock with unreachable proxy
# URLs. `--frozen` installs the pinned artifacts (public CDN) without touching
# the lock, and works offline-of-the-index (only the artifact CDN is needed).
#
# `--extra all` pulls in every runtime feature extra (a2a, rerank, deepagents,
# memory, search, excel). A local dev environment is expected to be able to run
# and validate any example config, and without this an extras-dependent config
# fails with "requires the 'memory' extra, which is not installed" — a packaging
# gap rather than a real problem with the config. Deployment artifacts are
# unaffected: dao_ai._extras still selects the minimal set per config, so the
# published wheel and generated requirements stay lean.
SYNC := $(UV) sync --frozen --extra all
# Deliberate re-lock after a real dependency change. Forced to public PyPI so
# the lock never picks up the internal mirror host. NOTE: run this where
# pypi.org is reachable (Databricks sandbox / CI) — the corp mirror blocks the
# public index API, so this will fail on a mirror-bound laptop by design.
LOCK := UV_INDEX_URL=https://pypi.org/simple $(UV) lock
BUILD := $(UV) build
PYTHON := $(UV) run python
EXPORT := $(UV) pip freeze --exclude-editable | grep -v -E "(pyspark|databricks-connect)"
PUBLISH := $(UV) run twine upload
PYTEST := $(UV) run pytest -v -s --timeout=120 --timeout-method=thread
RUFF_CHECK := $(UV) run ruff check --fix --ignore E501 
RUFF_FORMAT := $(UV) run ruff format 
FIND := $(shell which find)
RM := rm -rf
CD := cd

.PHONY: all clean distclean dist check check-lock lock lock-local format publish help test unit integration

# Internal PyPI index/proxy hosts that must never appear in the committed
# uv.lock: their URLs aren't reachable outside Databricks and break
# `uv sync --frozen` / the Databricks Apps uv build path for customers. Two
# distinct internal indexes occur, depending on where the lock was generated:
#   * the corp CDN mirror (``pypi-proxy.{dev,cloud,...}.databricks.com``) — a
#     transparent passthrough of the public CDN; it poisons wheel/sdist URLs.
#   * the serverless build proxy (``node.host.local:<port>/pypi/...``) — it
#     poisons the ``source`` registry-index field; wheel URLs stay public.
# Re-lock against public PyPI (`make lock`, or `make lock-local` on-corp) if this
# guard trips. See the ADR on Apps dependency management.
LOCK_FILE := $(TOP_DIR)/uv.lock
# Extended-regex alternation of forbidden internal index/proxy hosts.
FORBIDDEN_LOCK_HOST := pypi-proxy|node\.host\.local
# `make lock-local` re-locks against the ambient internal index (for on-corp use
# when the public index API is unreachable), then rewrites the recorded internal
# host back to public infrastructure, yielding a lock equivalent to a clean-room
# re-lock. The corp mirror is a transparent passthrough of the public CDN
# (identical package paths, hashes, and upload-times — only the host differs),
# so its artifact (``.../packages/...``) URLs are host-swapped to the public CDN
# and its recorded index (``.../simple``) is normalized to the canonical public
# index; the serverless proxy poisons only the registry field, likewise
# normalized to the public index. The corp mirror has appeared under multiple
# subdomains (pypi-proxy.dev... and pypi-proxy.cloud...); the rewrite matches both.
MIRROR_LOCK_HOST := pypi-proxy\.(dev|cloud)\.databricks\.com
SERVERLESS_LOCK_HOST := node\.host\.local(:[0-9]+)?
PUBLIC_LOCK_HOST := files.pythonhosted.org
PUBLIC_LOCK_INDEX := https://pypi.org/simple

all: dist

install: depends 
	$(SYNC) 

dist:
	$(BUILD)

depends:
	@$(SYNC)

check: check-lock
	$(RUFF_CHECK) $(SRC_DIR) $(TEST_DIR)

check-lock:
	@if grep -qE "$(FORBIDDEN_LOCK_HOST)" "$(LOCK_FILE)"; then \
		echo "ERROR: $(LOCK_FILE) references an internal PyPI index/proxy (matches: $(FORBIDDEN_LOCK_HOST))."; \
		echo "       These URLs are unreachable outside Databricks and break customer installs."; \
		echo "       Re-lock against public PyPI: make lock (or make lock-local on-corp)."; \
		exit 1; \
	fi
	@echo "uv.lock is clean (no internal index/proxy references)."

# Re-resolve dependencies and regenerate uv.lock against PUBLIC PyPI after a
# real dependency change in pyproject.toml. Everyday work uses `make install`
# (frozen) — do NOT run a bare `uv sync`/`uv lock`, which re-poisons the lock
# with the internal mirror host. Run this where pypi.org is reachable, then
# `make check-lock` to confirm the result is clean before committing.
lock:
	$(LOCK)
	@$(MAKE) check-lock

# On-corp-network alternative to `make lock` for when the public index API
# (pypi.org) is blocked but the internal mirror is reachable. Re-resolves via
# the ambient mirror, then rewrites the recorded mirror host back to the public
# CDN (safe because the mirror is a transparent passthrough — see above), and
# verifies the result. Only valid for public packages; a mirror-only dependency
# would produce a URL that doesn't exist on the public CDN (caught by a later
# `uv sync --frozen`). Prefer `make lock` where pypi.org is reachable.
lock-local:
	$(UV) lock
	@sed -E -i.bak \
		-e 's#https://$(MIRROR_LOCK_HOST)/simple/?#$(PUBLIC_LOCK_INDEX)#g' \
		-e 's#https://$(MIRROR_LOCK_HOST)/#https://$(PUBLIC_LOCK_HOST)/#g' \
		-e 's#https?://$(SERVERLESS_LOCK_HOST)/pypi/v[0-9]+/simple/?#$(PUBLIC_LOCK_INDEX)#g' \
		"$(LOCK_FILE)" && rm -f "$(LOCK_FILE).bak"
	@$(MAKE) check-lock

format: check depends
	$(RUFF_FORMAT) $(SRC_DIR) $(TEST_DIR) 

publish: dist
	$(PUBLISH) $(DIST_DIR)/*

clean: 
	$(FIND) $(SRC_DIR) $(TEST_DIR) -name \*.pyc -exec rm -f {} \;
	$(FIND) $(SRC_DIR) $(TEST_DIR) -name \*.pyo -exec rm -f {} \;

distclean: clean
	$(RM) $(DIST_DIR)
	$(RM) $(SRC_DIR)/*.egg-info 
	$(RM) $(TOP_DIR)/.mypy_cache
	$(FIND) $(SRC_DIR) $(TEST_DIR) \( -name __pycache__ -a -type d \) -prune -exec rm -rf {} \;

schema: depends
	@$(UV) run --quiet python -c "from dao_ai.config import AppConfig; import json; print(json.dumps(AppConfig.model_json_schema(), indent=2))"

test: 
	$(PYTEST) -ra --tb=short $(TEST_DIR)

unit: 
	$(PYTEST) -ra --tb=short -m unit $(TEST_DIR)

integration: 
	$(PYTEST) -ra --tb=short -m integration $(TEST_DIR)

help:
	$(info TOP_DIR: $(TOP_DIR))
	$(info SRC_DIR: $(SRC_DIR))
	$(info TEST_DIR: $(TEST_DIR))
	$(info DIST_DIR: $(DIST_DIR))
	$(info )
	$(info $$> make [all|dist|install|clean|distclean|check|check-lock|lock|lock-local|format|depends|publish|schema|test|unit|integration|help])
	$(info )
	$(info       all          - build library: [$(LIB)]. This is the default)
	$(info       dist         - build library: [$(LIB)])
	$(info       install      - installs: [$(LIB)])
	$(info       uninstall    - uninstalls: [$(LIB)])
	$(info       clean        - removes build artifacts)
	$(info       distclean    - removes library)
	$(info       check        - lint source (runs check-lock first))
	$(info       check-lock   - fail if uv.lock references the internal PyPI proxy)
	$(info       lock         - re-resolve uv.lock against public PyPI (run where pypi.org is reachable))
	$(info       lock-local   - re-lock on corp network via mirror, rewrite host to public CDN)
	$(info       format       - format source code)
	$(info       depends      - installs library dependencies)
	$(info       publish      - publish library)
	$(info       schema       - print JSON schema for AppConfig)
	$(info       test         - run all tests)
	$(info       unit         - run unit tests only)
	$(info       integration  - run integration tests only)
	$(info       help         - show this help message)
	@true

