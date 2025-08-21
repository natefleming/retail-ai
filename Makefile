TOP_DIR := .
SRC_DIR := $(TOP_DIR)/src
TEST_DIR := $(TOP_DIR)/tests
DIST_DIR := $(TOP_DIR)/dist
REQUIREMENTS_FILE := $(TOP_DIR)/requirements.txt
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
SYNC := $(UV) sync 
BUILD := $(UV) build 
PYTHON := $(UV) run python 
EXPORT := $(UV) pip freeze --exclude-editable | grep -v -E "(databricks-vectorsearch|pyspark|databricks-connect)" 
PUBLISH := $(UV) run twine upload
PYTEST := $(UV) run pytest -v -s
RUFF_CHECK := $(UV) run ruff check --fix --ignore E501 
RUFF_FORMAT := $(UV) run ruff format 
FIND := $(shell which find)
RM := rm -rf
CD := cd

.PHONY: all clean distclean dist check format publish help 

all: dist

install: depends 
	$(SYNC) 

dist: install
	$(BUILD)

depends: 
	@$(SYNC) 
	@$(EXPORT) > $(REQUIREMENTS_FILE)

check: 
	$(RUFF_CHECK) $(SRC_DIR) $(TEST_DIR) 

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
	@$(PYTHON) -c "from dao_ai.config import AppConfig; import json; print(json.dumps(AppConfig.model_json_schema(), indent=2))"

test: 
	$(PYTEST) $(TEST_DIR)

help:
	$(info TOP_DIR: $(TOP_DIR))
	$(info SRC_DIR: $(SRC_DIR))
	$(info TEST_DIR: $(TEST_DIR))
	$(info DIST_DIR: $(DIST_DIR))
	$(info )
	$(info $$> make [all|dist|install|clean|distclean|format|depends|publish|schema|test|help])
	$(info )
	$(info       all          - build library: [$(LIB)]. This is the default)
	$(info       dist         - build library: [$(LIB)])
	$(info       install      - installs: [$(LIB)])
	$(info       uninstall    - uninstalls: [$(LIB)])
	$(info       clean        - removes build artifacts)
	$(info       distclean    - removes library)
	$(info       format       - format source code)
	$(info       depends      - installs library dependencies)
	$(info       publish      - publish library)
	$(info       schema       - print JSON schema for AppConfig)
	$(info       test         - run tests)
	$(info       help         - show this help message)
	@true

