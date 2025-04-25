TOP_DIR := .
SRC_DIR := $(TOP_DIR)/retail_ai
DIST_DIR := $(TOP_DIR)/dist
TEST_DIR := $(TOP_DIR)/tests
LIB_NAME := retail_ai
LIB_VERSION := $(shell grep -m 1 version pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)
LIB := $(LIB_NAME)-$(LIB_VERSION)-py3-none-any.whl
TARGET := $(DIST_DIR)/$(LIB)

ifeq ($(OS),Windows_NT)
    PYTHON := py.exe
else
    PYTHON := python3
endif

UV := uv
UV_SYNC := $(UV) sync 
UV_BUILD := $(UV) build 
PYTEST := pytest -xvs
RUFF := $(UV) run ruff check --fix --ignore E501
ISORT := $(UV) run isort
FIND := $(shell which find)
RM := rm -rf
CD := cd

.PHONY: all clean distclean dist format help 

all: dist

install: depends 
	$(UV_SYNC) 

dist: install
	$(UV_BUILD)

depends: 
	$(UV_SYNC) 

format: depends
	$(ISORT) $(SRC_DIR) 
	$(RUFF) $(SRC_DIR) 

test:
	$(PYTEST) $(TEST_DIR)

clean: 
	$(FIND) $(SRC_DIR) $(TEST_DIR) -name \*.pyc -exec rm -f {} \;
	$(FIND) $(SRC_DIR) $(TEST_DIR) -name \*.pyo -exec rm -f {} \;


distclean: clean
	$(RM) $(DIST_DIR)
	$(RM) $(SRC_DIR)/*.egg-info 
	$(RM) $(TOP_DIR)/.mypy_cache
	$(FIND) $(SRC_DIR) \( -name __pycache__ -a -type d \) -prune -exec rm -rf {} \;

test: 
	$(PYTEST) $(TEST_DIR)

help:
	$(info TOP_DIR: $(TOP_DIR))
	$(info SRC_DIR: $(SRC_DIR))
	$(info TEST_DIR: $(TEST_DIR))
	$(info DIST_DIR: $(DIST_DIR))
	$(info )
	$(info $$> make [all|dist|install|clean|distclean|format|depends|test])
	$(info )
	$(info       all          - build library: [$(LIB)]. This is the default)
	$(info       dist         - build library: [$(LIB)])
	$(info       install      - installs: [$(LIB)])
	$(info       uninstall    - uninstalls: [$(LIB)])
	$(info       clean        - removes build artifacts)
	$(info       distclean    - removes library)
	$(info       format       - format source code)
	$(info       depends      - installs library dependencies)
	$(info       test         - run tests)
	$(info       help         - show this help message)
	$(info )
	@true

