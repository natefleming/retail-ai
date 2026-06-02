"""Shared helpers for dao_ai.mcp tests.

This is NOT a ``conftest.py`` on purpose: dao-ai's sibling test files do
``from conftest import has_databricks_env`` and rely on the rootdir
``tests/conftest.py`` being unambiguously on sys.path. Adding a subdirectory
``conftest.py`` here would shadow that resolution and break unrelated tests.
Instead, the MCP tests import ``mcp_config`` from this module explicitly.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SAMPLE_YAML = """
parameters:
  retail_space_id:
    description: "retail genie space"
  inventory_space_id:
    description: "inventory genie space"
  warehouse_id:
    description: "shared warehouse"
  lakebase_project:
    default: "test-lakebase"
    description: "lakebase project"
  lakebase_branch:
    default: "production"
    description: "lakebase branch"
  vs_endpoint:
    description: "vector search endpoint"
  vs_index:
    description: "vector search index full name"

resources:
  models:
    embedding_model: &embedding_model
      name: databricks-gte-large-en

  warehouses:
    default: &wh
      warehouse_id: ${var.warehouse_id}

  databases:
    lakebase: &db
      project: ${var.lakebase_project}
      branch: ${var.lakebase_branch}

  genie_rooms:
    retail: &retail
      space_id: ${var.retail_space_id}
    inventory: &inventory
      space_id: ${var.inventory_space_id}

  vector_stores:
    products: &products
      embedding_model: *embedding_model
      index:
        name: ${var.vs_index}
      endpoint:
        name: ${var.vs_endpoint}

retrievers:
  products_retriever: &products_retriever
    vector_store: *products
    columns: [product_id, brand_name, product_name]
    search_parameters:
      num_results: 10
      query_type: HYBRID

tools:
  product_vector_search:
    name: product_vector_search
    function:
      type: factory
      name: dao_ai.tools.create_vector_search_tool
      args:
        retriever: *products_retriever
        name: product_vector_search
        description: "Search the product catalog."

  ask_retail:
    name: ask_retail
    function:
      type: factory
      name: dao_ai.tools.create_genie_toolkit
      args:
        name: ask_retail
        description: "Ask about retail products."
        genie_room: *retail
        lru_cache_parameters: &lru
          warehouse: *wh
          capacity: 50
          time_to_live_seconds: 600

  ask_inventory:
    name: ask_inventory
    function:
      type: factory
      name: dao_ai.tools.create_genie_toolkit
      args:
        name: ask_inventory
        description: "Ask about inventory levels."
        genie_room: *inventory
        lru_cache_parameters: *lru
"""


@contextmanager
def mcp_config(tmp_path: Path) -> Iterator[str]:
    """Write a synthetic MCP-only config YAML and set the env vars its refs need."""
    path = tmp_path / "dao_ai_mcp.yaml"
    path.write_text(SAMPLE_YAML)
    prior = {
        "RETAIL_SPACE_ID": os.environ.get("RETAIL_SPACE_ID"),
        "INVENTORY_SPACE_ID": os.environ.get("INVENTORY_SPACE_ID"),
        "WAREHOUSE_ID": os.environ.get("WAREHOUSE_ID"),
        "VS_ENDPOINT": os.environ.get("VS_ENDPOINT"),
        "VS_INDEX": os.environ.get("VS_INDEX"),
    }
    os.environ["RETAIL_SPACE_ID"] = "01f00000000000000000000000000001"
    os.environ["INVENTORY_SPACE_ID"] = "01f00000000000000000000000000002"
    os.environ["WAREHOUSE_ID"] = "wh-test"
    os.environ["VS_ENDPOINT"] = "test-vs-endpoint"
    os.environ["VS_INDEX"] = "main.test.products_index"
    try:
        yield str(path)
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
