"""Validation scenarios for dao-ai OBO scope emission on both surfaces.

Each scenario builds an :class:`AppConfig` programmatically and exercises:
1. ``generate_user_api_scopes`` (the shared OBO scope generator) — confirms
   dao-ai produces the canonical OBO scope strings for that config shape.
2. ``build_auth_policy`` — confirms the Model Serving ``UserAuthPolicy``
   carries the same strings (the two surfaces are unified now).

A separate driver (``run_apps_probe.py``) then takes each scenario's
expected scope set and PATCHes a real probe app on FEVM with those exact
strings to prove the platform accepts the full set (not just individual
strings as Phase 0 already proved).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from dao_ai.config import (
    AppConfig,
    ConnectionModel,
    DatabaseModel,
    FunctionModel,
    GenieRoomModel,
    IndexModel,
    InferenceEndpointModel,
    ResourcesModel,
    SchemaModel,
    TableModel,
    VectorStoreModel,
    VolumeModel,
    WarehouseModel,
)

_SCHEMA = SchemaModel(catalog_name="retail_consumer_goods", schema_name="hardware_store")


@dataclass
class Scenario:
    """Self-contained validation case.

    Attributes:
        name: Identifier (used as app name suffix + log key).
        positive: True for "expect scopes emitted"; False for "expect ai-gateway
            absent" (the negative case for AI Gateway gating).
        build: Callable that returns the configured :class:`AppConfig`.
        expected_scopes: Set of canonical OBO scope strings the generator
            MUST produce. ``catalog.*:read`` auto-additions are added by the
            generator when tables/functions are present.
        forbidden_scopes: Strings that MUST NOT appear in the output (e.g.,
            ``ai-gateway`` for the negative scenarios).
    """

    name: str
    positive: bool
    build: Callable[[], AppConfig]
    expected_scopes: set[str] = field(default_factory=set)
    forbidden_scopes: set[str] = field(default_factory=set)


def _config(**res) -> AppConfig:
    return AppConfig(resources=ResourcesModel(**res))


SCENARIOS: list[Scenario] = [
    # ---- LLM + AI Gateway gating ----
    Scenario(
        name="llm-sp-no-gw",
        positive=True,
        build=lambda: _config(
            models={
                "claude": InferenceEndpointModel(
                    name="databricks-claude-opus-4-7",
                    on_behalf_of_user=False,
                    ai_gateway=False,
                )
            }
        ),
        expected_scopes=set(),  # SP-side LLM → no user scopes
        forbidden_scopes={"ai-gateway"},
    ),
    Scenario(
        name="llm-obo-no-gw",
        positive=True,
        build=lambda: _config(
            models={
                "claude": InferenceEndpointModel(
                    name="databricks-claude-opus-4-7",
                    on_behalf_of_user=True,
                    ai_gateway=False,
                )
            }
        ),
        expected_scopes={"serving.serving-endpoints"},
        forbidden_scopes={"ai-gateway"},
    ),
    Scenario(
        name="llm-obo-with-gw",
        positive=True,
        build=lambda: _config(
            models={
                "claude": InferenceEndpointModel(
                    name="databricks-claude-opus-4-7",
                    on_behalf_of_user=True,
                    ai_gateway=True,
                )
            }
        ),
        expected_scopes={"serving.serving-endpoints", "ai-gateway"},
    ),
    Scenario(
        name="llm-sp-with-gw-NEGATIVE",
        positive=False,
        build=lambda: _config(
            models={
                "claude": InferenceEndpointModel(
                    name="databricks-claude-opus-4-7",
                    on_behalf_of_user=False,
                    ai_gateway=True,
                )
            }
        ),
        expected_scopes=set(),  # ai-gateway not emitted without OBO
        forbidden_scopes={"ai-gateway"},
    ),
    # ---- Per-resource canonical strings + MCP companion pairing ----
    Scenario(
        name="warehouse-obo",
        positive=True,
        build=lambda: _config(
            warehouses={
                "wh": WarehouseModel(
                    name="shared",
                    warehouse_id="d1be2f7fe7faacb1",
                    on_behalf_of_user=True,
                )
            }
        ),
        expected_scopes={"sql", "mcp.functions"},
    ),
    Scenario(
        name="genie-obo",
        positive=True,
        build=lambda: _config(
            genie_rooms={
                "g": GenieRoomModel(
                    name="genie",
                    space_id="01f0000000000000",
                    on_behalf_of_user=True,
                )
            }
        ),
        expected_scopes={"genie", "mcp.genie"},
    ),
    Scenario(
        name="vector-search-obo",
        positive=True,
        build=lambda: _config(
            vector_stores={
                "vs": VectorStoreModel(
                    index=IndexModel(schema=_SCHEMA, name="products_index"),
                    on_behalf_of_user=True,
                )
            }
        ),
        expected_scopes={"vector-search", "mcp.vectorsearch", "serving.serving-endpoints"},
    ),
    Scenario(
        name="volume-obo",
        positive=True,
        build=lambda: _config(
            volumes={
                "v": VolumeModel(
                    schema=_SCHEMA, name="sample_vol", on_behalf_of_user=True
                )
            }
        ),
        expected_scopes={"files"},
    ),
    Scenario(
        name="connection-obo",
        positive=True,
        build=lambda: _config(
            connections={
                "c": ConnectionModel(
                    name="aan-test-genie-one", on_behalf_of_user=True
                )
            }
        ),
        expected_scopes={
            "catalog.connections",
            "mcp.external",
            "serving.serving-endpoints",
        },
    ),
    Scenario(
        name="lakebase-obo",
        positive=True,
        build=lambda: _config(
            databases={
                "lb": DatabaseModel(
                    project="lakebase-test", on_behalf_of_user=True
                )
            }
        ),
        expected_scopes={"postgres"},
    ),
    Scenario(
        name="table-and-function-obo",
        positive=True,
        build=lambda: _config(
            tables={
                "t": TableModel(
                    schema=_SCHEMA, name="products", on_behalf_of_user=True
                )
            },
            functions={
                "f": FunctionModel(
                    schema=_SCHEMA, name="find_product", on_behalf_of_user=True
                )
            },
        ),
        expected_scopes={
            "sql",
            "mcp.functions",
            "catalog.catalogs:read",
            "catalog.schemas:read",
            "catalog.tables:read",
        },
    ),
    # ---- Mixed: every OBO resource type, ai_gateway ON ----
    Scenario(
        name="mixed-all-obo-with-gw",
        positive=True,
        build=lambda: _config(
            models={
                "claude": InferenceEndpointModel(
                    name="databricks-claude-opus-4-7",
                    on_behalf_of_user=True,
                    ai_gateway=True,
                )
            },
            warehouses={
                "wh": WarehouseModel(
                    name="shared",
                    warehouse_id="d1be2f7fe7faacb1",
                    on_behalf_of_user=True,
                )
            },
            genie_rooms={
                "g": GenieRoomModel(
                    name="genie", space_id="01f0", on_behalf_of_user=True
                )
            },
            vector_stores={
                "vs": VectorStoreModel(
                    index=IndexModel(schema=_SCHEMA, name="products_index"),
                    on_behalf_of_user=True,
                )
            },
            volumes={
                "v": VolumeModel(schema=_SCHEMA, name="vol", on_behalf_of_user=True)
            },
            connections={
                "c": ConnectionModel(name="conn", on_behalf_of_user=True)
            },
            databases={
                "lb": DatabaseModel(project="lb", on_behalf_of_user=True)
            },
            tables={
                "t": TableModel(schema=_SCHEMA, name="products", on_behalf_of_user=True)
            },
        ),
        expected_scopes={
            "serving.serving-endpoints",
            "ai-gateway",
            "sql",
            "mcp.functions",
            "genie",
            "mcp.genie",
            "vector-search",
            "mcp.vectorsearch",
            "files",
            "catalog.connections",
            "mcp.external",
            "postgres",
            "catalog.catalogs:read",
            "catalog.schemas:read",
            "catalog.tables:read",
        },
    ),
]


def get_scenario(name: str) -> Scenario:
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(f"no scenario named {name}")
