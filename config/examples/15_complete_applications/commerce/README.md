# Commerce — reference commerce agent for B2C + B2B (LangGraph Commerce Agent v2.1)

A **self-contained** implementation of the LangGraph Commerce Agent v2.1 reference architecture on dao-ai, packaged so that everything a variant needs — configs, tables, functions, seed data — lives under this one directory tree.

## Variants

| Variant | Orchestration | Config | README |
|---|---|---|---|
| **Swarm** | 11-agent pipeline (supervisor → planner → handler/UCP → composer) with LLM-routed planner and direct handler → composer handoff. | [`commerce_swarm/commerce_swarm.yaml`](commerce_swarm/commerce_swarm.yaml) | [Details](commerce_swarm/README.md) |
| **Supervisor** | Supervisor + planner + specialists variant that keeps the same agents but routes them through a central supervisor. | [`commerce_supervisor/commerce_supervisor.yaml`](commerce_supervisor/commerce_supervisor.yaml) | [Details](commerce_supervisor/README.md) |

Both variants target the same Unity Catalog schema (`retail_consumer_goods.commerce_swarm`), share the same Delta tables, and register the same Unity Catalog functions — so the provisioning steps below are identical across variants.

## Directory layout

```
commerce/
├── README.md                       # this file
├── data/                           # DDL + seed data for the 10 commerce tables (products, orders, inventory, …)
├── functions/                      # UC function DDL (find_product, get_order_history, check_stock, …)
├── commerce_swarm/
│   ├── commerce_swarm.yaml
│   └── README.md
└── commerce_supervisor/
    ├── commerce_supervisor.yaml
    └── README.md
```

## Deploy

From the `dao-ai` repo root, pick a variant and run:

```bash
# Swarm variant
dao-ai pipeline --deploy --run \
  -c config/examples/15_complete_applications/commerce/commerce_swarm/commerce_swarm.yaml \
  -p <profile> --deployment-target apps

# Supervisor variant
dao-ai pipeline --deploy --run \
  -c config/examples/15_complete_applications/commerce/commerce_supervisor/commerce_supervisor.yaml \
  -p <profile> --deployment-target apps
```

The pipeline runs five stages against Databricks:

1. **provision-lakebase** — creates the `commerce-swarm` Lakebase project (autoscaling_min_cu: 0, scale-to-zero).
2. **ingest-and-transform** — creates the tables under `retail_consumer_goods.commerce_swarm` and loads seed rows from `data/`.
3. **unity-catalog-tools** — registers the six UC functions from `functions/`.
4. **provision-vector-search** — builds three Delta-Sync indexes (products, FAQs, policies) on the shared VS endpoint.
5. **deploy-agents** — deploys the agent as a Databricks App.

## Iterating locally

```bash
# Schema-validate the config
dao-ai validate -c config/examples/15_complete_applications/commerce/commerce_swarm/commerce_swarm.yaml

# Chat locally against deployed backing resources
dao-ai chat -c config/examples/15_complete_applications/commerce/commerce_swarm/commerce_swarm.yaml
```
