# 21. Provisioning From Git

**A complete project loaded from a git locator — no clone, no upload, nothing on your machine**

Every asset this config references lives beside it in the repository. Point
dao-ai at a git locator and the checkout supplies the DDL, the seed data, the UC
function SQL, the skill markdown, and two flavours of Python tool code. A user
with no dao-ai project checked out anywhere can provision the whole thing from a
notebook cell.

That makes this example the end-to-end test for config-relative resolution: each
piece of the layout exercises exactly one anchor, and if any anchor resolved
against the wrong directory the corresponding resource would be missing at
provision time or the tool absent at inference time.

## What each piece proves

| Asset | Config field | Anchor being tested | Symptom if it breaks |
|---|---|---|---|
| `data/products.sql` + `data/products.csv` | `datasets[].ddl` / `.data` | Config directory in the checkout, **plus** volume staging for Spark | `products` table missing or empty |
| `data/store_hours.sql` + `.json` | `datasets[].ddl` / `.data` | Same anchor, driver-side pandas read (no staging) | `store_hours` missing or empty |
| `functions/*.sql` | `unity_catalog_functions[].ddl` | Config directory in the checkout | UC function not created |
| `skills/from_git/product-lookup/SKILL.md` | `resources.skills[].path` | `_skill_base_dir` | Agent silently loses its workflow instructions |
| `src/from_git_tools/` | *(none — auto-discovered)* | Colocated `src/` on `sys.path` | `find_aisle` tool fails to import |
| `from_git_lib/` | `app.code_paths` | `code_paths` entry's **parent** on `sys.path` | `apply_contractor_discount` tool fails to import |

The last three were genuinely broken for `AppConfig.from_git` until they were
anchored on the checkout path rather than the locator string — a locator like
`gh:org/repo@main#examples/21_from_git/from_git.yaml` has no parent directory, so
resolution fell through to the working directory and produced no error, just
missing tools.

## Layout

```
examples/21_from_git/
├── from_git.yaml                          # the config
├── data/
│   ├── products.sql                       # DDL, run with USE IDENTIFIER(:database)
│   ├── products.csv                       # csv seed  → staged to a UC volume for Spark
│   ├── store_hours.sql
│   └── store_hours.json                   # json seed → read on the driver with pandas
├── functions/
│   ├── find_product_by_sku.sql            # {catalog_name}.{schema_name} placeholders
│   └── find_store_hours_by_city.sql
├── skills/from_git/product-lookup/
│   └── SKILL.md
├── src/from_git_tools/                    # colocated src/ → imports prefix-free
│   └── aisle.py                           # `type: python`  (a tool object)
└── from_git_lib/                          # app.code_paths → keeps its package prefix
    └── pricing.py                         # `type: factory` (called with args:)
```

## Run it

### From a notebook (nothing checked out)

[`notebooks/08_provision_from_git.py`](../../notebooks/08_provision_from_git.py)
is the runnable version — it takes a git locator as a widget, provisions every
resource in dependency order, exercises the tools in-process, and deploys to both
Model Serving and Apps.

```python
%uv pip install dao-ai
%restart_python
```

```python
from dao_ai.config import AppConfig

config = AppConfig.from_git(
    "gh:natefleming/dao-ai@main#examples/21_from_git/from_git.yaml",
    params={"catalog": "my_catalog", "schema": "dao_ai_from_git"},
)
```

See [docs/python-api.md → Provisioning a whole project from a
notebook](../../docs/python-api.md#provisioning-a-whole-project-from-a-notebook)
for the provisioning loop and where the checkout lands.

### From the CLI

```bash
# Quote the locator — `#` starts a comment in every common shell.
dao-ai validate -c 'gh:natefleming/dao-ai@main#examples/21_from_git/from_git.yaml'

dao-ai workflow up -c 'gh:natefleming/dao-ai@main#examples/21_from_git/from_git.yaml' \
  -p my-profile --param catalog=my_catalog --param schema=dao_ai_from_git
```

### From a local clone

Identical, because the anchors are the same either way:

```bash
dao-ai workflow up -c examples/21_from_git/from_git.yaml -p my-profile
```

## Why two seed formats

Serverless Spark executors cannot read the driver-local checkout, so a
`csv` / `parquet` / `orc` / `delta` seed is copied into a managed volume
`<catalog>.<schema>.dao_ai_staging` and Spark is handed the `/Volumes/...` path.
`json` and `excel` are read on the driver with pandas and are never staged. This
example ships one of each so both paths are covered — after provisioning you
should see a `dao_ai_staging` volume you never declared, alongside the
`dao_ai_from_git_assets` volume you did.

## Validating a run

```sql
SELECT COUNT(*) FROM my_catalog.dao_ai_from_git.products;      -- 12
SELECT COUNT(*) FROM my_catalog.dao_ai_from_git.store_hours;   -- 6

SELECT * FROM my_catalog.dao_ai_from_git.find_product_by_sku(ARRAY('DRL10045'));
SELECT * FROM my_catalog.dao_ai_from_git.find_store_hours_by_city('Chicago');

SHOW VOLUMES IN my_catalog.dao_ai_from_git;   -- dao_ai_from_git_assets, dao_ai_staging
```

One prompt covers the rest, because it needs a different anchor per tool:

> I'm a contractor — what does SKU DRL10045 cost me, where do I find it, and is
> the Chicago store open on Sunday?

- the price comes from `find_product_by_sku_uc` (colocated DDL),
- the aisle from `find_aisle` (colocated `src/`) — its reply includes the module's
  own file path, so you can read the provenance straight out of the answer,
- the contractor price from `apply_contractor_discount` (`code_paths` + `args:`),
- the hours from `find_store_hours_by_city_uc` (the json-seeded table),
- and the card formatting from `SKILL.md`.

Ask it in a notebook, against the Model Serving endpoint, and in the deployed
app: all three have to carry the repository's code, and a wrong answer in only
one of them points at packaging rather than at resolution.

## Cleanup

The schema is dedicated so the whole example drops in one statement:

```sql
DROP SCHEMA IF EXISTS my_catalog.dao_ai_from_git CASCADE;
```

Delete the vector index and the serving endpoint separately — they are not owned
by the schema:

```bash
databricks vector-search-indexes delete my_catalog.dao_ai_from_git.products_index
databricks serving-endpoints delete from_git_demo_dao
```

## See also

- [docs/python-api.md](../../docs/python-api.md) — `from_git` / `from_source`, `cache_dir`
- [docs/cli-reference.md](../../docs/cli-reference.md) — locator grammar, `--from`, caching, private repos
- [`13_orchestration`](../13_orchestration/) — skills in depth, including volume-backed skills
- [`14_basic_tools`](../14_basic_tools/) — `type: python` and `type: factory` tools on their own
