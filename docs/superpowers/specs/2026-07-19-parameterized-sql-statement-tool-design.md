# Parameterized SQL statement tool (Warehouse + Lakebase)

**Date:** 2026-07-19
**Status:** Approved (design)
**Area:** `src/dao_ai/tools/sql.py`, `src/dao_ai/config.py`

## Problem

`create_execute_statement_tool` (`src/dao_ai/tools/sql.py:18`) generates a tool
that runs a single SQL statement fixed at configuration time. The generated tool
exposes **no parameters** to the LLM — its only argument is the framework-injected
`runtime`. It also targets Databricks **SQL warehouses only**; there is no
equivalent for running a statement against a **Lakebase / Postgres** database,
even though `DatabaseModel` already exposes an async connection pool and
`aexecute_query`.

We want three things:

1. Allow a statement to declare **optional parameters** whose values are bound at
   runtime — some supplied by the **LLM**, some pulled from the runtime
   **`Context`** (mix per parameter).
2. Let the same factory target either a **`WarehouseModel`** (existing path) or a
   **`DatabaseModel`** (new Lakebase path), dispatching on type.
3. Expose a **first-class `type: sql`** config model (`SqlToolModel`) analogous to
   `type: genie` / `type: ai_search`, so the tool is declaratively configurable
   with typed fields and JSON-schema validation — not only via `type: factory`.

This gives dao-ai a parameterized-statement tool for both backends, analogous to
how UC functions are exposed as tools — but for author-controlled SQL with typed,
safely-bound parameters.

## Scope

**In scope**
- Backend dispatch: one public factory delegating to two public backend factories.
- Typed, optional parameters with per-param value source (LLM vs. Context).
- Native bound parameters for both backends (no string interpolation).
- Consistent text-table result output across both backends.
- First-class `type: sql` config model (`SqlToolModel`) in the `AnyTool` union,
  plus `schemas/model_config_schema.json` regeneration via `make schema`.
- Backward compatibility with existing callers and tests.
- Fix stale docstring references to the non-existent `create_execute_sql_tool`.

**Out of scope (flagged)**
- No SQL translation between placeholder styles — the author writes SQL in the
  native placeholder syntax of their chosen backend.
- No read-only / SELECT-only enforcement, row limits, or new statement timeouts.
  The SQL is author-controlled (fixed at config time); this is not open-ended
  LLM-authored SQL. Guardrails are a separate future concern.

## Design

### Public factory API (`src/dao_ai/tools/sql.py`)

```python
def create_execute_statement_tool(
    target: WarehouseModel | DatabaseModel | dict,
    statement: str,
    params: list[StatementParam] | None = None,
    name: str = "execute_sql_tool",
    description: str | None = None,
) -> tool: ...

def create_warehouse_statement_tool(
    warehouse: WarehouseModel | dict,
    statement: str,
    params: list[StatementParam] | None = None,
    name: str = "execute_sql_tool",
    description: str | None = None,
) -> tool: ...

def create_lakebase_statement_tool(
    database: DatabaseModel | dict,
    statement: str,
    params: list[StatementParam] | None = None,
    name: str = "execute_sql_tool",
    description: str | None = None,
) -> tool: ...
```

`create_execute_statement_tool` coerces a `dict` to the correct model, then
uses `isinstance` narrowing on the `WarehouseModel | DatabaseModel` union to
delegate to the matching public backend factory. Unknown types raise
`ValueError`. Both backend factories are public so callers who already know
their target can skip the dispatch.

### Parameter declaration model

```python
class ParamSource(str, Enum):
    LLM = "llm"          # value supplied by the model; appears in the tool schema
    CONTEXT = "context"  # value pulled from runtime Context; never seen by the model

class StatementParam(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str                                       # marker name used in the SQL
    type: Literal["string", "int", "float", "bool"] = "string"
    source: ParamSource = ParamSource.LLM           # defaults to LLM (common case)
    required: bool = True
    default: Any | None = None
    description: str | None = None                  # feeds the LLM tool schema
    context_key: str | None = None                  # source=context; defaults to name
```

### Parameter flow at runtime

- The inner tool accepts LLM-supplied parameters plus the injected `runtime`.
  Only `source == LLM` params are surfaced in the tool's `args_schema`
  (built from each LLM param's `name`, `type`, `required`, `description`) so the
  model sees a properly typed schema. When there are no LLM params, the tool
  takes no model-facing arguments (preserving today's zero-arg behavior).
- **LLM params:** read from the model-supplied values; enforce `required`;
  apply `default` when absent and optional.
- **Context params:** resolved via `getattr(runtime.context, context_key or name, default)`.
  `Context` is `extra="allow"`, so arbitrary attributes resolve. If a `required`
  context param is missing (and has no default), return an error string.
- All resolved values are assembled into a single `{name: value}` map, then bound
  natively per backend.

### Backend execution

**Warehouse** (`create_warehouse_statement_tool`) — sync inner tool, unchanged
path plus parameters:
- SQL author uses named markers: `:name`.
- Bind via `parameters=[StatementParameterListItem(name=k, value=str(v)) ...]`
  passed to `workspace_client.statement_execution.execute_statement(...)`.
  (`StatementParameterListItem` fields: `name`, `type`, `value`.)
- Existing polling, status handling, and text-table formatting retained.

**Lakebase** (`create_lakebase_statement_tool`) — `async def` inner tool:
- SQL author uses psycopg markers: `%(name)s`.
- Execute via the existing `await database.aexecute_query(statement, {name: value})`,
  which returns `list[dict]` (async pool uses `row_factory=dict_row`).
- Format that `list[dict]` into the **same** text table the warehouse path emits
  (`"col | col"` header, dashed rule, rows, `"(N rows returned)"` footer) so tool
  output is consistent regardless of backend. Empty result → the same
  "no results" / "empty result set" style messages.

### Placeholder syntax

The two backends use different native placeholder syntaxes (`:name` for the
warehouse SDK, `%(name)s` for psycopg). The SQL author writes the statement in
the syntax matching the configured `target`. This is documented in the factory
docstrings. No translation layer is introduced (keeps us off the error-prone
rewriting path; native binding is injection-safe on both backends).

### Result formatting

A shared helper formats `columns` + `rows` into the text table used today, so
both backends produce identical output shape. The warehouse path feeds it column
names from the response manifest and `result.data_array`; the Lakebase path feeds
it `list(dict.keys())` and `list(dict.values())` from `aexecute_query`.

### First-class config model (`src/dao_ai/config.py`)

Follows the established "typed wrapper over a factory" pattern used by
`GenieToolModel` (`config.py:6282`) and `LakebaseSearchToolModel` (`config.py:6444`).

1. Add `SQL = "sql"` to the `FunctionType` enum (`config.py:5349`).
2. Add `SqlToolModel(BaseFunctionModel)`:

```python
class SqlToolModel(BaseFunctionModel):
    """First-class SQL tool that delegates to
    ``dao_ai.tools.sql.create_execute_statement_tool``.

    Equivalent to ``type: factory + name:
    dao_ai.tools.sql.create_execute_statement_tool``, but with typed fields.
    Runs a fixed SQL statement (with optional bound parameters) against a SQL
    warehouse or a Lakebase / Postgres database.
    """
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.SQL] = Field(
        default=FunctionType.SQL,
        description="Function type discriminator. Must be 'sql'.",
    )
    warehouse: Optional[WarehouseModel] = Field(
        default=None,
        description="SQL warehouse to run the statement against. Mutually "
                    "exclusive with 'database'; exactly one is required.",
    )
    database: Optional[DatabaseModel] = Field(
        default=None,
        description="Lakebase / Postgres database to run the statement against. "
                    "Mutually exclusive with 'warehouse'; exactly one is required.",
    )
    statement: str = Field(
        description="SQL statement to execute. Use ':name' markers for a warehouse "
                    "target, '%(name)s' markers for a Lakebase/Postgres target.",
    )
    params: Optional[list[StatementParam]] = Field(
        default=None,
        description="Optional bound parameters. LLM-sourced params appear in the "
                    "tool schema; context-sourced params bind from runtime Context.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Tool name visible to the LLM. Defaults to 'execute_sql_tool'.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Tool description shown to the LLM during function calling.",
    )

    @model_validator(mode="after")
    def _require_exactly_one_target(self) -> Self:
        if bool(self.warehouse) == bool(self.database):
            raise ValueError(
                "SqlToolModel requires exactly one of 'warehouse' or 'database'."
            )
        return self

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools.sql import create_execute_statement_tool
        return [create_execute_statement_tool(
            target=self.warehouse or self.database,
            statement=self.statement,
            params=self.params,
            name=self.name or "execute_sql_tool",
            description=self.description,
        )]
```

3. Add `SqlToolModel` to the `AnyTool` union (`config.py:6849`).
4. Regenerate `schemas/model_config_schema.json` via `make schema` (per repo
   convention for any config model change).

The backend is expressed as separate mutually-exclusive `warehouse:` / `database:`
fields (not a single union), so the YAML call site names its backend explicitly.
A `model_validator(mode="after")` enforces that exactly one is set; `as_tools()`
passes whichever is present to the factory (the factory still dispatches on type).
This mirrors `AiSearchToolModel`'s mutually-exclusive `retriever` / `vector_store`
precedent. Because `SqlToolModel` subclasses `BaseFunctionModel`, it automatically
inherits `human_in_the_loop`, `audit`, and `call_limit` support — e.g. a mutating
statement can be gated behind human approval with no extra work.

### `StatementParam` location

`StatementParam` and `ParamSource` are defined in `src/dao_ai/tools/sql.py` and
imported into `config.py` for `SqlToolModel`. This keeps the parameter model next
to the factory that consumes it and avoids a config→tools dependency inversion.
(If an import cycle arises, fall back to defining them in `config.py` and
importing into `sql.py`; verify during implementation.)

## Naming

- Keep the **"statement"** family — the tool runs any SQL (including DML/DDL), so
  "query" would be misleadingly read-only. Matches the Databricks SDK
  `statement_execution` / `StatementResponse` / `StatementParameterListItem`
  surface.
- Default LLM-facing alias stays **`"execute_sql_tool"`** (unchanged), preserving
  backward compatibility with any config relying on the default name.
- First-class config discriminator is **`type: sql`** (concise, matches
  `genie` / `ai_search` / `lakebase_search`).
- Fix the stale docstring references to `create_execute_sql_tool`
  (`sql.py:45,54`) — that name never existed.

## Backward compatibility

- `params=None` → no LLM-facing args; identical to today's zero-arg tool.
- Existing `(warehouse, statement)` callers keep working: `WarehouseModel`
  dispatches to the warehouse path, which is behaviorally unchanged aside from
  optional parameter binding.
- The default alias is unchanged (`execute_sql_tool`), so existing tests
  asserting the default name continue to pass.

## Example configuration

First-class `type: sql`, warehouse target, LLM-supplied parameter:

```yaml
resources:
  warehouses:
    shared_warehouse: &shared_warehouse
      name: "Shared Warehouse"
      warehouse_id: ${var.warehouse_id}
      on_behalf_of_user: true

tools:
  store_lookup:
    name: store_lookup
    function:
      type: sql
      warehouse: *shared_warehouse
      statement: >
        SELECT store_id, name, city, active
        FROM retail.ops.stores
        WHERE store_id = :store_id
      description: "Look up a store by its id."
      params:
        - name: store_id
          type: int
          description: "The numeric store identifier to look up."
```

Lakebase target, mixed LLM + context params (`store_num` bound from `Context`):

```yaml
resources:
  databases:
    retail_database: &retail_database
      name: "Retail Database"
      project: "retail-consumer-goods"
      on_behalf_of_user: true

tools:
  category_inventory:
    name: category_inventory
    function:
      type: sql
      database: *retail_database
      statement: >
        SELECT product_id, product_name, on_hand
        FROM inventory
        WHERE store_num = %(store_num)s AND category = %(category)s
        ORDER BY on_hand ASC
      description: "List on-hand inventory for a product category at the current store."
      params:
        - name: category
          type: string
          description: "Product category to filter by (e.g. 'paint', 'plumbing')."
        - name: store_num
          source: context      # pulled from runtime Context, not the LLM
          type: int
          # context_key defaults to the param name ("store_num")
```

## Testing

Extend `tests/dao_ai/test_sql_tool.py` and `test_sql_tool_integration.py`:

- Dispatch: `WarehouseModel` → warehouse path; `DatabaseModel` → Lakebase path;
  `dict` coercion for both; unknown type → `ValueError`.
- LLM params: present in `args_schema` with correct types; required enforcement;
  default application; values bound (assert `StatementParameterListItem` list for
  warehouse, `{name: value}` dict passed to a mocked `aexecute_query` for Lakebase).
- Context params: resolved from a stubbed `Context`; missing required → error
  string; not present in the LLM schema.
- Mixed LLM + context params in one statement.
- Result formatting parity: warehouse and Lakebase produce the same text-table
  shape for equivalent rows; empty-result messages for both.
- Backward compat: `params=None` yields a zero-arg tool; default alias remains
  `execute_sql_tool`.
- `SqlToolModel`: `type: sql` parses with a `warehouse:` target and with a
  `database:` target; `as_tools()` returns a working tool; setting neither or both
  of `warehouse`/`database` is rejected by the validator; missing statement rejected.

## Files touched

- `src/dao_ai/tools/sql.py` — new models + factories + shared formatter; docstring fixes.
- `src/dao_ai/tools/__init__.py` — export the two new public factories and
  `StatementParam` / `ParamSource`.
- `src/dao_ai/config.py` — `FunctionType.SQL`, `SqlToolModel`, `AnyTool` union entry.
- `schemas/model_config_schema.json` — regenerated via `make schema`.
- `tests/dao_ai/test_sql_tool.py`, `tests/dao_ai/test_sql_tool_integration.py` — updated/added tests.

## Follow-ups

- Optional guardrails (read-only enforcement, row limits, timeouts) if the tool
  is later pointed at LLM-authored SQL rather than fixed statements.
