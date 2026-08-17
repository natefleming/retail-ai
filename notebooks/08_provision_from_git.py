# Databricks notebook source
# MAGIC %md
# MAGIC # Provision a dao-ai project from a git locator
# MAGIC
# MAGIC Nothing about this notebook assumes the project is checked out. The config,
# MAGIC its DDL, its seed data, its Unity Catalog function SQL, its skill markdown,
# MAGIC and two flavours of Python tool code all arrive with the git checkout that
# MAGIC `AppConfig.from_git` produces, and every relative path resolves against the
# MAGIC config's own directory *inside* that checkout.
# MAGIC
# MAGIC The default locator points at
# MAGIC [`examples/21_from_git`](https://github.com/natefleming/dao-ai/tree/main/examples/21_from_git),
# MAGIC which exists to exercise one anchor per asset — see its README for the table
# MAGIC of what breaks if any single anchor is wrong. Point `config-path` at your own
# MAGIC repository to provision that instead.
# MAGIC
# MAGIC What runs here, in dependency order: schemas → volumes → datasets → Unity
# MAGIC Catalog functions → vector indexes, then the agent is exercised in-process and
# MAGIC deployed to Model Serving and Databricks Apps.

# COMMAND ----------

# Dependency bootstrap, matching the pipeline step notebooks. Install dao-ai —
# which pulls its own transitive deps, python-dotenv included — from the newest
# ../dist wheel if one is there, else the published PyPI package. This notebook
# builds an agent graph and deploys it, so it needs every optional feature
# extra: ``[all]``. The spec is single-quoted in the magic so a dev wheel's
# ``+local`` version tag and the bracket survive shell expansion.
#
# Note: with no ../dist wheel this installs the *published* dao-ai, not your
# working tree. Run ``uv build`` from the repo root first to test local changes.
import glob, os

from packaging.version import Version

# Newest by *version*, not by filename: a lexical sort puts 0.2.8 above
# 0.2.10. ``Version`` also parses a dev wheel's ``+local`` tag correctly.
def _wheel_version(wheel: str) -> Version:
    return Version(os.path.basename(wheel).split("-")[1])

_wheels = sorted(glob.glob("../dist/dao_ai-*.whl"), key=_wheel_version, reverse=True)
_dao_ai_dep = (_wheels[0] if _wheels else "dao-ai") + "[all]"

# MAGIC %uv pip install --quiet '{_dao_ai_dep}'
# MAGIC %restart_python

# COMMAND ----------

# Record the installed dao-ai version plus the key libraries resolved under
# it, so each run's logs capture exactly what was installed. Alphabetical; the
# list is short and hand-curated.
from importlib.metadata import version

print(f"dao-ai=={version('dao-ai')}")
print(f"databricks-langchain=={version('databricks-langchain')}")
print(f"databricks-sdk=={version('databricks-sdk')}")
print(f"langchain=={version('langchain')}")
print(f"langgraph=={version('langgraph')}")
print(f"mlflow=={version('mlflow')}")

# COMMAND ----------

# `from_git` shells out to git, so it has to be on the driver's PATH. Fail here
# with a clear message rather than inside a subprocess call five cells down.
import shutil
import subprocess

git_executable: str | None = shutil.which("git")
if git_executable is None:
    raise RuntimeError(
        "`git` is not on PATH. AppConfig.from_git delegates to the git CLI, so a "
        "locator cannot be resolved without it. Use a compute image that ships "
        "git, or load the config from a local path instead."
    )

print(subprocess.run([git_executable, "--version"], capture_output=True, text=True).stdout.strip())

# COMMAND ----------

# A git locator, not a path. Grammar:
#   git+<scheme>://<host>/<owner>/<repo>[@<ref>][#<in-repo-path>]
#   gh:<owner>/<repo>[@<ref>][#<in-repo-path>]
# `@ref` may be a branch, tag, or full 40-character SHA; omit `#path` (or point
# it at a directory) and the config is discovered. Pin a tag or SHA for a
# repository you do not control — resolving a locator runs its code.
dbutils.widgets.text(
    name="config-path",
    defaultValue="gh:natefleming/dao-ai@main#examples/21_from_git/from_git.yaml",
)

# The example's `parameters:` block declares both of these with defaults; they
# are surfaced as widgets so a run can be pointed at a scratch schema.
dbutils.widgets.text(name="catalog", defaultValue="retail_consumer_goods")
dbutils.widgets.text(name="schema", defaultValue="dao_ai_from_git")

# Where the checkout lands. Empty means the default, `~/.dao-ai/git` on the
# *driver's local disk* — which does not survive a cluster restart. Set this to
# a `/Volumes/...` path to keep checkouts across restarts. Prefer a per-user
# destination: anyone who can write to a shared one can change code that later
# lands on `sys.path`.
dbutils.widgets.text(name="cache-dir", defaultValue="")

# `both` deploys to Model Serving *and* Databricks Apps, which is the point of
# the exercise: the repository's Python has to reach every runtime, not just the
# notebook. Set `none` to provision and test in-process only.
dbutils.widgets.dropdown(
    name="deploy-mode",
    defaultValue="both",
    choices=["none", "model_serving", "apps", "both"],
)

locator: str | None = dbutils.widgets.get("config-path") or None
if not locator:
    raise ValueError(
        "Nothing to load: the `config-path` widget is empty. Set it to a git "
        "locator, e.g. `gh:natefleming/dao-ai@main#examples/21_from_git/from_git.yaml`."
    )

catalog: str = dbutils.widgets.get("catalog")
schema: str = dbutils.widgets.get("schema")
cache_dir_widget: str | None = dbutils.widgets.get("cache-dir") or None
deploy_mode_widget: str = dbutils.widgets.get("deploy-mode")

print(f"locator:    {locator}")
print(f"catalog:    {catalog}")
print(f"schema:     {schema}")
print(f"cache-dir:  {cache_dir_widget or '(default: ~/.dao-ai/git)'}")
print(f"deploy:     {deploy_mode_widget}")

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the config from the locator
# MAGIC
# MAGIC `cache_dir` is the typed front door for the checkout destination; the
# MAGIC `$DAO_AI_GIT_CACHE` environment variable does the same thing for call sites
# MAGIC you do not control. Constructing a `GitSource` is only necessary when you
# MAGIC need an option a bare string cannot express (`cache_dir`, `token`,
# MAGIC `refresh`) — otherwise pass the locator straight to `from_git`.
# MAGIC
# MAGIC For a private repository set the token from a secret scope rather than
# MAGIC inlining it in the locator, which `parse_git_locator` rejects:
# MAGIC
# MAGIC ```python
# MAGIC os.environ["DAO_AI_GIT_TOKEN"] = dbutils.secrets.get("my-scope", "git-token")
# MAGIC ```

# COMMAND ----------

from pathlib import Path

from dao_ai.config import AppConfig
from dao_ai.git_source import GitSource

params: dict[str, str] = {"catalog": catalog, "schema": schema}

source: GitSource = (
    GitSource(locator, cache_dir=Path(cache_dir_widget))
    if cache_dir_widget
    else GitSource(locator)
)

config: AppConfig = AppConfig.from_git(source, params=params)

# `source_config_path` keeps the locator — what was typed, and what messages
# should echo. `local_config_path` is the real file inside the checkout, and the
# one to use for anything that touches the filesystem. The default cache layout
# embeds the resolved commit SHA in that path, so it doubles as provenance.
print(f"source_config_path: {config.source_config_path}")
print(f"local_config_path:  {config.local_config_path}")
print(f"substituted:        {config.substitution_vars}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confirm every anchor resolved inside the checkout
# MAGIC
# MAGIC Each of these was resolved at load time. A wrong anchor does not raise — it
# MAGIC silently yields a nonexistent directory or a `None`, and the failure surfaces
# MAGIC much later as a missing table or an absent tool. Assert them here instead.

# COMMAND ----------

import sys
from pathlib import Path

from dao_ai.code_paths import resolve_code_path
from dao_ai.skills import _skill_base_dir

checkout_config: Path = Path(config.local_config_path).resolve()
checkout_dir: Path = checkout_config.parent

if not checkout_config.is_file():
    raise RuntimeError(
        f"`local_config_path` does not point at a real file: {checkout_config}. "
        "The config directory is what every relative `ddl`, `data`, `skills`, and "
        "`code_paths` entry is anchored on, so nothing below can work."
    )

print(f"checkout dir: {checkout_dir}")
print(f"contents:     {sorted(p.name for p in checkout_dir.iterdir())}")

# skills/ — anchored via `_skill_base_dir`, which must be the config's directory.
skill_base: Path = _skill_base_dir(config)
print(f"\nskill base:   {skill_base}")
if skill_base.resolve() != checkout_dir:
    raise RuntimeError(f"skills anchored on {skill_base}, expected {checkout_dir}")

for skill_name, skill in config.resources.skills.items():
    skill_path: Path = (checkout_dir / str(skill.path)).resolve()
    marker: Path = skill_path / "SKILL.md"
    print(f"  {skill_name}: {marker} exists={marker.is_file()}")
    if not marker.is_file():
        raise RuntimeError(f"skill {skill_name!r} has no SKILL.md at {marker}")

# app.code_paths — the resolved entry's *parent* goes on sys.path, so a package
# named in `code_paths` keeps its own name in imports.
print("\ncode_paths:")
for entry in config.app.code_paths or []:
    resolved: Path | None = resolve_code_path(entry, config)
    print(f"  {entry} -> {resolved}")
    if resolved is None:
        raise RuntimeError(
            f"code_path {entry!r} did not resolve. Its parent is what goes on "
            "sys.path, so the module it provides would be unimportable."
        )
    if str(resolved.parent) not in sys.path:
        raise RuntimeError(f"{resolved.parent} is not on sys.path")

# The colocated `src/` convention needs no declaration — it is discovered and
# prepended at load time, which is why its packages import prefix-free.
src_dir: Path = checkout_dir / "src"
if src_dir.is_dir():
    print(f"\nsrc/: {src_dir} on sys.path={str(src_dir) in sys.path}")
    if str(src_dir) not in sys.path:
        raise RuntimeError(f"{src_dir} exists but was not added to sys.path")

# datasets / UC functions — `_base_path` is stamped at load time and is what
# `resolve_asset_path` joins a relative `ddl` or `data` against.
print("\ndataset assets:")
for dataset in config.datasets or []:
    for asset in (dataset.ddl, dataset.data):
        if asset is None:
            continue
        asset_path: Path = dataset.resolve_asset_path(asset)
        print(f"  {dataset.table.name}: {asset} -> {asset_path} exists={asset_path.is_file()}")
        if not asset_path.is_file():
            raise RuntimeError(f"dataset asset {asset!r} missing at {asset_path}")

print("\nunity catalog function assets:")
for uc_function in config.unity_catalog_functions or []:
    ddl_path: Path = uc_function.resolve_asset_path(uc_function.ddl)
    print(f"  {uc_function.function.name}: {ddl_path} exists={ddl_path.is_file()}")
    if not ddl_path.is_file():
        raise RuntimeError(f"UC function DDL missing at {ddl_path}")

print("\nall anchors resolved inside the checkout")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schemas and volumes
# MAGIC
# MAGIC First, because everything below lands in them.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

from dao_ai.config import SchemaModel, VolumeModel

w: WorkspaceClient = WorkspaceClient()

for _, schema_model in config.schemas.items():
    schema_model: SchemaModel
    _ = schema_model.create(w=w)
    print(f"schema: {schema_model.full_name}")

for _, volume in config.resources.volumes.items():
    volume: VolumeModel
    _ = volume.create(w=w)
    print(f"volume: {volume.full_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Nested volume paths
# MAGIC
# MAGIC `VolumePathModel` is the typed way to address a directory inside a volume,
# MAGIC and `create()` makes the directory. Worth exercising explicitly: a volume
# MAGIC path is a FUSE mount, not a POSIX filesystem, and directory semantics are
# MAGIC where the two differ most.

# COMMAND ----------

from pathlib import Path

from dao_ai.config import VolumePathModel

for volume_name, volume in config.resources.volumes.items():
    seed_path: VolumePathModel = VolumePathModel(volume=volume, path="seed")
    seed_path.create(w=w)

    seed_dir: Path = seed_path.as_path()
    print(f"{volume_name}: {seed_dir} exists={seed_dir.is_dir()}")

    # Round-trip a file through the nested directory: `create()` proving the
    # directory exists is weaker than reading back what was written to it.
    marker: Path = seed_dir / "provisioned_from.txt"
    marker.write_text(f"{config.source_config_path}\n{config.local_config_path}\n")
    print(f"  wrote {marker}")
    print(f"  read back: {marker.read_text().strip()!r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Datasets
# MAGIC
# MAGIC The seed files are inside the driver-local checkout, which serverless Spark
# MAGIC executors cannot read. So `csv` / `parquet` / `orc` / `delta` are copied into
# MAGIC a managed volume `<catalog>.<schema>.dao_ai_staging` — created for you — and
# MAGIC Spark is handed the `/Volumes/...` path instead. `json` and `excel` are read
# MAGIC on the driver with pandas and are never staged.

# COMMAND ----------

from dao_ai.config import DatasetModel

for dataset in config.datasets or []:
    dataset: DatasetModel
    dataset.create()
    print(f"dataset: {dataset.table.full_name} (format={dataset.format})")

# COMMAND ----------

# Row counts, so an empty table is caught here rather than as a confusing
# "no results" answer from the agent later.
for dataset in config.datasets or []:
    dataset: DatasetModel
    count: int = spark.table(dataset.table.full_name).count()
    print(f"{dataset.table.full_name}: {count} rows")
    if count == 0:
        raise RuntimeError(
            f"{dataset.table.full_name} is empty. The DDL ran but the seed file "
            f"({dataset.data}) produced no rows — check the staging path and the "
            "declared format."
        )

# COMMAND ----------

# The staging volume is created implicitly by the csv/parquet path. Listing the
# volumes confirms both it and the config's own declared volume are present.
for schema_model in config.schemas.values():
    volume_names: list[str] = [
        v.name
        for v in w.volumes.list(
            catalog_name=schema_model.catalog_name,
            schema_name=schema_model.schema_name,
        )
    ]
    print(f"{schema_model.full_name}: {sorted(volume_names)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog functions
# MAGIC
# MAGIC The DDL comes out of the checkout; `{catalog_name}` / `{schema_name}` in the
# MAGIC SQL are substituted from the function's declared schema.

# COMMAND ----------

from dao_ai.config import UnityCatalogFunctionSqlModel

for uc_function in config.unity_catalog_functions or []:
    uc_function: UnityCatalogFunctionSqlModel
    uc_function.create()
    print(f"function: {uc_function.function.full_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector indexes
# MAGIC
# MAGIC `resources.vector_stores` is a discriminated union: AI Search stores are
# MAGIC created with `create()`, Lakebase stores are provisioned with
# MAGIC `provision(dimension=...)`. Narrow on the type rather than probing for
# MAGIC attributes.

# COMMAND ----------

from dao_ai.config import AiSearchVectorStoreModel, LakebaseVectorStoreModel

for store_name, vector_store in config.resources.vector_stores.items():
    if isinstance(vector_store, AiSearchVectorStoreModel):
        vector_store.create()
        print(f"ai search index: {vector_store.index.full_name}")
    elif isinstance(vector_store, LakebaseVectorStoreModel):
        raise RuntimeError(
            f"{store_name} is a Lakebase store, which needs an embedding "
            "dimension this notebook does not know. Call "
            "`vector_store.provision(dimension=...)` yourself."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise the agent in-process
# MAGIC
# MAGIC Everything the agent needs now exists, and the repository's Python is already
# MAGIC on `sys.path` from the load above. A single prompt covers all of it, because
# MAGIC it needs a different anchor per tool: the UC functions came from colocated
# MAGIC DDL, `find_aisle` from the colocated `src/`, `apply_contractor_discount` from
# MAGIC an `app.code_paths` entry (called with its YAML `args:`), the search tool from
# MAGIC the index built off the staged seed, and the answer's shape from `SKILL.md`.

# COMMAND ----------

import mlflow
from mlflow.pyfunc import ChatModel

mlflow.langchain.autolog(run_tracer_inline=True)

app: ChatModel = config.as_chat_model()

# COMMAND ----------

from typing import Any

from rich import print as pprint

from dao_ai.models import process_messages

input_example: dict[str, Any] = {
    "messages": [
        {
            "role": "user",
            "content": (
                "I'm a contractor - what does SKU DRL10045 cost me, where do I "
                "find it in the store, and is the Chicago store open on Sunday?"
            ),
        }
    ],
    "custom_inputs": {
        "configurable": {
            "thread_id": "from-git-1",
            "user_id": "provision_from_git",
        }
    },
}

response = process_messages(app=app, **input_example)
pprint(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy
# MAGIC
# MAGIC Both targets have to carry the repository's Python: Model Serving through the
# MAGIC logged model's `code_paths`, Apps through the synced app source. A tool that
# MAGIC works in the cell above but not in a deployment points at packaging rather
# MAGIC than at path resolution.

# COMMAND ----------

from dao_ai.config import ServingMode

modes: list[ServingMode] = {
    "none": [],
    "model_serving": [ServingMode.MODEL_SERVING],
    "apps": [ServingMode.APPS],
    "both": [ServingMode.MODEL_SERVING, ServingMode.APPS],
}[deploy_mode_widget]

print(f"deploying to: {[m.value for m in modes] or '(nothing)'}")

# COMMAND ----------

for mode in modes:
    print(f"=== {mode.value} ===")
    # Only Model Serving needs an MLflow model logged and registered; Apps deploy
    # from the config plus the PyPI package.
    if mode is ServingMode.MODEL_SERVING:
        config.create_agent()
    config.deploy_agent(mode=mode)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What now exists
# MAGIC
# MAGIC Nothing below was uploaded by hand — every artifact traces back to the
# MAGIC locator at the top.

# COMMAND ----------

print(f"locator:  {config.source_config_path}")
print(f"checkout: {config.local_config_path}")
print()
for schema_model in config.schemas.values():
    print(f"schema:   {schema_model.full_name}")
for dataset in config.datasets or []:
    print(f"table:    {dataset.table.full_name}")
for uc_function in config.unity_catalog_functions or []:
    print(f"function: {uc_function.function.full_name}")
for vector_store in config.resources.vector_stores.values():
    if isinstance(vector_store, AiSearchVectorStoreModel):
        print(f"index:    {vector_store.index.full_name}")
if config.app.registered_model:
    print(f"model:    {config.app.registered_model.full_name}")
if config.app.endpoint_name:
    print(f"endpoint: {config.app.endpoint_name}")
