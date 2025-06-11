from pathlib import Path
from typing import Any, Sequence

import sqlparse
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.catalog import (
    CatalogInfo,
    SchemaInfo,
    VolumeInfo,
    VolumeType,
)
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex
from loguru import logger
from pyspark.sql import SparkSession

from retail_ai.config import (
    DatasetModel,
    SchemaModel,
    VectorStoreModel,
    VolumeModel,
    UnityCatalogFunctionSqlModel, UnityCatalogFunctionSqlTestModel, FunctionModel
)
from retail_ai.providers.base import ServiceProvider
from retail_ai.vector_search import endpoint_exists, index_exists
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from databricks_langchain import DatabricksFunctionClient, UCFunctionToolkit
from databricks.sdk.service.catalog import FunctionInfo
from unitycatalog.ai.core.base import FunctionExecutionResult

class DatabricksProvider(ServiceProvider):
    def __init__(
        self, 
        w: WorkspaceClient | None = None, 
        vsc: VectorSearchClient | None = None, 
        dfs: DatabricksFunctionClient | None = None,
    ) -> None:
        if w is None:
            w = WorkspaceClient()
        if vsc is None:
            vsc = VectorSearchClient()
        if dfs is None:
            dfs = DatabricksFunctionClient(w=w)
        self.w = w
        self.vsc = vsc
        self.dfs = dfs

    def create_catalog(self, schema: SchemaModel) -> CatalogInfo:
        catalog_info: CatalogInfo
        try:
            catalog_info = self.w.catalogs.get(name=schema.catalog_name)
        except NotFound:
            logger.debug(f"Creating catalog: {schema.catalog_name}")
            catalog_info = self.w.catalogs.create(name=schema.catalog_name)
        return catalog_info

    def create_schema(self, schema: SchemaModel) -> SchemaInfo:
        catalog_info: CatalogInfo = self.create_catalog(schema)
        schema_info: SchemaInfo
        try:
            schema_info = self.w.schemas.get(full_name=schema.full_name)
        except NotFound:
            logger.debug(f"Creating schema: {schema.full_name}")
            schema_info = self.w.schemas.create(
                name=schema.schema_name, catalog_name=catalog_info.name
            )
        return schema_info

    def create_volume(self, volume: VolumeModel) -> VolumeInfo:
        schema_info: SchemaInfo = self.create_schema(volume.schema_model)
        volume_info: VolumeInfo
        try:
            volume_info = self.w.volumes.read(name=volume.full_name)
        except NotFound:
            logger.debug(f"Creating volume: {volume.full_name}")
            volume_info = self.w.volumes.create(
                catalog_name=schema_info.catalog_name,
                schema_name=schema_info.name,
                name=volume.name,
                volume_type=VolumeType.MANAGED,
            )
        return volume_info

    def create_dataset(self, dataset: DatasetModel) -> None:
        current_dir: Path = "file:///" / Path.cwd().relative_to("/")

        # Get or create Spark session
        spark: SparkSession = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError(
                "No active Spark session found. This method requires Spark to be available."
            )

        table: str = dataset.table.full_name
        ddl_path: Path = Path(dataset.ddl)
        data_path: Path = Path(dataset.data)
        format: str = dataset.format
        read_options: dict[str, Any] = dataset.read_options or {}

        statements: Sequence[str] = sqlparse.parse(ddl_path.read_text())
        for statement in statements:
            logger.debug(statement)
            spark.sql(
                str(statement), args={"database": dataset.table.schema_model.full_name}
            )

        if format == "sql":
            data_statements: Sequence[str] = sqlparse.parse(data_path.read_text())
            for statement in data_statements:
                logger.debug(statement)
                spark.sql(
                    str(statement),
                    args={"database": dataset.table.schema_model.full_name},
                )
        else:
            logger.debug(f"Writing to: {table}")
            data_path = current_dir / data_path
            spark.read.format(format).options(**read_options).load(
                data_path.as_posix()
            ).write.mode("overwrite").saveAsTable(table)

    def create_vector_store(self, vector_store: VectorStoreModel) -> None:
        if not endpoint_exists(self.vsc, vector_store.endpoint.name):
            self.vsc.create_endpoint_and_wait(
                name=vector_store.endpoint.name,
                endpoint_type=vector_store.endpoint.type,
                verbose=True,
            )

        logger.debug(f"Endpoint named {vector_store.endpoint.name} is ready.")

        if not index_exists(
            self.vsc, vector_store.endpoint.name, vector_store.index.full_name
        ):
            logger.debug(
                f"Creating index {vector_store.index.full_name} on endpoint {vector_store.endpoint.name}..."
            )
            self.vsc.create_delta_sync_index_and_wait(
                endpoint_name=vector_store.endpoint.name,
                index_name=vector_store.index.full_name,
                source_table_name=vector_store.source_table.full_name,
                pipeline_type="TRIGGERED",
                primary_key=vector_store.primary_key,
                embedding_source_column=vector_store.embedding_source_column,
                embedding_model_endpoint_name=vector_store.embedding_model.name,
                columns_to_sync=vector_store.columns,
            )
        else:
            self.vsc.get_index(
                vector_store.endpoint.name, vector_store.index.full_name
            ).sync()

        logger.debug(
            f"index {vector_store.index.full_name} on table {vector_store.source_table.full_name} is ready"
        )

    def get_vector_index(self, vector_store: VectorStoreModel) -> None:
        index: VectorSearchIndex = self.vsc.get_index(
            vector_store.endpoint.name, vector_store.index.full_name
        )
        return index


    def create_sql_function(self, unity_catalog_function: UnityCatalogFunctionSqlModel) -> None:
        function: FunctionModel = unity_catalog_function.function
        schema: SchemaModel = function.schema_model
        ddl_path: Path = Path(unity_catalog_function.ddl)

        statements: Sequence[str] = [str(s) for s in sqlparse.parse(ddl_path.read_text())]
        for sql in statements:
            sql = sql.replace("{catalog_name}", schema.catalog_name)
            sql = sql.replace("{schema_name}", schema.schema_name)

            logger.info(function.name)
            _: FunctionInfo = self.dfs.create_function(sql_function_body=sql)

            if unity_catalog_function.test:
                logger.info(unity_catalog_function.test.parameters)

                result: FunctionExecutionResult = self.dfs.execute_function(
                    function_name=function.full_name,
                    parameters=unity_catalog_function.test.parameters
                )

                if result.error:
                    logger.error(result.error)
                else:
                    logger.info(f"Function {function.full_name} executed successfully.")
                    logger.info(f"Result: {result}")