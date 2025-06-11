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
from loguru import logger
from pyspark.sql import SparkSession

from retail_ai.config import DatasetModel, SchemaModel, VolumeModel
from retail_ai.providers.base import ServiceProvider


class DatabricksProvider(ServiceProvider):
    def __init__(self, w: WorkspaceClient | None = None) -> None:
        if w is None:
            w = WorkspaceClient()
        self.w = w

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
        data_path: Path = current_dir / Path(dataset.data)
        format: str = dataset.format
        read_options: dict[str, Any] = dataset.read_options or {}

        statements: Sequence[str] = sqlparse.parse(ddl_path.read_text())
        for statement in statements:
            logger.debug(statement)
            spark.sql(
                str(statement), args={"database": dataset.table.schema_model.full_name}
            )

        if format == "sql":
            data_statements: Sequence[str] = sqlparse.parse(ddl_path.read_text())
            for statement in data_statements:
                logger.debug(statement)
                spark.sql(
                    str(statement),
                    args={"database": dataset.table.schema_model.full_name},
                )
        else:
            logger.debug(f"Writing to: {table}")
            spark.read.format(format).options(**read_options).load(
                data_path.as_posix()
            ).write.mode("overwrite").saveAsTable(table)
