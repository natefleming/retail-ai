from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.catalog import (
    CatalogInfo,
    SchemaInfo,
    VolumeInfo,
    VolumeType,
)
import re
from pathlib import Path
from typing import Any, Sequence

from retail_ai.config import SchemaModel, VolumeModel, DatasetModel
from retail_ai.providers.base import ServiceProvider

from loguru import logger

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
        from pyspark.sql import SparkSession
        
        # Get or create Spark session
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active Spark session found. This method requires Spark to be available.")
        
        table: str = dataset.table.full_name
        ddl_path: Path = Path(dataset.ddl)
        
        current_dir: Path = Path.cwd()
        data_path: Path = current_dir / Path(dataset.data)
        
        format: str = dataset.format
        read_options: dict[str, Any] = dataset.read_options or {}
        
        logger.info(f"Creating dataset: {table}")
        logger.debug(f"DDL path: {ddl_path}")
        logger.debug(f"Data path: {data_path}")
        logger.debug(f"Format: {format}")
        
        # Execute DDL statements
        if ddl_path.exists():
            logger.debug(f"Reading DDL from: {ddl_path}")
            ddl_content = ddl_path.read_text()
            statements: Sequence[str] = [s.strip() for s in re.split(r"\s*;\s*", ddl_content) if s.strip()]
            
            for statement in statements:
                logger.debug(f"Executing DDL: {statement}")
                spark.sql(statement, args={"database": dataset.table.schema_model.full_name})
        else:
            logger.warning(f"DDL file not found: {ddl_path}")
        
        # Load data based on format
        if format == "sql":
            # Execute SQL statements for data loading
            if data_path.exists():
                logger.debug(f"Reading SQL data from: {data_path}")
                data_content = data_path.read_text()
                data_statements: Sequence[str] = [s.strip() for s in re.split(r"\s*;\s*", data_content) if s.strip()]
                
                for statement in data_statements:
                    logger.debug(f"Executing SQL data statement: {statement}")
                    spark.sql(statement, args={"database": dataset.table.schema_model.full_name})
            else:
                logger.warning(f"SQL data file not found: {data_path}")
        else:
            if data_path.exists():
                logger.debug(f"Loading {format} data from: {data_path}")

                data_uri = data_path.as_uri() if data_path.is_absolute() else f"file:///{data_path.as_posix()}"
                
                df = spark.read.format(format).options(**read_options).load(data_uri)
                df.write.mode("overwrite").saveAsTable(table)
                
                logger.info(f"Successfully loaded data into table: {table}")
            else:
                logger.warning(f"Data file not found: {data_path}")
        
        logger.info(f"Dataset creation completed: {table}")
        