from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.catalog import (
    CatalogInfo,
    SchemaInfo,
    VolumeInfo,
    VolumeType,
)

from retail_ai.config import SchemaModel, VolumeModel
from retail_ai.providers.base import ServiceProvider

from loguru import logger

class DatabricksProvider(ServiceProvider):
    def __init__(self, w: WorkspaceClient | None = None) -> None:
        if w is None:
            w = WorkspaceClient()
        self.w = w

    def create_catalog(self, schema: SchemaModel) -> CatalogInfo:
        logger.debug(f"Creating catalog: {schema.catalog_name}")
        catalog_info: CatalogInfo
        try:
            catalog_info = self.w.catalogs.get(name=schema.catalog_name)
        except NotFound:
            catalog_info = self.w.catalogs.create(name=schema.catalog_name)
        return catalog_info

    def create_schema(self, schema: SchemaModel) -> SchemaInfo:
        logger.debug(f"Creating schema: {schema.full_name}")
        catalog_info: CatalogInfo = self.create_catalog(schema)
        schema_info: SchemaInfo
        try:
            schema_info = self.w.schemas.get(full_name=schema.full_name)
        except NotFound:
            schema_info = self.w.schemas.create(
                name=schema.schema_name, catalog_name=catalog_info.name
            )
        return schema_info

    def create_volume(self, volume: VolumeModel) -> VolumeInfo:
        logger.debug(f"Creating volume: {volume.full_name}")
        schema_info: SchemaInfo = self.create_schema(volume.schema_model)
        volume_info: VolumeInfo
        try:
            volume_info = self.w.volumes.read(name=volume.full_name)
        except NotFound:
            volume_info = self.w.volumes.create(
                catalog_name=schema_info.catalog_name,
                schema_name=schema_info.name,
                name=volume.name,
                volume_type=VolumeType.MANAGED,
            )
        return volume_info
