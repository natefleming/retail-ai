from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.catalog import (CatalogInfo, SchemaInfo,
                                            VolumeInfo, VolumeType)


def _volume_as_path(self: VolumeInfo) -> Path:
  return Path(f"/Volumes/{self.catalog_name}/{self.schema_name}/{self.name}")

# monkey patch
VolumeInfo.as_path = _volume_as_path


def get_or_create_catalog(name: str, w: WorkspaceClient | None = None) -> CatalogInfo:
  if w is None:
    w = WorkspaceClient()

  catalog: CatalogInfo
  try:
    catalog = w.catalogs.get(name=name)
  except NotFound:
    catalog = w.catalogs.create(name=name)
  return catalog


def get_or_create_database(catalog: CatalogInfo, name: str, w: WorkspaceClient | None = None) -> SchemaInfo:
  if w is None:
    w = WorkspaceClient()

  database: SchemaInfo
  try:
    database = w.schemas.get(full_name=f"{catalog.name}.{name}")
  except NotFound:
    database = w.schemas.create(name=name, catalog_name=catalog.name)
  return database


def get_or_create_volume(
  catalog: CatalogInfo,
  database: SchemaInfo, 
  name: str, 
  volume_type: VolumeType = VolumeType.MANAGED,
  w: WorkspaceClient | None = None
) -> VolumeInfo:
  if w is None:
    w = WorkspaceClient()
    
  volume: VolumeInfo
  try:
    volume = w.volumes.read(name=f"{database.full_name}.{name}")
  except NotFound:
    volume = w.volumes.create(
      catalog_name=catalog.name, 
      schema_name=database.name, 
      name=name, 
      volume_type=VolumeType.MANAGED
    )
  return volume
