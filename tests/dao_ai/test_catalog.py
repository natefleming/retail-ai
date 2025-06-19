from pathlib import Path
from unittest.mock import Mock

import pytest
from databricks.sdk.service.catalog import VolumeInfo

from dao_ai.catalog import _volume_as_path, full_name


@pytest.mark.unit
def test_volume_as_path() -> None:
    """Test the _volume_as_path function."""
    # Create a mock VolumeInfo object
    volume = Mock(spec=VolumeInfo)
    volume.catalog_name = "test_catalog"
    volume.schema_name = "test_schema"
    volume.name = "test_volume"

    # Test the function
    result = _volume_as_path(volume)

    assert isinstance(result, Path)
    assert str(result) == "/Volumes/test_catalog/test_schema/test_volume"


@pytest.mark.unit
def test_volume_info_monkey_patch() -> None:
    """Test that VolumeInfo has the as_path method after monkey patching."""
    # Test that the method exists on the class
    assert hasattr(VolumeInfo, "as_path")

    # Create a mock VolumeInfo object and set attributes
    volume = Mock()
    volume.catalog_name = "my_catalog"
    volume.schema_name = "my_schema"
    volume.name = "my_volume"

    # Call the monkey-patched method directly
    result = VolumeInfo.as_path(volume)

    assert isinstance(result, Path)
    assert str(result) == "/Volumes/my_catalog/my_schema/my_volume"


@pytest.mark.unit
def test_full_name_with_catalog_and_schema() -> None:
    """Test full_name with both catalog and schema."""
    schema_info = {"catalog_name": "production", "schema_name": "retail"}

    result = full_name("products", schema=schema_info)

    assert result == "production.retail.products"


@pytest.mark.unit
def test_full_name_with_catalog_only() -> None:
    """Test full_name with catalog only."""
    schema_info = {"catalog_name": "production"}

    result = full_name("my_schema", schema=schema_info)

    assert result == "production.my_schema"


@pytest.mark.unit
def test_full_name_with_no_catalog() -> None:
    """Test full_name with no catalog or schema."""
    result = full_name("simple_name")

    assert result == "simple_name"


@pytest.mark.unit
def test_full_name_with_empty_schema() -> None:
    """Test full_name with empty schema dictionary."""
    result = full_name("table_name", schema={})

    assert result == "table_name"


@pytest.mark.unit
def test_full_name_with_none_values() -> None:
    """Test full_name with None values in schema."""
    schema_info = {"catalog_name": None, "schema_name": None}

    result = full_name("entity", schema=schema_info)

    assert result == "entity"


@pytest.mark.unit
def test_full_name_with_partial_schema() -> None:
    """Test full_name with only schema_name but no catalog_name."""
    schema_info = {"schema_name": "my_schema"}

    result = full_name("my_table", schema=schema_info)

    assert result == "my_table"


@pytest.mark.unit
def test_full_name_with_kwargs() -> None:
    """Test full_name with additional kwargs (should be ignored)."""
    schema_info = {"catalog_name": "test_catalog", "schema_name": "test_schema"}

    result = full_name("entity", schema=schema_info, extra_param="ignored")

    assert result == "test_catalog.test_schema.entity"


@pytest.mark.unit
def test_full_name_edge_cases() -> None:
    """Test full_name with edge cases."""
    # Empty strings should be treated as falsy
    schema_info = {"catalog_name": "", "schema_name": ""}

    result = full_name("entity", schema=schema_info)

    assert result == "entity"


@pytest.mark.unit
def test_volume_as_path_with_special_characters() -> None:
    """Test _volume_as_path with special characters in names."""
    volume = Mock(spec=VolumeInfo)
    volume.catalog_name = "test-catalog_123"
    volume.schema_name = "schema.with.dots"
    volume.name = "volume_with_underscores"

    result = _volume_as_path(volume)

    assert (
        str(result)
        == "/Volumes/test-catalog_123/schema.with.dots/volume_with_underscores"
    )
