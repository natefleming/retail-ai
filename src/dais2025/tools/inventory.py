"""
Inventory Tools

This module contains tool creation functions for inventory-related operations including
inventory lookup by SKU and store-specific inventory queries.
"""

from typing import Any, Callable

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementResponse, StatementState
from langchain_core.tools import tool
from loguru import logger

from dao_ai.config import (
    SchemaModel,
    WarehouseModel,
)


def create_find_inventory_by_sku_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)
    """Create a Unity Catalog tool for finding inventory by SKU."""

    @tool
    def find_inventory_by_sku(skus: list[str]) -> tuple:
        """
        Find inventory details by one or more SKUs using Unity Catalog functions.
        This tool retrieves detailed inventory information across all stores for products based on their SKU codes.

        Args:
            skus (list[str]): One or more unique identifiers to retrieve.
                             SKU values are between 5-8 alpha numeric characters.
                             Examples: ["PET-KCP-001", "DUN-KCP-002"]

        Returns:
            (tuple): A tuple containing inventory information with fields like:
                inventory_id BIGINT
                ,sku STRING
                ,product_id BIGINT
                ,store_id INT (or similar store identifier)
                ,store_quantity INT
                ,warehouse STRING
                ,warehouse_quantity INT
                ,retail_amount DECIMAL(11, 2)
                ,popularity_rating STRING
                ,department STRING
                ,aisle_location STRING
                ,is_closeout BOOLEAN
                (Additional fields may be available depending on Unity Catalog function definition)
        """
        logger.debug(f"find_inventory_by_sku: {skus}")

        # Convert list to SQL array format
        skus_str = ", ".join([f"'{sku}'" for sku in skus])

        # Execute the Unity Catalog function with basic query
        sql_query = f"""
            SELECT * FROM {schema.full_name}.find_inventory_by_sku(ARRAY({skus_str}))
        """

        # Get workspace client and execute query
        w = WorkspaceClient()

        response: StatementResponse = w.statement_execution.execute_statement(
            warehouse_id=warehouse.warehouse_id,
            statement=sql_query,
            wait_timeout="30s",
        )

        if response.status.state != StatementState.SUCCEEDED:
            logger.error(f"Query failed: {response.status}")
            return ()

        # Convert results to DataFrame and then to tuple
        if response.result and response.result.data_array:
            # Try to get column names from different possible locations
            columns = None

            # Try the manifest approach first (older SDK versions)
            if hasattr(response.result, "manifest") and hasattr(
                response.result.manifest, "schema"
            ):
                columns = [col.name for col in response.result.manifest.schema.columns]
            # Try the schema approach (newer SDK versions)
            elif hasattr(response.result, "schema") and hasattr(
                response.result.schema, "columns"
            ):
                columns = [col.name for col in response.result.schema.columns]
            # Fallback: try to infer from the first row of data
            elif response.result.data_array and len(response.result.data_array) > 0:
                # Use generic column names based on the number of columns
                num_cols = (
                    len(response.result.data_array[0])
                    if response.result.data_array[0]
                    else 0
                )
                columns = [f"col_{i}" for i in range(num_cols)]
            else:
                logger.warning("Could not determine column names from response")
                columns = []

            if columns:
                df = pd.DataFrame(response.result.data_array, columns=columns)
                logger.debug(f"Found {len(df)} inventory records")
                return tuple(df.to_dict("records"))
            else:
                logger.error("No columns found in response")
                return ()

        return ()

    return find_inventory_by_sku


def create_find_store_inventory_by_sku_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)
    """Create a Unity Catalog tool for finding store-specific inventory by SKU."""

    @tool
    def find_store_inventory_by_sku(store: str, skus: list[str]) -> tuple:
        """
        Find store-specific inventory details by one or more SKUs using Unity Catalog functions.
        This tool retrieves detailed inventory information for a specific store based on SKU codes.

        Args:
            store (str): The store identifier to retrieve inventory for (can be store ID or store name)
            skus (list[str]): One or more unique identifiers to retrieve.
                             SKU values are between 5-8 alpha numeric characters.
                             Examples: ["PET-KCP-001", "DUN-KCP-002"]

        Returns:
            (tuple): A tuple containing store-specific inventory information with fields like:
                inventory_id BIGINT
                ,sku STRING
                ,product_id BIGINT
                ,store_id INT (or similar store identifier)
                ,store_quantity INT
                ,warehouse STRING
                ,warehouse_quantity INT
                ,retail_amount DECIMAL(11, 2)
                ,popularity_rating STRING
                ,department STRING
                ,aisle_location STRING
                ,is_closeout BOOLEAN
                (Additional fields may be available depending on Unity Catalog function definition)
        """
        logger.debug(f"find_store_inventory_by_sku: store={store}, skus={skus}")

        # Get workspace client
        w = WorkspaceClient()

        # First, determine if store is an ID or name and get the store ID
        store_id = None

        # Check if store is already a numeric ID
        if store.isdigit():
            store_id = int(store)
        else:
            # Look up store ID by name
            store_lookup_query = f"""
                SELECT store_id FROM {schema.full_name}.dim_stores 
                WHERE store_name = '{store}'
                LIMIT 1
            """

            store_response = w.statement_execution.execute_statement(
                warehouse_id=warehouse.warehouse_id,
                statement=store_lookup_query,
                wait_timeout="30s",
            )

            if store_response.status.state != StatementState.SUCCEEDED:
                logger.error(f"Store lookup query failed: {store_response.status}")
                return ()

            if (
                store_response.result
                and store_response.result.data_array
                and len(store_response.result.data_array) > 0
            ):
                store_id = store_response.result.data_array[0][0]
                logger.debug(f"Found store ID {store_id} for store name '{store}'")
            else:
                logger.error(f"Store '{store}' not found in dim_stores table")
                return ()

        # Convert list to SQL array format
        skus_str = ", ".join([f"'{sku}'" for sku in skus])

        # Execute the Unity Catalog function with basic query
        sql_query = f"""
            SELECT * FROM {schema.full_name}.find_store_inventory_by_sku({store_id}, ARRAY({skus_str}))
        """

        response = w.statement_execution.execute_statement(
            warehouse_id=warehouse.warehouse_id,
            statement=sql_query,
            wait_timeout="30s",
        )

        if response.status.state != StatementState.SUCCEEDED:
            logger.error(f"Query failed: {response.status}")
            return ()

        # Convert results to DataFrame and then to tuple
        if response.result and response.result.data_array:
            # Try to get column names from different possible locations
            columns = None

            # Try the manifest approach first (older SDK versions)
            if hasattr(response.result, "manifest") and hasattr(
                response.result.manifest, "schema"
            ):
                columns = [col.name for col in response.result.manifest.schema.columns]
            # Try the schema approach (newer SDK versions)
            elif hasattr(response.result, "schema") and hasattr(
                response.result.schema, "columns"
            ):
                columns = [col.name for col in response.result.schema.columns]
            # Fallback: try to infer from the first row of data
            elif response.result.data_array and len(response.result.data_array) > 0:
                # Use generic column names based on the number of columns
                num_cols = (
                    len(response.result.data_array[0])
                    if response.result.data_array[0]
                    else 0
                )
                columns = [f"col_{i}" for i in range(num_cols)]
            else:
                logger.warning("Could not determine column names from response")
                columns = []

            if columns:
                df = pd.DataFrame(response.result.data_array, columns=columns)
                logger.debug(f"Found {len(df)} store inventory records")
                return tuple(df.to_dict("records"))
            else:
                logger.error("No columns found in response")
                return ()

        return ()

    return find_store_inventory_by_sku


def create_find_nearby_stores_inventory_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)
    """Create a tool for finding inventory at nearby stores."""

    @tool
    def find_nearby_stores_inventory(
        reference_store: str, skus: list[str], radius_miles: float = 5.0
    ) -> tuple:
        """
        Find inventory for products at stores near a reference location.
        This tool helps customers find products at nearby store locations when the primary store is out of stock.

        Args:
            reference_store (str): The reference store name or ID to search around (e.g., "Downtown Market")
            skus (list[str]): One or more SKU codes to check inventory for
                             Examples: ["ADI-GAZ-001", "ADI-SMB-001"]
            radius_miles (float): Search radius in miles (default: 5.0)

        Returns:
            (tuple): A tuple containing nearby store inventory information with fields:
                store_name STRING
                ,store_address STRING
                ,distance_miles FLOAT
                ,sku STRING
                ,store_quantity INT
                ,retail_amount DECIMAL(11, 2)
                ,aisle_location STRING
                ,store_phone STRING
                ,estimated_travel_time_minutes INT
        """
        logger.debug(
            f"find_nearby_stores_inventory: reference_store={reference_store}, skus={skus}, radius={radius_miles}"
        )

        # Mock data for stores near Downtown Market in SF
        nearby_stores_data = []

        for sku in skus:
            # Marina Market - 2.1 miles from Downtown Market
            nearby_stores_data.append(
                {
                    "store_name": "Marina Market",
                    "store_address": "2200 Chestnut Street, San Francisco, CA 94123",
                    "distance_miles": 2.1,
                    "sku": sku,
                    "store_quantity": 12 if sku == "ADI-GAZ-001" else 8,
                    "retail_amount": 89.99 if "GAZ" in sku else 94.99,
                    "aisle_location": "Aisle 3A",
                    "store_phone": "415-555-0102",
                    "estimated_travel_time_minutes": 15,
                }
            )

            # Mission Market - 2.8 miles from Downtown Market
            nearby_stores_data.append(
                {
                    "store_name": "Mission Market",
                    "store_address": "2128 Mission Street, San Francisco, CA 94110",
                    "distance_miles": 2.8,
                    "sku": sku,
                    "store_quantity": 18 if sku == "ADI-GAZ-001" else 14,
                    "retail_amount": 89.99 if "GAZ" in sku else 94.99,
                    "aisle_location": "Aisle 6C",
                    "store_phone": "415-555-0103",
                    "estimated_travel_time_minutes": 20,
                }
            )

            # Union Square Market - 1.2 miles from Downtown Market
            nearby_stores_data.append(
                {
                    "store_name": "Union Square Market",
                    "store_address": "350 Post Street, San Francisco, CA 94108",
                    "distance_miles": 1.2,
                    "sku": sku,
                    "store_quantity": 6 if sku == "ADI-GAZ-001" else 10,
                    "retail_amount": 89.99 if "GAZ" in sku else 94.99,
                    "aisle_location": "Aisle 2B",
                    "store_phone": "415-555-0104",
                    "estimated_travel_time_minutes": 8,
                }
            )

        logger.debug(f"Found {len(nearby_stores_data)} nearby store inventory records")
        return tuple(nearby_stores_data)

    return find_nearby_stores_inventory


def create_place_item_hold_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)
    """Create a tool for placing holds on items at specific stores."""

    @tool
    def place_item_hold(
        sku: str,
        store_name: str,
        size: str = None,
        customer_name: str = None,
        hold_duration_hours: int = 24,
    ) -> dict:
        """
        Place a hold on an item at a specific store for a customer.
        This tool allows store associates to reserve items for customers for pickup.

        Args:
            sku (str): The SKU code of the item to place on hold (e.g., "ADI-GAZ-001")
            store_name (str): The name of the store where the hold should be placed (e.g., "Marina Market")
            size (str, optional): The size of the item if applicable (e.g., "10", "Large", "XL")
            customer_name (str, optional): The customer's name for the hold (if not provided, will use "Customer")
            hold_duration_hours (int): Duration of the hold in hours (default: 24)

        Returns:
            (dict): A dictionary containing hold confirmation details:
                hold_id STRING - Unique identifier for the hold
                sku STRING - Product SKU
                store_name STRING - Store where item is held
                size STRING - Item size (if applicable)
                customer_name STRING - Customer name
                hold_expires_at STRING - When the hold expires
                confirmation_message STRING - Success message
                store_phone STRING - Store phone number for customer reference
                pickup_instructions STRING - Instructions for pickup
        """
        logger.debug(
            f"place_item_hold: sku={sku}, store={store_name}, size={size}, customer={customer_name}"
        )

        # Generate a unique hold ID
        import uuid
        from datetime import datetime, timedelta

        hold_id = f"HOLD-{uuid.uuid4().hex[:8].upper()}"
        current_time = datetime.now()
        expiry_time = current_time + timedelta(hours=hold_duration_hours)

        # Use default customer name if not provided
        if not customer_name:
            customer_name = "Customer"

        # Mock store phone numbers and details
        store_details = {
            "Marina Market": {
                "phone": "415-555-0102",
                "address": "2200 Chestnut Street, San Francisco, CA 94123",
            },
            "Mission Market": {
                "phone": "415-555-0103",
                "address": "2128 Mission Street, San Francisco, CA 94110",
            },
            "Union Square Market": {
                "phone": "415-555-0104",
                "address": "350 Post Street, San Francisco, CA 94108",
            },
            "Downtown Market": {
                "phone": "415-555-0101",
                "address": "123 Market Street, San Francisco, CA 94102",
            },
        }

        store_info = store_details.get(
            store_name,
            {"phone": "415-555-0100", "address": "Store address not available"},
        )

        # Create size description
        size_desc = f" in size {size}" if size else ""

        # Create product description based on SKU
        product_descriptions = {
            "ADI-GAZ-001": "Adidas Gazelle Sneakers",
            "ADI-SMB-001": "Adidas Samba Classic Sneakers",
            "ADI-STS-001": "Adidas Stan Smith Classic Sneakers",
            "ADI-SUP-001": "Adidas Superstar Classic Sneakers",
            "ADI-CAM-001": "Adidas Campus Classic Sneakers",
            "NIK-AF1-001": "Nike Air Force 1 Low",
            "CON-CHK-001": "Converse Chuck Taylor All Star",
            "VAN-OLD-001": "Vans Old Skool",
        }

        product_name = product_descriptions.get(sku, f"Product {sku}")

        hold_result = {
            "hold_id": hold_id,
            "sku": sku,
            "product_name": product_name,
            "store_name": store_name,
            "store_phone": store_info["phone"],
            "store_address": store_info["address"],
            "size": size,
            "customer_name": customer_name,
            "hold_duration_hours": hold_duration_hours,
            "hold_expires_at": expiry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "confirmation_message": f"Successfully placed a {hold_duration_hours}-hour hold on {product_name}{size_desc} at {store_name} for {customer_name}.",
            "pickup_instructions": f"Please bring a valid ID when picking up your item. Ask for hold ID {hold_id} at customer service.",
            "hold_created_at": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        logger.debug(f"Created hold {hold_id} for {customer_name}")
        return hold_result

    return place_item_hold
