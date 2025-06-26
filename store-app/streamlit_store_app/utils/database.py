"""Database utilities for the Streamlit Store App."""

import os
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

# Global constants
STORES_TABLE = "retail_consumer_goods.store_ops.dim_stores"


def get_connection():
    """Get a connection to Databricks SQL warehouse."""
    cfg = Config()
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=lambda: cfg.authenticate,
    )


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stores_from_databricks() -> pd.DataFrame:
    """Get stores from the actual Databricks dim_stores table."""
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                query_sql = f"""
                SELECT 
                    store_id,
                    store_name,
                    store_address,
                    store_city,
                    store_state,
                    store_zipcode,
                    store_country,
                    store_phone,
                    store_email,
                    store_area_sqft,
                    is_open_24_hours,
                    latitude,
                    longitude,
                    region_id
                FROM {STORES_TABLE}
                WHERE store_country = 'USA'
                ORDER BY store_name
                """
                cursor.execute(query_sql)

                # Fetch results and convert to DataFrame
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

                if not rows:
                    st.warning("No stores found in the database. Using fallback data.")
                    return get_stores_fallback()

                df = pd.DataFrame(rows, columns=columns)

                # Convert to the format expected by the app
                stores_data = []
                for _, row in df.iterrows():
                    store_data = {
                        "id": row["store_id"],
                        "name": row["store_name"],
                        "address": row["store_address"],
                        "city": row["store_city"],
                        "state": row["store_state"],
                        "zip_code": row["store_zipcode"],
                        "phone": row["store_phone"],
                        "email": row["store_email"],
                        "size_sqft": int(row["store_area_sqft"])
                        if row["store_area_sqft"]
                        else 3000,
                        "rating": 4.5,  # Default rating since not in dim_stores
                        "type": "flagship"
                        if row["store_area_sqft"] and row["store_area_sqft"] > 3500
                        else "express",
                        "is_24_hours": row["is_open_24_hours"],
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                        "region_id": row["region_id"],
                        # Default hours - could be enhanced with actual hours data
                        "hours": {
                            "monday": {"open": "08:00", "close": "22:00"},
                            "tuesday": {"open": "08:00", "close": "22:00"},
                            "wednesday": {"open": "08:00", "close": "22:00"},
                            "thursday": {"open": "08:00", "close": "22:00"},
                            "friday": {"open": "08:00", "close": "23:00"},
                            "saturday": {"open": "09:00", "close": "23:00"},
                            "sunday": {"open": "09:00", "close": "21:00"},
                        }
                        if not row["is_open_24_hours"]
                        else {
                            "monday": {"open": "24/7", "close": "24/7"},
                            "tuesday": {"open": "24/7", "close": "24/7"},
                            "wednesday": {"open": "24/7", "close": "24/7"},
                            "thursday": {"open": "24/7", "close": "24/7"},
                            "friday": {"open": "24/7", "close": "24/7"},
                            "saturday": {"open": "24/7", "close": "24/7"},
                            "sunday": {"open": "24/7", "close": "24/7"},
                        },
                    }
                    stores_data.append(store_data)

                return pd.DataFrame(stores_data)

    except Exception as e:
        st.error(f"Error connecting to Databricks: {str(e)}")
        st.info("Using fallback store data.")
        return get_stores_fallback()


@st.cache_data(ttl=3600)  # Cache for 1 hour since fallback data doesn't change
def get_stores_fallback() -> pd.DataFrame:
    """Fallback store data when database is not available."""
    return pd.DataFrame(
        [
            {
                "id": "001",
                "name": "BrickMart Downtown",
                "address": "123 Main St",
                "city": "New York",
                "state": "NY",
                "zip_code": "10001",
                "phone": "(212) 555-1234",
                "email": "downtown@brickmart.com",
                "size_sqft": 5000,
                "rating": 4.5,
                "type": "flagship",
                "is_24_hours": True,
                "latitude": 40.7128,
                "longitude": -74.006,
                "region_id": "1",
                "hours": {
                    "monday": {"open": "24/7", "close": "24/7"},
                    "tuesday": {"open": "24/7", "close": "24/7"},
                    "wednesday": {"open": "24/7", "close": "24/7"},
                    "thursday": {"open": "24/7", "close": "24/7"},
                    "friday": {"open": "24/7", "close": "24/7"},
                    "saturday": {"open": "24/7", "close": "24/7"},
                    "sunday": {"open": "24/7", "close": "24/7"},
                },
            },
            {
                "id": "002",
                "name": "BrickMart Uptown",
                "address": "456 Broadway",
                "city": "New York",
                "state": "NY",
                "zip_code": "10002",
                "phone": "(212) 555-5678",
                "email": "uptown@brickmart.com",
                "size_sqft": 3000,
                "rating": 4.3,
                "type": "express",
                "is_24_hours": False,
                "latitude": 40.7138,
                "longitude": -74.007,
                "region_id": "1",
                "hours": {
                    "monday": {"open": "08:00", "close": "22:00"},
                    "tuesday": {"open": "08:00", "close": "22:00"},
                    "wednesday": {"open": "08:00", "close": "22:00"},
                    "thursday": {"open": "08:00", "close": "22:00"},
                    "friday": {"open": "08:00", "close": "23:00"},
                    "saturday": {"open": "09:00", "close": "23:00"},
                    "sunday": {"open": "09:00", "close": "21:00"},
                },
            },
            {
                "id": "080",
                "name": "BrickMart Financial District",
                "address": "343 Sansome St",
                "city": "San Francisco",
                "state": "CA",
                "zip_code": "94104",
                "phone": "(415) 555-1234",
                "email": "fidi@brickmart.com",
                "size_sqft": 3200,
                "rating": 4.7,
                "type": "flagship",
                "is_24_hours": False,
                "latitude": 37.7936,
                "longitude": -122.4014,
                "region_id": "6",
                "hours": {
                    "monday": {"open": "08:00", "close": "22:00"},
                    "tuesday": {"open": "08:00", "close": "22:00"},
                    "wednesday": {"open": "08:00", "close": "22:00"},
                    "thursday": {"open": "08:00", "close": "22:00"},
                    "friday": {"open": "08:00", "close": "23:00"},
                    "saturday": {"open": "09:00", "close": "23:00"},
                    "sunday": {"open": "09:00", "close": "21:00"},
                },
            },
        ]
    )


def get_stores():
    """Get list of available stores from Databricks or fallback data."""
    # Check if we should use mock data from config
    config = st.session_state.get("config", {})
    if config.get("database", {}).get("mock", True):
        return get_stores_fallback()
    else:
        return get_stores_from_databricks()


def query(sql: str) -> pd.DataFrame:
    """
    Mock database query function that returns sample data.
    In production, this would connect to a real database.

    Args:
        sql: SQL query string

    Returns:
        DataFrame containing query results
    """
    # Sample data for different query types
    if "FROM sales" in sql:
        return (
            pd.DataFrame(
                {
                    "amount": [random.uniform(100, 1000) for _ in range(10)],
                    "created_at": [
                        datetime.now() - timedelta(hours=i) for i in range(10)
                    ],
                }
            )
            .agg({"amount": "sum"})
            .iloc[0]
        )

    elif "FROM orders" in sql:
        if "COUNT" in sql:
            return random.randint(5, 15)
        else:
            return pd.DataFrame(
                {
                    "id": range(1, 11),
                    "customer_name": [f"Customer {i}" for i in range(1, 11)],
                    "status": random.choices(
                        ["pending", "processing", "completed"], k=10
                    ),
                    "amount": [random.uniform(50, 500) for _ in range(10)],
                    "created_at": [
                        datetime.now() - timedelta(hours=i) for i in range(10)
                    ],
                }
            )

    elif "FROM inventory" in sql:
        if "COUNT" in sql and "stock_level < reorder_point" in sql:
            return random.randint(3, 8)
        elif "COUNT" in sql and "stock_level = 0" in sql:
            return random.randint(1, 4)
        elif "SUM" in sql:
            return random.uniform(10000, 50000)
        else:
            return pd.DataFrame(
                {
                    "id": range(1, 11),
                    "name": [f"Product {i}" for i in range(1, 11)],
                    "category": random.choices(
                        ["Electronics", "Clothing", "Food", "Home"], k=10
                    ),
                    "stock_level": [random.randint(0, 100) for _ in range(10)],
                    "reorder_point": [20] * 10,
                    "unit_price": [random.uniform(10, 200) for _ in range(10)],
                }
            )

    elif "FROM staff" in sql:
        if "COUNT" in sql:
            if "active" in sql:
                return random.randint(8, 12)
            elif "on_leave" in sql:
                return random.randint(1, 3)
            elif "scheduled" in sql:
                return random.randint(4, 8)
        else:
            return pd.DataFrame(
                {
                    "id": range(1, 6),
                    "name": [f"Employee {i}" for i in range(1, 6)],
                    "role": random.choices(["Store Associate", "Manager"], k=5),
                    "start_time": [
                        "9:00 AM",
                        "10:00 AM",
                        "11:00 AM",
                        "2:00 PM",
                        "3:00 PM",
                    ],
                    "end_time": [
                        "5:00 PM",
                        "6:00 PM",
                        "7:00 PM",
                        "10:00 PM",
                        "11:00 PM",
                    ],
                }
            )

    elif "FROM alerts" in sql:
        alert_types = {
            "inventory": [
                "Low stock alert: 5 items below reorder point",
                "Out of stock: Product X needs immediate reorder",
            ],
            "order": [
                "High volume of pending orders",
                "Order #123 delayed in processing",
            ],
            "staff": [
                "Coverage needed for evening shift",
                "Staff meeting reminder: 3 PM today",
            ],
        }

        if "type = 'inventory'" in sql:
            messages = alert_types["inventory"]
        elif "type = 'order'" in sql:
            messages = alert_types["order"]
        elif "type = 'staff'" in sql:
            messages = alert_types["staff"]
        else:
            messages = sum(alert_types.values(), [])

        return pd.DataFrame(
            {
                "id": range(len(messages)),
                "type": ["warning"] * len(messages),
                "message": messages,
                "status": ["active"] * len(messages),
                "priority": ["high"] * len(messages),
            }
        )

    return pd.DataFrame()
