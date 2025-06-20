import pandas as pd
import pytest
from src.calculations.metrics import (
    analyze_inventory_status,
    calculate_daily_metrics,
    calculate_staff_metrics,
    generate_alerts,
)


@pytest.fixture
def sample_sales_data():
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "amount": [100, 200, 300],
            "status": ["completed", "pending", "completed"],
        }
    )


@pytest.fixture
def sample_inventory_data():
    return pd.DataFrame(
        {
            "sku": ["YM123", "TS456", "HG789"],
            "quantity": [5, 25, 2],
            "min_stock": [10, 15, 5],
        }
    )


@pytest.fixture
def sample_staff_data():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "status": ["active", "break", "active"],
            "role": ["associate", "manager", "cashier"],
        }
    )


def test_calculate_daily_metrics(sample_sales_data):
    metrics = calculate_daily_metrics(sample_sales_data)
    assert isinstance(metrics, dict)
    assert "daily_sales" in metrics
    assert "pending_orders" in metrics
    assert metrics["daily_sales"] > 0


def test_analyze_inventory_status(sample_inventory_data):
    status = analyze_inventory_status(sample_inventory_data)
    assert isinstance(status, dict)
    assert "low_inventory_items" in status
    assert "restock_needed" in status
    assert isinstance(status["restock_needed"], list)


def test_calculate_staff_metrics(sample_staff_data):
    metrics = calculate_staff_metrics(sample_staff_data)
    assert isinstance(metrics, dict)
    assert "staff_count" in metrics
    assert "utilization_rate" in metrics
    assert 0 <= metrics["utilization_rate"] <= 1


def test_generate_alerts():
    metrics = {"pending_orders": 5, "low_inventory_items": 2, "utilization_rate": 0.9}
    alerts = generate_alerts(metrics)
    assert isinstance(alerts, list)
    assert len(alerts) > 0
    assert all(isinstance(alert, dict) for alert in alerts)
    assert all("type" in alert for alert in alerts)
