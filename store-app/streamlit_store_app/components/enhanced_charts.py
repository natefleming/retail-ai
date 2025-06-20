"""
Enhanced Chart Components for Store Operations App

This module provides enhanced data visualization using popular third-party Streamlit components
to create beautiful, interactive charts for better business insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_echarts import st_echarts
import numpy as np
from datetime import datetime, timedelta


def create_sales_performance_chart(sales_data: pd.DataFrame):
    """
    Create an interactive sales performance chart using ECharts.
    
    Args:
        sales_data: DataFrame with sales data
    """
    
    # Prepare data for ECharts
    dates = sales_data['Date'].dt.strftime('%Y-%m-%d').tolist()
    sales_values = sales_data['Sales'].tolist()
    
    options = {
        "title": {
            "text": "Daily Sales Performance",
            "left": "center",
            "textStyle": {"fontSize": 20, "fontWeight": "bold"}
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "cross"},
            "formatter": "Date: {b}<br/>Sales: ${c:,.0f}"
        },
        "legend": {"data": ["Sales"], "top": "10%"},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "toolbox": {
            "feature": {
                "saveAsImage": {"title": "Save as Image"},
                "dataZoom": {"title": {"zoom": "Zoom", "back": "Reset Zoom"}},
                "restore": {"title": "Restore"},
                "magicType": {"type": ["line", "bar"], "title": {"line": "Line", "bar": "Bar"}}
            }
        },
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": dates,
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {
            "type": "value",
            "axisLabel": {"formatter": "${value}"}
        },
        "series": [
            {
                "name": "Sales",
                "type": "line",
                "stack": "Total",
                "smooth": True,
                "lineStyle": {"width": 3},
                "areaStyle": {"opacity": 0.3},
                "data": sales_values,
                "markPoint": {
                    "data": [
                        {"type": "max", "name": "Max"},
                        {"type": "min", "name": "Min"}
                    ]
                },
                "markLine": {
                    "data": [{"type": "average", "name": "Average"}]
                }
            }
        ]
    }
    
    st_echarts(options=options, height="400px", key="sales_performance")


def create_inventory_status_gauge(low_stock_count: int, total_products: int):
    """
    Create a gauge chart showing inventory health using ECharts.
    
    Args:
        low_stock_count: Number of products with low stock
        total_products: Total number of products
    """
    
    # Calculate inventory health percentage
    health_percentage = ((total_products - low_stock_count) / total_products) * 100
    
    options = {
        "title": {
            "text": "Inventory Health",
            "left": "center",
            "textStyle": {"fontSize": 18, "fontWeight": "bold"}
        },
        "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
        "series": [
            {
                "name": "Inventory Health",
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "center": ["50%", "75%"],
                "radius": "90%",
                "min": 0,
                "max": 100,
                "splitNumber": 8,
                "axisLine": {
                    "lineStyle": {
                        "width": 6,
                        "color": [
                            [0.3, "#FF6E76"],
                            [0.7, "#FDDD60"], 
                            [1, "#58D9F9"]
                        ]
                    }
                },
                "pointer": {
                    "icon": "path://M12.8,0.7l12,40.1H0.7L12.8,0.7z",
                    "length": "12%",
                    "width": 20,
                    "offsetCenter": [0, "-60%"],
                    "itemStyle": {"color": "auto"}
                },
                "axisTick": {"length": 12, "lineStyle": {"color": "auto", "width": 2}},
                "splitLine": {"length": 20, "lineStyle": {"color": "auto", "width": 5}},
                "axisLabel": {
                    "color": "#464646",
                    "fontSize": 20,
                    "distance": -60,
                    "rotate": "tangential"
                },
                "title": {"offsetCenter": [0, "-10%"], "fontSize": 20},
                "detail": {
                    "fontSize": 30,
                    "offsetCenter": [0, "-35%"],
                    "valueAnimation": True,
                    "color": "auto"
                },
                "data": [{"value": round(health_percentage, 1), "name": "Health %"}]
            }
        ]
    }
    
    st_echarts(options=options, height="300px", key="inventory_gauge")


def create_category_performance_radar(category_data: pd.DataFrame):
    """
    Create a radar chart showing performance across categories using ECharts.
    
    Args:
        category_data: DataFrame with category performance metrics
    """
    
    categories = category_data['Category'].tolist()
    sales_values = category_data['Sales'].tolist()
    
    # Normalize values to 0-100 scale for radar chart
    max_sales = max(sales_values)
    normalized_sales = [(val/max_sales)*100 for val in sales_values]
    
    options = {
        "title": {
            "text": "Category Performance Radar",
            "left": "center",
            "textStyle": {"fontSize": 18, "fontWeight": "bold"}
        },
        "tooltip": {"trigger": "item"},
        "legend": {"data": ["Sales Performance"], "top": "10%"},
        "radar": {
            "indicator": [{"name": cat, "max": 100} for cat in categories],
            "center": ["50%", "60%"],
            "radius": "75%"
        },
        "series": [
            {
                "name": "Category Performance",
                "type": "radar",
                "data": [
                    {
                        "value": normalized_sales,
                        "name": "Sales Performance",
                        "areaStyle": {"opacity": 0.3}
                    }
                ]
            }
        ]
    }
    
    st_echarts(options=options, height="400px", key="category_radar")


def create_advanced_plotly_dashboard(sales_data: pd.DataFrame, inventory_data: pd.DataFrame):
    """
    Create an advanced dashboard using Plotly with multiple subplots.
    
    Args:
        sales_data: DataFrame with sales data
        inventory_data: DataFrame with inventory data
    """
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sales Trend', 'Category Distribution', 'Stock Levels', 'Price vs Stock'),
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Sales trend line chart
    fig.add_trace(
        go.Scatter(
            x=sales_data['Date'],
            y=sales_data['Sales'],
            mode='lines+markers',
            name='Daily Sales',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Category pie chart
    category_sales = sales_data.groupby('Category')['Sales'].sum()
    fig.add_trace(
        go.Pie(
            labels=category_sales.index,
            values=category_sales.values,
            name="Category Sales",
            hole=0.4,
            marker_colors=['#3b82f6', '#ef4444', '#10b981', '#f59e0b']
        ),
        row=1, col=2
    )
    
    # Stock levels bar chart
    fig.add_trace(
        go.Bar(
            x=inventory_data['Product'],
            y=inventory_data['Stock'],
            name='Stock Levels',
            marker_color=['#ef4444' if status == 'Critical' else '#f59e0b' if status == 'Low Stock' else '#10b981' 
                         for status in inventory_data['Status']]
        ),
        row=2, col=1
    )
    
    # Price vs Stock scatter
    fig.add_trace(
        go.Scatter(
            x=inventory_data['Stock'],
            y=inventory_data['Price'],
            mode='markers+text',
            text=inventory_data['Product'],
            textposition="top center",
            name='Price vs Stock',
            marker=dict(
                size=12,
                color=inventory_data['Price'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Price ($)")
            )
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Store Operations Dashboard",
        title_x=0.5,
        title_font_size=24
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Sales ($)", row=1, col=1)
    fig.update_xaxes(title_text="Product", row=2, col=1)
    fig.update_yaxes(title_text="Stock Quantity", row=2, col=1)
    fig.update_xaxes(title_text="Stock Quantity", row=2, col=2)
    fig.update_yaxes(title_text="Price ($)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True, key="advanced_dashboard")


def create_real_time_metrics_chart():
    """
    Create a real-time metrics chart that updates with simulated data.
    """
    
    # Generate simulated real-time data
    current_time = datetime.now()
    time_points = [current_time - timedelta(minutes=x) for x in range(30, 0, -1)]
    
    # Simulate real-time metrics
    sales_per_minute = np.random.normal(50, 15, 30)
    customers_per_minute = np.random.normal(8, 3, 30)
    
    options = {
        "title": {
            "text": "Real-Time Store Metrics",
            "left": "center",
            "textStyle": {"fontSize": 18, "fontWeight": "bold"}
        },
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["Sales/min", "Customers/min"], "top": "10%"},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": [t.strftime('%H:%M') for t in time_points]
        },
        "yAxis": [
            {
                "type": "value",
                "name": "Sales ($)",
                "position": "left",
                "axisLabel": {"formatter": "${value}"}
            },
            {
                "type": "value", 
                "name": "Customers",
                "position": "right",
                "axisLabel": {"formatter": "{value}"}
            }
        ],
        "series": [
            {
                "name": "Sales/min",
                "type": "line",
                "yAxisIndex": 0,
                "data": [round(x, 1) for x in sales_per_minute],
                "smooth": True,
                "lineStyle": {"color": "#3b82f6", "width": 2},
                "areaStyle": {"opacity": 0.2}
            },
            {
                "name": "Customers/min",
                "type": "line", 
                "yAxisIndex": 1,
                "data": [max(0, round(x)) for x in customers_per_minute],
                "smooth": True,
                "lineStyle": {"color": "#10b981", "width": 2}
            }
        ]
    }
    
    st_echarts(options=options, height="350px", key="realtime_metrics")


def create_heatmap_chart(data: pd.DataFrame):
    """
    Create a heatmap showing sales by hour and day using Plotly.
    
    Args:
        data: DataFrame with datetime and sales data
    """
    
    # Create sample hourly data
    hours = list(range(24))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Generate sample heatmap data
    np.random.seed(42)
    heatmap_data = np.random.randint(10, 100, size=(7, 24))
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=hours,
        y=days,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Sales: $%{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Sales Heatmap by Day and Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True, key="sales_heatmap")


def show_charts_demo():
    """Demonstrate all enhanced chart components."""
    
    st.title("📊 Enhanced Charts Demo")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    sales_data = pd.DataFrame({
        'Date': dates,
        'Sales': [1000 + i*50 + np.random.randint(-200, 300) for i in range(len(dates))],
        'Store': ['Store A' if i%2==0 else 'Store B' for i in range(len(dates))],
        'Category': [['Electronics', 'Clothing', 'Home', 'Sports'][i%4] for i in range(len(dates))]
    })
    
    inventory_data = pd.DataFrame({
        'Product': ['iPhone 15', 'Samsung Galaxy', 'Nike Shoes', 'Levi Jeans', 'Coffee Maker'],
        'Stock': [25, 30, 45, 60, 15],
        'Price': [999, 899, 120, 89, 159],
        'Category': ['Electronics', 'Electronics', 'Sports', 'Clothing', 'Home'],
        'Status': ['Low Stock', 'In Stock', 'In Stock', 'In Stock', 'Critical']
    })
    
    # Chart demonstrations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sales Performance")
        create_sales_performance_chart(sales_data)
        
        st.subheader("🎯 Category Performance")
        category_data = sales_data.groupby('Category')['Sales'].sum().reset_index()
        create_category_performance_radar(category_data)
    
    with col2:
        st.subheader("📊 Inventory Health")
        low_stock_count = len(inventory_data[inventory_data['Status'].isin(['Low Stock', 'Critical'])])
        create_inventory_status_gauge(low_stock_count, len(inventory_data))
        
        st.subheader("⏱️ Real-Time Metrics")
        create_real_time_metrics_chart()
    
    st.markdown("---")
    
    st.subheader("🔥 Advanced Dashboard")
    create_advanced_plotly_dashboard(sales_data, inventory_data)
    
    st.markdown("---")
    
    st.subheader("🌡️ Sales Heatmap")
    create_heatmap_chart(sales_data)


if __name__ == "__main__":
    show_charts_demo() 