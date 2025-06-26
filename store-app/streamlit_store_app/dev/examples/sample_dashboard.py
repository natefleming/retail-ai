"""
Sample Dashboard Component

An example component showing various Streamlit features.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def show_component():
    """Display the sample dashboard component."""
    st.header("📊 Sample Dashboard Component")
    
    st.markdown("""
    This is an example component created in the Development Playground. 
    You can use this as a template for your own components!
    """)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Revenue",
            value="$45,230",
            delta="12%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Active Users",
            value="1,234",
            delta="-3%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Conversion Rate",
            value="3.4%",
            delta="0.8%",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="Avg Order Value",
            value="$87.50",
            delta="5%",
            delta_color="normal"
        )
    
    st.divider()
    
    # Charts section
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 Revenue Trend")
        
        # Generate sample time series data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        revenue_data = pd.DataFrame({
            'Date': dates,
            'Revenue': np.random.normal(1000, 200, len(dates)).cumsum()
        })
        
        fig = px.line(
            revenue_data, 
            x='Date', 
            y='Revenue',
            title='Daily Revenue Trend',
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("🥧 Sales by Category")
        
        # Sample pie chart data
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports']
        values = [35, 25, 15, 15, 10]
        
        fig_pie = px.pie(
            values=values,
            names=categories,
            title='Sales Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig_pie.update_layout(height=400)
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.divider()
    
    # Interactive controls
    st.subheader("🎛️ Interactive Controls")
    
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar", "Line", "Area"],
            index=1
        )
    
    with control_col2:
        time_period = st.selectbox(
            "Time Period",
            ["Last 7 days", "Last 30 days", "Last 90 days"],
            index=1
        )
    
    with control_col3:
        show_trend = st.checkbox("Show Trend Line", value=True)
    
    # Dynamic chart based on controls
    st.subheader(f"📊 {chart_type} Chart - {time_period}")
    
    # Generate data based on time period
    days = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}[time_period]
    sample_dates = pd.date_range(end=datetime.now(), periods=days)
    sample_values = np.random.randint(50, 200, days)
    
    chart_data = pd.DataFrame({
        'Date': sample_dates,
        'Value': sample_values
    })
    
    if chart_type == "Bar":
        fig_dynamic = px.bar(chart_data, x='Date', y='Value')
    elif chart_type == "Line":
        fig_dynamic = px.line(chart_data, x='Date', y='Value')
    else:  # Area
        fig_dynamic = px.area(chart_data, x='Date', y='Value')
    
    if show_trend and chart_type != "Bar":
        # Add trend line
        z = np.polyfit(range(len(sample_values)), sample_values, 1)
        p = np.poly1d(z)
        fig_dynamic.add_trace(
            go.Scatter(
                x=sample_dates,
                y=p(range(len(sample_values))),
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color='red')
            )
        )
    
    fig_dynamic.update_layout(height=350)
    st.plotly_chart(fig_dynamic, use_container_width=True)
    
    # Success message
    st.success("🎉 Component loaded successfully! Edit this file to customize it.")


if __name__ == "__main__":
    show_component() 