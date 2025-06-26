"""Performance Metrics tab for VP of Retail Operations."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def show_performance_metrics_tab():
    """Display comprehensive performance metrics and benchmarking."""

    st.markdown("#### 📈 Performance Metrics & Benchmarking")

    # Performance metric selector
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_category = st.selectbox(
            "Metric Category:",
            [
                "Financial Performance",
                "Operational Efficiency",
                "Customer Metrics",
                "Employee Performance",
            ],
            key="perf_metric_category",
        )

    with col2:
        st.selectbox(
            "Comparison Type:",
            [
                "Year-over-Year",
                "Quarter-over-Quarter",
                "Month-over-Month",
                "Peer Benchmark",
            ],
            key="perf_comparison_type",
        )

    with col3:
        st.selectbox(
            "Geographic Scope:",
            ["National", "Regional", "Market", "Top/Bottom Performers"],
            key="perf_geographic_scope",
        )

    with col4:
        st.selectbox(
            "Time Period:",
            ["Last 30 Days", "Last Quarter", "Last 6 Months", "Last Year"],
            key="perf_time_period",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Display metrics based on selected category
    if metric_category == "Financial Performance":
        show_financial_performance_metrics()
    elif metric_category == "Operational Efficiency":
        show_operational_efficiency_metrics()
    elif metric_category == "Customer Metrics":
        show_customer_performance_metrics()
    else:  # Employee Performance
        show_employee_performance_metrics()


def show_financial_performance_metrics():
    """Display financial performance metrics and analysis."""

    st.markdown("### 💰 Financial Performance Analysis")

    # Key financial metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">$47.8M</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+12.3%</div>
            </div>
            <div class="metric-label">Revenue</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">34.6%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+1.8%</div>
            </div>
            <div class="metric-label">Gross Margin</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">12.4%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+2.1%</div>
            </div>
            <div class="metric-label">Operating Margin</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">$8.7M</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+18.2%</div>
            </div>
            <div class="metric-label">EBITDA</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">+8.9%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+3.2pp</div>
            </div>
            <div class="metric-label">Same-Store Sales</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Financial performance breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Revenue Breakdown by Channel")

        revenue_channels = pd.DataFrame(
            {
                "Channel": [
                    "In-Store",
                    "Online",
                    "Mobile App",
                    "Phone Orders",
                    "Curbside",
                ],
                "Revenue_M": [32.4, 8.7, 4.2, 1.8, 0.7],
                "Growth_Rate": [10.2, 18.5, 24.3, -5.2, 45.8],
            }
        )

        fig_revenue = px.bar(
            revenue_channels,
            x="Channel",
            y="Revenue_M",
            color="Growth_Rate",
            title="Revenue by Channel ($M)",
            color_continuous_scale="RdYlGn",
        )
        fig_revenue.update_layout(height=350)
        st.plotly_chart(fig_revenue, use_container_width=True)

    with col2:
        st.markdown("#### 💹 Profitability Trends")

        # Generate monthly profitability data
        months = pd.date_range(start="2023-12-01", periods=12, freq="ME")
        profitability_data = pd.DataFrame(
            {
                "Month": months,
                "Gross_Margin": np.random.normal(34.6, 2.1, 12),
                "Operating_Margin": np.random.normal(12.4, 1.8, 12),
                "Net_Margin": np.random.normal(8.2, 1.5, 12),
            }
        )

        fig_profit = go.Figure()
        fig_profit.add_trace(
            go.Scatter(
                x=profitability_data["Month"],
                y=profitability_data["Gross_Margin"],
                mode="lines",
                name="Gross Margin",
                line=dict(color="#1f4e79"),
            )
        )
        fig_profit.add_trace(
            go.Scatter(
                x=profitability_data["Month"],
                y=profitability_data["Operating_Margin"],
                mode="lines",
                name="Operating Margin",
                line=dict(color="#28a745"),
            )
        )
        fig_profit.add_trace(
            go.Scatter(
                x=profitability_data["Month"],
                y=profitability_data["Net_Margin"],
                mode="lines",
                name="Net Margin",
                line=dict(color="#dc3545"),
            )
        )

        fig_profit.update_layout(
            height=350, title="Profitability Trends (%)", yaxis_title="Margin (%)"
        )
        st.plotly_chart(fig_profit, use_container_width=True)

    # Regional financial performance comparison
    st.markdown("#### 🌎 Regional Financial Performance Comparison")

    regional_financial = pd.DataFrame(
        {
            "Region": ["West Coast", "East Coast", "Southeast", "Central", "Northeast"],
            "Revenue_M": [12.4, 11.8, 8.7, 9.2, 5.7],
            "Gross_Margin": [36.2, 34.8, 37.1, 32.4, 33.6],
            "Operating_Margin": [14.1, 12.8, 15.3, 10.2, 11.7],
            "Profit_M": [1.75, 1.51, 1.33, 0.94, 0.67],
        }
    )

    # Create financial performance heatmap
    fig_heatmap = px.imshow(
        regional_financial.set_index("Region")[
            ["Revenue_M", "Gross_Margin", "Operating_Margin", "Profit_M"]
        ].T,
        title="Regional Financial Performance Heatmap",
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig_heatmap.update_layout(height=300)
    st.plotly_chart(fig_heatmap, use_container_width=True)


def show_operational_efficiency_metrics():
    """Display operational efficiency metrics and analysis."""

    st.markdown("### ⚙️ Operational Efficiency Analysis")

    # Key operational metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">8.4x</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+0.7x</div>
            </div>
            <div class="metric-label">Inventory Turnover</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">87.2%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+3.4%</div>
            </div>
            <div class="metric-label">Store Utilization</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">94.8%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+2.7%</div>
            </div>
            <div class="metric-label">Labor Efficiency</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">91.5%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+4.1%</div>
            </div>
            <div class="metric-label">Supply Chain Efficiency</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">$4.23</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">-$0.18</div>
            </div>
            <div class="metric-label">Cost per Transaction</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Operational efficiency breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📦 Inventory Performance")

        inventory_metrics = pd.DataFrame(
            {
                "Metric": [
                    "Stock Accuracy",
                    "Turnover Rate",
                    "Stockout Rate",
                    "Overstock Rate",
                    "Dead Stock",
                ],
                "Current": [97.2, 8.4, 2.8, 5.1, 1.3],
                "Target": [98.0, 9.0, 2.0, 4.0, 1.0],
                "Industry_Avg": [94.5, 7.2, 4.2, 6.8, 2.1],
            }
        )

        fig_inventory = go.Figure()
        fig_inventory.add_trace(
            go.Bar(
                x=inventory_metrics["Metric"],
                y=inventory_metrics["Current"],
                name="Current",
                marker_color="#1f4e79",
            )
        )
        fig_inventory.add_trace(
            go.Bar(
                x=inventory_metrics["Metric"],
                y=inventory_metrics["Target"],
                name="Target",
                marker_color="#28a745",
            )
        )
        fig_inventory.add_trace(
            go.Bar(
                x=inventory_metrics["Metric"],
                y=inventory_metrics["Industry_Avg"],
                name="Industry Avg",
                marker_color="#ffc107",
            )
        )

        fig_inventory.update_layout(height=350, barmode="group")
        st.plotly_chart(fig_inventory, use_container_width=True)

    with col2:
        st.markdown("#### 🕐 Time-Based Efficiency")

        time_metrics = pd.DataFrame(
            {
                "Process": [
                    "Order Processing",
                    "Checkout Time",
                    "Restocking",
                    "Customer Service",
                    "Delivery",
                ],
                "Avg_Time_Minutes": [12.3, 3.8, 45.2, 8.7, 127.4],
                "Efficiency_Score": [92.1, 88.4, 85.7, 91.3, 79.2],
            }
        )

        fig_time = px.scatter(
            time_metrics,
            x="Avg_Time_Minutes",
            y="Efficiency_Score",
            size=[20] * 5,
            hover_name="Process",
            title="Process Time vs Efficiency",
            color="Efficiency_Score",
            color_continuous_scale="RdYlGn",
        )
        fig_time.update_layout(height=350)
        st.plotly_chart(fig_time, use_container_width=True)

    # Store performance ranking
    st.markdown("#### 🏪 Store Performance Ranking")

    # Generate mock store performance data
    store_performance = pd.DataFrame(
        {
            "Store_ID": [f"ST-{str(i).zfill(3)}" for i in range(1, 21)],
            "Location": [f"Store {i}" for i in range(1, 21)],
            "Efficiency_Score": np.random.uniform(75, 98, 20),
            "Revenue_M": np.random.uniform(0.8, 2.4, 20),
            "Customer_Satisfaction": np.random.uniform(4.2, 4.9, 20),
            "Employee_Score": np.random.uniform(80, 95, 20),
        }
    ).round(2)

    # Sort by efficiency score
    store_performance = store_performance.sort_values(
        "Efficiency_Score", ascending=False
    )

    # Display top 10 performers
    col1, col2 = st.columns([2, 1])

    with col1:
        # Performance scatter plot
        fig_stores = px.scatter(
            store_performance,
            x="Revenue_M",
            y="Efficiency_Score",
            size="Customer_Satisfaction",
            color="Employee_Score",
            hover_name="Location",
            title="Store Performance: Revenue vs Efficiency",
            color_continuous_scale="Viridis",
        )
        fig_stores.update_layout(height=400)
        st.plotly_chart(fig_stores, use_container_width=True)

    with col2:
        st.markdown("**Top 10 Performing Stores**")

        for i, row in store_performance.head(10).iterrows():
            rank_color = (
                "#28a745"
                if row["Efficiency_Score"] >= 90
                else "#ffc107"
                if row["Efficiency_Score"] >= 85
                else "#dc3545"
            )

            st.markdown(
                f"""
            <div style="
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 6px;
                margin-bottom: 6px;
                border-left: 3px solid {rank_color};
                font-size: 13px;
            ">
                <div style="font-weight: 600;">{row["Location"]}</div>
                <div style="color: #6c757d;">
                    Efficiency: {row["Efficiency_Score"]:.1f}%<br>
                    Revenue: ${row["Revenue_M"]:.1f}M
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def show_customer_performance_metrics():
    """Display customer performance metrics and analysis."""

    st.markdown("### 👥 Customer Performance Analysis")

    # Customer metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">4.7/5.0</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+0.3</div>
            </div>
            <div class="metric-label">Customer Satisfaction</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">87.3%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+2.1%</div>
            </div>
            <div class="metric-label">Customer Retention</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">$67.45</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+5.2%</div>
            </div>
            <div class="metric-label">Average Order Value</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">$247</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+12%</div>
            </div>
            <div class="metric-label">Customer Lifetime Value</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">68</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+5</div>
            </div>
            <div class="metric-label">Net Promoter Score</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Customer segmentation analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Customer Segmentation Performance")

        segments = pd.DataFrame(
            {
                "Segment": [
                    "VIP Customers",
                    "Regular Customers",
                    "Occasional Buyers",
                    "New Customers",
                ],
                "Count": [2847, 15632, 8945, 4231],
                "Avg_Spend": [234.50, 89.20, 45.30, 67.80],
                "Satisfaction": [4.8, 4.7, 4.5, 4.4],
                "Retention": [95.2, 88.1, 72.4, 65.8],
            }
        )

        fig_segments = px.scatter(
            segments,
            x="Count",
            y="Avg_Spend",
            size="Satisfaction",
            color="Retention",
            hover_name="Segment",
            title="Customer Segments: Count vs Spend",
            color_continuous_scale="Viridis",
        )
        fig_segments.update_layout(height=350)
        st.plotly_chart(fig_segments, use_container_width=True)

    with col2:
        st.markdown("#### 📈 Customer Journey Metrics")

        journey_data = pd.DataFrame(
            {
                "Stage": [
                    "Awareness",
                    "Interest",
                    "Consideration",
                    "Purchase",
                    "Loyalty",
                ],
                "Conversion_Rate": [45.2, 32.8, 28.5, 18.7, 12.3],
                "Customer_Count": [18420, 8325, 5847, 3446, 2264],
                "Avg_Time_Days": [0, 2.3, 5.7, 8.2, 45.6],
            }
        )

        fig_journey = go.Figure()
        fig_journey.add_trace(
            go.Scatter(
                x=journey_data["Stage"],
                y=journey_data["Conversion_Rate"],
                mode="lines+markers",
                name="Conversion Rate (%)",
                yaxis="y",
                line=dict(color="#1f4e79", width=3),
                marker=dict(size=10),
            )
        )

        fig_journey.add_trace(
            go.Bar(
                x=journey_data["Stage"],
                y=journey_data["Customer_Count"],
                name="Customer Count",
                yaxis="y2",
                opacity=0.6,
                marker_color="#28a745",
            )
        )

        fig_journey.update_layout(
            height=350,
            title="Customer Journey Performance",
            yaxis=dict(title="Conversion Rate (%)", side="left"),
            yaxis2=dict(title="Customer Count", side="right", overlaying="y"),
        )

        st.plotly_chart(fig_journey, use_container_width=True)


def show_employee_performance_metrics():
    """Display employee performance metrics and analysis."""

    st.markdown("### 👨‍💼 Employee Performance Analysis")

    # Employee metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">4.2/5.0</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+0.4</div>
            </div>
            <div class="metric-label">Employee Satisfaction</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">108.4%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+8.4%</div>
            </div>
            <div class="metric-label">Productivity Index</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">12.8%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">-3.2%</div>
            </div>
            <div class="metric-label">Turnover Rate</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">94.7%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+12.3%</div>
            </div>
            <div class="metric-label">Training Completion</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">87.6%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+4.8%</div>
            </div>
            <div class="metric-label">Performance Score</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Employee performance analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Performance by Department")

        dept_performance = pd.DataFrame(
            {
                "Department": [
                    "Sales",
                    "Customer Service",
                    "Inventory",
                    "Operations",
                    "Management",
                ],
                "Avg_Performance": [89.2, 87.8, 85.4, 88.7, 92.1],
                "Employee_Count": [45, 32, 28, 18, 12],
                "Satisfaction": [4.3, 4.1, 4.0, 4.2, 4.5],
            }
        )

        fig_dept = px.bar(
            dept_performance,
            x="Department",
            y="Avg_Performance",
            color="Satisfaction",
            title="Department Performance Score",
            color_continuous_scale="Viridis",
        )
        fig_dept.update_layout(height=350)
        st.plotly_chart(fig_dept, use_container_width=True)

    with col2:
        st.markdown("#### 🎯 Training Impact Analysis")

        training_impact = pd.DataFrame(
            {
                "Training_Program": [
                    "Customer Service",
                    "Technical Skills",
                    "Leadership",
                    "Product Knowledge",
                    "Safety",
                ],
                "Completion_Rate": [96.5, 89.2, 78.4, 92.8, 98.1],
                "Performance_Improvement": [8.7, 12.3, 15.2, 6.9, 4.2],
                "Employee_Feedback": [4.4, 4.1, 4.6, 4.2, 3.9],
            }
        )

        fig_training = px.scatter(
            training_impact,
            x="Completion_Rate",
            y="Performance_Improvement",
            size="Employee_Feedback",
            hover_name="Training_Program",
            title="Training Completion vs Performance Impact",
            color="Employee_Feedback",
            color_continuous_scale="RdYlGn",
        )
        fig_training.update_layout(height=350)
        st.plotly_chart(fig_training, use_container_width=True)

    # Performance improvement recommendations
    st.markdown("#### 💡 Performance Improvement Recommendations")

    col1, col2, col3 = st.columns(3)

    recommendations = [
        {
            "category": "Training",
            "recommendations": [
                "Enhance technical skills training program",
                "Implement leadership development for high performers",
                "Add AI tool training modules",
            ],
        },
        {
            "category": "Recognition",
            "recommendations": [
                "Expand employee recognition program",
                "Implement peer-to-peer feedback system",
                "Create performance milestone rewards",
            ],
        },
        {
            "category": "Process",
            "recommendations": [
                "Streamline inventory management processes",
                "Implement flexible scheduling options",
                "Deploy performance analytics dashboard",
            ],
        },
    ]

    for i, rec_group in enumerate(recommendations):
        with [col1, col2, col3][i]:
            st.markdown(f"**{rec_group['category']} Improvements**")
            for rec in rec_group["recommendations"]:
                st.markdown(f"• {rec}")

    # Performance trends
    st.markdown("#### 📈 Employee Performance Trends")

    months = pd.date_range(start="2023-12-01", periods=12, freq="ME")
    performance_trends = pd.DataFrame(
        {
            "Month": months,
            "Overall_Performance": np.random.normal(87.6, 3.2, 12),
            "Employee_Satisfaction": np.random.normal(4.2, 0.3, 12),
            "Turnover_Rate": np.random.normal(12.8, 2.1, 12),
        }
    )

    fig_trends = go.Figure()
    fig_trends.add_trace(
        go.Scatter(
            x=performance_trends["Month"],
            y=performance_trends["Overall_Performance"],
            mode="lines+markers",
            name="Performance Score (%)",
            line=dict(color="#1f4e79", width=3),
        )
    )

    fig_trends.add_trace(
        go.Scatter(
            x=performance_trends["Month"],
            y=performance_trends["Employee_Satisfaction"] * 20,  # Scale for visibility
            mode="lines+markers",
            name="Satisfaction (scaled)",
            line=dict(color="#28a745", width=3),
            yaxis="y2",
        )
    )

    fig_trends.update_layout(
        height=350,
        title="Employee Performance Trends",
        yaxis=dict(title="Performance Score (%)", side="left"),
        yaxis2=dict(title="Satisfaction Score", side="right", overlaying="y"),
    )

    st.plotly_chart(fig_trends, use_container_width=True)
