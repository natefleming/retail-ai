"""Strategic Insights tab for VP of Retail Operations."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def show_strategic_insights_tab():
    """Display strategic insights focused on key business areas and ROI metrics."""

    st.markdown("#### 🎯 Strategic Focus Areas & Business Value")

    # Strategic focus area selector
    col1, col2, col3 = st.columns(3)

    with col1:
        focus_area = st.selectbox(
            "Strategic Focus Area:",
            [
                "Customer Experience & Monetization",
                "Employee Productivity Enhancement",
                "Supply Chain Resiliency",
                "AI-Powered Operations",
                "ROI & Investment Analysis",
            ],
            key="strategic_focus_area",
        )

    with col2:
        st.selectbox(
            "Time Horizon:",
            ["Current Quarter", "Next Quarter", "FY 2024", "3-Year Outlook"],
            key="strategic_time_horizon",
        )

    with col3:
        st.selectbox(
            "Metric View:",
            ["Financial Impact", "Operational Metrics", "Risk Assessment"],
            key="strategic_metric_view",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Display content based on selected focus area
    if focus_area == "Customer Experience & Monetization":
        show_customer_experience_insights()
    elif focus_area == "Employee Productivity Enhancement":
        show_employee_productivity_insights()
    elif focus_area == "Supply Chain Resiliency":
        show_supply_chain_insights()
    elif focus_area == "AI-Powered Operations":
        show_ai_operations_insights()
    else:  # ROI & Investment Analysis
        show_roi_investment_analysis()


def show_customer_experience_insights():
    """Display customer experience and monetization insights."""

    st.markdown("### 💝 Customer Experience & Monetization Strategy")

    # CX KPIs Overview
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
                <div class="metric-value">$247</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+12%</div>
            </div>
            <div class="metric-label">Customer Lifetime Value</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">87.3%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+2.1%</div>
            </div>
            <div class="metric-label">Retention Rate</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
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
    with col5:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">34.2%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+8.7%</div>
            </div>
            <div class="metric-label">Personalization Rate</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Customer Journey Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🛍️ Customer Journey Optimization")

        journey_stages = [
            "Awareness",
            "Consideration",
            "Purchase",
            "Post-Purchase",
            "Loyalty",
        ]
        satisfaction_scores = [4.2, 4.5, 4.8, 4.6, 4.3]
        improvement_potential = [0.3, 0.2, 0.1, 0.4, 0.7]

        fig_journey = go.Figure()

        fig_journey.add_trace(
            go.Scatter(
                x=journey_stages,
                y=satisfaction_scores,
                mode="lines+markers",
                name="Current Satisfaction",
                line=dict(color="#1f4e79", width=3),
                marker=dict(size=10),
            )
        )

        fig_journey.add_trace(
            go.Scatter(
                x=journey_stages,
                y=[s + p for s, p in zip(satisfaction_scores, improvement_potential)],
                mode="lines+markers",
                name="Potential with AI",
                line=dict(color="#28a745", width=3, dash="dash"),
                marker=dict(size=8),
            )
        )

        fig_journey.update_layout(
            height=300,
            yaxis_title="Satisfaction Score",
            yaxis=dict(range=[3.5, 5.0]),
            margin=dict(l=0, r=0, t=20, b=0),
        )

        st.plotly_chart(fig_journey, use_container_width=True)

    with col2:
        st.markdown("#### 💰 Revenue Impact by Initiative")

        initiatives = [
            "Personalized Recommendations",
            "Loyalty Program Enhancement",
            "Omnichannel Experience",
            "Customer Service AI",
            "Mobile App Optimization",
        ]

        current_impact = [2.3, 1.8, 3.1, 1.2, 0.9]
        potential_impact = [4.2, 3.5, 4.8, 2.8, 2.1]

        fig_impact = go.Figure()

        fig_impact.add_trace(
            go.Bar(
                y=initiatives,
                x=current_impact,
                name="Current Impact ($M)",
                orientation="h",
                marker_color="#667eea",
            )
        )

        fig_impact.add_trace(
            go.Bar(
                y=initiatives,
                x=potential_impact,
                name="Potential Impact ($M)",
                orientation="h",
                marker_color="#28a745",
                opacity=0.7,
            )
        )

        fig_impact.update_layout(
            height=300,
            xaxis_title="Revenue Impact ($M)",
            margin=dict(l=0, r=0, t=20, b=0),
            barmode="overlay",
        )

        st.plotly_chart(fig_impact, use_container_width=True)

    # CX Investment Portfolio
    st.markdown("#### 📊 Customer Experience Investment Portfolio")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Investment allocation sunburst
        cx_investments = pd.DataFrame(
            {
                "Category": [
                    "Technology",
                    "Technology",
                    "Technology",
                    "People",
                    "People",
                    "Process",
                    "Process",
                ],
                "Initiative": [
                    "AI/ML Platforms",
                    "Mobile Apps",
                    "Data Analytics",
                    "Staff Training",
                    "Customer Service",
                    "Journey Mapping",
                    "Feedback Systems",
                ],
                "Investment_M": [2.8, 1.2, 1.5, 1.8, 2.3, 0.7, 0.9],
                "ROI_Expected": [145, 112, 178, 89, 134, 67, 98],
            }
        )

        fig_sunburst = px.sunburst(
            cx_investments,
            path=["Category", "Initiative"],
            values="Investment_M",
            title="CX Investment Allocation ($M)",
        )
        fig_sunburst.update_layout(height=350)
        st.plotly_chart(fig_sunburst, use_container_width=True)

    with col2:
        st.markdown("**Strategic Recommendations**")

        recommendations = [
            {
                "priority": "High",
                "action": "Deploy AI-driven personalization",
                "impact": "$3.2M potential revenue",
                "timeline": "Q2 2024",
            },
            {
                "priority": "Medium",
                "action": "Enhance loyalty program",
                "impact": "8% retention improvement",
                "timeline": "Q3 2024",
            },
            {
                "priority": "High",
                "action": "Omnichannel integration",
                "impact": "$2.1M efficiency gain",
                "timeline": "Q1 2024",
            },
        ]

        for rec in recommendations:
            priority_color = "#dc3545" if rec["priority"] == "High" else "#ffc107"

            st.markdown(
                f"""
            <div style="
                background-color: #f8f9fa;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 10px;
                border-left: 4px solid {priority_color};
            ">
                <div style="font-weight: 600; color: #495057; margin-bottom: 5px;">
                    {rec["priority"]} Priority: {rec["action"]}
                </div>
                <div style="font-size: 13px; color: #6c757d;">
                    Impact: {rec["impact"]}<br>
                    Timeline: {rec["timeline"]}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def show_employee_productivity_insights():
    """Display employee productivity enhancement insights."""

    st.markdown("### 👥 Employee Productivity Enhancement")

    # Productivity KPIs
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">108.4%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+8.4%</div>
            </div>
            <div class="metric-label">Overall Productivity</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
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
    with col3:
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
    with col4:
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
    with col5:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">67.3%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+15.2%</div>
            </div>
            <div class="metric-label">AI Tool Adoption</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Productivity analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Productivity Trends by Department")

        departments = [
            "Sales",
            "Customer Service",
            "Inventory",
            "Operations",
            "Management",
        ]
        productivity_data = pd.DataFrame(
            {
                "Department": departments,
                "Q1_2023": np.random.uniform(85, 105, 5),
                "Q4_2023": np.random.uniform(95, 115, 5),
                "Target_2024": [110, 112, 108, 105, 115],
            }
        )

        fig_prod = go.Figure()
        fig_prod.add_trace(
            go.Bar(
                x=departments,
                y=productivity_data["Q1_2023"],
                name="Q1 2023",
                marker_color="#dc3545",
            )
        )
        fig_prod.add_trace(
            go.Bar(
                x=departments,
                y=productivity_data["Q4_2023"],
                name="Q4 2023",
                marker_color="#1f4e79",
            )
        )
        fig_prod.add_trace(
            go.Scatter(
                x=departments,
                y=productivity_data["Target_2024"],
                mode="markers",
                name="2024 Target",
                marker=dict(color="#28a745", size=12, symbol="diamond"),
            )
        )

        fig_prod.update_layout(height=300, yaxis_title="Productivity Index")
        st.plotly_chart(fig_prod, use_container_width=True)

    with col2:
        st.markdown("#### 🤖 AI Tools Impact on Productivity")

        ai_tools = [
            "Inventory AI",
            "Customer AI",
            "Schedule AI",
            "Analytics AI",
            "Training AI",
        ]
        adoption_rates = [78, 65, 72, 59, 43]
        productivity_gains = [15.2, 12.8, 9.4, 18.7, 8.3]

        fig_ai = px.scatter(
            x=adoption_rates,
            y=productivity_gains,
            size=[20] * 5,
            hover_name=ai_tools,
            title="AI Adoption vs Productivity Gain",
            labels={"x": "Adoption Rate (%)", "y": "Productivity Gain (%)"},
        )
        fig_ai.update_layout(height=300)
        st.plotly_chart(fig_ai, use_container_width=True)


def show_supply_chain_insights():
    """Display supply chain resiliency insights."""

    st.markdown("### 🚛 Supply Chain Resiliency")

    # Supply chain KPIs
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">97.2%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+2.1%</div>
            </div>
            <div class="metric-label">Inventory Accuracy</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">2.8%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">-1.3%</div>
            </div>
            <div class="metric-label">Stockout Rate</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">5.2 days</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">-0.8</div>
            </div>
            <div class="metric-label">Lead Time</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">94.6%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+3.2%</div>
            </div>
            <div class="metric-label">Supplier Reliability</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">$0.12/unit</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">-$0.03</div>
            </div>
            <div class="metric-label">Cost Efficiency</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Supply chain risk assessment
    st.markdown("#### ⚠️ Supply Chain Risk Assessment")

    risk_data = pd.DataFrame(
        {
            "Risk_Factor": [
                "Supplier Disruption",
                "Demand Volatility",
                "Transportation Delays",
                "Quality Issues",
                "Inventory Excess",
            ],
            "Probability": [0.25, 0.35, 0.20, 0.15, 0.30],
            "Impact": [8.5, 7.2, 5.8, 9.1, 6.3],
            "Mitigation_Status": [
                "In Progress",
                "Completed",
                "Planned",
                "In Progress",
                "Completed",
            ],
        }
    )

    risk_data["Risk_Score"] = risk_data["Probability"] * risk_data["Impact"]

    fig_risk = px.scatter(
        risk_data,
        x="Probability",
        y="Impact",
        size="Risk_Score",
        color="Mitigation_Status",
        hover_name="Risk_Factor",
        title="Supply Chain Risk Matrix",
    )
    fig_risk.update_layout(height=400)
    st.plotly_chart(fig_risk, use_container_width=True)


def show_ai_operations_insights():
    """Display AI-powered operations insights."""

    st.markdown("### 🤖 AI-Powered Operations")

    # AI operations KPIs
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">94.7%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+2.3%</div>
            </div>
            <div class="metric-label">AI Model Accuracy</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">67.8%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+12.5%</div>
            </div>
            <div class="metric-label">Automation Rate</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">12.3x faster</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+3.2x</div>
            </div>
            <div class="metric-label">Decision Speed</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">87.4%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+15.1%</div>
            </div>
            <div class="metric-label">Error Reduction</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">$2.8M</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+$0.7M</div>
            </div>
            <div class="metric-label">Cost Savings</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # AI implementation roadmap
    st.markdown("#### 🛣️ AI Implementation Roadmap")

    roadmap_data = pd.DataFrame(
        {
            "Initiative": [
                "Demand Forecasting",
                "Dynamic Pricing",
                "Inventory Optimization",
                "Customer Segmentation",
                "Predictive Maintenance",
            ],
            "Status": ["Deployed", "Pilot", "Development", "Planning", "Research"],
            "Business_Value": [2.3, 1.8, 3.2, 1.5, 0.9],
            "Implementation_Timeline": [
                "Q4 2023",
                "Q1 2024",
                "Q2 2024",
                "Q3 2024",
                "Q4 2024",
            ],
            "Risk_Level": ["Low", "Medium", "Low", "High", "Medium"],
        }
    )

    fig_roadmap = px.bar(
        roadmap_data,
        x="Initiative",
        y="Business_Value",
        color="Status",
        title="AI Initiative Business Value ($M)",
        color_discrete_map={
            "Deployed": "#28a745",
            "Pilot": "#17a2b8",
            "Development": "#ffc107",
            "Planning": "#fd7e14",
            "Research": "#6c757d",
        },
    )
    fig_roadmap.update_layout(height=350)
    st.plotly_chart(fig_roadmap, use_container_width=True)


def show_roi_investment_analysis():
    """Display ROI and investment analysis."""

    st.markdown("### 💰 ROI & Investment Analysis")

    # ROI overview
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">142%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+23%</div>
            </div>
            <div class="metric-label">Overall ROI</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">18 months</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">-6 months</div>
            </div>
            <div class="metric-label">Payback Period</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">$8.4M</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+$2.1M</div>
            </div>
            <div class="metric-label">NPV</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">34.7%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+8.2%</div>
            </div>
            <div class="metric-label">IRR</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">2.4x</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+0.7x</div>
            </div>
            <div class="metric-label">Investment Efficiency</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ROI analysis by investment category
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Hard ROI Metrics")

        hard_roi_data = pd.DataFrame(
            {
                "Category": [
                    "Technology",
                    "Process Optimization",
                    "Automation",
                    "Training",
                    "Infrastructure",
                ],
                "Investment_M": [5.2, 2.8, 3.7, 1.9, 4.1],
                "Returns_M": [8.9, 4.2, 6.8, 2.7, 5.3],
                "ROI_Percent": [171, 150, 184, 142, 129],
            }
        )

        fig_hard_roi = px.scatter(
            hard_roi_data,
            x="Investment_M",
            y="Returns_M",
            size="ROI_Percent",
            hover_name="Category",
            title="Investment vs Returns ($M)",
        )
        fig_hard_roi.add_trace(
            go.Scatter(
                x=[0, 6],
                y=[0, 6],
                mode="lines",
                name="Break-even Line",
                line=dict(dash="dash", color="red"),
            )
        )
        fig_hard_roi.update_layout(height=350)
        st.plotly_chart(fig_hard_roi, use_container_width=True)

    with col2:
        st.markdown("#### 🎯 Soft ROI Metrics")

        soft_metrics = [
            {
                "metric": "Customer Satisfaction",
                "improvement": "+0.3 points",
                "value": "High",
            },
            {"metric": "Employee Engagement", "improvement": "+12%", "value": "Medium"},
            {"metric": "Brand Perception", "improvement": "+18%", "value": "High"},
            {
                "metric": "Market Position",
                "improvement": "+2 ranking",
                "value": "Medium",
            },
            {
                "metric": "Competitive Advantage",
                "improvement": "Significant",
                "value": "High",
            },
        ]

        for metric in soft_metrics:
            value_color = "#28a745" if metric["value"] == "High" else "#ffc107"

            st.markdown(
                f"""
            <div style="
                background-color: #f8f9fa;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 8px;
                border-left: 4px solid {value_color};
            ">
                <div style="font-weight: 600; color: #495057;">
                    {metric["metric"]}: {metric["improvement"]}
                </div>
                <div style="font-size: 12px; color: #6c757d;">
                    Business Value: {metric["value"]}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Investment prioritization matrix
    st.markdown("#### 🎯 Investment Prioritization Matrix")

    investment_options = pd.DataFrame(
        {
            "Initiative": [
                "AI Customer Analytics",
                "Supply Chain AI",
                "Store Automation",
                "Mobile Platform",
                "Data Infrastructure",
            ],
            "Expected_ROI": [185, 167, 143, 129, 156],
            "Implementation_Ease": [7, 5, 6, 8, 4],
            "Strategic_Importance": [9, 8, 6, 7, 9],
            "Investment_Required": [3.2, 4.8, 6.1, 2.3, 5.7],
        }
    )

    fig_matrix = px.scatter(
        investment_options,
        x="Implementation_Ease",
        y="Expected_ROI",
        size="Investment_Required",
        color="Strategic_Importance",
        hover_name="Initiative",
        title="Investment Prioritization: ROI vs Implementation Ease",
        color_continuous_scale="Viridis",
    )
    fig_matrix.update_layout(height=400)
    st.plotly_chart(fig_matrix, use_container_width=True)
