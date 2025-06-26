"""Team Insights page for store managers."""

import streamlit as st
import streamlit_modal as modal

from components.chat import show_chat_container
from components.styles import load_css


def main():
    """Main team insights page."""
    # Load CSS
    load_css()

    # Page header
    col1, col2 = st.columns([8, 2])
    with col1:
        st.title("👥 Team Insights")
        st.markdown("**Monitor team performance and manage staff effectively**")

    with col2:
        if st.button("🤖 AI Assistant", use_container_width=True):
            st.session_state.show_chat = True

    # Create the chat modal
    chat_modal = modal.Modal(
        "AI Assistant", key="team_insights_chat_modal", max_width=800
    )

    # Handle chat modal
    if st.session_state.get("show_chat", False):
        chat_modal.open()
        st.session_state.show_chat = False

    # Modal content
    if chat_modal.is_open():
        with chat_modal.container():
            # Get chat config with fallback
            chat_config = st.session_state.get("config", {}).get(
                "chat",
                {
                    "placeholder": "How can I help you with team insights?",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            )

            # Show the chat container
            show_chat_container(chat_config)

    # Add custom CSS for better tab styling (same as other pages)
    st.markdown(
        """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px !important;
        padding: 12px 24px !important;
        background-color: #f8f9fa !important;
        border-radius: 8px 8px 0px 0px !important;
        border: 1px solid #dee2e6 !important;
        border-bottom: none !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #495057 !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 22px !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef !important;
        color: #212529 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
    }
    .stTabs [aria-selected="true"] p {
        color: white !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Main content in tabs - fully tab-based experience
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Performance", "Scheduling", "Analytics", "Feedback"]
    )

    with tab1:
        show_performance_overview()

    with tab2:
        show_scheduling_overview()

    with tab3:
        show_team_analytics()

    with tab4:
        show_feedback_management()


def show_performance_overview():
    """Display team performance overview."""
    # Team overview stats at top of tab
    st.markdown("#### 👥 Team Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="team-stat-card active">
                <div class="stat-icon">👥</div>
                <div class="stat-value">12/15</div>
                <div class="stat-label">Staff Present</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="team-stat-card performance">
                <div class="stat-icon">📊</div>
                <div class="stat-value">94%</div>
                <div class="stat-label">Avg Performance</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="team-stat-card tasks">
                <div class="stat-icon">✅</div>
                <div class="stat-value">47/52</div>
                <div class="stat-label">Tasks Complete</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="team-stat-card satisfaction">
                <div class="stat-icon">😊</div>
                <div class="stat-value">4.6/5</div>
                <div class="stat-label">Team Satisfaction</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("### 🏆 Team Performance")

    # Top performers
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 🌟 Top Performers Today")

        performers = [
            {
                "name": "Sarah Chen",
                "role": "Store Associate",
                "department": "Women's Fashion",
                "score": 98,
                "metrics": {
                    "BOPIS Orders": 15,
                    "Customer Assists": 8,
                    "Sales": "$2,450",
                },
                "achievements": [
                    "Customer Service Excellence",
                    "Sales Target Exceeded",
                ],
            },
            {
                "name": "Mike Rodriguez",
                "role": "Store Associate",
                "department": "Electronics",
                "score": 95,
                "metrics": {
                    "BOPIS Orders": 12,
                    "Customer Assists": 6,
                    "Sales": "$3,200",
                },
                "achievements": ["Tech Expert", "Upselling Champion"],
            },
            {
                "name": "Emma Wilson",
                "role": "Store Associate",
                "department": "Customer Service",
                "score": 92,
                "metrics": {
                    "Customer Assists": 18,
                    "Returns Processed": 12,
                    "Satisfaction": "4.9/5",
                },
                "achievements": ["Customer Satisfaction Leader"],
            },
        ]

        for performer in performers:
            show_performer_card(performer)

    with col2:
        st.markdown("#### 📊 Performance Distribution")

        performance_data = [
            {"range": "90-100%", "count": 5, "color": "#28a745"},
            {"range": "80-89%", "count": 6, "color": "#ffc107"},
            {"range": "70-79%", "count": 3, "color": "#fd7e14"},
            {"range": "Below 70%", "count": 1, "color": "#dc3545"},
        ]

        for data in performance_data:
            st.markdown(
                f"""
                <div class="performance-bar">
                    <div class="performance-label">{data["range"]}</div>
                    <div class="performance-count" style="background-color: {data["color"]}">
                        {data["count"]} employees
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("#### ⚠️ Needs Attention")

        attention_items = [
            {
                "employee": "James Park",
                "issue": "Late arrivals (3 times)",
                "severity": "medium",
            },
            {
                "employee": "Lisa Wong",
                "issue": "Low customer satisfaction",
                "severity": "high",
            },
            {
                "employee": "Emma Wilson",
                "issue": "Missed training session",
                "severity": "low",
            },
        ]

        for item in attention_items:
            severity_colors = {"high": "#dc3545", "medium": "#ffc107", "low": "#28a745"}
            st.markdown(
                f"""
                <div class="attention-item" style="border-left-color: {severity_colors[item["severity"]]}">
                    <div class="attention-employee">{item["employee"]}</div>
                    <div class="attention-issue">{item["issue"]}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )


def show_scheduling_overview():
    """Display scheduling overview and management."""
    st.markdown("### 📅 Schedule Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📋 Current Week Schedule")

        # Weekly schedule grid
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        shifts = ["Morning", "Afternoon", "Evening"]

        schedule_data = {
            "Mon": {"Morning": 5, "Afternoon": 4, "Evening": 3},
            "Tue": {"Morning": 5, "Afternoon": 4, "Evening": 3},
            "Wed": {"Morning": 4, "Afternoon": 5, "Evening": 3},
            "Thu": {"Morning": 5, "Afternoon": 4, "Evening": 3},
            "Fri": {"Morning": 6, "Afternoon": 5, "Evening": 4},
            "Sat": {"Morning": 6, "Afternoon": 6, "Evening": 4},
            "Sun": {"Morning": 4, "Afternoon": 4, "Evening": 3},
        }

        # Create schedule table
        for shift in shifts:
            cols = st.columns(8)
            with cols[0]:
                st.markdown(f"**{shift}**")
            for i, day in enumerate(days):
                with cols[i + 1]:
                    count = schedule_data[day][shift]
                    color = (
                        "#28a745"
                        if count >= 5
                        else "#ffc107"
                        if count >= 3
                        else "#dc3545"
                    )
                    st.markdown(
                        f"<div style='text-align: center; color: {color}; font-weight: bold;'>{count}</div>",
                        unsafe_allow_html=True,
                    )

    with col2:
        st.markdown("#### ⚠️ Schedule Alerts")

        alerts = [
            {
                "type": "Understaffed",
                "details": "Electronics dept needs coverage 2-6 PM (Mike Rodriguez sick)",
                "severity": "high",
                "action": "Need 1 more associate",
            },
            {
                "type": "Overtime Risk",
                "details": "Sarah Chen - 42 hours this week",
                "severity": "medium",
                "action": "Consider reducing Friday shift",
            },
            {
                "type": "Break Coverage",
                "details": "No coverage for lunch breaks Thu",
                "severity": "medium",
                "action": "Adjust break schedule",
            },
            {
                "type": "Training Due",
                "details": "3 employees need safety training",
                "severity": "low",
                "action": "Schedule training session",
            },
        ]

        for alert in alerts:
            show_schedule_alert(alert)

        st.markdown("#### 📝 Quick Actions")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Add Shift", use_container_width=True):
                st.info("Shift creation form would open")
        with col_b:
            if st.button("Request Coverage", use_container_width=True):
                st.info("Coverage request form would open")


def show_team_analytics():
    """Display team analytics and insights."""
    st.markdown("### 📈 Team Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Department Performance")

        departments = [
            {
                "name": "Women's Fashion",
                "staff": 4,
                "avg_performance": 91,
                "sales_per_hour": "$285",
                "customer_satisfaction": 4.7,
            },
            {
                "name": "Electronics",
                "staff": 3,
                "avg_performance": 88,
                "sales_per_hour": "$420",
                "customer_satisfaction": 4.5,
            },
            {
                "name": "Men's Fashion",
                "staff": 3,
                "avg_performance": 85,
                "sales_per_hour": "$195",
                "customer_satisfaction": 4.3,
            },
            {
                "name": "Customer Service",
                "staff": 2,
                "avg_performance": 94,
                "sales_per_hour": "N/A",
                "customer_satisfaction": 4.8,
            },
        ]

        for dept in departments:
            st.markdown(
                f"""
                <div class="department-card">
                    <div class="dept-header">
                        <span class="dept-name">{dept["name"]}</span>
                        <span class="dept-staff">{dept["staff"]} staff</span>
                    </div>
                    <div class="dept-metrics">
                        <div class="dept-metric">
                            <span class="metric-label">Performance:</span>
                            <span class="metric-value">{dept["avg_performance"]}%</span>
                        </div>
                        <div class="dept-metric">
                            <span class="metric-label">Sales/Hour:</span>
                            <span class="metric-value">{dept["sales_per_hour"]}</span>
                        </div>
                        <div class="dept-metric">
                            <span class="metric-label">Satisfaction:</span>
                            <span class="metric-value">{dept["customer_satisfaction"]}/5</span>
                        </div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### 📅 Trends & Insights")

        insights = [
            {
                "title": "Peak Performance Hours",
                "data": "2-4 PM shows highest productivity",
                "trend": "up",
                "impact": "positive",
            },
            {
                "title": "Training Impact",
                "data": "Recent training improved scores by 12%",
                "trend": "up",
                "impact": "positive",
            },
            {
                "title": "Staffing Efficiency",
                "data": "Weekend overstaffing by 15%",
                "trend": "down",
                "impact": "negative",
            },
            {
                "title": "Customer Satisfaction",
                "data": "Consistent 4.6+ rating for 3 weeks",
                "trend": "stable",
                "impact": "positive",
            },
        ]

        for insight in insights:
            trend_icon = {"up": "📈", "down": "📉", "stable": "➡️"}[insight["trend"]]
            impact_color = {
                "positive": "#28a745",
                "negative": "#dc3545",
                "neutral": "#6c757d",
            }[insight["impact"]]

            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-header">
                        <span class="insight-title">{insight["title"]}</span>
                        <span class="insight-trend">{trend_icon}</span>
                    </div>
                    <div class="insight-data" style="color: {impact_color}">
                        {insight["data"]}
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )


def show_feedback_management():
    """Display feedback and communication management."""
    st.markdown("### 💬 Team Feedback & Communication")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📝 Recent Feedback")

        feedback_items = [
            {
                "employee": "Sarah Chen",
                "type": "Positive",
                "message": "Excellent customer service with VIP client today",
                "date": "2 hours ago",
                "source": "Customer Review",
            },
            {
                "employee": "Mike Rodriguez",
                "type": "Suggestion",
                "message": "Requests additional training on new iPhone models",
                "date": "1 day ago",
                "source": "Employee Request",
            },
            {
                "employee": "Emma Wilson",
                "type": "Recognition",
                "message": "Handled difficult return situation professionally",
                "date": "2 days ago",
                "source": "Manager Observation",
            },
        ]

        for feedback in feedback_items:
            feedback_colors = {
                "Positive": "#28a745",
                "Suggestion": "#007bff",
                "Recognition": "#6f42c1",
                "Concern": "#dc3545",
            }

            st.markdown(
                f"""
                <div class="feedback-card">
                    <div class="feedback-header">
                        <span class="feedback-employee">{feedback["employee"]}</span>
                        <span class="feedback-type" style="background-color: {feedback_colors[feedback["type"]]}">
                            {feedback["type"]}
                        </span>
                    </div>
                    <div class="feedback-message">{feedback["message"]}</div>
                    <div class="feedback-meta">
                        <span>{feedback["source"]}</span> • <span>{feedback["date"]}</span>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### 📢 Team Announcements")

        if st.button("Create Announcement", type="primary", use_container_width=True):
            st.info("Announcement creation form would open")

        announcements = [
            {
                "title": "Holiday Schedule Released",
                "message": "Check your schedules for holiday season adjustments",
                "date": "Today",
                "priority": "high",
            },
            {
                "title": "New Product Training",
                "message": "Training session scheduled for Friday 2 PM",
                "date": "Yesterday",
                "priority": "medium",
            },
            {
                "title": "Team Building Event",
                "message": "Monthly team lunch this Saturday",
                "date": "3 days ago",
                "priority": "low",
            },
        ]

        for announcement in announcements:
            priority_colors = {"high": "#dc3545", "medium": "#ffc107", "low": "#28a745"}

            st.markdown(
                f"""
                <div class="announcement-card">
                    <div class="announcement-header">
                        <span class="announcement-title">{announcement["title"]}</span>
                        <span class="announcement-priority" style="background-color: {priority_colors[announcement["priority"]]}">
                            {announcement["priority"].upper()}
                        </span>
                    </div>
                    <div class="announcement-message">{announcement["message"]}</div>
                    <div class="announcement-date">{announcement["date"]}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )


def show_performer_card(performer):
    """Display a top performer card."""
    st.markdown(
        f"""
        <div class="performer-card">
            <div class="performer-header">
                <div class="performer-info">
                    <div class="performer-name">{performer["name"]}</div>
                    <div class="performer-role">{performer["role"]} - {performer["department"]}</div>
                </div>
                <div class="performer-score">{performer["score"]}%</div>
            </div>
            <div class="performer-details">
                {" • ".join([f"{k}: {v}" for k, v in performer["metrics"].items()])}
            </div>
            <div class="achievements">
                🏆 {" • ".join(performer["achievements"])}
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("---")


def show_schedule_alert(alert):
    """Display a schedule alert."""
    severity_colors = {"high": "#dc3545", "medium": "#ffc107", "low": "#28a745"}

    st.markdown(
        f"""
        <div class="schedule-alert-card" style="border-left-color: {severity_colors[alert["severity"]]}">
            <div class="alert-header">
                <span class="alert-type">{alert["type"]}</span>
                <span class="alert-severity">{alert["severity"].upper()}</span>
            </div>
            <div class="alert-details">{alert["details"]}</div>
            <div class="alert-action">Action: {alert["action"]}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )


# Add custom CSS for team components
st.markdown(
    """
    <style>
    /* Global font improvements */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Enhanced team stat cards - Clean styling without colored borders */
    .team-stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .team-stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .team-stat-card .stat-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .team-stat-card .stat-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        display: block;
        line-height: 1.2;
    }
    
    .team-stat-card .stat-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced performer cards - Clean styling without colored borders */
    .performer-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
    }
    
    .performer-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    }
    
    .performer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .performer-name {
        font-weight: 700;
        font-size: 1.25rem;
        color: #1e293b;
        line-height: 1.3;
    }
    
    .performer-role {
        color: #64748b;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .performer-score {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    .performer-details {
        color: #475569;
        line-height: 1.6;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    
    .achievements {
        color: #475569;
        font-size: 1rem;
        font-weight: 500;
        line-height: 1.6;
        padding: 0.75rem;
        background: rgba(248, 250, 252, 0.8);
        border-radius: 8px;
        border: 1px solid rgba(226, 232, 240, 0.6);
    }
    
    /* Performance distribution bars */
    .performance-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        padding: 0.75rem;
        background: rgba(248, 250, 252, 0.8);
        border-radius: 8px;
        border: 1px solid rgba(226, 232, 240, 0.6);
    }
    
    .performance-label {
        font-weight: 600;
        color: #334155;
        font-size: 1rem;
    }
    
    .performance-count {
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    
    /* Attention items */
    .attention-item {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(226, 232, 240, 0.6);
        border-left: 4px solid #ef4444;
        transition: all 0.3s ease;
    }
    
    .attention-item:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .attention-employee {
        font-weight: 700;
        color: #1e293b;
        font-size: 1rem;
        margin-bottom: 0.25rem;
    }
    
    .attention-issue {
        color: #64748b;
        font-size: 0.875rem;
    }
    
    /* Enhanced analytics cards - Clean styling without colored borders */
    .analytics-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .analytics-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    }
    
    .analytics-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .analytics-title {
        font-weight: 700;
        font-size: 1.25rem;
        color: #1e293b;
        line-height: 1.3;
    }
    
    .analytics-value {
        font-weight: 700;
        font-size: 1.5rem;
        color: #1e293b;
    }
    
    .analytics-details {
        color: #475569;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .analytics-details div {
        margin-bottom: 0.5rem;
        padding: 0.25rem 0;
    }
    
    .analytics-details strong {
        color: #334155;
        font-weight: 600;
    }
    
    /* Enhanced feedback cards - Clean styling without colored borders */
    .feedback-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
    }
    
    .feedback-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    .feedback-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .feedback-type {
        font-weight: 700;
        color: #1e293b;
        font-size: 1.125rem;
    }
    
    .feedback-rating {
        color: #f59e0b;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .feedback-message {
        color: #475569;
        line-height: 1.6;
        font-size: 1rem;
        margin-bottom: 0.75rem;
        font-style: italic;
    }
    
    .feedback-meta {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Enhanced announcement cards - Clean styling without colored borders */
    .announcement-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
    }
    
    .announcement-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    .announcement-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .announcement-title {
        font-weight: 700;
        color: #1e293b;
        font-size: 1.125rem;
    }
    
    .announcement-priority {
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    
    .announcement-message {
        color: #475569;
        line-height: 1.6;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .announcement-date {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Enhanced schedule alert cards - Clean styling without colored borders */
    .schedule-alert-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
    }
    
    .schedule-alert-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    .alert-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .alert-type {
        font-weight: 700;
        color: #1e293b;
        font-size: 1.125rem;
    }
    
    .alert-severity {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .alert-details {
        color: #475569;
        line-height: 1.6;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .alert-action {
        color: #3b82f6;
        font-weight: 600;
        font-size: 1rem;
        font-style: italic;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Page title styling */
    h1 {
        color: #1e293b;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h3 {
        color: #334155;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h4 {
        color: #475569;
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
    }
    
    /* Enhanced markdown text */
    .stMarkdown p {
        font-size: 1rem;
        line-height: 1.6;
        color: #64748b;
    }
    
    /* Success/info message styling */
    .stSuccess, .stInfo {
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 500;
    }
    </style>
""",
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    main()
