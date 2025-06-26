"""My Schedule page for store associates."""

from datetime import datetime, timedelta

import streamlit as st
import streamlit_modal as modal

from components.chat import show_chat_container
from components.styles import load_css


def main():
    """Main schedule page."""
    # Load CSS
    load_css()

    # Page header
    col1, col2 = st.columns([8, 2])
    with col1:
        st.title("📅 My Schedule")
        st.markdown("**View your work schedule and manage time off requests**")

    with col2:
        if st.button("🤖 AI Assistant", use_container_width=True):
            st.session_state.show_chat = True

    # Create the chat modal
    chat_modal = modal.Modal("AI Assistant", key="schedule_chat_modal", max_width=800)

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
                    "placeholder": "How can I help you with your schedule?",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            )

            # Show the chat container
            show_chat_container(chat_config)

    # Add custom CSS for better tab styling (same as daily operations)
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
    tab1, tab2, tab3 = st.tabs(["This Week", "Next Week", "Shift Requests"])

    with tab1:
        show_weekly_schedule("current")

    with tab2:
        show_weekly_schedule("next")

    with tab3:
        show_shift_requests()


def show_weekly_schedule(week_type):
    """Display weekly schedule."""
    if week_type == "current":
        # Current shift status at top of tab
        st.markdown("#### 🕐 Current Shift Status")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                <div class="current-shift-card">
                    <div class="shift-status active">
                        <span class="status-indicator">🟢</span>
                        <span class="status-text">Currently Working</span>
                    </div>
                    <div class="shift-time">
                        <div class="shift-start">Started: 8:00 AM</div>
                        <div class="shift-end">Ends: 4:00 PM</div>
                    </div>
                    <div class="shift-details">
                        <div>Department: Women's Fashion</div>
                        <div>Break: 12:00 - 12:30 PM</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="time-tracking-card">
                    <div class="time-header">⏱️ Time Tracking</div>
                    <div class="time-worked">
                        <div class="time-value">4h 23m</div>
                        <div class="time-label">Hours Worked Today</div>
                    </div>
                    <div class="time-remaining">
                        <div class="time-value">3h 37m</div>
                        <div class="time-label">Remaining</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
                <div class="week-summary-card">
                    <div class="week-header">📊 This Week</div>
                    <div class="week-stats">
                        <div class="week-stat">
                            <span class="stat-value">32/40</span>
                            <span class="stat-label">Hours</span>
                        </div>
                        <div class="week-stat">
                            <span class="stat-value">4/5</span>
                            <span class="stat-label">Days</span>
                        </div>
                        <div class="week-stat">
                            <span class="stat-value">94%</span>
                            <span class="stat-label">Performance</span>
                        </div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### This Week's Schedule")
        base_date = datetime.now()
    else:
        st.markdown("### Next Week's Schedule")
        base_date = datetime.now() + timedelta(weeks=1)

    # Generate week dates
    start_of_week = base_date - timedelta(days=base_date.weekday())

    # Mock schedule data
    schedule = [
        {
            "day": "Monday",
            "date": start_of_week,
            "shift": "8:00 AM - 4:00 PM",
            "department": "Women's Fashion",
            "break": "12:00 - 12:30 PM",
            "status": "completed" if week_type == "current" else "scheduled",
        },
        {
            "day": "Tuesday",
            "date": start_of_week + timedelta(days=1),
            "shift": "8:00 AM - 4:00 PM",
            "department": "Electronics",
            "break": "12:00 - 12:30 PM",
            "status": "completed" if week_type == "current" else "scheduled",
        },
        {
            "day": "Wednesday",
            "date": start_of_week + timedelta(days=2),
            "shift": "10:00 AM - 6:00 PM",
            "department": "Customer Service",
            "break": "2:00 - 2:30 PM",
            "status": "in_progress" if week_type == "current" else "scheduled",
        },
        {
            "day": "Thursday",
            "date": start_of_week + timedelta(days=3),
            "shift": "8:00 AM - 4:00 PM",
            "department": "Visual Merchandising",
            "break": "12:00 - 12:30 PM",
            "status": "scheduled",
        },
        {
            "day": "Friday",
            "date": start_of_week + timedelta(days=4),
            "shift": "9:00 AM - 5:00 PM",
            "department": "Women's Fashion",
            "break": "1:00 - 1:30 PM",
            "status": "scheduled",
        },
        {
            "day": "Saturday",
            "date": start_of_week + timedelta(days=5),
            "shift": "OFF",
            "department": "",
            "break": "",
            "status": "off",
        },
        {
            "day": "Sunday",
            "date": start_of_week + timedelta(days=6),
            "shift": "OFF",
            "department": "",
            "break": "",
            "status": "off",
        },
    ]

    for day_schedule in schedule:
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

        status_colors = {
            "completed": "#28a745",
            "in_progress": "#007bff",
            "scheduled": "#6c757d",
            "off": "#ffc107",
        }

        status_icons = {
            "completed": "✅",
            "in_progress": "🔄",
            "scheduled": "📅",
            "off": "🏖️",
        }

        with col1:
            st.markdown(
                f"""
                <div class="schedule-day">
                    <div class="day-name">{day_schedule["day"]}</div>
                    <div class="day-date">{day_schedule["date"].strftime("%m/%d")}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="schedule-shift">
                    <div class="shift-time">{day_schedule["shift"]}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="schedule-department">
                    <div class="department-name">{day_schedule["department"]}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
                <div class="schedule-break">
                    <div class="break-time">{day_schedule["break"]}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col5:
            st.markdown(
                f"""
                <div class="schedule-status">
                    <span style="color: {status_colors[day_schedule["status"]]}">
                        {status_icons[day_schedule["status"]]} {day_schedule["status"].replace("_", " ").title()}
                    </span>
                </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("---")


def show_shift_requests():
    """Display shift request management."""
    st.markdown("### 🔄 Shift Requests & Changes")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Request Time Off")

        st.date_input("Date", min_value=datetime.now().date())
        st.selectbox("Type", ["Vacation", "Sick Leave", "Personal", "Swap Shift"])
        st.text_area(
            "Reason", placeholder="Please provide a reason for your request..."
        )

        if st.button("Submit Request", type="primary"):
            st.success("Request submitted successfully! Your manager will review it.")

    with col2:
        st.markdown("#### Pending Requests")

        requests = [
            {
                "date": "2024-01-15",
                "type": "Vacation",
                "status": "Pending",
                "submitted": "2024-01-10",
            },
            {
                "date": "2024-01-22",
                "type": "Swap Shift",
                "status": "Approved",
                "submitted": "2024-01-08",
            },
        ]

        for req in requests:
            status_color = "#ffc107" if req["status"] == "Pending" else "#28a745"
            st.markdown(
                f"""
                <div class="request-item">
                    <div class="request-header">
                        <span class="request-date">{req["date"]}</span>
                        <span class="request-status" style="color: {status_color}">{req["status"]}</span>
                    </div>
                    <div class="request-details">
                        <div>Type: {req["type"]}</div>
                        <div>Submitted: {req["submitted"]}</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )


# Add custom CSS for schedule components
st.markdown(
    """
    <style>
    /* Global font improvements */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Enhanced shift status cards - Clean styling without colored borders */
    .current-shift-card, .time-tracking-card, .week-summary-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
        height: 100%;
    }
    
    .current-shift-card:hover, .time-tracking-card:hover, .week-summary-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .shift-status {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .status-indicator {
        font-size: 1.5rem;
    }
    
    .status-text {
        font-weight: 700;
        font-size: 1.25rem;
        color: #1e293b;
    }
    
    .shift-time {
        margin-bottom: 1rem;
    }
    
    .shift-start, .shift-end {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .shift-details {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .shift-details div {
        margin-bottom: 0.5rem;
    }
    
    .time-header, .week-header {
        font-weight: 700;
        font-size: 1.25rem;
        color: #1e293b;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .time-worked, .time-remaining {
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .time-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        display: block;
        margin-bottom: 0.25rem;
    }
    
    .time-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .week-stats {
        display: flex;
        justify-content: space-around;
        gap: 1rem;
    }
    
    .week-stat {
        text-align: center;
        flex: 1;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        display: block;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced schedule display */
    .schedule-day, .schedule-shift, .schedule-department, .schedule-break, .schedule-status {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(226, 232, 240, 0.6);
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .schedule-day:hover, .schedule-shift:hover, .schedule-department:hover, .schedule-break:hover, .schedule-status:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .day-name {
        font-weight: 700;
        font-size: 1.125rem;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    
    .day-date {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .shift-time {
        font-weight: 600;
        font-size: 1rem;
        color: #334155;
    }
    
    .department-name {
        font-weight: 600;
        font-size: 1rem;
        color: #475569;
    }
    
    .break-time {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Enhanced request status cards - Clean without colored borders */
    .request-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
    }
    
    .request-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    .request-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    
    .request-type {
        font-weight: 700;
        font-size: 1.125rem;
        color: #1e293b;
    }
    
    .request-status {
        font-weight: 600;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.9);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .request-details {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Enhanced form styling */
    .stSelectbox > div > div, .stTextArea > div > div > textarea, .stDateInput > div > div > input {
        font-size: 1rem;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
        background-color: white;
    }
    
    .stSelectbox > div > div:focus-within, .stTextArea > div > div:focus-within, .stDateInput > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
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
