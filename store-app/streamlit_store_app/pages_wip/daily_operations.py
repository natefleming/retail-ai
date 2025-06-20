"""Daily Operations page for store managers."""

import streamlit as st
import streamlit_modal as modal

from components.chat import show_chat_container
from components.navigation import show_nav
from components.styles import load_css


def main():
    """Main daily operations page."""
    # Load CSS
    load_css()

    # Show navigation
    show_nav()

    # Page header
    col1, col2 = st.columns([8, 2])
    with col1:
        st.title("📋 Daily Operations")
        st.markdown("**Manage today's priorities and store operations**")

    with col2:
        if st.button("🤖 AI Assistant", use_container_width=True):
            st.session_state.show_chat = True

    # Create the chat modal
    chat_modal = modal.Modal("AI Assistant", key="daily_ops_chat_modal", max_width=800)

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
                    "placeholder": "How can I help you with daily operations?",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            )

            # Show the chat container
            show_chat_container(chat_config)

    # Add custom CSS for better tab styling (same as homepage)
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
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #495057 !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 20px !important;
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
        [
            "Today's Priorities",
            "Deliveries & Vendors",
            "Store Metrics",
            "Schedule Management",
        ]
    )

    with tab1:
        show_daily_priorities()

    with tab2:
        show_deliveries_vendors()

    with tab3:
        show_store_metrics()

    with tab4:
        show_schedule_management()


def show_daily_priorities():
    """Display today's operational priorities."""
    # Operations overview stats at top of tab
    st.markdown("#### 📊 Operations Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="ops-stat-card tasks">
                <div class="stat-icon">✅</div>
                <div class="stat-value">5/8</div>
                <div class="stat-label">Tasks Complete</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="ops-stat-card urgent">
                <div class="stat-icon">🚨</div>
                <div class="stat-value">2</div>
                <div class="stat-label">Urgent Items</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="ops-stat-card deliveries">
                <div class="stat-icon">🚚</div>
                <div class="stat-value">3</div>
                <div class="stat-label">Deliveries Today</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="ops-stat-card meetings">
                <div class="stat-icon">👥</div>
                <div class="stat-value">2</div>
                <div class="stat-label">Meetings Scheduled</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("### 🎯 Today's Priorities")

    priorities = [
        {
            "id": 1,
            "task": "Morning inventory check",
            "status": "completed",
            "time": "08:00",
            "assigned_to": "Sarah Chen",
            "department": "All Departments",
            "notes": "Completed successfully - no major issues found",
            "priority": "high",
        },
        {
            "id": 2,
            "task": "Staff meeting - Holiday season prep",
            "status": "in_progress",
            "time": "09:30",
            "assigned_to": "All Staff",
            "department": "Conference Room",
            "notes": "Discussing holiday promotions and staffing",
            "priority": "high",
        },
        {
            "id": 3,
            "task": "Vendor delivery - Designer Collection",
            "status": "pending",
            "time": "11:00",
            "assigned_to": "Mike Rodriguez",
            "department": "Receiving",
            "notes": "New fall designer pieces - priority display",
            "priority": "high",
        },
        {
            "id": 4,
            "task": "Weekly sales report review",
            "status": "pending",
            "time": "15:00",
            "assigned_to": "Manager",
            "department": "Office",
            "notes": "Analyze performance metrics and trends",
            "priority": "medium",
        },
        {
            "id": 5,
            "task": "Evening shift handover",
            "status": "pending",
            "time": "18:00",
            "assigned_to": "Emma Wilson",
            "department": "All Departments",
            "notes": "Brief evening team on day's activities",
            "priority": "medium",
        },
        {
            "id": 6,
            "task": "Security system check",
            "status": "pending",
            "time": "19:30",
            "assigned_to": "Security Team",
            "department": "All Areas",
            "notes": "Monthly security system maintenance",
            "priority": "low",
        },
    ]

    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("Status", ["All", "Pending", "In Progress", "Completed"])
    with col2:
        st.selectbox("Priority", ["All", "High", "Medium", "Low"])
    with col3:
        if st.button("Add New Task", type="primary"):
            st.info("Task creation form would open here")

    # Display priorities
    for priority in priorities:
        show_priority_card(priority)


def show_deliveries_vendors():
    """Display delivery and vendor information."""
    st.markdown("### 🚚 Deliveries & Vendors")

    deliveries = [
        {
            "vendor": "Fashion Forward Distributors",
            "time": "10:00 AM",
            "status": "arrived",
            "items": ["Women's Fall Collection", "Designer Handbags", "Accessories"],
            "contact": "Jennifer Smith",
            "phone": "(555) 123-4567",
            "notes": "Priority items for holiday display",
        },
        {
            "vendor": "Tech Solutions Inc",
            "time": "2:00 PM",
            "status": "scheduled",
            "items": ["iPhone 15 Cases", "Wireless Chargers", "Smart Watches"],
            "contact": "David Chen",
            "phone": "(555) 987-6543",
            "notes": "New product launch items",
        },
        {
            "vendor": "Luxury Goods Co",
            "time": "4:00 PM",
            "status": "delayed",
            "items": ["Designer Jewelry", "Premium Watches", "Gift Sets"],
            "contact": "Maria Rodriguez",
            "phone": "(555) 456-7890",
            "notes": "Delayed due to traffic - ETA 4:30 PM",
        },
    ]

    for delivery in deliveries:
        show_delivery_card(delivery)


def show_store_metrics():
    """Display current store performance metrics."""
    st.markdown("### 📊 Store Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💰 Sales Performance")
        st.markdown(
            """
            <div class="metrics-card sales">
                <div class="metric-row">
                    <span class="metric-label">Today's Sales:</span>
                    <span class="metric-value">$28,750</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Yesterday:</span>
                    <span class="metric-value">$24,320</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Change:</span>
                    <span class="metric-value positive">+18.2%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Weekly Target:</span>
                    <span class="metric-value">$150,000</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Progress:</span>
                    <span class="metric-value">67%</span>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### 👥 Customer Metrics")
        st.markdown(
            """
            <div class="metrics-card customers">
                <div class="metric-row">
                    <span class="metric-label">Visitors Today:</span>
                    <span class="metric-value">247</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Conversion Rate:</span>
                    <span class="metric-value">68%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Avg Transaction:</span>
                    <span class="metric-value">$171.50</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Peak Hour:</span>
                    <span class="metric-value">2:00 - 4:00 PM</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Returns:</span>
                    <span class="metric-value">7 items</span>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 📈 Department Performance")

    departments = [
        {
            "name": "Women's Fashion",
            "sales": "$12,400",
            "target": "$15,000",
            "performance": "83%",
        },
        {
            "name": "Electronics",
            "sales": "$8,750",
            "target": "$8,000",
            "performance": "109%",
        },
        {
            "name": "Men's Fashion",
            "sales": "$4,200",
            "target": "$5,000",
            "performance": "84%",
        },
        {
            "name": "Accessories",
            "sales": "$3,400",
            "target": "$3,500",
            "performance": "97%",
        },
    ]

    for dept in departments:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**{dept['name']}**")
        with col2:
            st.write(f"Sales: {dept['sales']}")
        with col3:
            st.write(f"Target: {dept['target']}")
        with col4:
            performance_color = (
                "green"
                if float(dept["performance"].rstrip("%")) >= 100
                else "orange"
                if float(dept["performance"].rstrip("%")) >= 80
                else "red"
            )
            st.markdown(
                f"<span style='color: {performance_color}'>**{dept['performance']}**</span>",
                unsafe_allow_html=True,
            )


def show_schedule_management():
    """Display schedule management tools."""
    st.markdown("### 📅 Schedule Management")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👥 Current Shift")

        current_staff = [
            {
                "name": "Sarah Chen",
                "role": "Store Associate",
                "department": "Women's Fashion",
                "shift": "8 AM - 4 PM",
                "status": "active",
            },
            {
                "name": "Mike Rodriguez",
                "role": "Store Associate",
                "department": "Electronics",
                "shift": "8 AM - 4 PM",
                "status": "active",
            },
            {
                "name": "Emma Wilson",
                "role": "Store Associate",
                "department": "Customer Service",
                "shift": "10 AM - 6 PM",
                "status": "active",
            },
            {
                "name": "James Park",
                "role": "Visual Merchandiser",
                "department": "All Floors",
                "shift": "9 AM - 5 PM",
                "status": "break",
            },
        ]

        for staff in current_staff:
            status_color = {"active": "#28a745", "break": "#ffc107", "off": "#6c757d"}[
                staff["status"]
            ]
            st.markdown(
                f"""
                <div class="staff-card">
                    <div class="staff-header">
                        <span class="staff-name">{staff["name"]}</span>
                        <span class="staff-status" style="color: {status_color}">●</span>
                    </div>
                    <div class="staff-details">
                        <div>{staff["role"]} - {staff["department"]}</div>
                        <div>Shift: {staff["shift"]}</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### ⚠️ Schedule Alerts")

        alerts = [
            {
                "type": "Coverage Gap",
                "time": "3:00 - 4:00 PM",
                "department": "Electronics",
                "severity": "high",
            },
            {
                "type": "Overtime Alert",
                "employee": "Sarah Chen",
                "hours": "42/40",
                "severity": "medium",
            },
            {
                "type": "Break Reminder",
                "employee": "Mike Rodriguez",
                "due": "In 30 minutes",
                "severity": "low",
            },
        ]

        for alert in alerts:
            severity_colors = {"high": "#dc3545", "medium": "#ffc107", "low": "#28a745"}

            # Build HTML content step by step to avoid f-string issues
            html_content = f"""
                <div class="alert-card" style="border-left-color: {severity_colors[alert["severity"]]}">
                    <div class="alert-type">{alert["type"]}</div>
                    <div class="alert-details">
            """

            # Add the main detail (time, employee, or due)
            main_detail = alert.get("time", alert.get("employee", alert.get("due", "")))
            html_content += main_detail

            # Add department if present
            if "department" in alert:
                html_content += f" - {alert['department']}"

            # Add hours if present
            if "hours" in alert:
                html_content += f" ({alert['hours']})"

            # Close the HTML
            html_content += """
                    </div>
                </div>
            """

            st.markdown(html_content, unsafe_allow_html=True)


def show_priority_card(priority):
    """Display a priority task card."""
    priority_colors = {"high": "#dc3545", "medium": "#ffc107", "low": "#28a745"}

    col1, col2, col3 = st.columns([6, 2, 2])

    with col1:
        st.markdown(
            f"""
            <div class="priority-detail-card">
                <div class="priority-header">
                    <span class="priority-task">{priority["task"]}</span>
                    <span class="priority-level" style="background-color: {priority_colors[priority["priority"]]}">
                        {priority["priority"].upper()}
                    </span>
                </div>
                <div class="priority-details">
                    <div><strong>Time:</strong> {priority["time"]}</div>
                    <div><strong>Assigned to:</strong> {priority["assigned_to"]}</div>
                    <div><strong>Department:</strong> {priority["department"]}</div>
                    <div><strong>Notes:</strong> {priority["notes"]}</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        current_status = priority["status"]
        status_options = ["pending", "in_progress", "completed"]
        new_status = st.selectbox(
            "Status",
            status_options,
            index=status_options.index(current_status),
            key=f"priority_status_{priority['id']}",
        )

        if new_status != current_status:
            st.success(f"Status updated to {new_status.replace('_', ' ').title()}")

    with col3:
        if st.button(
            "Edit Task", key=f"edit_{priority['id']}", use_container_width=True
        ):
            st.info("Task editing form would open here")

    st.markdown("---")


def show_delivery_card(delivery):
    """Display a delivery card."""
    status_colors = {"arrived": "#28a745", "scheduled": "#007bff", "delayed": "#dc3545"}

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            f"""
            <div class="delivery-card">
                <div class="delivery-header">
                    <span class="vendor-name">{delivery["vendor"]}</span>
                    <span class="delivery-status" style="color: {status_colors[delivery["status"]]}">
                        {delivery["status"].upper()}
                    </span>
                </div>
                <div class="delivery-details">
                    <div><strong>Time:</strong> {delivery["time"]}</div>
                    <div><strong>Contact:</strong> {delivery["contact"]} - {delivery["phone"]}</div>
                    <div><strong>Items:</strong> {", ".join(delivery["items"])}</div>
                    <div><strong>Notes:</strong> {delivery["notes"]}</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        if delivery["status"] == "arrived":
            if st.button(
                "Process Delivery",
                key=f"process_{delivery['vendor']}",
                use_container_width=True,
            ):
                st.success("Delivery processing started!")
        elif delivery["status"] == "scheduled":
            if st.button(
                "Mark Arrived",
                key=f"arrived_{delivery['vendor']}",
                use_container_width=True,
            ):
                st.success("Delivery marked as arrived!")
        else:  # delayed
            if st.button(
                "Update ETA",
                key=f"update_{delivery['vendor']}",
                use_container_width=True,
            ):
                st.info("ETA update form would open here")

    st.markdown("---")


# Add custom CSS for operations components
st.markdown(
    """
    <style>
    /* Global font improvements */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Modern card styling with larger fonts - Clean styling without colored borders */
    .ops-stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .ops-stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .ops-stat-card .stat-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .ops-stat-card .stat-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        display: block;
        line-height: 1.2;
    }
    
    .ops-stat-card .stat-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced card styling - Clean without colored borders */
    .priority-detail-card, .delivery-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .priority-detail-card:hover, .delivery-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    }
    
    .priority-header, .delivery-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .priority-task, .vendor-name {
        font-weight: 700;
        font-size: 1.25rem;
        color: #1e293b;
        line-height: 1.3;
    }
    
    .priority-level {
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .delivery-status {
        font-weight: 700;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.9);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .priority-details, .delivery-details {
        color: #475569;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .priority-details div, .delivery-details div {
        margin-bottom: 0.5rem;
        padding: 0.25rem 0;
    }
    
    .priority-details strong, .delivery-details strong {
        color: #334155;
        font-weight: 600;
    }
    
    /* Enhanced metrics cards - Clean without colored borders */
    .metrics-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .metric-row:last-child {
        border-bottom: none;
    }
    
    .metric-label {
        color: #64748b;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .metric-value {
        font-weight: 700;
        color: #1e293b;
        font-size: 1.125rem;
    }
    
    .metric-value.positive {
        color: #10b981;
    }
    
    /* Enhanced staff cards - Clean without colored borders */
    .staff-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
    }
    
    .staff-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    .staff-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    
    .staff-name {
        font-weight: 700;
        color: #1e293b;
        font-size: 1.125rem;
    }
    
    .staff-status {
        font-size: 1.5rem;
    }
    
    .staff-details {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .staff-details div {
        margin-bottom: 0.25rem;
    }
    
    /* Enhanced alert cards - Clean without colored borders */
    .alert-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
    }
    
    .alert-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    .alert-type {
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-size: 1.125rem;
    }
    
    .alert-details {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        background-color: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #1e293b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
    
    /* Enhanced selectbox styling */
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
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
    
    /* Department performance styling */
    .stColumns > div {
        padding: 0.5rem;
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
