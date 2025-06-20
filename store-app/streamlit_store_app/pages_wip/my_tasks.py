"""My Tasks page for store associates."""

import streamlit as st
import streamlit_modal as modal

from components.chat import show_chat_container
from components.styles import load_css


def main():
    """Main tasks page."""
    # Load CSS
    load_css()

    # Page header
    col1, col2 = st.columns([8, 2])
    with col1:
        st.title("📋 My Tasks")
        st.markdown("**Manage your daily assignments and priorities**")

    with col2:
        if st.button("🤖 AI Assistant", use_container_width=True):
            st.session_state.show_chat = True

    # Create the chat modal
    chat_modal = modal.Modal("AI Assistant", key="tasks_chat_modal", max_width=800)

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
                    "placeholder": "How can I help you with your tasks?",
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
        ["All Tasks", "BOPIS Orders", "Restocking", "Customer Service"]
    )

    with tab1:
        show_all_tasks()

    with tab2:
        show_bopis_tasks()

    with tab3:
        show_restocking_tasks()

    with tab4:
        show_service_tasks()


def show_all_tasks():
    """Display all tasks with filters at the top of the tab."""
    # Task filters at top of tab
    st.markdown("#### 🔍 Task Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        priority_filter = st.selectbox("Priority", ["All", "High", "Medium", "Low"])
    with col2:
        type_filter = st.selectbox(
            "Type",
            ["All", "BOPIS", "Restocking", "Customer Service", "Visual Merchandising"],
        )
    with col3:
        status_filter = st.selectbox(
            "Status", ["All", "Pending", "In Progress", "Completed"]
        )

    st.markdown("---")

    # Mock tasks data focused on retail/fashion
    tasks = get_all_tasks()

    # Filter tasks
    filtered_tasks = tasks
    if priority_filter != "All":
        filtered_tasks = [
            t for t in filtered_tasks if t["priority"].title() == priority_filter
        ]
    if type_filter != "All":
        filtered_tasks = [t for t in filtered_tasks if t["type"] == type_filter]
    if status_filter != "All":
        filtered_tasks = [
            t
            for t in filtered_tasks
            if t["status"].replace("_", " ").title() == status_filter
        ]

    st.markdown("### 📋 All Tasks")

    if filtered_tasks:
        for task in filtered_tasks:
            show_task_card(task, "all")
    else:
        st.info("No tasks matching your filters.")


def show_bopis_tasks():
    """Display BOPIS tasks only."""
    st.markdown("### 🛒 BOPIS Orders")

    tasks = get_all_tasks()
    bopis_tasks = [t for t in tasks if t["type"] == "BOPIS"]

    if bopis_tasks:
        for task in bopis_tasks:
            show_task_card(task, "bopis")
    else:
        st.info("No BOPIS orders available.")


def show_restocking_tasks():
    """Display restocking tasks only."""
    st.markdown("### 📦 Restocking Tasks")

    tasks = get_all_tasks()
    restock_tasks = [t for t in tasks if t["type"] == "Restocking"]

    if restock_tasks:
        for task in restock_tasks:
            show_task_card(task, "restock")
    else:
        st.info("No restocking tasks available.")


def show_service_tasks():
    """Display customer service and visual merchandising tasks."""
    st.markdown("### 🤝 Customer Service & Visual Merchandising")

    tasks = get_all_tasks()
    service_tasks = [
        t for t in tasks if t["type"] in ["Customer Service", "Visual Merchandising"]
    ]

    if service_tasks:
        for task in service_tasks:
            show_task_card(task, "service")
    else:
        st.info("No customer service or visual merchandising tasks available.")


def get_all_tasks():
    """Get all tasks data."""
    return [
        {
            "id": 1,
            "type": "BOPIS",
            "title": "Order #B2024-0156",
            "customer": "Sarah Johnson",
            "items": ["Designer Handbag", "Silk Scarf", "Sunglasses"],
            "priority": "high",
            "due": "10:30 AM",
            "status": "pending",
            "location": "Customer Service Desk",
            "notes": "VIP customer - handle with care",
        },
        {
            "id": 2,
            "type": "BOPIS",
            "title": "Order #B2024-0157",
            "customer": "Michael Chen",
            "items": ["Wireless Headphones"],
            "priority": "medium",
            "due": "11:15 AM",
            "status": "pending",
            "location": "Electronics Section",
            "notes": "Gift wrapping requested",
        },
        {
            "id": 3,
            "type": "Restocking",
            "title": "Women's Designer Section",
            "location": "Floor 2 - Designer",
            "items": ["Fall Jackets", "Evening Dresses", "Accessories"],
            "priority": "high",
            "due": "12:00 PM",
            "status": "in_progress",
            "notes": "New collection display setup required",
        },
        {
            "id": 4,
            "type": "Customer Service",
            "title": "Personal Shopping Appointment",
            "customer": "Emma Rodriguez",
            "location": "Private Styling Room",
            "priority": "high",
            "due": "2:00 PM",
            "status": "pending",
            "notes": "Wedding guest outfit consultation - budget $500-800",
        },
        {
            "id": 5,
            "type": "Visual Merchandising",
            "title": "Electronics Display Update",
            "location": "Electronics Section",
            "items": ["iPhone 15 Cases", "Smart Watches", "Headphones"],
            "priority": "medium",
            "due": "3:00 PM",
            "status": "pending",
            "notes": "Highlight new arrivals and promotions",
        },
        {
            "id": 6,
            "type": "Restocking",
            "title": "Men's Footwear",
            "location": "Men's Department",
            "items": ["Dress Shoes", "Sneakers", "Boots"],
            "priority": "low",
            "due": "4:00 PM",
            "status": "pending",
            "notes": "Check sizes and arrange by style",
        },
    ]


def show_task_card(task, task_type):
    """Display a detailed task card with actions."""
    priority_colors = {"high": "#dc3545", "medium": "#ffc107", "low": "#28a745"}

    with st.container():
        col1, col2, col3 = st.columns([6, 2, 2])

        with col1:
            # Build HTML content step by step to avoid f-string issues
            html_content = f"""
                <div class="task-detail-card {task_type}">
                    <div class="task-header">
                        <span class="task-title">{task["title"]}</span>
                        <span class="task-priority" style="background-color: {priority_colors[task["priority"]]}; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                            {task["priority"].upper()}
                        </span>
                    </div>
                    <div class="task-details">
                        <div><strong>Due:</strong> {task["due"]}</div>
                        <div><strong>Location:</strong> {task["location"]}</div>
            """

            # Add customer info if present
            if "customer" in task:
                html_content += (
                    f"<div><strong>Customer:</strong> {task['customer']}</div>"
                )

            # Add items info if present
            if "items" in task:
                items_str = ", ".join(task["items"])
                html_content += f"<div><strong>Items:</strong> {items_str}</div>"

            # Add notes and close the HTML
            html_content += f"""
                        <div><strong>Notes:</strong> {task["notes"]}</div>
                    </div>
                </div>
            """

            st.markdown(html_content, unsafe_allow_html=True)

        with col2:
            status_options = ["pending", "in_progress", "completed"]
            current_status = task["status"]
            new_status = st.selectbox(
                "Status",
                status_options,
                index=status_options.index(current_status),
                key=f"status_{task_type}_{task['id']}",
            )

            if new_status != current_status:
                st.success(f"Status updated to {new_status.replace('_', ' ').title()}")

        with col3:
            if task["type"] == "BOPIS":
                if st.button(
                    "Start Picking",
                    key=f"action_{task_type}_{task['id']}",
                    use_container_width=True,
                ):
                    st.success("Started picking order!")
            elif task["type"] == "Restocking":
                if st.button(
                    "Begin Restock",
                    key=f"action_{task_type}_{task['id']}",
                    use_container_width=True,
                ):
                    st.success("Restocking task started!")
            elif task["type"] == "Customer Service":
                if st.button(
                    "Start Service",
                    key=f"action_{task_type}_{task['id']}",
                    use_container_width=True,
                ):
                    st.success("Customer service initiated!")
            elif task["type"] == "Visual Merchandising":
                if st.button(
                    "Start Setup",
                    key=f"action_{task_type}_{task['id']}",
                    use_container_width=True,
                ):
                    st.success("Visual merchandising started!")

        st.markdown("---")


# Add custom CSS for task cards
st.markdown(
    """
    <style>
    /* Global font improvements */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Enhanced task cards - Clean styling without colored borders */
    .task-detail-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .task-detail-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    }
    
    .task-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .task-title {
        font-weight: 700;
        font-size: 1.25rem;
        color: #1e293b;
        line-height: 1.3;
    }
    
    .task-priority {
        font-weight: 600;
        font-size: 0.875rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: white;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .task-details {
        color: #475569;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .task-details div {
        margin-bottom: 0.5rem;
        padding: 0.25rem 0;
    }
    
    .task-details strong {
        color: #334155;
        font-weight: 600;
    }
    
    /* Enhanced form styling */
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
