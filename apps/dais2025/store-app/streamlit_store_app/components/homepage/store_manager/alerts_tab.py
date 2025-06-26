"""Store manager alerts tab with interactive functionality."""

import streamlit as st
from streamlit_card import card
from .demo_alerts import show_demo_alert_controls, show_demo_alerts_display, DemoAlertSystem
import time


def show_manager_alerts_tab():
    """Display the Alerts tab with interactive counters and scrollable alert containers."""
    
    # Note: Demo alerts are now shown on the main dashboard
    # This tab shows regular store alerts only
    
    # Add custom CSS for scrollable alert containers and interactive elements
    st.markdown(
        """
    <style>
    /* =============================================================================
       REFACTORED ALERT TAB STYLES
       - Consolidated duplicate styles and removed redundant padding properties
       - Created shared base classes to reduce repetition
       - Maintained exact visual appearance while improving maintainability
       - Added clear section organization with improved comments
       ============================================================================= */
    
    /* BASE STYLES - Shared properties to reduce duplication */
    .base-card {
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .base-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
    }
    
    .base-gradient-bg {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    }
    
    /* SHARED PADDING CLASSES - Standardized spacing system */
    .padding-standard { padding: 20px; }          /* Most cards and containers */
    .padding-large { padding: 24px; }             /* Modal cards and important sections */
    .padding-medium { padding: 16px; }            /* Info items and smaller cards */
    .padding-small { padding: 6px 12px; }         /* Badges and small elements */
    
    /* METRIC CONTAINERS */
    .metric-container {
        @extend .base-card, .base-gradient-bg, .padding-standard;
        margin: 8px 0;
    }
    
    .metric-value-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #1e293b;
        line-height: 1;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
    }
    
    /* BADGES - Consolidated badge system with shared base styles */
    .base-badge {
        @extend .padding-small;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border: 1px solid;
    }
    
    .metric-badge {
        @extend .base-badge;
    }
    
    .metric-badge.positive {
        background: #dcfce7;
        color: #16a34a;
        border-color: #bbf7d0;
    }
    
    .metric-badge.neutral {
        background: #e0f2fe;
        color: #0369a1;
        border-color: #bae6fd;
    }
    
    .metric-badge.info {
        background: #fef3c7;
        color: #d97706;
        border-color: #fde68a;
    }
    
    /* ALERT CONTAINERS AND ITEMS */
    .alerts-container {
        @extend .base-gradient-bg;
        height: 400px;
        overflow-y: auto;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .alert-item {
        @extend .base-card, .base-gradient-bg;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        cursor: pointer;
    }
    
    .alert-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, currentColor 50%, transparent 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .alert-item:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
    }
    
    .alert-item:hover::before {
        opacity: 0.3;
    }
    
    /* ALERT STATE VARIANTS - Streamlined color system */
    .alert-item.urgent {
        border-left-color: #dc2626;
        color: #dc2626;
    }
    
    .alert-item.urgent:hover {
        background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
        border-color: #fecaca;
    }
    
    .alert-item.important {
        border-left-color: #d97706;
        color: #d97706;
    }
    
    .alert-item.important:hover {
        background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
        border-color: #fed7aa;
    }
    
    .alert-item.resolved {
        opacity: 0.6;
        background: linear-gradient(135deg, #f1f5f9 0%, #f8fafc 100%);
        border-left-color: #64748b;
    }
    
    /* ALERT CONTENT STRUCTURE */
    .alert-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .alert-type {
        font-weight: 700;
        color: #1e293b;
        font-size: 1rem;
        letter-spacing: -0.01em;
    }
    
    .alert-severity {
        @extend .base-badge;
        font-size: 0.75rem;
        color: white;
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.3);
    }
    
    .alert-severity.important {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        box-shadow: 0 2px 8px rgba(217, 119, 6, 0.3);
    }
    
    .alert-time {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .alert-message {
        color: #374151;
        margin-bottom: 1rem;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .alert-actions {
        display: flex;
        gap: 0.75rem;
        align-items: center;
    }
    
    .click-hint {
        font-size: 0.75rem;
        color: #3b82f6;
        font-style: italic;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* CARD COMPONENTS - Unified card system */
    .modern-card {
        @extend .base-card, .base-gradient-bg, .padding-large;
        margin: 16px 0;
    }
    
    .alert-card {
        @extend .base-card, .base-gradient-bg, .padding-standard;
        margin-bottom: 16px;
        border-left: 4px solid;
    }
    
    .alert-card.warning {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
    }
    
    .alert-card.info {
        border-left-color: #3b82f6;
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
    }
    
    .alert-card.success {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);
    }
    
    .region-card {
        @extend .base-gradient-bg, .padding-medium;
        border-radius: 12px;
        margin-bottom: 12px;
        border-left: 4px solid;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .region-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* LAYOUT COMPONENTS */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.01em;
    }
    
    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 16px;
    }
    
    .info-item {
        @extend .padding-medium;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        border-left: 4px solid #e2e8f0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .info-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    }
    
    /* TYPOGRAPHY - Consolidated text styling */
    .info-label {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        color: #64748b;
    }
    
    .info-value {
        font-size: 16px;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.4;
    }
    
    .alert-title {
        font-weight: 700;
        font-size: 16px;
        color: #1e293b;
        margin-bottom: 8px;
        letter-spacing: -0.01em;
    }
    
    .alert-description {
        color: #64748b;
        margin-bottom: 8px;
        line-height: 1.5;
    }
    
    .alert-action {
        font-size: 14px;
        color: #3b82f6;
        font-weight: 600;
    }
    
    .opportunity-impact {
        font-size: 14px;
        color: #059669;
        font-weight: 600;
        font-style: italic;
    }
    
    .region-name {
        font-weight: 700;
        font-size: 14px;
        color: #1e293b;
        margin-bottom: 8px;
    }
    
    .region-details {
        font-size: 13px;
        line-height: 1.5;
        color: #64748b;
    }
    
    .region-metric {
        font-weight: 600;
        color: #1e293b;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # # Auto-refresh logic for demo alerts
    # demo_active = st.session_state.get("demo_start_time") is not None
    
    # if demo_active:
    #     # Import demo system
    #     from .demo_alerts import DemoAlertSystem
    #     import time
        
    #     # Initialize demo system
    #     demo_system = DemoAlertSystem()
    #     demo_system.initialize_demo_state()
        
    #     # Check for new alerts and auto-refresh if needed
    #     new_alerts = demo_system.check_for_new_alerts()
    #     if new_alerts:
    #         # Show toast notifications for new alerts
    #         for alert in new_alerts:
    #             st.toast(f"{alert['sound_effect']} New {alert['severity'].upper()} Alert: {alert['type']}", icon="🚨")
    #         # Auto-refresh when new alerts appear
    #         st.rerun()
        
    #     # Auto-refresh every 3 seconds during demo
    #     time.sleep(3)
    #     st.rerun()

    # Initialize alert state
    if "resolved_alerts" not in st.session_state:
        st.session_state.resolved_alerts = set()
    if "show_alert_modal" not in st.session_state:
        st.session_state.show_alert_modal = False
    if "modal_alert_type" not in st.session_state:
        st.session_state.modal_alert_type = ""
    if "selected_alert_id" not in st.session_state:
        st.session_state.selected_alert_id = None

    # All alerts data with detailed information
    all_alerts = [
        {
            "id": 1,
            "type": "Personal Styling",
            "message": "Platinum Member Emma Rodriguez arriving in 1 hour - Personal stylist still unassigned",
            "severity": "urgent",
            "action": "Assign personal stylist",
            "time": "5 seconds ago",
            "details": {
                "customer_name": "Emma Rodriguez",
                "membership_tier": "Platinum Member (5+ years)",
                "appointment_time": "11:00 AM (58 minutes from now)",
                "service_type": "Personal Shopping - Women's Professional Wear",
                "avg_purchase": "$850 per visit",
                "last_visit": "3 weeks ago, purchased $1,200 business wardrobe",
                "original_stylist": "Jessica Martinez (called in sick)",
                "backup_failed": "Auto-reassignment system failed",
                "available_stylists": [
                    {
                        "name": "Victoria Chen",
                        "rating": "4.9/5",
                        "specialty": "Women's Fashion",
                        "status": "Available",
                    },
                    {
                        "name": "Emma Wilson",
                        "rating": "4.7/5",
                        "specialty": "Customer Service & Cross-trained",
                        "status": "Available",
                    },
                    {
                        "name": "James Park",
                        "rating": "4.8/5",
                        "specialty": "Visual Merchandising",
                        "status": "Busy until 11:30 AM",
                    },
                ],
            },
        },
        {
            "id": 2,
            "type": "Critical Stock",
            "message": "Designer Jeans - only 2 left",
            "severity": "urgent",
            "action": "Reorder now",
            "time": "20 min ago",
        },
        {
            "id": 3,
            "type": "Staff Coverage",
            "message": "Mike Rodriguez called in sick - Electronics dept needs coverage for 2-6 PM shift",
            "severity": "urgent",
            "action": "Find coverage",
            "time": "30 min ago",
        },
        {
            "id": 4,
            "type": "Delivery Update",
            "message": "New designer collection arriving tomorrow - Prepare display area",
            "severity": "important",
            "action": "Prep display area",
            "time": "1 hour ago",
        },
        {
            "id": 5,
            "type": "Schedule Change",
            "message": "Staff meeting moved to 3 PM in conference room",
            "severity": "important",
            "action": "Update team",
            "time": "2 hours ago",
        },
        {
            "id": 6,
            "type": "Personal Styling",
            "message": "Platinum Member Emma Rodriguez arriving at 2 PM for wardrobe consultation",
            "severity": "important",
            "action": "Prep personal shopper",
            "time": "2 hours ago",
        },
        {
            "id": 7,
            "type": "Delivery Delay",
            "message": "Designer collection delayed to 4:30 PM",
            "severity": "important",
            "action": "Update team",
            "time": "3 hours ago",
        },
    ]

    # Calculate real-time counters
    urgent_alerts = [
        a
        for a in all_alerts
        if a["severity"] == "urgent" and a["id"] not in st.session_state.resolved_alerts
    ]
    important_alerts = [
        a
        for a in all_alerts
        if a["severity"] == "important"
        and a["id"] not in st.session_state.resolved_alerts
    ]
    resolved_alerts = [
        a for a in all_alerts if a["id"] in st.session_state.resolved_alerts
    ]
    total_active = len(urgent_alerts) + len(important_alerts)

    urgent_count = len(urgent_alerts)
    important_count = len(important_alerts)
    resolved_count = len(resolved_alerts)

    # # Show demo alerts if demo is active
    # if demo_active:
    #     st.markdown("---")
    #     st.markdown("### 🎬 Live Demo Alerts")
        
    #     # Get active demo alerts
    #     active_demo_alerts = demo_system.get_active_alerts()
        
    #     if active_demo_alerts:
    #         # Show demo alerts in a compact format
    #         with st.container(height=300):
    #             from .demo_alerts import show_demo_alerts_display
    #             show_demo_alerts_display(demo_system)
    #     else:
    #         st.info("🎬 Demo running... Waiting for alerts to trigger...")
        
    #     st.markdown("---")

    # Create tabs for different alert types
    tab1, tab2, tab3, tab4 = st.tabs([
        f"🚨 Urgent ({urgent_count})",
        f"⚠️ Important ({important_count})", 
        f"✅ Resolved ({resolved_count})",
        f"📋 All Active ({total_active})"
    ])

    # Function to display alerts in a consistent format
    def display_alerts(alerts, tab_name):
        """Display alerts in a consistent format with details buttons."""
        if len(alerts) > 0:
            with st.container(height=400):
                for alert in alerts:
                    is_resolved = alert["id"] in st.session_state.resolved_alerts
                    severity_label = (
                        "RESOLVED" if is_resolved else alert["severity"].upper()
                    )

                    # Color coding for alert cards
                    if alert["severity"] == "urgent":
                        border_color = "#dc3545"
                        bg_color = "#fff5f5" if not is_resolved else "#f8f9fa"
                        severity_bg = "#dc3545"
                        severity_text = "white"
                    elif alert["severity"] == "important":
                        border_color = "#ffc107"
                        bg_color = "#fffbf0" if not is_resolved else "#f8f9fa"
                        severity_bg = "#ffc107"
                        severity_text = "#212529"
                    else:
                        border_color = "#6c757d"
                        bg_color = "#f8f9fa"
                        severity_bg = "#6c757d"
                        severity_text = "white"

                    # Create columns for alert content and button
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        # Alert card display
                        st.markdown(
                            f"""
                        <div style="
                            background: {bg_color};
                            border-left: 4px solid {border_color};
                            border-radius: 8px;
                            padding: 1rem;
                            margin-bottom: 0.75rem;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            opacity: {"0.6" if is_resolved else "1"};
                        ">
                            <div style="
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                                margin-bottom: 0.5rem;
                            ">
                                <span style="
                                    font-weight: 700;
                                    color: #495057;
                                    font-size: 0.9rem;
                                ">{alert["type"]}</span>
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <span style="
                                        background: {severity_bg};
                                        color: {severity_text};
                                        padding: 0.2rem 0.5rem;
                                        border-radius: 12px;
                                        font-size: 0.85rem;
                                        font-weight: 700;
                                    ">{severity_label}</span>
                                    <span style="
                                        font-size: 0.75rem;
                                        color: #6c757d;
                                    ">{alert["time"]}</span>
                                </div>
                            </div>
                            <div style="
                                color: #495057;
                                margin-bottom: 0.5rem;
                                line-height: 1.4;
                            ">{alert["message"]}</div>
                            <div style="
                                color: #007bff;
                                font-size: 0.8rem;
                            ">→ {alert["action"]}</div>
                            <div class="click-hint">Click "Details" for more information</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    with col2:
                        # Details button for each alert
                        if st.button(
                            "Details",
                            key=f"alert_details_{alert['id']}_{tab_name}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_alert_id = alert["id"]
                            st.rerun()
        else:
            st.info(f"No {tab_name.lower()} alerts to display.")

    # Display alerts in each tab
    with tab1:
        st.markdown("### 🚨 Urgent Alerts - Immediate Action Required")
        display_alerts(urgent_alerts, "Urgent")

    with tab2:
        st.markdown("### ⚠️ Important Alerts - Action Needed")
        display_alerts(important_alerts, "Important")

    with tab3:
        st.markdown("### ✅ Resolved Alerts")
        display_alerts(resolved_alerts, "Resolved")

    with tab4:
        st.markdown("### 📋 All Active Alerts")
        all_active_alerts = urgent_alerts + important_alerts
        display_alerts(all_active_alerts, "All Active")

    # Alert details modal using st.dialog - ONLY triggered by Details buttons
    if st.session_state.selected_alert_id:
        selected_alert = next(
            (a for a in all_alerts if a["id"] == st.session_state.selected_alert_id),
            None,
        )

        if selected_alert:
            try:

                @st.dialog("Alert Details", width="large")
                def show_alert_details():
                    # Add modern CSS styling
                    st.markdown(
                        """
                    <style>
                    /* =============================================================================
                       MODAL-SPECIFIC STYLES
                       - Inherits base styles from main CSS
                       - Modal-specific overrides and dialog styling only
                       - Removed duplicate declarations already covered by base styles
                       ============================================================================= */
                    
                    /* MODAL CARD OVERRIDES - Specific styling for dialog context */
                    .modern-card {
                        background: white;
                        border: 1px solid rgba(0, 0, 0, 0.04);
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
                    }
                    
                    .modern-card:hover {
                        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.12);
                    }
                    
                    /* MODAL HEADER ADJUSTMENTS */
                    .section-header {
                        color: #1a1a1a;
                        margin-bottom: 20px;
                        gap: 12px;
                        letter-spacing: -0.02em;
                    }
                    
                    /* MODAL GRID LAYOUT */
                    .info-grid {
                        margin-bottom: 24px;
                    }
                    
                    /* MODAL INFO ITEMS - Simplified background for modal context */
                    .info-item {
                        background: #f8fafc;
                        transition: all 0.2s ease;
                    }
                    
                    .info-item:hover {
                        background: #f1f5f9;
                    }
                    
                    /* MODAL TYPOGRAPHY ADJUSTMENTS */
                    .info-label {
                        font-size: 12px;
                        margin-bottom: 8px;
                        opacity: 0.7;
                    }
                    
                    .info-value {
                        color: #1a1a1a;
                    }
                    
                    /* STATUS BADGES - Modal-specific status styling */
                    .status-badge {
                        display: inline-flex;
                        align-items: center;
                        gap: 6px;
                        padding: 6px 12px;
                        border-radius: 20px;
                        font-size: 13px;
                        font-weight: 600;
                        letter-spacing: 0.02em;
                    }
                    
                    .status-available {
                        background: #dcfce7;
                        color: #166534;
                        border: 1px solid #bbf7d0;
                    }
                    
                    .status-busy {
                        background: #fef3c7;
                        color: #92400e;
                        border: 1px solid #fde68a;
                    }
                    
                    /* RECOMMENDATION BADGE - Modal-specific recommendation styling */
                    .recommended-badge {
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        color: white;
                        padding: 4px 10px;
                        border-radius: 12px;
                        font-size: 11px;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                        margin-left: 8px;
                    }
                    
                    /* DIALOG POSITIONING AND LAYOUT - Critical for modal functionality */
                    div[data-testid="stDialog"] {
                        position: fixed !important;
                        top: 0 !important;
                        left: 0 !important;
                        right: 0 !important;
                        bottom: 0 !important;
                        display: flex !important;
                        align-items: flex-start !important;
                        justify-content: center !important;
                        z-index: 1000 !important;
                        background: rgba(0, 0, 0, 0.5) !important;
                        width: 100vw !important;
                        height: 100vh !important;
                        padding-top: 2rem !important;
                        overflow-y: auto !important;
                    }
                    
                    div[data-testid="stDialog"] > div {
                        max-width: 1000px !important;
                        width: 85vw !important;
                        position: relative !important;
                        background: white !important;
                        border-radius: 16px !important;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
                        margin: 0 auto !important;
                        max-height: calc(100vh - 4rem) !important;
                        overflow-y: auto !important;
                    }
                    </style>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Create a scrollable container for the modal content
                    with st.container(height=650):
                        # Modern header for Personal Styling alerts
                        if selected_alert["type"] == "Personal Styling":
                            st.markdown(
                                """
                            <div style="
                                background: linear-gradient(135deg, #dc2626 0%, #b91c1c 50%, #991b1b 100%);
                                color: white;
                                padding: 32px;
                                border-radius: 20px;
                                margin-bottom: 32px;
                                position: relative;
                                overflow: hidden;
                                border: 3px solid #fca5a5;
                                box-shadow: 0 0 30px rgba(220, 38, 38, 0.4);
                                animation: pulse-urgent 2s infinite;
                            ">
                                <style>
                                @keyframes pulse-urgent {
                                    0% { box-shadow: 0 0 30px rgba(220, 38, 38, 0.4); }
                                    50% { box-shadow: 0 0 50px rgba(220, 38, 38, 0.8); }
                                    100% { box-shadow: 0 0 30px rgba(220, 38, 38, 0.4); }
                                }
                                </style>
                                <div style="
                                    position: absolute;
                                    top: -50%;
                                    right: -20%;
                                    width: 200px;
                                    height: 200px;
                                    background: rgba(255, 255, 255, 0.1);
                                    border-radius: 50%;
                                    filter: blur(40px);
                                "></div>
                                <div style="position: relative; z-index: 2;">
                                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                                        <div style="display: flex; align-items: center; gap: 16px;">
                                            <div style="
                                                background: rgba(255, 255, 255, 0.2);
                                                padding: 16px;
                                                border-radius: 20px;
                                                backdrop-filter: blur(10px);
                                            ">
                                                <span style="font-size: 32px;">⚠️</span>
                                            </div>
                                            <div>
                                                <div style="
                                                    background: #fef2f2;
                                                    color: #dc2626;
                                                    padding: 8px 16px;
                                                    border-radius: 25px;
                                                    font-size: 14px;
                                                    font-weight: 900;
                                                    text-transform: uppercase;
                                                    letter-spacing: 1px;
                                                    margin-bottom: 8px;
                                                    border: 2px solid #fca5a5;
                                                ">URGENT - IMMEDIATE ACTION REQUIRED</div>
                                                <h1 style="
                                                    margin: 0;
                                                    font-size: 28px;
                                                    font-weight: 900;
                                                    letter-spacing: -0.02em;
                                                ">STYLIST UNASSIGNED</h1>
                                            </div>
                                        </div>
                                        <div style="text-align: right;">
                                            <div style="
                                                background: rgba(255, 255, 255, 0.2);
                                                padding: 12px 20px;
                                                border-radius: 15px;
                                                backdrop-filter: blur(10px);
                                                border: 1px solid rgba(255, 255, 255, 0.3);
                                            ">
                                                <div style="
                                                    font-size: 24px;
                                                    font-weight: 900;
                                                    margin-bottom: 4px;
                                                ">⏰ 58 MIN</div>
                                                <div style="
                                                    font-size: 12px;
                                                    opacity: 0.9;
                                                    text-transform: uppercase;
                                                    letter-spacing: 1px;
                                                ">Until Appointment</div>
                                            </div>
                                        </div>
                                    </div>
                                    <div style="
                                        background: rgba(255, 255, 255, 0.15);
                                        padding: 20px;
                                        border-radius: 15px;
                                        backdrop-filter: blur(10px);
                                        border: 1px solid rgba(255, 255, 255, 0.2);
                                    ">
                                        <div style="
                                            font-size: 18px;
                                            font-weight: 600;
                                            line-height: 1.6;
                                            margin-bottom: 12px;
                                        ">
                                            <span style="
                                                background: #fbbf24;
                                                color: #92400e;
                                                padding: 4px 8px;
                                                border-radius: 8px;
                                                font-weight: 900;
                                                margin-right: 8px;
                                            ">PLATINUM MEMBER</span>
                                            Emma Rodriguez arriving in 1 hour
                                        </div>
                                        <div style="
                                            font-size: 16px;
                                            opacity: 0.95;
                                            line-height: 1.5;
                                        ">
                                            • <strong>Risk of service disruption and customer dissatisfaction</strong>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        else:
                            # Standard header for other alert types
                            st.markdown(f"### {selected_alert['message']}")
                            st.markdown(
                                f"**Severity:** {selected_alert['severity'].title()}"
                            )
                            st.markdown(f"**Time:** {selected_alert['time']}")

                        # Show additional details if available
                        if "details" in selected_alert:
                            details = selected_alert["details"]

                            # Issue Details Section - MOVED TO TOP
                            st.markdown(
                                f"""
                            <div class="modern-card" style="border-left: 4px solid #ef4444;">
                                <div class="section-header">
                                    <span style="color: #ef4444; font-size: 20px;">⚠️</span>
                                    Issue Analysis
                                </div>
                                <div style="
                                    background: #fef2f2;
                                    border-radius: 12px;
                                    padding: 20px;
                                    border: 1px solid #fecaca;
                                ">
                                    <div style="margin-bottom: 12px;">
                                        <span style="color: #991b1b; font-weight: 600; font-size: 14px;">Original Stylist:</span>
                                        <span style="color: #1f2937; margin-left: 8px;">{details["original_stylist"]}</span>
                                    </div>
                                    <div>
                                        <span style="color: #991b1b; font-weight: 600; font-size: 14px;">Root Cause:</span>
                                        <span style="color: #1f2937; margin-left: 8px;">{details["backup_failed"]}</span>
                                    </div>
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Customer Information Section - CLEANED UP
                            st.markdown(
                                f"""
                            <div class="modern-card">
                                <div class="section-header">
                                    <span style="
                                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                        -webkit-background-clip: text;
                                        -webkit-text-fill-color: transparent;
                                        background-clip: text;
                                        font-size: 20px;
                                    ">👤</span>
                                    Customer Profile
                                </div>
                                <div class="info-grid">
                                    <div class="info-item" style="border-left-color: #667eea;">
                                        <div class="info-label" style="color: #667eea;">Customer</div>
                                        <div class="info-value">{details["customer_name"]}</div>
                                    </div>
                                    <div class="info-item" style="border-left-color: #f59e0b;">
                                        <div class="info-label" style="color: #f59e0b;">Membership</div>
                                        <div class="info-value">{details["membership_tier"]}</div>
                                    </div>
                                    <div class="info-item" style="border-left-color: #ef4444;">
                                        <div class="info-label" style="color: #ef4444;">Appointment</div>
                                        <div class="info-value">{details["appointment_time"]}</div>
                                    </div>
                                    <div class="info-item" style="border-left-color: #10b981;">
                                        <div class="info-label" style="color: #10b981;">Average Purchase</div>
                                        <div class="info-value">{details["avg_purchase"]}</div>
                                    </div>
                                </div>
                                <div style="
                                    background: #f8fafc;
                                    border-radius: 12px;
                                    padding: 16px;
                                    border: 1px solid #e2e8f0;
                                ">
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 14px;">
                                        <div>
                                            <span style="color: #64748b; font-weight: 600;">Last Visit:</span><br>
                                            <span style="color: #1e293b;">{details["last_visit"]}</span>
                                        </div>
                                        <div>
                                            <span style="color: #64748b; font-weight: 600;">Service Type:</span><br>
                                            <span style="color: #1e293b;">{details["service_type"]}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Available Stylists Section
                            st.markdown("---")
                            st.markdown("### 👥 Available Stylists")

                            # Integrated stylist recommendation analysis for Victoria Chen
                            st.markdown(
                                """
                            <div style="
                                background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                                border: 2px solid #10b981;
                                border-radius: 16px;
                                padding: 20px;
                                margin-bottom: 20px;
                                box-shadow: 0 4px 20px rgba(16, 185, 129, 0.1);
                            ">
                                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                                    <span style="
                                        background: #10b981;
                                        color: white;
                                        width: 28px;
                                        height: 28px;
                                        border-radius: 50%;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        font-size: 14px;
                                        font-weight: 700;
                                    ">1</span>
                                    <div style="
                                        font-size: 16px;
                                        font-weight: 700;
                                        color: #064e3b;
                                    ">RECOMMENDED: Victoria Chen - Optimal Match for Emma Rodriguez</div>
                                </div>
                                <div style="
                                    display: grid;
                                    grid-template-columns: 1fr 1fr;
                                    gap: 12px;
                                    font-size: 13px;
                                    line-height: 1.5;
                                ">
                                    <div style="color: #166534;">
                                        <strong>✓ Specialty Match:</strong> Women's Fashion expert (3+ years)
                                    </div>
                                    <div style="color: #166534;">
                                        <strong>✓ Brand Expertise:</strong> Certified in Emma's preferred brands
                                    </div>
                                    <div style="color: #166534;">
                                        <strong>✓ Customer History:</strong> 2 similar Platinum member successes
                                    </div>
                                    <div style="color: #166534;">
                                        <strong>✓ Schedule Fit:</strong> Available now, full 90-min dedication
                                    </div>
                                </div>
                                <div style="
                                    margin-top: 12px;
                                    padding-top: 12px;
                                    border-top: 1px solid #bbf7d0;
                                    font-size: 13px;
                                    color: #166534;
                                    font-weight: 600;
                                    text-align: center;
                                ">
                                    🎯 Compatibility Score: 95% - Highest match for this appointment
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Display each stylist card individually with properly integrated buttons
                            for i, stylist in enumerate(details["available_stylists"]):
                                if stylist["status"] == "Available":
                                    status_bg = "#dcfce7"
                                    status_color = "#166534"
                                    status_border = "#bbf7d0"
                                    status_icon = "✓"
                                else:
                                    status_bg = "#fef3c7"
                                    status_color = "#92400e"
                                    status_border = "#fde68a"
                                    status_icon = "⏳"

                                # Special styling for recommended stylist
                                if stylist["name"] == "Victoria Chen":
                                    card_style = "border: 2px solid #10b981; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);"
                                    recommended_badge = '<span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-left: 8px;">OPTIMAL MATCH</span>'
                                else:
                                    card_style = "border: 1px solid #e2e8f0;"
                                    recommended_badge = ""

                                # Create a container for each stylist with integrated button using columns
                                col_info, col_button = st.columns([4, 1])

                                with col_info:
                                    st.markdown(
                                        f"""
                                    <div style="
                                        background: white;
                                        border-radius: 16px;
                                        padding: 24px;
                                        margin: 12px 0px;
                                        {card_style}
                                        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
                                        transition: all 0.3s ease;
                                        height: 120px;
                                        display: flex;
                                        align-items: center;
                                    ">
                                        <div style="display: flex; align-items: center; gap: 12px; width: 100%;">
                                            <div style="
                                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                                color: white;
                                                width: 40px;
                                                height: 40px;
                                                border-radius: 12px;
                                                display: flex;
                                                align-items: center;
                                                justify-content: center;
                                                font-size: 18px;
                                            ">👤</div>
                                            <div style="flex: 1;">
                                                <div style="
                                                    font-size: 18px;
                                                    font-weight: 700;
                                                    color: #1a1a1a;
                                                    letter-spacing: -0.01em;
                                                    margin-bottom: 4px;
                                                ">{stylist["name"]}{recommended_badge}</div>
                                                <div style="
                                                    font-size: 14px;
                                                    color: #64748b;
                                                    font-weight: 500;
                                                    margin-bottom: 8px;
                                                ">{stylist["specialty"]}</div>
                                                <div style="display: flex; align-items: center; gap: 12px;">
                                                    <div style="
                                                        display: inline-flex;
                                                        align-items: center;
                                                        gap: 6px;
                                                        padding: 6px 12px;
                                                        border-radius: 20px;
                                                        font-size: 13px;
                                                        font-weight: 600;
                                                        background: {status_bg};
                                                        color: {status_color};
                                                        border: 1px solid {status_border};
                                                    ">
                                                        {status_icon} {stylist["status"]}
                                                    </div>
                                                    <div style="
                                                        font-size: 14px;
                                                        color: #f59e0b;
                                                        font-weight: 600;
                                                    ">⭐ {stylist["rating"]}</div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    """,
                                        unsafe_allow_html=True,
                                    )

                                with col_button:
                                    # Add some top margin to align with the card
                                    st.markdown(
                                        "<div style='margin-top: 50px;'></div>",
                                        unsafe_allow_html=True,
                                    )

                                    if stylist["status"] == "Available":
                                        button_type = (
                                            "primary"
                                            if stylist["name"] == "Victoria Chen"
                                            else "secondary"
                                        )
                                        button_text = (
                                            "Assign"
                                            if stylist["name"] == "Victoria Chen"
                                            else "Assign"
                                        )

                                        if st.button(
                                            button_text,
                                            key=f"assign_{stylist['name'].replace(' ', '_')}_individual",
                                            type=button_type,
                                            use_container_width=True,
                                        ):
                                            # Enhanced orchestrated response demonstration
                                            st.success(
                                                f"✅ {stylist['name']} assigned to Emma Rodriguez's appointment!"
                                            )

                                            # Show orchestrated workflow details
                                            st.info("""
                                            **🔄 Automated Workflow Initiated:**
                                            
                                            **📱 Staff Notification:**
                                            • Victoria Chen notified via mobile app
                                            • Customer profile and preferences sent
                                            • Appointment details and preparation time provided
                                            
                                            **🧠 AI-Powered Preparation:**
                                            • Emma's style profile and purchase history loaded
                                            • Recommended items pre-selected based on preferences
                                            • Inventory availability confirmed for suggested pieces
                                            
                                            **📋 System Updates:**
                                            • Appointment status updated across all systems
                                            • Customer service team notified of assignment
                                            • Performance tracking initiated for service quality
                                            
                                            **⏰ Timeline:** All actions completed in <5 seconds
                                            """)

                                            st.balloons()
                                            st.session_state.selected_alert_id = None
                                            st.rerun()
                                    else:
                                        st.button(
                                            "N/A",
                                            key=f"unavailable_{stylist['name'].replace(' ', '_')}",
                                            disabled=True,
                                            use_container_width=True,
                                            help=f"{stylist['name']} is {stylist['status']}",
                                        )
                        else:
                            # For alerts without detailed information
                            st.markdown("---")

                        if st.button("Close", key="close_alert_dialog"):
                            st.session_state.selected_alert_id = None
                            st.rerun()

                # Show the dialog
                show_alert_details()

            except Exception as e:
                st.error(f"Dialog error: {e}")
                # Fallback: Show details in an expander
                with st.expander(
                    f"📋 {selected_alert['type']} - Alert Details", expanded=True
                ):
                    st.markdown(f"### {selected_alert['message']}")
                    st.markdown(f"**Severity:** {selected_alert['severity'].title()}")
                    st.markdown(f"**Time:** {selected_alert['time']}")
                    st.markdown(f"**Recommended Action:** {selected_alert['action']}")

                    if "details" in selected_alert:
                        details = selected_alert["details"]
                        st.markdown("### Customer Information")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Customer:** {details['customer_name']}")
                            st.markdown(f"**Membership:** {details['membership_tier']}")
                            st.markdown(
                                f"**Appointment:** {details['appointment_time']}"
                            )
                            st.markdown(f"**Service:** {details['service_type']}")

                        with col2:
                            st.markdown(f"**Avg Purchase:** {details['avg_purchase']}")
                            st.markdown(f"**Last Visit:** {details['last_visit']}")
                            st.markdown(
                                f"**Original Stylist:** {details['original_stylist']}"
                            )
                            st.markdown(f"**Issue:** {details['backup_failed']}")

                        st.markdown("### Available Stylists")
                        for stylist in details["available_stylists"]:
                            status_color = (
                                "#28a745"
                                if stylist["status"] == "Available"
                                else "#ffc107"
                            )
                            st.markdown(
                                f"""
                            <div style="
                                background: #f8f9fa;
                                border-radius: 8px;
                                padding: 0.75rem;
                                margin-bottom: 0.5rem;
                                border-left: 3px solid {status_color};
                            ">
                                <strong>{stylist["name"]}</strong> - {stylist["specialty"]}<br>
                                Rating: {stylist["rating"]} | Status: <span style="color: {status_color}; font-weight: bold;">{stylist["status"]}</span>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Add individual assign buttons for fallback section too
                            if stylist["status"] == "Available":
                                if st.button(
                                    f"Assign {stylist['name']}",
                                    key=f"assign_{stylist['name'].replace(' ', '_')}_fallback",
                                ):
                                    st.success(
                                        f"✅ {stylist['name']} assigned to Emma Rodriguez's appointment!"
                                    )
                                    st.info(
                                        "🔄 Automated workflow initiated - all systems updated!"
                                    )
                                    st.session_state.selected_alert_id = None
                                    st.rerun()

                    if st.button("Close", key="close_simple_dialog"):
                        st.session_state.selected_alert_id = None
                        st.rerun()
