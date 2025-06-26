"""Associate my tasks tab."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


def show_my_tasks_tab():
    """Display the My Tasks tab with realistic retail tasks organized for maximum efficiency."""
    
    # Get current time for dynamic time calculations
    now = datetime.now()
    
    # Calculate personal shopping appointment time - next closest convenient hour
    current_minute = now.minute
    current_hour = now.hour
    
    if current_minute < 30:
        # If before X:30, schedule for (X+1):30
        personal_appointment_time = now.replace(hour=current_hour + 1, minute=30, second=0, microsecond=0)
    else:
        # If X:30 or later, schedule for (X+2):00
        personal_appointment_time = now.replace(hour=current_hour + 2, minute=0, second=0, microsecond=0)
    
    # Calculate time until appointment
    time_until_appointment = personal_appointment_time - now
    minutes_until = int(time_until_appointment.total_seconds() // 60)
    
    # Other dynamic times remain the same
    bopis_due_time = now - timedelta(hours=2, minutes=30)  # Overdue by 2.5 hours
    return_time = now + timedelta(hours=2, minutes=5)
    
    # Format times for display
    personal_appointment_str = personal_appointment_time.strftime("%I:%M %p")
    bopis_due_str = bopis_due_time.strftime("%I:%M %p")
    return_time_str = return_time.strftime("%I:%M %p")
    
    # Calculate relative time strings
    personal_relative = f"{minutes_until} min"
    bopis_overdue_hours = int((now - bopis_due_time).total_seconds() // 3600)
    bopis_overdue_minutes = int(((now - bopis_due_time).total_seconds() % 3600) // 60)
    return_relative = "2h 5min"
    
    # Calculate upcoming task times
    upcoming_times = {
        "inventory": now + timedelta(hours=3),
        "visual": now + timedelta(hours=3, minutes=30), 
        "restocking": now + timedelta(hours=4),
        "customer_service": now + timedelta(hours=5, minutes=30)
    }
    
    # Initialize session state for modals
    if 'show_personal_modal' not in st.session_state:
        st.session_state.show_personal_modal = False
    if 'show_bopis_modal' not in st.session_state:
        st.session_state.show_bopis_modal = False
    if 'show_returns_modal' not in st.session_state:
        st.session_state.show_returns_modal = False
    if 'selected_task' not in st.session_state:
        st.session_state.selected_task = None
    
    # Modal functions
    @st.dialog("Personal Shopping Appointment", width="large")
    def show_personal_shopping_modal():
            st.markdown(
                """
            <style>
            /* Modal styling */
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
                max-width: 900px !important;
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

            # Header with urgent styling - dynamic time
            st.markdown(
                f"""
            <div style="
                background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 50%, #6d28d9 100%);
                color: white;
                padding: 32px;
                border-radius: 20px;
                margin-bottom: 32px;
                position: relative;
                overflow: hidden;
                border: 3px solid #c4b5fd;
                box-shadow: 0 0 30px rgba(139, 92, 246, 0.4);
            ">
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
                                <span style="font-size: 32px;">🛍️</span>
                            </div>
                            <div>
                                <div style="
                                    background: #fef3c7;
                                    color: #92400e;
                                    padding: 8px 16px;
                                    border-radius: 25px;
                                    font-size: 14px;
                                    font-weight: 900;
                                    text-transform: uppercase;
                                    letter-spacing: 1px;
                                    margin-bottom: 8px;
                                    border: 2px solid #fbbf24;
                                ">URGENT - PERSONAL STYLING APPOINTMENT</div>
                                <h1 style="
                                    margin: 0;
                                    font-size: 28px;
                                    font-weight: 900;
                                    letter-spacing: -0.02em;
                                ">ASSIGNMENT NOTIFICATION</h1>
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
                                ">⏰ {personal_relative.upper()}</div>
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
                            Emma Rodriguez arriving in {personal_relative}
                        </div>
                        <div style="
                            font-size: 16px;
                            opacity: 0.95;
                            line-height: 1.5;
                        ">
                            • <strong>Personal styling appointment assigned to you</strong><br>
                            • <strong>Customer intelligence and preparation recommendations included</strong>
                        </div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Customer Information Section
            st.markdown("### 👤 Customer Information")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                <div style="
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 4px solid #8b5cf6;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #64748b;">Customer</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">Emma Rodriguez</div>
                <div style="font-size: 14px; color: #64748b;">VIP Gold Member (3+ years)</div>
                </div>
                
                <div style="
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 4px solid #10b981;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #64748b;">Appointment</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">{personal_appointment_str} (90 minutes)</div>
                    <div style="font-size: 14px; color: #64748b;">Personal Shopping - Women's Fashion Consultation</div>
                </div>
                
                <div style="
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 4px solid #f59e0b;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #64748b;">Budget Range</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">$500 - $800</div>
                    <div style="font-size: 14px; color: #64748b;">Professional wardrobe refresh</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    """
                <div style="
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 4px solid #3b82f6;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #64748b;">Contact Info</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">(555) 123-4567</div>
                    <div style="font-size: 14px; color: #64748b;">emma.rodriguez@email.com</div>
                </div>
                
                <div style="
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 4px solid #ef4444;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #64748b;">Style Profile</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">Business Casual</div>
                    <div style="font-size: 14px; color: #64748b;">Size M (US 8-10), prefers navy/black/burgundy</div>
                </div>
                
                <div style="
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 4px solid #8b5cf6;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #64748b;">Occasion</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">New Job at Law Firm</div>
                    <div style="font-size: 14px; color: #64748b;">Needs complete wardrobe refresh</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # AI Customer Intelligence
            st.markdown("---")
            st.markdown("### 🧠 AI Customer Intelligence")

            col1_ai, col2_ai = st.columns(2)

            with col1_ai:
                st.info("""**Purchase Pattern**

• Shops quarterly for seasonal updates  
• Prefers quality over quantity  
• Average basket: 3-4 coordinated pieces  
• Low return rate (2.1%)
• Price-conscious but values investment pieces""")

                st.info("""**Brand Preferences**

• Ann Taylor, J.Crew, Banana Republic  
• Avoids fast fashion  
• Color palette: Navy, black, burgundy, cream  
• Classic cuts with modern details""")

            with col2_ai:
                st.info("""**Style Intelligence**

• Professional wardrobe focus  
• Prefers coordinated sets  
• Size consistency: M (tops), 8-10 (bottoms)  
• Comfortable in classic silhouettes
• Values versatile pieces""")

                st.info("""**Current Needs Assessment**

• Career transition wardrobe  
• Law firm dress code compliance  
• Business travel pieces  
• Mix & match capabilities  
• Quality fabrics for daily wear""")

            st.success("""**🎯 AI Styling Recommendations**

• **Start with basics**: Navy and black blazers, coordinating pants  
• **Add versatile pieces**: Silk blouses, quality knitwear, classic dresses  
• **Focus on fabrics**: Wool blends, ponte knits, wrinkle-resistant materials  
• **Key accessories**: Professional handbag, classic pumps, simple jewelry  
• **Personal note**: She always asks about care instructions - have fabric guide ready!""")

            # Action Buttons
            st.markdown("---")
            col1_action, col2_action, col3_action = st.columns(3)

            with col1_action:
                if st.button("✅ Accept Assignment", type="primary", use_container_width=True):
                    st.success("✅ Assignment accepted! Customer preparation initiated.")
                    st.balloons()
                    st.session_state.show_personal_modal = False
                    st.rerun()

            with col2_action:
                if st.button("📱 Call Customer", use_container_width=True):
                    st.info("📱 Calling (555) 123-4567...")

            with col3_action:
                if st.button("❌ Close", use_container_width=True):
                    st.session_state.show_personal_modal = False
                    st.rerun()

    @st.dialog("BOPIS Order Details", width="large")
    def show_bopis_modal():
        task = st.session_state.selected_task
        
        # Calculate how long overdue
        overdue_str = f"{bopis_overdue_hours}h {bopis_overdue_minutes}m"
        
        st.markdown(
            f"""
        <div style="
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #b91c1c 100%);
            color: white;
            padding: 32px;
            border-radius: 20px;
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
            border: 3px solid #fca5a5;
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.4);
        ">
            <div style="position: relative; z-index: 2;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div style="
                            background: rgba(255, 255, 255, 0.2);
                            padding: 16px;
                            border-radius: 20px;
                            backdrop-filter: blur(10px);
                        ">
                            <span style="font-size: 32px;">📦</span>
                        </div>
                        <div>
                            <div style="
                                background: #fee2e2;
                                color: #991b1b;
                                padding: 8px 16px;
                                border-radius: 25px;
                                font-size: 14px;
                                font-weight: 900;
                                text-transform: uppercase;
                                letter-spacing: 1px;
                                margin-bottom: 8px;
                                border: 2px solid #fca5a5;
                            ">OVERDUE - BOPIS ORDER</div>
                            <h1 style="
                                margin: 0;
                                font-size: 28px;
                                font-weight: 900;
                                letter-spacing: -0.02em;
                            ">IMMEDIATE ATTENTION REQUIRED</h1>
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
                            ">OVERDUE</div>
                            <div style="
                                font-size: 12px;
                                opacity: 0.9;
                                text-transform: uppercase;
                                letter-spacing: 1px;
                            ">By {overdue_str}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Order Information
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📋 Order Information")
            st.markdown(f"""
            **Customer:** {task['details']['customer']}
            **Order Number:** {task['details']['order_number']}
            **Total:** {task['details']['total']}
            **Payment Status:** {task['details']['payment_status']}
            **Pickup Location:** {task['details']['location']}
            **Due Time:** {bopis_due_str} (overdue by {overdue_str})
            """)

        with col2:
            st.markdown("### 📦 Items Ordered")
            for item in task['details']['items']:
                st.markdown(f"• {item}")

        st.markdown("---")

        st.warning("""
        **PRIORITY ACTIONS REQUIRED:**
        
        • Customer may be waiting or will arrive soon
        • Ensure all items are present and in good condition  
        • Process pickup immediately upon customer arrival
        • Apologize for any wait time
        """)

        # Pickup Process
        st.markdown("### ✅ Pickup Process")
        pickup_steps = [
            "Verify customer ID matches order name",
            "Locate all items from secure pickup area", 
            "Check items for damage or defects",
            "Process pickup in POS system",
            "Provide receipt and thank customer"
        ]

        for i, step in enumerate(pickup_steps, 1):
            st.markdown(f"{i}. {step}")

        # Action Buttons
        st.markdown("---")
        col1_action, col2_action, col3_action = st.columns(3)

        with col1_action:
            if st.button("✅ Process Pickup", type="primary", use_container_width=True):
                st.success("✅ BOPIS order processed successfully!")
                st.balloons()
                st.session_state.show_bopis_modal = False
                st.rerun()

        with col2_action:
            if st.button("📱 Call Customer", use_container_width=True):
                st.info("📱 Calling customer...")

        with col3_action:
            if st.button("❌ Close", use_container_width=True):
                st.session_state.show_bopis_modal = False
                st.rerun()

    @st.dialog("Return Processing Details", width="large")
    def show_returns_modal():
        task = st.session_state.selected_task
        
        st.markdown(
            f"""
        <div style="
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
            color: white;
            padding: 32px;
            border-radius: 20px;
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
            border: 3px solid #93c5fd;
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.4);
        ">
            <div style="position: relative; z-index: 2;">
                <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 20px;">
                    <div style="
                        background: rgba(255, 255, 255, 0.2);
                        padding: 16px;
                        border-radius: 20px;
                        backdrop-filter: blur(10px);
                    ">
                        <span style="font-size: 32px;">🔄</span>
                    </div>
                    <div>
                        <div style="
                            background: #dbeafe;
                            color: #1e40af;
                            padding: 8px 16px;
                            border-radius: 25px;
                            font-size: 14px;
                            font-weight: 900;
                            text-transform: uppercase;
                            letter-spacing: 1px;
                            margin-bottom: 8px;
                            border: 2px solid #93c5fd;
                        ">RETURN PROCESSING</div>
                        <h1 style="
                            margin: 0;
                            font-size: 28px;
                            font-weight: 900;
                            letter-spacing: -0.02em;
                        ">CUSTOMER RETURN</h1>
                        <div style="
                            font-size: 16px;
                            opacity: 0.9;
                            margin-top: 8px;
                        ">Scheduled for {return_time_str}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Return Information
        st.markdown("### 📝 Return Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Customer:** {task['details']['customer']}
            **Return Reason:** {task['details']['return_reason']}
            **Return Value:** {task['details']['return_value']}
            **Return Method:** {task['details']['return_method']}
            **Scheduled Time:** {return_time_str}
            """)
            
        with col2:
            st.markdown("### 📦 Items to Return")
            for item in task['details']['items']:
                st.markdown(f"• {item}")

        st.markdown("---")

        # Return Process Steps
        st.markdown("### 🔄 Return Process Steps")
        return_steps = [
            "Verify return eligibility and condition",
            "Check return policy compliance", 
            "Process return in POS system",
            "Handle refund or exchange as requested",
            "Update inventory system",
            "File return documentation"
        ]

        for i, step in enumerate(return_steps, 1):
            st.markdown(f"{i}. {step}")

        # Action Buttons
        st.markdown("---")
        col1_action, col2_action, col3_action = st.columns(3)

        with col1_action:
            if st.button("✅ Process Return", type="primary", use_container_width=True):
                st.success("✅ Return processed successfully!")
                st.balloons()
                st.session_state.show_returns_modal = False
                st.rerun()

        with col2_action:
            if st.button("🔍 Check Policy", use_container_width=True):
                st.info("Opening return policy guidelines...")

        with col3_action:
            if st.button("❌ Close", use_container_width=True):
                st.session_state.show_returns_modal = False
                st.rerun()

    # Show modals if flags are set
    if st.session_state.show_personal_modal:
        show_personal_shopping_modal()
        
    if st.session_state.show_bopis_modal:
        show_bopis_modal()
        
    if st.session_state.show_returns_modal:
        show_returns_modal()
    
    # Task Overview Section
    st.markdown("### Task Overview")
    
    # Task summary metrics using consistent styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">3</div>
                    <div class="metric-badge negative">Due within 30 min</div>
                </div>
                <div class="metric-label"><span style="color: #dc2626; font-weight: 700;">URGENT</span> Tasks</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
    
    with col2:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">5</div>
                    <div class="metric-badge info">4 BOPIS, 1 Returns</div>
                </div>
                <div class="metric-label">Customer Orders</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
    
    with col3:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">2</div>
                    <div class="metric-badge neutral">Next: {personal_appointment_str}</div>
                </div>
                <div class="metric-label">Personal Shopping</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
    
    with col4:
        with st.container():
            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-value-container">
                    <div class="metric-value">6</div>
                    <div class="metric-badge positive">+2 from yesterday</div>
                </div>
                <div class="metric-label">Completed Today</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Main task sections
    col_main, col_side = st.columns([2.5, 1])
    
    with col_main:
        # Priority Tasks Section
        st.markdown("#### PRIORITY TASKS <span style='color: #64748b; font-weight: 400; font-size: 14px;'>(Next 2 Hours)</span>", unsafe_allow_html=True)
        
        priority_tasks = [
            {
                "id": 1,
                "title": "Personal Shopping Appointment",
                "subtitle": "Emma Rodriguez - Platinum Member",
                "time": f"{personal_appointment_str} ({personal_relative})",
                "duration": "90 minutes",
                "type": "personal_shopping",
                "priority": "urgent",
                "details": {
                    "customer": "Emma Rodriguez",
                    "member_level": "Platinum (5+ years)",
                    "service": "Professional wardrobe consultation",
                    "budget": "$500-800",
                    "preferences": "Business casual, navy/black/burgundy, size M",
                    "notes": "New job at law firm - needs complete wardrobe refresh"
                },
                "prep_items": [
                    "Review customer purchase history",
                    "Prepare fitting room in Women's section",
                    "Pull sample pieces in preferred colors",
                    "Check inventory for recommended brands"
                ]
            },
            {
                "id": 2,
                "title": "BOPIS Order Ready for Pickup",
                "subtitle": "Sarah Johnson - Order #B2024-0156",
                "time": f"Overdue (due {bopis_due_str})",
                "duration": "15 minutes",
                "type": "bopis",
                "priority": "overdue",
                "details": {
                    "customer": "Sarah Johnson",
                    "order_number": "B2024-0156",
                    "items": ["Navy Blazer (Size 8)", "Black Dress Pants (Size 8)", "White Silk Blouse (Size M)"],
                    "total": "$247.89",
                    "payment_status": "Paid",
                    "location": "Customer Service Counter"
                }
            },
            {
                "id": 3,
                "title": "Customer Return Processing",
                "subtitle": "Michael Chen - Online return",
                "time": f"{return_time_str} ({return_relative})",
                "duration": "20 minutes",
                "type": "returns",
                "priority": "normal",
                "details": {
                    "customer": "Michael Chen",
                    "return_reason": "Size exchange",
                    "items": ["Men's Dress Shirt (L → XL)", "Tie (Navy)"],
                    "return_value": "$89.99",
                    "return_method": "Online return label"
                }
            }
        ]
        
        for task in priority_tasks:
            # Task card with priority styling
            if task["priority"] == "urgent":
                border_color = "#f59e0b"
                bg_color = "#fef3c7"
                priority_label = "URGENT"
                priority_color = "#dc2626"
            elif task["priority"] == "overdue":
                border_color = "#ef4444"
                bg_color = "#fee2e2"
                priority_label = "OVERDUE"
                priority_color = "#dc2626"
            else:
                border_color = "#3b82f6"
                bg_color = "#dbeafe"
                priority_label = "NORMAL"
                priority_color = "#3b82f6"
            
            st.markdown(
                f"""
                <div style="
                    border-left: 4px solid {border_color};
                    background: {bg_color};
                    padding: 16px;
                    border-radius: 8px;
                    margin-bottom: 12px;
                ">
                    <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 8px;">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
                                <span style="
                                    background: {priority_color}; 
                                    color: white; 
                                    padding: 2px 8px; 
                                    border-radius: 4px; 
                                    font-size: 11px; 
                                    font-weight: 700;
                                    text-transform: uppercase;
                                ">{priority_label}</span>
                                <div style="font-weight: 700; font-size: 16px; color: #1e293b;">
                                    {task['title']}
                                </div>
                            </div>
                            <div style="color: #64748b; font-size: 14px; margin: 4px 0;">
                                {task['subtitle']}
                            </div>
                            <div style="color: #1e293b; font-weight: 600; font-size: 14px;">
                                <strong>Time:</strong> {task['time']} • <strong>Duration:</strong> {task['duration']}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Action buttons for each task
            col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
            
            with col_btn1:
                if st.button(f"Details", key=f"details_{task['id']}", use_container_width=True):
                    st.session_state.selected_task = task
                    if task["type"] == "personal_shopping":
                        st.session_state.show_personal_modal = True
                    elif task["type"] == "bopis":
                        st.session_state.show_bopis_modal = True
                    elif task["type"] == "returns":
                        st.session_state.show_returns_modal = True
                    st.rerun()

            with col_btn2:
                if task["type"] == "personal_shopping":
                    if st.button(f"Prep Checklist", key=f"prep_{task['id']}", use_container_width=True):
                        show_personal_shopping_prep(task)
                elif task["type"] == "bopis":
                    if st.button(f"Locate Order", key=f"locate_{task['id']}", use_container_width=True):
                        st.info(f"Order location: {task['details']['location']}")
                else:
                    if st.button(f"Review Items", key=f"review_{task['id']}", use_container_width=True):
                        st.info("Opening return details...")
            
            with col_btn3:
                if st.button(f"Start Task", key=f"start_{task['id']}", use_container_width=True, type="primary"):
                    st.success(f"Started: {task['title']}")
                    st.balloons()
            
            with col_btn4:
                if st.button(f"Snooze 15min", key=f"snooze_{task['id']}", use_container_width=True):
                    st.info("Task snoozed for 15 minutes")
            
            st.markdown("<br>", unsafe_allow_html=True)

        # Upcoming Tasks Section with dynamic times
        st.markdown("#### UPCOMING TASKS <span style='color: #64748b; font-weight: 400; font-size: 14px;'>(Later Today)</span>", unsafe_allow_html=True)
        
        upcoming_tasks = [
            {"task": "Inventory Count - Designer Handbags", "time": upcoming_times["inventory"].strftime("%I:%M %p"), "type": "inventory"},
            {"task": "Visual Merchandising - Window Display", "time": upcoming_times["visual"].strftime("%I:%M %p"), "type": "visual"},
            {"task": "Stock Replenishment - Fall Collection", "time": upcoming_times["restocking"].strftime("%I:%M %p"), "type": "restocking"},
            {"task": "Customer Follow-up Calls", "time": upcoming_times["customer_service"].strftime("%I:%M %p"), "type": "customer_service"}
        ]
        
        for i, task in enumerate(upcoming_tasks):
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 12px;
                    background: #f8fafc;
                    border-radius: 6px;
                    margin-bottom: 6px;
                    border: 1px solid #e2e8f0;
                ">
                    <div>
                        <span style="font-weight: 600; color: #1e293b;">▶ {task['task']}</span>
                    </div>
                    <div style="color: #64748b; font-size: 14px; font-weight: 600;">
                        {task['time']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_side:
        # Quick Actions Panel
        st.markdown("#### QUICK ACTIONS")
        
        if st.button("Check New BOPIS Orders", use_container_width=True):
            st.info("Checking for new orders...")
        
        if st.button("Report Low Stock", use_container_width=True):
            show_stock_report_form()
        
        if st.button("Request Team Help", use_container_width=True):
            st.success("Help request sent to floor manager")
        
        if st.button("Customer Needs Assistance", use_container_width=True):
            st.info("Alert sent: Customer needs help in your area")
        
        if st.button("Take Break", use_container_width=True):
            st.success("Break started - coverage requested")

        st.markdown("---")
        
        # Current Focus Section
        st.markdown("#### CURRENT ASSIGNMENT")
        
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 16px;
                border: 1px solid #e2e8f0;
            ">
                <div style="font-size: 18px; font-weight: 700; margin-bottom: 8px;">
                    Women's Fashion
                </div>
                <div style="font-size: 14px; opacity: 0.9;">
                    Designer Area • Solo Coverage
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Performance Today
        st.markdown("#### TODAY'S PERFORMANCE")
        
        performance_metrics = [
            {"label": "Tasks Completed", "value": "6/8", "color": "#10b981"},
            {"label": "Customer Interactions", "value": "18", "color": "#3b82f6"},
            {"label": "Sales Assisted", "value": "$3,250", "color": "#f59e0b"},
            {"label": "Response Time", "value": "2.3 min", "color": "#8b5cf6"}
        ]
        
        for metric in performance_metrics:
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    padding: 8px 0;
                    border-bottom: 1px solid #e5e7eb;
                ">
                    <span style="color: #64748b; font-size: 14px;">• {metric['label']}</span>
                    <span style="color: {metric['color']}; font-weight: 700;">{metric['value']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )


def show_personal_shopping_prep(task):
    """Show personal shopping preparation checklist."""
    st.success(f"""
    **Personal Shopping Preparation - {task['details']['customer']}**
    
    **Prep Checklist:**
    {chr(10).join([f"☐ {item}" for item in task['prep_items']])}
    
    **Customer Notes:**
    • {task['details']['notes']}
    • Budget: {task['details']['budget']}
    • Preferences: {task['details']['preferences']}
    """)


def show_stock_report_form():
    """Show stock reporting form."""
    with st.form("stock_report"):
        st.markdown("**Report Low Stock Item**")
        
        department = st.selectbox("Department", 
            ["Women's Fashion", "Men's Fashion", "Shoes", "Accessories", "Beauty"])
        
        item_description = st.text_input("Item Description")
        
        current_stock = st.number_input("Current Stock Level", min_value=0, value=0)
        
        urgency = st.radio("Urgency Level", 
            ["Low - Can wait until next delivery", 
             "Medium - Needed within 2-3 days", 
             "High - Needed immediately"])
        
        notes = st.text_area("Additional Notes (optional)")
        
        if st.form_submit_button("Submit Stock Report"):
            st.success("Stock report submitted to inventory team!")
            st.balloons()
