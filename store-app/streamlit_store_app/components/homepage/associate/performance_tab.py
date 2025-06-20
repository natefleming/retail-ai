"""Associate BOPIS order management and performance tab."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


def show_performance_tab():
    """Display the BOPIS Order Management tab with order queue and performance metrics."""
    
    # Header section
    st.markdown("### 📦 BOPIS Order Management")
    
    # BOPIS performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            """
            <div style="background: #f8fafc; border-radius: 12px; padding: 16px; border-left: 4px solid #ef4444; margin-bottom: 16px;">
                <div style="font-size: 24px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">2</div>
                <div style="font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600;">Overdue Orders</div>
                <div style="font-size: 11px; color: #ef4444; font-weight: 600;">Need immediate attention</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div style="background: #f8fafc; border-radius: 12px; padding: 16px; border-left: 4px solid #f59e0b; margin-bottom: 16px;">
                <div style="font-size: 24px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">5</div>
                <div style="font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600;">Ready for Pickup</div>
                <div style="font-size: 11px; color: #f59e0b; font-weight: 600;">Customers notified</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div style="background: #f8fafc; border-radius: 12px; padding: 16px; border-left: 4px solid #3b82f6; margin-bottom: 16px;">
                <div style="font-size: 24px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">8</div>
                <div style="font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600;">Being Prepared</div>
                <div style="font-size: 11px; color: #3b82f6; font-weight: 600;">In progress</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            """
            <div style="background: #f8fafc; border-radius: 12px; padding: 16px; border-left: 4px solid #10b981; margin-bottom: 16px;">
                <div style="font-size: 24px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">23</div>
                <div style="font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600;">Completed Today</div>
                <div style="font-size: 11px; color: #10b981; font-weight: 600;">95% on time</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Main content area
    col_main, col_side = st.columns([2.5, 1])
    
    with col_main:
        # BOPIS Order Queue
        st.markdown("#### 🚨 URGENT: Overdue Orders")
        
        overdue_orders = [
            {
                "order_id": "B2024-0156",
                "customer": "Sarah Johnson",
                "phone": "(555) 123-4567",
                "total": "$247.89",
                "due_time": "2:30 PM",
                "overdue_hours": 2.5,
                "items": ["Navy Blazer (Size 8)", "Black Dress Pants (Size 8)", "White Silk Blouse (Size M)"],
                "location": "Customer Service Counter",
                "payment_status": "Paid"
            },
            {
                "order_id": "B2024-0162",
                "customer": "Michael Torres",
                "phone": "(555) 789-0123",
                "total": "$89.95",
                "due_time": "1:15 PM",
                "overdue_hours": 3.8,
                "items": ["Men's Casual Shirt (L)", "Blue Jeans (32x30)"],
                "location": "Customer Service Counter",
                "payment_status": "Paid"
            }
        ]
        
        for order in overdue_orders:
            st.markdown(
                f"""
                <div style="border-left: 4px solid #ef4444; background: #fee2e2; padding: 16px; border-radius: 8px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div>
                            <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">
                                Order #{order['order_id']} - {order['customer']}
                            </div>
                            <div style="font-size: 14px; color: #64748b;">
                                📞 {order['phone']} • Due: {order['due_time']} • Total: {order['total']}
                            </div>
                        </div>
                        <div style="background: #dc2626; color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 700;">
                            ⚠️ OVERDUE {order['overdue_hours']}H
                        </div>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <div style="font-size: 12px; color: #64748b; font-weight: 600; margin-bottom: 4px;">ITEMS:</div>
                        <div style="font-size: 14px; color: #1e293b;">
                            {' • '.join(order['items'])}
                        </div>
                    </div>
                    <div style="font-size: 12px; color: #64748b;">
                        📍 Location: {order['location']} • Payment: {order['payment_status']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Action buttons for overdue orders
            col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
            
            with col_btn1:
                if st.button(f"📞 Call Customer", key=f"call_{order['order_id']}", use_container_width=True):
                    st.info(f"Calling {order['phone']}...")
            
            with col_btn2:
                if st.button(f"📦 Locate Items", key=f"locate_{order['order_id']}", use_container_width=True):
                    st.info(f"Items located at: {order['location']}")
            
            with col_btn3:
                if st.button(f"✅ Process Pickup", key=f"pickup_{order['order_id']}", use_container_width=True, type="primary"):
                    st.success(f"✅ Order {order['order_id']} processed for pickup!")
                    st.balloons()
            
            with col_btn4:
                if st.button(f"❌ Mark Missing", key=f"missing_{order['order_id']}", use_container_width=True):
                    show_missing_items_form(order['order_id'])
            
            st.markdown("<br>", unsafe_allow_html=True)

        # Ready for Pickup Orders
        st.markdown("#### 📬 Ready for Pickup")
        
        ready_orders = [
            {
                "order_id": "B2024-0171",
                "customer": "Emma Martinez",
                "phone": "(555) 456-7890",
                "total": "$125.47",
                "ready_time": "3:45 PM",
                "items": ["Summer Dress (M)", "Sandals (Size 8)"],
                "pickup_window": "Until 6:00 PM today"
            },
            {
                "order_id": "B2024-0168",
                "customer": "David Chen",
                "phone": "(555) 234-5678",
                "total": "$78.99",
                "ready_time": "4:20 PM",
                "items": ["Polo Shirt (L)", "Shorts (Size 32)"],
                "pickup_window": "Until 6:00 PM today"
            }
        ]
        
        for order in ready_orders:
            st.markdown(
                f"""
                <div style="border-left: 4px solid #10b981; background: #ecfdf5; padding: 12px; border-radius: 8px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <div style="font-weight: 600; color: #1e293b; margin-bottom: 4px;">
                                #{order['order_id']} - {order['customer']} • {order['total']}
                            </div>
                            <div style="font-size: 14px; color: #64748b;">
                                📞 {order['phone']} • Ready: {order['ready_time']} • Pickup: {order['pickup_window']}
                            </div>
                            <div style="font-size: 13px; color: #059669; font-weight: 500; margin-top: 4px;">
                                Items: {' • '.join(order['items'])}
                            </div>
                        </div>
                        <div style="background: #059669; color: white; padding: 4px 8px; border-radius: 12px; font-size: 11px; font-weight: 700;">
                            ✅ READY
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # In Progress Orders
        st.markdown("#### 🔄 Orders Being Prepared")
        
        in_progress_orders = [
            {
                "order_id": "B2024-0175",
                "customer": "Lisa Thompson",
                "estimated_ready": "5:30 PM",
                "items": 4,
                "progress": 75,
                "assigned_to": "You"
            },
            {
                "order_id": "B2024-0178",
                "customer": "James Wilson",
                "estimated_ready": "6:00 PM",
                "items": 2,
                "progress": 50,
                "assigned_to": "Jessica M."
            }
        ]
        
        for order in in_progress_orders:
            progress_color = "#10b981" if order["progress"] >= 75 else "#f59e0b" if order["progress"] >= 50 else "#ef4444"
            
            st.markdown(
                f"""
                <div style="border-left: 4px solid {progress_color}; background: #f8fafc; padding: 12px; border-radius: 8px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <div style="font-weight: 600; color: #1e293b;">#{order['order_id']} - {order['customer']}</div>
                            <div style="font-size: 13px; color: #64748b;">
                                Est. Ready: {order['estimated_ready']} • {order['items']} items • Assigned: {order['assigned_to']}
                            </div>
                        </div>
                        <div style="color: {progress_color}; font-weight: 600; font-size: 14px;">
                            {order['progress']}%
                        </div>
                    </div>
                    <div style="background: #e2e8f0; height: 6px; border-radius: 3px; overflow: hidden;">
                        <div style="background: {progress_color}; height: 100%; width: {order['progress']}%; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_side:
        # Quick Actions Panel
        st.markdown("#### ⚡ BOPIS Actions")
        
        if st.button("📱 Scan Order Barcode", use_container_width=True):
            st.info("Opening barcode scanner...")
        
        if st.button("🔍 Search Order #", use_container_width=True):
            show_order_search_form()
        
        if st.button("📞 Call Stockroom", use_container_width=True):
            st.info("Calling stockroom extension 2255")
        
        if st.button("❓ BOPIS Help Guide", use_container_width=True):
            show_bopis_help_guide()
        
        if st.button("📋 Print Pickup List", use_container_width=True):
            st.success("Pickup list sent to printer")

        st.markdown("---")
        
        # Today's BOPIS Performance
        st.markdown("#### 📊 Your BOPIS Stats")
        
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px; border-radius: 12px; text-align: center; margin-bottom: 16px;">
                <div style="font-size: 18px; font-weight: 700; margin-bottom: 8px;">BOPIS Performance</div>
                <div style="font-size: 24px; font-weight: 900; margin-bottom: 4px;">95%</div>
                <div style="font-size: 12px; opacity: 0.9;">On-time completion rate</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        bopis_stats = [
            {"label": "Orders Processed", "value": "23", "target": "25", "color": "#10b981"},
            {"label": "Avg. Prep Time", "value": "12 min", "target": "15 min", "color": "#10b981"},
            {"label": "Customer Satisfaction", "value": "4.8/5", "target": "4.5/5", "color": "#10b981"},
            {"label": "Items Located", "value": "98%", "target": "95%", "color": "#10b981"}
        ]
        
        for stat in bopis_stats:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #e5e7eb;">
                    <span style="font-size: 12px; font-weight: 600; color: #1e293b;">{stat['label']}</span>
                    <div style="text-align: right;">
                        <div style="font-size: 12px; color: {stat['color']}; font-weight: 700;">{stat['value']}</div>
                        <div style="font-size: 10px; color: #64748b;">Target: {stat['target']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        
        # BOPIS Tips
        st.markdown("#### 💡 Quick Tips")
        
        tips = [
            "Check customer ID before releasing orders",
            "Verify phone number for security",
            "Offer to carry items to customer's car",
            "Apologize for any wait time",
            "Mention other services (alterations, gift wrap)"
        ]
        
        for i, tip in enumerate(tips, 1):
            st.markdown(
                f"""
                <div style="background: #f0f9ff; border-left: 3px solid #0369a1; padding: 8px; margin-bottom: 6px; border-radius: 4px;">
                    <div style="font-size: 11px; color: #0369a1; font-weight: 600; text-transform: uppercase; margin-bottom: 2px;">Tip #{i}</div>
                    <div style="font-size: 12px; color: #1e293b;">{tip}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        
        # Shift Goals
        st.markdown("#### 🎯 Today's Goals")
        
        goals = [
            {"name": "Process 25 BOPIS orders", "current": 23, "target": 25},
            {"name": "Maintain 95% on-time rate", "current": 95, "target": 95},
            {"name": "Zero customer complaints", "current": 0, "target": 0}
        ]
        
        for goal in goals:
            if goal["target"] == 0:
                progress = 100 if goal["current"] == 0 else 0
                status = "✅ Achieved" if goal["current"] == 0 else "❌ Not Met"
            else:
                progress = min(100, (goal["current"] / goal["target"]) * 100)
                status = f"{goal['current']}/{goal['target']}"
            
            color = "#10b981" if progress >= 90 else "#f59e0b" if progress >= 70 else "#ef4444"
            
            st.markdown(
                f"""
                <div style="background: #f8fafc; border-radius: 8px; padding: 12px; margin-bottom: 8px; border-left: 3px solid {color};">
                    <div style="font-size: 12px; font-weight: 600; color: #1e293b; margin-bottom: 4px;">{goal['name']}</div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="background: #e2e8f0; height: 4px; border-radius: 2px; flex: 1; margin-right: 8px;">
                            <div style="background: {color}; height: 100%; width: {progress}%; border-radius: 2px;"></div>
                        </div>
                        <div style="font-size: 11px; color: {color}; font-weight: 600;">{status}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def show_missing_items_form(order_id):
    """Show form to report missing items for an order."""
    with st.form(f"missing_items_{order_id}"):
        st.markdown(f"**❌ Report Missing Items - Order #{order_id}**")
        
        missing_reason = st.selectbox("Reason for Missing Items", 
            ["Items not found in stockroom", "Items damaged/defective", "Items sold from display", "Inventory error", "Other"])
        
        missing_items = st.text_area("Which items are missing?", placeholder="List specific items that cannot be located")
        
        action_taken = st.selectbox("Action Taken",
            ["Notified stockroom manager", "Checked all possible locations", "Contacted other departments", "Customer notified of delay"])
        
        estimated_resolution = st.selectbox("Estimated Resolution Time",
            ["Within 30 minutes", "Within 2 hours", "By end of day", "Next business day", "Unknown"])
        
        notes = st.text_area("Additional Notes", placeholder="Any other relevant information...")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("📤 Submit Report", type="primary"):
                st.error(f"❌ Missing items report submitted for order #{order_id}")
                st.info("Manager and customer have been notified")
        
        with col2:
            if st.form_submit_button("❌ Cancel"):
                st.info("Report cancelled")


def show_order_search_form():
    """Show order search form."""
    with st.form("order_search"):
        st.markdown("**🔍 Search BOPIS Order**")
        
        search_type = st.radio("Search by:", ["Order Number", "Customer Phone", "Customer Name"])
        
        if search_type == "Order Number":
            search_query = st.text_input("Order Number", placeholder="B2024-XXXX")
        elif search_type == "Customer Phone":
            search_query = st.text_input("Phone Number", placeholder="(555) 123-4567")
        else:
            search_query = st.text_input("Customer Name", placeholder="First Last")
        
        if st.form_submit_button("🔍 Search Order", type="primary"):
            if search_query:
                st.success(f"Searching for: {search_query}")
                # Mock search result
                st.info("Order found: B2024-0156 - Sarah Johnson - Ready for pickup")
            else:
                st.warning("Please enter search criteria")


def show_bopis_help_guide():
    """Show BOPIS help guide."""
    st.info("""
    **📋 BOPIS Quick Reference Guide**
    
    **Order Processing Steps:**
    1. Locate items in stockroom or sales floor
    2. Check items for damage/defects
    3. Package items if needed
    4. Update order status to "Ready"
    5. Send pickup notification to customer
    
    **Customer Pickup Process:**
    1. Verify customer ID matches order
    2. Confirm phone number for security
    3. Process pickup in POS system
    4. Provide receipt
    5. Thank customer and offer assistance
    
    **Escalation Contacts:**
    • Missing items: Call stockroom ext. 2255
    • System issues: Call IT helpdesk ext. 4400
    • Customer complaints: Call manager ext. 1100
    """)
