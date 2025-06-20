"""Associate schedule tab."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


def show_schedule_tab():
    """Display the Schedule tab with comprehensive shift information and schedule management."""
    
    # Header Section
    st.markdown("### 📅 My Schedule Management")
    
    # Quick stats overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 This Week", "32.5 hrs", delta="2.5 hrs over target")
    
    with col2:
        st.metric("💰 Est. Earnings", "$487.50", delta="+$37.50 OT")
    
    with col3:
        st.metric("🔄 Next Break", "2:30 PM", delta="1h 15m from now")
    
    with col4:
        st.metric("📞 Coverage Status", "Solo", delta="Until 3:00 PM")

    st.markdown("---")

    # Main content area
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        # Today's Schedule Details
        st.markdown("#### 🕐 Today's Schedule")
        
        # Current shift breakdown
        today_schedule = [
            {
                "time": "9:00 AM",
                "event": "Shift Start - Women's Fashion",
                "status": "completed",
                "duration": "",
                "notes": "Clocked in on time"
            },
            {
                "time": "10:30 AM", 
                "event": "Floor Coverage - Designer Area",
                "status": "completed",
                "duration": "1.5 hrs",
                "notes": "Solo coverage period"
            },
            {
                "time": "12:00 PM",
                "event": "Lunch Break",
                "status": "completed", 
                "duration": "30 min",
                "notes": "Took break with Lisa"
            },
            {
                "time": "1:30 PM",
                "event": "Personal Shopping - Emma R.",
                "status": "current",
                "duration": "90 min",
                "notes": "VIP customer consultation"
            },
            {
                "time": "3:00 PM",
                "event": "Team Coverage Returns",
                "status": "upcoming",
                "duration": "2 hrs", 
                "notes": "Jessica joins for support"
            },
            {
                "time": "4:30 PM",
                "event": "Inventory Count",
                "status": "upcoming",
                "duration": "1 hr",
                "notes": "Designer handbags section"
            },
            {
                "time": "5:30 PM",
                "event": "Shift End & Clock Out",
                "status": "upcoming",
                "duration": "",
                "notes": "8.5 hour shift complete"
            }
        ]
        
        for item in today_schedule:
            # Status styling
            if item["status"] == "completed":
                status_color = "#10b981"
                bg_color = "#ecfdf5"
                status_icon = "✅"
            elif item["status"] == "current":
                status_color = "#f59e0b"
                bg_color = "#fef3c7"
                status_icon = "🔄"
            else:
                status_color = "#6b7280"
                bg_color = "#f9fafb"
                status_icon = "⏰"
            
            st.markdown(
                f"""
                <div style="border-left: 4px solid {status_color}; background: {bg_color}; padding: 16px; border-radius: 8px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                                <span style="font-size: 16px;">{status_icon}</span>
                                <span style="font-weight: 700; color: #1e293b;">{item['time']}</span>
                                <span style="color: #64748b;">•</span>
                                <span style="font-weight: 600; color: #1e293b;">{item['event']}</span>
                            </div>
                            <div style="color: #64748b; font-size: 14px; margin-left: 24px;">{item['notes']}</div>
                        </div>
                        <div style="text-align: right; color: {status_color}; font-weight: 600;">{item['duration']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Weekly Schedule Overview
        st.markdown("#### 📅 This Week's Schedule")
        
        # Create weekly schedule data
        weekly_schedule = [
            {
                "day": "Monday (Today)",
                "date": "Dec 16",
                "shift": "9:00 AM - 5:30 PM",
                "department": "Women's Fashion",
                "hours": "8.5",
                "status": "active",
                "notes": "Personal shopping appointment at 1:30 PM"
            },
            {
                "day": "Tuesday", 
                "date": "Dec 17",
                "shift": "8:00 AM - 4:30 PM",
                "department": "Electronics", 
                "hours": "8.0",
                "status": "scheduled",
                "notes": "Holiday inventory prep"
            },
            {
                "day": "Wednesday",
                "date": "Dec 18", 
                "shift": "10:00 AM - 6:30 PM",
                "department": "Women's Fashion",
                "hours": "8.0",
                "status": "scheduled",
                "notes": "Evening shift coverage"
            },
            {
                "day": "Thursday",
                "date": "Dec 19",
                "shift": "OFF",
                "department": "---",
                "hours": "0",
                "status": "off",
                "notes": "Requested day off"
            },
            {
                "day": "Friday",
                "date": "Dec 20",
                "shift": "9:00 AM - 7:00 PM",
                "department": "Women's Fashion", 
                "hours": "9.5",
                "status": "scheduled",
                "notes": "Weekend prep + overtime"
            },
            {
                "day": "Saturday",
                "date": "Dec 21",
                "shift": "8:00 AM - 2:00 PM", 
                "department": "Store Support",
                "hours": "5.5",
                "status": "scheduled",
                "notes": "Holiday rush support"
            },
            {
                "day": "Sunday",
                "date": "Dec 22",
                "shift": "OFF",
                "department": "---",
                "hours": "0", 
                "status": "off",
                "notes": "Weekly rest day"
            }
        ]
        
        # Display weekly schedule in a clean table format
        for day_info in weekly_schedule:
            if day_info["status"] == "active":
                border_color = "#f59e0b"
                bg_color = "#fef3c7"
            elif day_info["status"] == "off":
                border_color = "#6b7280"
                bg_color = "#f3f4f6"
            else:
                border_color = "#3b82f6"
                bg_color = "#dbeafe"
            
            st.markdown(
                f"""
                <div style="border: 1px solid {border_color}; background: {bg_color}; border-radius: 8px; padding: 12px; margin-bottom: 6px;">
                    <div style="display: grid; grid-template-columns: 2fr 1fr 2fr 2fr 1fr; gap: 16px; align-items: center;">
                        <div>
                            <div style="font-weight: 700; color: #1e293b;">{day_info['day']}</div>
                            <div style="font-size: 12px; color: #64748b;">{day_info['date']}</div>
                        </div>
                        <div style="font-weight: 600; color: #1e293b;">{day_info['shift']}</div>
                        <div style="color: #64748b;">{day_info['department']}</div>
                        <div style="font-size: 12px; color: #64748b;">{day_info['notes']}</div>
                        <div style="text-align: right; font-weight: 600; color: {border_color};">{day_info['hours']}h</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_side:
        # Quick Actions Panel
        st.markdown("#### ⚡ Schedule Actions")
        
        if st.button("📱 Request Time Off", use_container_width=True):
            show_time_off_request()
        
        if st.button("🔄 Find Shift Coverage", use_container_width=True):
            show_shift_coverage_request()
        
        if st.button("⏰ Adjust Break Time", use_container_width=True):
            st.info("Break adjustment request sent to manager")
        
        if st.button("📞 Report Schedule Issue", use_container_width=True):
            st.info("Schedule issue reported to HR")
        
        if st.button("💰 View Payroll Info", use_container_width=True):
            show_payroll_summary()

        st.markdown("---")
        
        # Schedule Summary
        st.markdown("#### 📊 Week Summary")
        
        # Weekly totals
        total_hours = 32.5
        regular_hours = 30.0
        overtime_hours = 2.5
        total_days = 5
        
        st.markdown(
            f"""
            <div style="
                background: #f8fafc;
                border-radius: 12px;
                padding: 16px;
                border: 1px solid #e2e8f0;
                margin-bottom: 16px;
            ">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: 700; color: #1e293b;">{total_hours}</div>
                        <div style="font-size: 12px; color: #64748b;">Total Hours</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: 700; color: #1e293b;">{total_days}</div>
                        <div style="font-size: 12px; color: #64748b;">Work Days</div>
                    </div>
                </div>
                <div style="border-top: 1px solid #e2e8f0; padding-top: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: #64748b; font-size: 14px;">Regular:</span>
                        <span style="font-weight: 600;">{regular_hours}h</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #64748b; font-size: 14px;">Overtime:</span>
                        <span style="font-weight: 600; color: #f59e0b;">{overtime_hours}h</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Department Assignments
        st.markdown("#### 🏪 Department Rotation")
        
        departments = [
            {"name": "Women's Fashion", "hours": "26h", "percentage": "80%"},
            {"name": "Electronics", "hours": "8h", "percentage": "25%"},
            {"name": "Store Support", "hours": "5.5h", "percentage": "17%"}
        ]
        
        for dept in departments:
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 8px 0;
                    border-bottom: 1px solid #e5e7eb;
                ">
                    <div>
                        <div style="font-weight: 600; color: #1e293b; font-size: 14px;">{dept['name']}</div>
                        <div style="color: #64748b; font-size: 12px;">{dept['percentage']} of week</div>
                    </div>
                    <div style="font-weight: 600; color: #3b82f6;">
                        {dept['hours']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        
        # Upcoming Alerts
        st.markdown("#### 🔔 Schedule Alerts")
        
        alerts = [
            {"type": "overtime", "message": "Approaching 40hr limit this week", "urgent": True},
            {"type": "coverage", "message": "Solo coverage Tue 2-4 PM", "urgent": False},
            {"type": "holiday", "message": "Holiday schedule starts Dec 23", "urgent": False}
        ]
        
        for alert in alerts:
            alert_color = "#fee2e2" if alert["urgent"] else "#f0f9ff"
            text_color = "#dc2626" if alert["urgent"] else "#0369a1"
            icon = "⚠️" if alert["urgent"] else "ℹ️"
            
            st.markdown(
                f"""
                <div style="
                    background: {alert_color};
                    color: {text_color};
                    padding: 8px;
                    border-radius: 6px;
                    margin-bottom: 6px;
                    font-size: 12px;
                ">
                    {icon} {alert['message']}
                </div>
                """,
                unsafe_allow_html=True
            )


def show_time_off_request():
    """Show time off request form."""
    with st.form("time_off_request"):
        st.markdown("**Request Time Off**")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
        
        request_type = st.selectbox("Request Type", 
            ["Vacation", "Personal Day", "Sick Leave", "Family Emergency", "Other"])
        
        reason = st.text_area("Reason (optional)")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Submit Request", type="primary"):
                st.success("✅ Time off request submitted to manager!")
                st.balloons()
        with col2:
            if st.form_submit_button("Cancel"):
                st.info("Request cancelled")


def show_shift_coverage_request():
    """Show shift coverage request form."""
    with st.form("coverage_request"):
        st.markdown("**Find Shift Coverage**")
        
        col1, col2 = st.columns(2)
        with col1:
            shift_date = st.date_input("Date Needing Coverage")
        with col2:
            shift_time = st.selectbox("Shift Time", 
                ["Morning (8 AM - 2 PM)", "Afternoon (2 PM - 8 PM)", "Evening (4 PM - 10 PM)", "Full Day"])
        
        reason = st.selectbox("Reason",
            ["Personal Emergency", "Medical Appointment", "Family Event", "School Commitment", "Other"])
        
        notes = st.text_area("Additional Notes")
        
        if st.form_submit_button("Request Coverage", type="primary"):
            st.success("✅ Coverage request posted to team board!")
            st.info("💬 Available associates will be notified")


def show_payroll_summary():
    """Show payroll information summary."""
    st.markdown("**💰 Estimated Earnings This Week**")
    
    # Calculate earnings
    regular_rate = 15.50
    overtime_rate = 23.25  # 1.5x regular
    regular_hours = 30.0
    overtime_hours = 2.5
    
    regular_pay = regular_hours * regular_rate
    overtime_pay = overtime_hours * overtime_rate
    gross_pay = regular_pay + overtime_pay
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Regular Pay", f"${regular_pay:.2f}", f"{regular_hours}h @ ${regular_rate}")
        st.metric("Overtime Pay", f"${overtime_pay:.2f}", f"{overtime_hours}h @ ${overtime_rate}")
    
    with col2:
        st.metric("Gross Pay", f"${gross_pay:.2f}", f"Before taxes/deductions")
        st.metric("Est. Take Home", f"${gross_pay * 0.78:.2f}", "~22% taxes/deductions")
    
    st.info("💡 **Tip:** Work 2.5 more hours this week to maximize your overtime pay!")
    
    if st.button("📊 View Full Payroll Details"):
        st.info("Opening detailed payroll portal...")
