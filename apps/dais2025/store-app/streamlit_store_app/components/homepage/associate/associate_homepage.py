"""Main store associate homepage coordinator."""

import streamlit as st
from datetime import datetime, timedelta

from .dashboard_tab import show_associate_dashboard_tab
from .my_tasks_tab import show_my_tasks_tab
from .performance_tab import show_performance_tab
from .products_tab import show_product_lookup_tab
from .schedule_tab import show_schedule_tab


def show_shift_status_bar():
    """Display persistent shift status information."""
    # Calculate shift progress
    shift_start = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    shift_end = datetime.now().replace(hour=17, minute=0, second=0, microsecond=0)
    current_time = datetime.now()
    
    # Calculate time remaining
    time_remaining = shift_end - current_time
    hours_remaining = int(time_remaining.total_seconds() // 3600)
    minutes_remaining = int((time_remaining.total_seconds() % 3600) // 60)
    
    # Calculate shift progress percentage
    total_shift_duration = (shift_end - shift_start).total_seconds()
    elapsed_duration = (current_time - shift_start).total_seconds()
    progress_percent = max(0, min(100, (elapsed_duration / total_shift_duration) * 100))
    
    # Determine status
    if time_remaining.total_seconds() <= 0:
        status_text = "Shift Complete"
        status_color = "#10b981"
        time_text = "Time to clock out!"
    elif time_remaining.total_seconds() <= 1800:  # 30 minutes or less
        status_text = "Shift Ending Soon"
        status_color = "#f59e0b"
        time_text = f"{hours_remaining}h {minutes_remaining}m remaining"
    else:
        status_text = "On Shift"
        status_color = "#3b82f6"
        time_text = f"{hours_remaining}h {minutes_remaining}m remaining"
    
    # Current assignment
    current_department = "Women's Fashion - Designer Area"
    
    # Quick stats
    tasks_completed = 6
    total_tasks = 8
    break_due = current_time.hour >= 13 and current_time.minute >= 0  # Break due after 1 PM
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        ">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="
                        background: {status_color};
                        color: white;
                        padding: 6px 12px;
                        border-radius: 20px;
                        font-size: 14px;
                        font-weight: 600;
                    ">{status_text}</div>
                    <div style="color: #475569; font-weight: 600;">{time_text}</div>
                    <div style="color: #64748b;">• {current_department}</div>
                </div>
                <div style="display: flex; align-items: center; gap: 20px;">
                    <div style="text-align: center;">
                        <div style="font-size: 18px; font-weight: 700; color: #1e293b;">{tasks_completed}/{total_tasks}</div>
                        <div style="font-size: 12px; color: #64748b;">Tasks Done</div>
                    </div>
                    {'<div style="background: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 8px; font-size: 12px; font-weight: 600;">Break Due!</div>' if break_due else ''}
                </div>
            </div>
            <div style="
                background: #e2e8f0;
                height: 6px;
                border-radius: 3px;
                overflow: hidden;
            ">
                <div style="
                    background: {status_color};
                    height: 100%;
                    width: {progress_percent}%;
                    transition: width 0.3s ease;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_associate_homepage(selected_nav: str = "My Dashboard"):
    """Display homepage content for store associates based on navigation selection."""    
    
    # Calculate dynamic appointment time (same logic as my_tasks_tab.py)
    from datetime import datetime
    now = datetime.now()
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
    
    # Initialize session state for toast notification
    if 'associate_task_notification_shown' not in st.session_state:
        st.session_state.associate_task_notification_shown = False
    
    # Show task assignment toast notification on first load
    if not st.session_state.associate_task_notification_shown:
        # Enhanced urgent toast notification with custom styling
        st.markdown(
            """
            <style>
            .urgent-task-toast {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 9999;
                background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 50%, #6d28d9 100%);
                color: white;
                padding: 20px 24px;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(139, 92, 246, 0.4), 0 0 30px rgba(139, 92, 246, 0.3);
                border: 3px solid #c4b5fd;
                animation: taskPulse 2s infinite, slideInRight 0.5s ease-out;
                max-width: 420px;
                font-weight: 600;
                font-size: 16px;
                line-height: 1.4;
            }
            
            @keyframes taskPulse {
                0% { box-shadow: 0 20px 60px rgba(139, 92, 246, 0.4), 0 0 30px rgba(139, 92, 246, 0.3); }
                50% { box-shadow: 0 25px 80px rgba(139, 92, 246, 0.6), 0 0 50px rgba(139, 92, 246, 0.5); }
                100% { box-shadow: 0 20px 60px rgba(139, 92, 246, 0.4), 0 0 30px rgba(139, 92, 246, 0.3); }
            }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            .task-badge {
                background: #fbbf24;
                color: #92400e;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 8px;
                display: inline-block;
                animation: badgePulse 1.5s infinite;
            }
            
            @keyframes badgePulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }
            
            .assignment-highlight {
                background: rgba(255,255,255,0.2);
                padding: 2px 8px;
                border-radius: 8px;
                font-weight: 900;
                display: inline-block;
                animation: highlight 2s infinite;
            }
            
            @keyframes highlight {
                0%, 100% { background: rgba(255,255,255,0.2); }
                50% { background: rgba(255,255,255,0.4); }
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Use enhanced toast with dynamic timing
        st.toast(f"🚨⚡ NEW URGENT ASSIGNMENT ⚡🚨\n\nPersonal Shopping with Platinum Member Emma Rodriguez!\n\n🛍️ STARTING IN {minutes_until} MINUTES - VIEW DETAILS NOW! 🛍️", icon="🎯")
        
        # Add a visual overlay notification with dynamic timing
        st.markdown(
            f"""
            <div class="urgent-task-toast">
                <div class="task-badge">🔥 NEW ASSIGNMENT</div>
                <div style="font-size: 20px; margin-bottom: 8px;">🎯 PERSONAL SHOPPING TASK</div>
                <div style="margin-bottom: 12px;">
                    <strong>Platinum Member:</strong> Emma Rodriguez<br>
                    <strong>Start Time:</strong> <span class="assignment-highlight">{minutes_until} MINUTES</span>
                </div>
                <div style="font-size: 14px; opacity: 0.95;">
                    🛍️ Professional wardrobe consultation<br>
                    💰 Budget: $500-800<br>
                    📍 Women's Fashion - Designer Area<br>
                    🚀 <strong>Click "My Tasks" for full details!</strong>
                </div>
            </div>
            
            <script>
                setTimeout(function() {{
                    const toast = document.querySelector('.urgent-task-toast');
                    if (toast) {{
                        toast.style.animation = 'taskPulse 2s infinite, fadeOut 1s ease-in-out 8s forwards';
                    }}
                }}, 100);
                
                setTimeout(function() {{
                    const toast = document.querySelector('.urgent-task-toast');
                    if (toast) {{
                        toast.remove();
                    }}
                }}, 10000);
            </script>
            """,
            unsafe_allow_html=True
        )
        
        st.session_state.associate_task_notification_shown = True
    
    # Map navigation selection to content
    if selected_nav == "My Dashboard":
        show_associate_dashboard_tab()
    elif selected_nav == "My Tasks":
        show_my_tasks_tab()
    elif selected_nav == "My Schedule":
        show_schedule_tab()
    elif selected_nav == "Product Lookup":
        show_product_lookup_tab()
    elif selected_nav == "Performance" or selected_nav == "Customer Service":
        # Customer Service maps to performance for now
        show_performance_tab()
    else:
        # Default to dashboard for any unrecognized selection
        show_associate_dashboard_tab()
