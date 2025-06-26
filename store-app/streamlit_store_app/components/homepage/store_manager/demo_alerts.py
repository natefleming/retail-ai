"""Demo Alert System for Timed Alert Simulations."""

import streamlit as st
import time
from datetime import datetime, timedelta
import random


class DemoAlertSystem:
    """Manages demo alerts with timed triggers."""
    
    def __init__(self):
        self.demo_alerts = [
            {
                "id": "demo_1",
                "type": "VIP Customer Alert",
                "message": "🏆 Platinum Member Michael Thompson just walked in - Needs immediate personal shopper",
                "severity": "urgent",
                "trigger_after": 5,  # Changed from 10 to 5 seconds
                "demo_impact": "High-value customer ($2,500 avg purchase)",
                "sound_effect": "🔔"
            },
            {
                "id": "demo_2", 
                "type": "Inventory Critical",
                "message": "⚠️ Premium Handbags down to last 3 units - Weekend rush approaching",
                "severity": "urgent",
                "trigger_after": 20,  # Adjusted to maintain spacing
                "demo_impact": "Revenue risk: $15K+ potential weekend sales",
                "sound_effect": "🚨"
            },
            {
                "id": "demo_3",
                "type": "Staff Emergency",
                "message": "📞 Sarah Chen called in sick - Electronics dept needs coverage for 2-6 PM shift",
                "severity": "urgent", 
                "trigger_after": 40,  # Adjusted to maintain spacing
                "demo_impact": "Customer service impact during peak hours",
                "sound_effect": "⚡"
            },
            {
                "id": "demo_4",
                "type": "System Alert",
                "message": "💳 POS Terminal #3 showing connectivity issues - May need restart",
                "severity": "important",
                "trigger_after": 65,  # Adjusted to maintain spacing
                "demo_impact": "Checkout delays possible",
                "sound_effect": "⚠️"
            },
            {
                "id": "demo_5",
                "type": "Opportunity Alert",
                "message": "📈 Men's Winter Coats selling 45% above forecast - Consider extending promotion",
                "severity": "important",
                "trigger_after": 85,  # Adjusted to maintain spacing
                "demo_impact": "Revenue opportunity: Extend successful promotion",
                "sound_effect": "💡"
            }
        ]
    
    def initialize_demo_state(self):
        """Initialize demo state variables."""
        if "demo_start_time" not in st.session_state:
            st.session_state.demo_start_time = None
        if "demo_active_alerts" not in st.session_state:
            st.session_state.demo_active_alerts = []
        if "demo_resolved_alerts" not in st.session_state:
            st.session_state.demo_resolved_alerts = set()
        if "last_alert_sound" not in st.session_state:
            st.session_state.last_alert_sound = ""
    
    def start_demo(self):
        """Start the demo timer."""
        st.session_state.demo_start_time = time.time()
        st.session_state.demo_active_alerts = []
        st.session_state.demo_resolved_alerts = set()
        st.success("🎬 Demo started! New alerts will appear automatically...")
    
    def stop_demo(self):
        """Stop the demo and reset."""
        st.session_state.demo_start_time = None
        st.session_state.demo_active_alerts = []
        st.session_state.demo_resolved_alerts = set()
        st.session_state.last_alert_sound = ""
        st.info("Demo stopped and reset.")
    
    def check_for_new_alerts(self):
        """Check if any new alerts should be triggered."""
        if st.session_state.demo_start_time is None:
            return []
        
        elapsed_time = time.time() - st.session_state.demo_start_time
        new_alerts = []
        
        for alert in self.demo_alerts:
            # Check if alert should be triggered and hasn't been added yet
            if (elapsed_time >= alert["trigger_after"] and 
                alert["id"] not in [a["id"] for a in st.session_state.demo_active_alerts]):
                
                # Add timestamp when alert is triggered
                triggered_alert = alert.copy()
                triggered_alert["timestamp"] = datetime.now()
                triggered_alert["time_display"] = "Just now"
                
                st.session_state.demo_active_alerts.append(triggered_alert)
                new_alerts.append(triggered_alert)
                st.session_state.last_alert_sound = alert["sound_effect"]
        
        return new_alerts
    
    def get_active_alerts(self):
        """Get all currently active demo alerts."""
        # Update time displays for existing alerts
        for alert in st.session_state.demo_active_alerts:
            if alert["id"] not in st.session_state.demo_resolved_alerts:
                elapsed = datetime.now() - alert["timestamp"]
                minutes = int(elapsed.total_seconds() / 60)
                if minutes == 0:
                    alert["time_display"] = "Just now"
                elif minutes == 1:
                    alert["time_display"] = "1 minute ago"
                else:
                    alert["time_display"] = f"{minutes} minutes ago"
        
        return [a for a in st.session_state.demo_active_alerts 
                if a["id"] not in st.session_state.demo_resolved_alerts]
    
    def resolve_alert(self, alert_id):
        """Mark an alert as resolved."""
        st.session_state.demo_resolved_alerts.add(alert_id)
    
    def get_demo_status(self):
        """Get current demo status."""
        if st.session_state.demo_start_time is None:
            return "Not started", 0
        
        elapsed = time.time() - st.session_state.demo_start_time
        total_demo_time = max([a["trigger_after"] for a in self.demo_alerts]) + 30
        
        if elapsed >= total_demo_time:
            return "Demo complete", 100
        
        progress = min((elapsed / total_demo_time) * 100, 100)
        return f"Running ({int(elapsed)}s elapsed)", progress


def show_demo_alert_controls():
    """Display demo alert controls."""
    demo_system = DemoAlertSystem()
    demo_system.initialize_demo_state()
    
    st.markdown("### 🎬 Demo Alert System")
    st.markdown("*Alerts appear on main Dashboard - no navigation required*")
    
    # Demo controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start Demo", use_container_width=True, type="primary"):
            demo_system.start_demo()
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Demo", use_container_width=True):
            demo_system.stop_demo()
            st.rerun()
    
    # Status and progress
    status, progress = demo_system.get_demo_status()
    
    if st.session_state.demo_start_time is None:
        st.info("💡 **Ready to Start!** First alert appears on **Dashboard** in **5 seconds**")
    else:
        st.write(f"**Status:** {status}")
        if progress > 0:
            st.progress(progress / 100)
    
    # Check for new alerts and show notification
    new_alerts = demo_system.check_for_new_alerts()
    if new_alerts:
        for alert in new_alerts:
            st.toast(f"{alert['sound_effect']} New {alert['severity'].upper()} Alert: {alert['type']}", icon="🚨")
        time.sleep(0.1)  # Small delay to show toast
        st.rerun()
    
    # Show upcoming alerts timeline
    if st.session_state.demo_start_time is not None:
        st.markdown("#### ⏰ Alert Timeline")
        elapsed = time.time() - st.session_state.demo_start_time
        
        upcoming_shown = False
        for i, alert in enumerate(demo_system.demo_alerts[:3]):  # Show first 3 for space
            time_until = alert["trigger_after"] - elapsed
            if time_until > 0 and alert["id"] not in [a["id"] for a in st.session_state.demo_active_alerts]:
                if not upcoming_shown:
                    upcoming_shown = True
                st.markdown(f"• **{alert['type']}** - in {int(time_until)} seconds")
            elif alert["id"] in [a["id"] for a in st.session_state.demo_active_alerts]:
                st.markdown(f"• ✅ **{alert['type']}** - triggered")
    
    return demo_system


def show_demo_alerts_display(demo_system):
    """Display active demo alerts."""
    active_alerts = demo_system.get_active_alerts()
    
    if not active_alerts:
        if st.session_state.demo_start_time is not None:
            st.info("🎬 Demo running... Waiting for alerts to trigger...")
        else:
            st.info("💡 Click 'Start Demo' to begin receiving simulated alerts")
        return
    
    # Add some custom CSS for demo alerts
    st.markdown("""
        <style>
        .demo-alert {
            border: 2px solid #3b82f6;
            border-radius: 12px;
            padding: 16px;
            margin: 12px 0;
            background: linear-gradient(135deg, #dbeafe 0%, #ffffff 50%);
            position: relative;
            animation: slideIn 0.5s ease-out;
        }
        
        .demo-alert.urgent {
            border-color: #dc2626;
            background: linear-gradient(135deg, #fee2e2 0%, #ffffff 50%);
        }
        
        .demo-alert.important {
            border-color: #d97706;
            background: linear-gradient(135deg, #fed7d7 0%, #ffffff 50%);
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(-100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        .demo-badge {
            position: absolute;
            top: -8px;
            right: 12px;
            background: #3b82f6;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚨 Live Demo Alerts")
    
    for alert in active_alerts:
        severity_class = alert["severity"]
        
        st.markdown(f"""
            <div class="demo-alert {severity_class}">
                <div class="demo-badge">DEMO</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <strong style="color: #1e293b; font-size: 16px;">{alert["type"]}</strong>
                    <span style="font-size: 12px; color: #64748b;">{alert["time_display"]}</span>
                </div>
                <div style="margin-bottom: 8px; color: #374151;">
                    {alert["message"]}
                </div>
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 12px;">
                    💼 <strong>Demo Impact:</strong> {alert["demo_impact"]}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Action button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(f"✅ Resolve", key=f"resolve_{alert['id']}", use_container_width=True):
                demo_system.resolve_alert(alert["id"])
                st.success(f"Alert resolved: {alert['type']}")
                time.sleep(0.5)
                st.rerun()


def show_demo_alert_simulator():
    """Main function to show the complete demo alert simulator."""
    st.markdown("## 🎬 Alert Demo System")
    st.markdown("This system simulates realistic store alerts at timed intervals for demonstrations.")
    
    # Show demo controls
    demo_system = show_demo_alert_controls()
    
    # Manual refresh option instead of auto-refresh
    if st.session_state.demo_start_time is not None:
        if st.button("🔄 Check for New Alerts", use_container_width=True):
            st.rerun()
    
    # Show active demo alerts
    show_demo_alerts_display(demo_system) 