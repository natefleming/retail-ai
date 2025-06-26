"""Metric display components for the Streamlit Store App."""

import streamlit as st


def display_metric_card(label: str, value: str):
    """
    Display a metric card with a label and value.

    Args:
        label: The metric label
        value: The metric value to display
    """
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>{value}</h3>
            <p>{label}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )


def display_alert(message: str, alert_type: str = "warning"):
    """
    Display an alert message.

    Args:
        message: The alert message to display
        alert_type: The type of alert (info, warning, error)
    """
    icon = {"info": "ℹ️", "warning": "⚠️", "error": "🚨"}.get(alert_type, "ℹ️")

    st.markdown(
        f"""
        <div class="alert-item alert-{alert_type}">
            <span class="alert-icon">{icon}</span>
            <span class="alert-message">{message}</span>
        </div>
    """,
        unsafe_allow_html=True,
    )


def display_quick_access_card(title: str, description: str, icon: str = "📦"):
    """Display a quick access card with consistent styling."""
    st.markdown(
        f"""
        <div style="background: white; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4>{icon} {title}</h4>
            <p>{description}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )
