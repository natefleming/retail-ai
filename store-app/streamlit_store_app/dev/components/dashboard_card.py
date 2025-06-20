"""
Dashboard Card Component

Created in the Development Playground.
"""

import streamlit as st
import plotly.express as px
import pandas as pd


def show_component():
    """Main function to display this component."""
    st.header("🎨 Dashboard Card")
    
    st.markdown("""
    Welcome to your new component! 
    
    **Getting Started:**
    1. Edit this file in your IDE
    2. Use the reload button to see changes
    3. Build amazing visualizations!
    """)
    
    # Example interactive element
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sample Chart")
        # Sample data
        data = pd.DataFrame({
            'x': range(10),
            'y': [i**2 for i in range(10)]
        })
        
        fig = px.line(data, x='x', y='y', title='Sample Interactive Chart')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎛️ Controls")
        
        number = st.slider("Pick a number", 0, 100, 50)
        st.metric("Your Number", number, delta=number-50)
        
        color = st.color_picker("Pick a color", "#00f900")
        st.markdown(f'<div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">Your color!</div>', unsafe_allow_html=True)
    
    # Add your component logic here
    st.info("💡 Edit this file to customize your component!")


if __name__ == "__main__":
    show_component()
