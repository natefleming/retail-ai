#!/usr/bin/env python3
"""
Test script for VP Dashboard components
"""

import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all required modules can be imported."""
    try:
        import streamlit as st

        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Streamlit: {e}")
        return False

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Plotly: {e}")
        return False

    try:
        import pandas as pd

        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Pandas: {e}")
        return False

    try:
        import numpy as np

        print("✅ Numpy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Numpy: {e}")
        return False

    return True


def test_vp_component():
    """Test that the VP dashboard component can be imported."""
    try:
        from components.homepage.vp_executive import (
            show_geographical_drill_down,
            show_performance_trends,
            show_strategic_insights,
            show_vp_executive_metrics,
            show_vp_homepage,
        )

        print("✅ VP Executive dashboard components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import VP dashboard components: {e}")
        return False


def test_config():
    """Test that the configuration includes the VP role."""
    try:
        from utils.config import load_config

        config = load_config()

        if "vp_retail_operations" in config.get("roles", {}):
            print("✅ VP role found in configuration")
        else:
            print("❌ VP role not found in configuration")
            return False

        if "vp_retail_operations" in config.get("employees", {}):
            print("✅ VP employee found in configuration")
        else:
            print("❌ VP employee not found in configuration")
            return False

        return True
    except Exception as e:
        print(f"❌ Failed to load or validate configuration: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Testing VP Dashboard Integration...")
    print("=" * 50)

    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check dependencies.")
        sys.exit(1)

    # Test VP component
    if not test_vp_component():
        print("\n❌ VP component tests failed. Please check the component files.")
        sys.exit(1)

    # Test configuration
    if not test_config():
        print("\n❌ Configuration tests failed. Please check config.yaml.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("🎉 All tests passed! VP Dashboard integration is ready.")
    print("\n📋 VP of Retail Operations features:")
    print("   • Executive-level KPI dashboard")
    print("   • Geographical drill-down analysis")
    print("   • AI-powered strategic insights")
    print("   • Performance trends analytics")
    print("   • Real-time network oversight")

    print("\n🔧 To use the VP dashboard:")
    print("   1. Start the Streamlit app: streamlit run app.py")
    print("   2. Select 'Vp Retail Operations' from the role dropdown")
    print("   3. Access the executive dashboard with 4 main tabs")
