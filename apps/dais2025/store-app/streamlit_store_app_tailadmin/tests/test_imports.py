"""Test basic imports and functionality."""

import pytest


def test_tailadmin_imports():
    """Test that all TailAdmin components can be imported."""
    from components.tailadmin import (
        TAILADMIN_COLORS,
        create_tailadmin_card,
        create_tailadmin_metric_card,
        get_tailadmin_color,
        inject_tailadmin_css,
    )

    # Test that color system works
    assert TAILADMIN_COLORS["brand"]["500"] == "#465fff"
    assert get_tailadmin_color("brand") == "#465fff"
    assert get_tailadmin_color("success", "500") == "#12b76a"


def test_page_imports():
    """Test that all page modules can be imported."""
    from pages.components_demo import show_components_demo
    from pages.homepage import show_homepage
    from pages.implementation_guide import show_implementation_guide
    from pages.vp_dashboard_clean import show_vp_dashboard_clean
    from pages.vp_dashboard_enhanced import show_vp_dashboard_enhanced

    # Test that functions exist and are callable
    assert callable(show_homepage)
    assert callable(show_vp_dashboard_clean)
    assert callable(show_vp_dashboard_enhanced)
    assert callable(show_implementation_guide)
    assert callable(show_components_demo)


def test_create_metric_card():
    """Test metric card creation."""
    from components.tailadmin import create_tailadmin_metric_card

    card_html = create_tailadmin_metric_card(
        icon="💰", value="$45.2K", label="Revenue", change="12.5%", change_type="positive"
    )

    assert isinstance(card_html, str)
    assert "💰" in card_html
    assert "$45.2K" in card_html
    assert "Revenue" in card_html
    assert "12.5%" in card_html


def test_create_card():
    """Test basic card creation."""
    from components.tailadmin import create_tailadmin_card

    card_html = create_tailadmin_card(title="Test Card", content="<p>Test content</p>")

    assert isinstance(card_html, str)
    assert "Test Card" in card_html
    assert "Test content" in card_html
