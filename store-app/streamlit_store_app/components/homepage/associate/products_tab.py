"""Associate product lookup and stock info tab."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


def show_product_lookup_tab():
    """Display the Product Lookup tab with tools associates need for customer service."""
    
    # Header section with search
    st.markdown("### 🔍 Product Lookup & Stock Info")
    
    # Quick search bar
    search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
    
    with search_col1:
        search_query = st.text_input("🔍 Search products (name, SKU, or scan barcode)", placeholder="e.g. 'red dress size 8' or 'SKU-12345'")
    
    with search_col2:
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if search_query:
                st.success(f"Searching for: {search_query}")
            else:
                st.warning("Please enter a search term")
    
    with search_col3:
        if st.button("📷 Scan Item", use_container_width=True):
            st.info("Opening barcode scanner...")

    st.markdown("---")

    # Main content area
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        # Quick stats for associates
        st.markdown("#### 📊 Your Department Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                """
                <div style="background: #f8fafc; border-radius: 12px; padding: 16px; border-left: 4px solid #3b82f6; margin-bottom: 16px;">
                    <div style="font-size: 24px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">342</div>
                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600;">Items in Section</div>
                    <div style="font-size: 11px; color: #10b981; font-weight: 600;">Women's Fashion</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                """
                <div style="background: #f8fafc; border-radius: 12px; padding: 16px; border-left: 4px solid #f59e0b; margin-bottom: 16px;">
                    <div style="font-size: 24px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">7</div>
                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600;">Need Restocking</div>
                    <div style="font-size: 11px; color: #f59e0b; font-weight: 600;">Your Tasks</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                """
                <div style="background: #f8fafc; border-radius: 12px; padding: 16px; border-left: 4px solid #10b981; margin-bottom: 16px;">
                    <div style="font-size: 24px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">15</div>
                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600;">Customer Inquiries</div>
                    <div style="font-size: 11px; color: #10b981; font-weight: 600;">Today</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                """
                <div style="background: #f8fafc; border-radius: 12px; padding: 16px; border-left: 4px solid #8b5cf6; margin-bottom: 16px;">
                    <div style="font-size: 24px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">23</div>
                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600;">New Arrivals</div>
                    <div style="font-size: 11px; color: #8b5cf6; font-weight: 600;">This Week</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Popular/Trending Items
        st.markdown("#### 🔥 Trending Items (Know These!)")
        
        trending_items = [
            {
                "name": "Autumn Wool Blazer",
                "sku": "AWB-2024-15",
                "price": "$89.99",
                "location": "A2-15 (Rack 3)",
                "sizes": "XS, S, M, L available",
                "colors": "Navy, Black, Burgundy",
                "notes": "Popular for business wear - customers often ask for matching pants"
            },
            {
                "name": "Designer Silk Scarf Collection",
                "sku": "DSC-2024-22", 
                "price": "$45.99",
                "location": "A1-08 (Display Case)",
                "sizes": "One size",
                "colors": "6 patterns available",
                "notes": "Gift item - mention gift wrapping service"
            },
            {
                "name": "Comfort Stretch Jeans",
                "sku": "CSJ-2024-31",
                "price": "$79.99",
                "location": "B2-12 (Wall Display)",
                "sizes": "Size 4-16 in stock",
                "colors": "Dark wash, Light wash, Black",
                "notes": "Best seller - recommend size up for comfort fit"
            }
        ]
        
        for item in trending_items:
            st.markdown(
                f"""
                <div style="background: #f8fafc; border-radius: 12px; padding: 20px; border-left: 4px solid #10b981; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div>
                            <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">{item['name']}</div>
                            <div style="font-size: 14px; color: #64748b;">SKU: {item['sku']} • {item['price']}</div>
                        </div>
                        <div style="background: #10b981; color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">
                            🔥 TRENDING
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
                        <div>
                            <div style="font-size: 12px; color: #64748b; font-weight: 600; margin-bottom: 2px;">LOCATION</div>
                            <div style="font-size: 14px; color: #1e293b; font-weight: 600;">{item['location']}</div>
                        </div>
                        <div>
                            <div style="font-size: 12px; color: #64748b; font-weight: 600; margin-bottom: 2px;">SIZES</div>
                            <div style="font-size: 14px; color: #1e293b;">{item['sizes']}</div>
                        </div>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <div style="font-size: 12px; color: #64748b; font-weight: 600; margin-bottom: 2px;">COLORS AVAILABLE</div>
                        <div style="font-size: 14px; color: #1e293b;">{item['colors']}</div>
                    </div>
                    <div style="background: #e0f2fe; padding: 8px; border-radius: 6px; border-left: 3px solid #0369a1;">
                        <div style="font-size: 12px; color: #64748b; font-weight: 600; margin-bottom: 2px;">💡 SALES TIP</div>
                        <div style="font-size: 13px; color: #0369a1; font-style: italic;">{item['notes']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Quick Stock Check Tool
        st.markdown("#### 📦 Quick Stock Check")
        
        stock_items = [
            {"item": "Black Business Pants (Size 8)", "location": "B3-22", "stock": "3 left", "status": "low"},
            {"item": "White Button-Down Shirts (M)", "location": "A2-18", "stock": "12 available", "status": "good"},
            {"item": "Designer Handbags (Black)", "location": "C1-05", "stock": "Out of stock", "status": "out"},
            {"item": "Casual Sweaters (Navy, L)", "location": "B1-14", "stock": "8 available", "status": "good"},
        ]
        
        for item in stock_items:
            if item["status"] == "out":
                border_color = "#ef4444"
                bg_color = "#fee2e2"
                status_badge = "❌ OUT"
                badge_color = "#dc2626"
            elif item["status"] == "low":
                border_color = "#f59e0b"
                bg_color = "#fef3c7"
                status_badge = "⚠️ LOW"
                badge_color = "#d97706"
            else:
                border_color = "#10b981"
                bg_color = "#ecfdf5"
                status_badge = "✅ GOOD"
                badge_color = "#059669"
            
            st.markdown(
                f"""
                <div style="border-left: 4px solid {border_color}; background: {bg_color}; padding: 12px; border-radius: 8px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <div style="font-weight: 600; color: #1e293b; margin-bottom: 4px;">{item['item']}</div>
                            <div style="font-size: 14px; color: #64748b;">📍 {item['location']} • Stock: {item['stock']}</div>
                        </div>
                        <div style="background: {badge_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 11px; font-weight: 700;">
                            {status_badge}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_side:
        # Quick Actions Panel
        st.markdown("#### ⚡ Quick Actions")
        
        if st.button("🏷️ Price Check", use_container_width=True):
            st.info("Open price scanner or enter SKU")
        
        if st.button("📋 Check Size Guide", use_container_width=True):
            st.info("Opening size chart reference")
        
        if st.button("🔄 Report Out of Stock", use_container_width=True):
            show_stock_report_form()
        
        if st.button("📞 Call for Stock Check", use_container_width=True):
            st.info("Calling stockroom extension 2255")
        
        if st.button("🎁 Gift Services Info", use_container_width=True):
            st.info("Gift wrapping available • Personal shopping consultations")

        st.markdown("---")
        
        # Your Restocking Tasks
        st.markdown("#### 📦 Your Restocking Tasks")
        
        restock_tasks = [
            {"item": "Silk Blouses", "location": "A2-16", "priority": "high", "time": "Due 2:00 PM"},
            {"item": "Fall Accessories", "location": "C1-08", "priority": "medium", "time": "Due 4:00 PM"},
            {"item": "Evening Wear", "location": "A3-22", "priority": "low", "time": "Due tomorrow"}
        ]
        
        for task in restock_tasks:
            if task["priority"] == "high":
                priority_color = "#dc2626"
                bg_color = "#fef2f2"
            elif task["priority"] == "medium":
                priority_color = "#d97706"
                bg_color = "#fffbeb"
            else:
                priority_color = "#059669"
                bg_color = "#f0fdf4"
            
            st.markdown(
                f"""
                <div style="background: {bg_color}; border-radius: 8px; padding: 12px; margin-bottom: 8px; border-left: 3px solid {priority_color};">
                    <div style="font-weight: 600; color: #1e293b; font-size: 14px; margin-bottom: 4px;">{task['item']}</div>
                    <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">📍 {task['location']}</div>
                    <div style="font-size: 11px; color: {priority_color}; font-weight: 600; text-transform: uppercase;">{task['time']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        
        # Customer Service Quick Reference
        st.markdown("#### 📞 Quick Reference")
        
        quick_refs = [
            {"title": "Return Policy", "info": "30 days with receipt"},
            {"title": "Price Match", "info": "Online competitors only"},
            {"title": "Alterations", "info": "Available Mon-Fri"},
            {"title": "Personal Shopping", "info": "Book 24hrs ahead"},
            {"title": "Gift Cards", "info": "No expiration date"}
        ]
        
        for ref in quick_refs:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #e5e7eb;">
                    <span style="font-size: 12px; font-weight: 600; color: #1e293b;">{ref['title']}</span>
                    <span style="font-size: 12px; color: #64748b;">{ref['info']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        
        # Daily Goals
        st.markdown("#### 🎯 Today's Goals")
        
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px; border-radius: 12px; text-align: center; margin-bottom: 16px;">
                <div style="font-size: 18px; font-weight: 700; margin-bottom: 8px;">Customer Service</div>
                <div style="font-size: 14px; opacity: 0.9;">Help 15+ customers today</div>
                <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">Progress: 8/15 ✨</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def show_stock_report_form():
    """Show stock reporting form for associates."""
    with st.form("associate_stock_report"):
        st.markdown("**🔄 Report Stock Issue**")
        
        issue_type = st.selectbox("Issue Type", 
            ["Out of Stock", "Low Stock (< 5 items)", "Damaged Items", "Misplaced Items", "Price Tag Missing"])
        
        item_location = st.text_input("Item Location", placeholder="e.g. A2-15, Rack 3")
        
        item_description = st.text_area("Item Description", placeholder="Be specific: brand, color, size, etc.")
        
        urgency = st.radio("Priority Level", 
            ["🔴 Urgent - Customer waiting", 
             "🟡 Medium - Needed today", 
             "🟢 Low - Can wait until tomorrow"])
        
        if st.form_submit_button("📤 Submit Report", type="primary"):
            st.success("✅ Stock report sent to inventory team!")
            st.info("📱 You'll receive an update within 30 minutes")
            st.balloons()
