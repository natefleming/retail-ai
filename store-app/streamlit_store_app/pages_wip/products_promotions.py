"""Products & Promotions page for store associates."""

import streamlit as st
import streamlit_modal as modal

from components.chat import show_chat_container
from components.styles import load_css


def main():
    """Main products and promotions page."""
    # Load CSS
    load_css()

    # Page header
    col1, col2 = st.columns([8, 2])
    with col1:
        st.title("🏷️ Products & Promotions")
        st.markdown("**Stay updated on current promotions and featured products**")

    with col2:
        if st.button("🤖 AI Assistant", use_container_width=True):
            st.session_state.show_chat = True

    # Create the chat modal
    chat_modal = modal.Modal("AI Assistant", key="products_chat_modal", max_width=800)

    # Handle chat modal
    if st.session_state.get("show_chat", False):
        chat_modal.open()
        st.session_state.show_chat = False

    # Modal content
    if chat_modal.is_open():
        with chat_modal.container():
            # Get chat config with fallback
            chat_config = st.session_state.get("config", {}).get(
                "chat",
                {
                    "placeholder": "How can I help you with products and promotions?",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            )

            # Show the chat container
            show_chat_container(chat_config)

    # Add custom CSS for better tab styling (same as other pages)
    st.markdown(
        """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px !important;
        padding: 12px 24px !important;
        background-color: #f8f9fa !important;
        border-radius: 8px 8px 0px 0px !important;
        border: 1px solid #dee2e6 !important;
        border-bottom: none !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #495057 !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 22px !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef !important;
        color: #212529 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
    }
    .stTabs [aria-selected="true"] p {
        color: white !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Main content in tabs - fully tab-based experience
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Current Promotions", "New Arrivals", "Trending", "Clearance"]
    )

    with tab1:
        show_current_promotions()

    with tab2:
        show_new_arrivals()

    with tab3:
        show_trending_items()

    with tab4:
        show_clearance_items()


def show_current_promotions():
    """Display current active promotions."""
    # Quick stats at top of tab
    st.markdown("#### 📊 Promotions Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="promo-stat-card active">
                <div class="stat-icon">🔥</div>
                <div class="stat-value">5</div>
                <div class="stat-label">Active Promotions</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="promo-stat-card new">
                <div class="stat-icon">🆕</div>
                <div class="stat-value">24</div>
                <div class="stat-label">New Arrivals</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="promo-stat-card trending">
                <div class="stat-icon">📈</div>
                <div class="stat-value">12</div>
                <div class="stat-label">Trending Items</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="promo-stat-card clearance">
                <div class="stat-icon">💸</div>
                <div class="stat-value">8</div>
                <div class="stat-label">Clearance Items</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("### 🔥 Active Promotions")

    promotions = [
        {
            "name": "Fall Fashion Sale",
            "discount": "40% off",
            "category": "Women's Apparel",
            "items": ["Fall Jackets", "Sweaters", "Boots"],
            "ends": "End of week",
            "code": "FALL40",
            "priority": "high",
            "description": "Seasonal clearance on fall collection",
        },
        {
            "name": "Tech Accessories Bundle",
            "discount": "Buy 2 Get 1 Free",
            "category": "Electronics",
            "items": ["iPhone Cases", "Wireless Chargers", "Screen Protectors"],
            "ends": "3 days",
            "code": "TECH3FOR2",
            "priority": "medium",
            "description": "Perfect for gift giving season",
        },
        {
            "name": "Designer Handbag Event",
            "discount": "25% off",
            "category": "Accessories",
            "items": ["Designer Handbags", "Wallets", "Clutches"],
            "ends": "Tomorrow",
            "code": "LUXURY25",
            "priority": "high",
            "description": "Exclusive designer collection sale",
        },
        {
            "name": "Men's Formal Wear",
            "discount": "30% off",
            "category": "Men's Apparel",
            "items": ["Suits", "Dress Shirts", "Ties"],
            "ends": "5 days",
            "code": "FORMAL30",
            "priority": "medium",
            "description": "Professional wardrobe essentials",
        },
        {
            "name": "Holiday Jewelry Sale",
            "discount": "50% off",
            "category": "Jewelry",
            "items": ["Necklaces", "Earrings", "Bracelets"],
            "ends": "2 days",
            "code": "SPARKLE50",
            "priority": "high",
            "description": "Perfect for holiday gifting",
        },
    ]

    for promo in promotions:
        show_promotion_card(promo)


def show_new_arrivals():
    """Display new arrival products."""
    st.markdown("### 🆕 New Arrivals")

    new_items = [
        {
            "name": "iPhone 15 Pro Cases",
            "category": "Electronics",
            "price": "$49.99",
            "location": "Electronics - E3",
            "stock": "High",
            "features": ["MagSafe Compatible", "Drop Protection", "Multiple Colors"],
            "arrival_date": "Yesterday",
        },
        {
            "name": "Winter Collection Coats",
            "category": "Women's Apparel",
            "price": "$199.99 - $399.99",
            "location": "Women's Fashion - W2",
            "stock": "Medium",
            "features": ["Waterproof", "Insulated", "Designer Styles"],
            "arrival_date": "2 days ago",
        },
        {
            "name": "Smart Fitness Watches",
            "category": "Electronics",
            "price": "$299.99",
            "location": "Electronics - E1",
            "stock": "Low",
            "features": ["Heart Rate Monitor", "GPS", "Water Resistant"],
            "arrival_date": "3 days ago",
        },
        {
            "name": "Designer Sneakers",
            "category": "Footwear",
            "price": "$159.99",
            "location": "Footwear - F4",
            "stock": "High",
            "features": ["Limited Edition", "Comfort Sole", "Premium Materials"],
            "arrival_date": "1 week ago",
        },
    ]

    for item in new_items:
        show_product_card(item, "new")


def show_trending_items():
    """Display trending products."""
    st.markdown("### 📈 Trending Items")

    trending_items = [
        {
            "name": "Wireless Earbuds Pro",
            "category": "Electronics",
            "price": "$179.99",
            "location": "Electronics - E2",
            "stock": "Medium",
            "trend_reason": "High customer demand",
            "sales_increase": "+45%",
        },
        {
            "name": "Oversized Blazers",
            "category": "Women's Apparel",
            "price": "$89.99",
            "location": "Women's Fashion - W1",
            "stock": "Low",
            "trend_reason": "Social media influence",
            "sales_increase": "+60%",
        },
        {
            "name": "Minimalist Watches",
            "category": "Accessories",
            "price": "$129.99",
            "location": "Accessories - A2",
            "stock": "High",
            "trend_reason": "Professional style trend",
            "sales_increase": "+35%",
        },
    ]

    for item in trending_items:
        show_product_card(item, "trending")


def show_clearance_items():
    """Display clearance products."""
    st.markdown("### 💸 Clearance Items")

    clearance_items = [
        {
            "name": "Summer Dresses",
            "category": "Women's Apparel",
            "original_price": "$79.99",
            "sale_price": "$29.99",
            "location": "Women's Fashion - W3",
            "stock": "Low",
            "discount": "62% off",
            "reason": "End of season",
        },
        {
            "name": "Last Gen Tablets",
            "category": "Electronics",
            "original_price": "$399.99",
            "sale_price": "$199.99",
            "location": "Electronics - E4",
            "stock": "Medium",
            "discount": "50% off",
            "reason": "New model released",
        },
        {
            "name": "Sandals Collection",
            "category": "Footwear",
            "original_price": "$59.99",
            "sale_price": "$19.99",
            "location": "Footwear - F2",
            "stock": "High",
            "discount": "67% off",
            "reason": "Seasonal clearance",
        },
    ]

    for item in clearance_items:
        show_product_card(item, "clearance")


def show_promotion_card(promo):
    """Display a promotion card."""

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            f"""
            <div class="promotion-detail-card">
                <div class="promo-header">
                    <span class="promo-name">{promo["name"]}</span>
                    <span class="promo-discount">{promo["discount"]}</span>
                </div>
                <div class="promo-details">
                    <div><strong>Category:</strong> {promo["category"]}</div>
                    <div><strong>Items:</strong> {", ".join(promo["items"])}</div>
                    <div><strong>Code:</strong> {promo["code"]}</div>
                    <div><strong>Ends:</strong> {promo["ends"]}</div>
                    <div><strong>Description:</strong> {promo["description"]}</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        if st.button(
            "Share Details", key=f"share_{promo['code']}", use_container_width=True
        ):
            st.success("Promotion details copied to share with customers!")

    st.markdown("---")


def show_product_card(item, card_type):
    """Display a product card."""
    col1, col2 = st.columns([3, 1])

    with col1:
        if card_type == "clearance":
            st.markdown(
                f"""
                <div class="product-detail-card {card_type}">
                    <div class="product-header">
                        <span class="product-name">{item["name"]}</span>
                        <span class="product-discount">{item["discount"]}</span>
                    </div>
                    <div class="product-pricing">
                        <span class="original-price">${item["original_price"]}</span>
                        <span class="sale-price">${item["sale_price"]}</span>
                    </div>
                    <div class="product-details">
                        <div><strong>Category:</strong> {item["category"]}</div>
                        <div><strong>Location:</strong> {item["location"]}</div>
                        <div><strong>Stock:</strong> {item["stock"]}</div>
                        <div><strong>Reason:</strong> {item["reason"]}</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )
        elif card_type == "trending":
            st.markdown(
                f"""
                <div class="product-detail-card {card_type}">
                    <div class="product-header">
                        <span class="product-name">{item["name"]}</span>
                        <span class="trend-indicator">{item["sales_increase"]}</span>
                    </div>
                    <div class="product-price">{item["price"]}</div>
                    <div class="product-details">
                        <div><strong>Category:</strong> {item["category"]}</div>
                        <div><strong>Location:</strong> {item["location"]}</div>
                        <div><strong>Stock:</strong> {item["stock"]}</div>
                        <div><strong>Trending:</strong> {item["trend_reason"]}</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:  # new arrivals
            st.markdown(
                f"""
                <div class="product-detail-card {card_type}">
                    <div class="product-header">
                        <span class="product-name">{item["name"]}</span>
                        <span class="arrival-date">{item["arrival_date"]}</span>
                    </div>
                    <div class="product-price">{item["price"]}</div>
                    <div class="product-details">
                        <div><strong>Category:</strong> {item["category"]}</div>
                        <div><strong>Location:</strong> {item["location"]}</div>
                        <div><strong>Stock:</strong> {item["stock"]}</div>
                        <div><strong>Features:</strong> {", ".join(item["features"])}</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        if st.button(
            "View Details", key=f"view_{item['name']}", use_container_width=True
        ):
            st.info("Product details would open in inventory system")

    st.markdown("---")


# Add custom CSS for products and promotions components
st.markdown(
    """
    <style>
    /* Global font improvements */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Enhanced promotion stat cards - Clean styling without colored borders */
    .promo-stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .promo-stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .promo-stat-card .stat-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .promo-stat-card .stat-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        display: block;
        line-height: 1.2;
    }
    
    .promo-stat-card .stat-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced promotion detail cards - Clean styling without colored borders */
    .promotion-detail-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .promotion-detail-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    }
    
    .promo-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .promo-name {
        font-weight: 700;
        font-size: 1.25rem;
        color: #1e293b;
        line-height: 1.3;
    }
    
    .promo-discount {
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
    }
    
    .promo-details {
        color: #475569;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .promo-details div {
        margin-bottom: 0.5rem;
        padding: 0.25rem 0;
    }
    
    .promo-details strong {
        color: #334155;
        font-weight: 600;
    }
    
    /* Enhanced product detail cards - Clean styling without colored borders */
    .product-detail-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .product-detail-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    }
    
    .product-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .product-name {
        font-weight: 700;
        font-size: 1.25rem;
        color: #1e293b;
        line-height: 1.3;
    }
    
    .product-discount, .trend-indicator, .arrival-date {
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .product-discount {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    
    .trend-indicator {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .arrival-date {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }
    
    .product-pricing {
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .original-price {
        text-decoration: line-through;
        color: #64748b;
        font-size: 1rem;
    }
    
    .sale-price, .product-price {
        font-weight: 700;
        font-size: 1.25rem;
        color: #1e293b;
    }
    
    .product-details {
        color: #475569;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .product-details div {
        margin-bottom: 0.5rem;
        padding: 0.25rem 0;
    }
    
    .product-details strong {
        color: #334155;
        font-weight: 600;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Page title styling */
    h1 {
        color: #1e293b;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h3 {
        color: #334155;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h4 {
        color: #475569;
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
    }
    
    /* Enhanced markdown text */
    .stMarkdown p {
        font-size: 1rem;
        line-height: 1.6;
        color: #64748b;
    }
    
    /* Success/info message styling */
    .stSuccess, .stInfo {
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 500;
    }
    </style>
""",
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    main()
