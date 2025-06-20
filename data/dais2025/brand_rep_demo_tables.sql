-- Brand Rep Product Education Demo Tables
-- These tables support the Nike Air Max brand training demo scenarios
USE IDENTIFIER(:database);
-- Customer Demographics and Brand Preferences
CREATE TABLE IF NOT EXISTS customer_brand_profiles (
    customer_id STRING,
    age_group STRING,
    lifestyle_category STRING,
    brand_preference STRING,
    price_sensitivity STRING,
    shopping_pattern STRING,
    fitness_level STRING,
    purchase_frequency STRING,
    avg_spend_per_visit DECIMAL(10,2),
    preferred_shopping_times STRING,
    store_id INT,
    last_purchase_date DATE,
    total_lifetime_value DECIMAL(12,2)
);

-- Product Performance Analytics
CREATE TABLE IF NOT EXISTS product_performance (
    product_id BIGINT,
    sku STRING,
    product_name STRING,
    brand STRING,
    category STRING,
    subcategory STRING,
    price DECIMAL(10,2),
    units_sold_6m INT,
    units_sold_3m INT,
    units_sold_1m INT,
    return_rate DECIMAL(5,2),
    avg_rating DECIMAL(3,2),
    review_count INT,
    store_id INT,
    seasonal_trend STRING,
    inventory_turnover DECIMAL(5,2),
    profit_margin DECIMAL(5,2)
);

-- Customer Feedback and Reviews
CREATE TABLE IF NOT EXISTS customer_feedback (
    feedback_id BIGINT,
    customer_id STRING,
    product_id BIGINT,
    sku STRING,
    rating INT,
    review_text STRING,
    feedback_category STRING,
    sentiment STRING,
    purchase_date DATE,
    review_date DATE,
    store_id INT,
    verified_purchase BOOLEAN,
    helpful_votes INT
);

-- Competitive Analysis Data
CREATE TABLE IF NOT EXISTS competitive_insights (
    insight_id BIGINT,
    customer_id STRING,
    primary_brand STRING,
    alternative_brand STRING,
    decision_factor STRING,
    price_comparison DECIMAL(10,2),
    feature_comparison STRING,
    customer_comment STRING,
    purchase_decision STRING,
    interaction_date DATE,
    store_id INT,
    associate_id STRING
);

-- Sales Conversations and Objections
CREATE TABLE IF NOT EXISTS sales_interactions (
    interaction_id BIGINT,
    customer_id STRING,
    associate_id STRING,
    product_discussed STRING,
    brand STRING,
    customer_objection STRING,
    objection_category STRING,
    resolution_strategy STRING,
    outcome STRING,
    interaction_date TIMESTAMP,
    store_id INT,
    duration_minutes INT,
    follow_up_needed BOOLEAN
); 