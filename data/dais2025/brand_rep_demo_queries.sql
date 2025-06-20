-- These queries demonstrate how Genie would answer the specific questions from the demo
USE IDENTIFIER(:database);
-- Demo Question 1: "Nike brand representative is coming to train us on Air Max SC. What should I know about our customers who buy Nike?"
-- Expected Response: Downtown Market Nike Customer Profile

-- Query 1a: Nike Customer Demographics
SELECT 
    age_group,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customer_brand_profiles WHERE brand_preference = 'Nike' AND store_id = 101), 1) as percentage
FROM customer_brand_profiles 
WHERE brand_preference = 'Nike' AND store_id = 101
GROUP BY age_group
ORDER BY customer_count DESC;

-- Query 1b: Nike Customer Shopping Patterns
SELECT 
    preferred_shopping_times,
    COUNT(*) as customers,
    ROUND(AVG(avg_spend_per_visit), 2) as avg_spend,
    price_sensitivity,
    COUNT(*) as count_by_sensitivity
FROM customer_brand_profiles 
WHERE brand_preference = 'Nike' AND store_id = 101
GROUP BY preferred_shopping_times, price_sensitivity
ORDER BY customers DESC;

-- Query 1c: Nike Customer Price Sensitivity Analysis
SELECT 
    price_sensitivity,
    COUNT(*) as customer_count,
    ROUND(AVG(avg_spend_per_visit), 2) as avg_spend_per_visit,
    ROUND(AVG(total_lifetime_value), 2) as avg_lifetime_value
FROM customer_brand_profiles 
WHERE brand_preference = 'Nike' AND store_id = 101
GROUP BY price_sensitivity;

-- Demo Question 2: "Show me how Nike Air Max products perform at our store."
-- Expected Response: Air Max Performance (Last 6 months)

-- Query 2a: Air Max Product Performance Summary
SELECT 
    product_name,
    sku,
    price,
    units_sold_6m,
    units_sold_3m,
    units_sold_1m,
    return_rate,
    avg_rating,
    seasonal_trend
FROM product_performance 
WHERE brand = 'Nike' AND product_name LIKE '%Air Max%' AND store_id = 101
ORDER BY units_sold_6m DESC;

-- Query 2b: Customer Feedback Themes for Air Max
SELECT 
    p.product_name,
    cf.feedback_category,
    COUNT(*) as feedback_count,
    ROUND(AVG(cf.rating), 1) as avg_rating,
    cf.sentiment
FROM customer_feedback cf
JOIN product_performance p ON cf.product_id = p.product_id
WHERE p.brand = 'Nike' AND p.product_name LIKE '%Air Max%' AND cf.store_id = 101
GROUP BY p.product_name, cf.feedback_category, cf.sentiment
ORDER BY p.product_name, feedback_count DESC;

-- Query 2c: Seasonal Trends and Return Analysis
SELECT 
    product_name,
    seasonal_trend,
    return_rate,
    CASE 
        WHEN return_rate > 10 THEN 'High Return Rate'
        WHEN return_rate > 5 THEN 'Moderate Return Rate'
        ELSE 'Low Return Rate'
    END as return_category,
    units_sold_6m
FROM product_performance 
WHERE brand = 'Nike' AND product_name LIKE '%Air Max%' AND store_id = 101;

-- Demo Question 3: "What do customers say when they choose Adidas over Nike?"
-- Expected Response: Competitive Insights

-- Query 3a: Nike vs Adidas Decision Factors
SELECT 
    decision_factor,
    COUNT(*) as frequency,
    primary_brand,
    alternative_brand,
    purchase_decision
FROM competitive_insights 
WHERE (primary_brand = 'Nike' AND alternative_brand = 'Adidas') 
   OR (primary_brand = 'Adidas' AND alternative_brand = 'Nike')
   AND store_id = 101
GROUP BY decision_factor, primary_brand, alternative_brand, purchase_decision
ORDER BY frequency DESC;

-- Query 3b: Common Customer Objections to Nike
SELECT 
    objection_category,
    customer_objection,
    COUNT(*) as frequency,
    outcome,
    ROUND(AVG(duration_minutes), 1) as avg_interaction_time
FROM sales_interactions 
WHERE brand = 'Nike' AND store_id = 101
GROUP BY objection_category, customer_objection, outcome
ORDER BY frequency DESC;

-- Query 3c: Price Comparison Analysis
SELECT 
    ci.decision_factor,
    ci.customer_comment,
    COUNT(*) as comment_frequency,
    ROUND(AVG(ci.price_comparison), 2) as avg_price_difference
FROM competitive_insights ci
WHERE ci.decision_factor = 'Price' 
  AND ((ci.primary_brand = 'Nike' AND ci.alternative_brand = 'Adidas') 
       OR (ci.primary_brand = 'Adidas' AND ci.alternative_brand = 'Nike'))
  AND ci.store_id = 101
GROUP BY ci.decision_factor, ci.customer_comment;

-- Demo Question 4: "How does Air Max SC cushioning compare to our top-selling Air Max 90?"
-- Expected Response: Quick Comparison for positioning

-- Query 4a: Air Max SC vs Air Max 90 Comparison
SELECT 
    product_name,
    sku,
    price,
    units_sold_6m,
    return_rate,
    avg_rating,
    review_count,
    profit_margin
FROM product_performance 
WHERE sku IN ('NIKE-AMSC-001', 'NIKE-AM90-001') AND store_id = 101;

-- Query 4b: Customer Feedback Comparison
SELECT 
    p.product_name,
    cf.feedback_category,
    COUNT(*) as feedback_count,
    ARRAY_JOIN(COLLECT_SET(cf.review_text), '^ ') as sample_reviews
FROM customer_feedback cf
JOIN product_performance p ON cf.product_id = p.product_id
WHERE p.sku IN ('NIKE-AMSC-001', 'NIKE-AM90-001') AND cf.store_id = 101
GROUP BY p.product_name, cf.feedback_category
ORDER BY p.product_name, feedback_count DESC;

-- Query 4c: Price Point Positioning Analysis
SELECT 
    'Air Max 90' as premium_option,
    'Air Max SC' as value_option,
    (SELECT price FROM product_performance WHERE sku = 'NIKE-AM90-001') as premium_price,
    (SELECT price FROM product_performance WHERE sku = 'NIKE-AMSC-001') as value_price,
    (SELECT price FROM product_performance WHERE sku = 'NIKE-AM90-001') - 
    (SELECT price FROM product_performance WHERE sku = 'NIKE-AMSC-001') as price_difference,
    ROUND(((SELECT price FROM product_performance WHERE sku = 'NIKE-AMSC-001') * 100.0 / 
           (SELECT price FROM product_performance WHERE sku = 'NIKE-AM90-001')), 1) as value_percentage;

-- Additional Analytics for Brand Rep Training

-- Query 5: Cross-sell Opportunities with Nike Products
SELECT 
    si.product_discussed,
    si.outcome,
    COUNT(*) as interaction_count,
    ROUND(AVG(si.duration_minutes), 1) as avg_interaction_time,
    ARRAY_JOIN(COLLECT_SET(DISTINCT si.resolution_strategy), '^ ') as successful_strategies
FROM sales_interactions si
WHERE si.brand = 'Nike' AND si.store_id = 101
GROUP BY si.product_discussed, si.outcome
ORDER BY interaction_count DESC;

-- Query 6: Best Times to Promote Nike Products
SELECT 
    cbp.preferred_shopping_times,
    COUNT(*) as nike_customers,
    ROUND(AVG(cbp.avg_spend_per_visit), 2) as avg_spend,
    cbp.price_sensitivity,
    COUNT(*) as customers_by_sensitivity
FROM customer_brand_profiles cbp
WHERE cbp.brand_preference = 'Nike' AND cbp.store_id = 101
GROUP BY cbp.preferred_shopping_times, cbp.price_sensitivity
ORDER BY nike_customers DESC; 