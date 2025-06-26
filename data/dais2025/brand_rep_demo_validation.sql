-- Brand Rep Demo Data Validation
-- These queries validate that our data supports the expected demo responses

-- Validation 1: Nike Customer Profile (Demo Question 1)
-- Expected: Ages 18-35 (68%), fitness enthusiasts and students, peak sales weekends/after 5PM
SELECT 
    'Nike Customer Demographics' as validation_check,
    COUNT(*) as total_nike_customers,
    SUM(CASE WHEN age_group IN ('18-25', '26-35') THEN 1 ELSE 0 END) as age_18_35_count,
    ROUND(SUM(CASE WHEN age_group IN ('18-25', '26-35') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as age_18_35_percentage,
    SUM(CASE WHEN lifestyle_category IN ('Student', 'Fitness Enthusiast') THEN 1 ELSE 0 END) as student_fitness_count,
    SUM(CASE WHEN preferred_shopping_times LIKE '%Weekend%' OR preferred_shopping_times LIKE '%After 5PM%' THEN 1 ELSE 0 END) as peak_time_shoppers,
    ROUND(AVG(avg_spend_per_visit), 2) as avg_spend
FROM customer_brand_profiles 
WHERE brand_preference = 'Nike' AND store_id = 101;

-- Validation 2: Air Max Performance (Demo Question 2)  
-- Expected: Air Max 90 (127 units), Air Max 270 (89 units), Air Max Plus (12 units), 8% return rate
SELECT 
    'Air Max Performance' as validation_check,
    product_name,
    units_sold_6m,
    return_rate,
    CASE 
        WHEN product_name = 'Nike Air Max 90' AND units_sold_6m = 127 THEN 'PASS'
        WHEN product_name = 'Nike Air Max 270' AND units_sold_6m = 89 THEN 'PASS'
        WHEN product_name = 'Nike Air Max Plus' AND units_sold_6m = 12 THEN 'PASS'
        ELSE 'CHECK'
    END as validation_status
FROM product_performance 
WHERE brand = 'Nike' AND product_name LIKE '%Air Max%' AND store_id = 101
ORDER BY units_sold_6m DESC;

-- Validation 3: Competitive Insights (Demo Question 3)
-- Expected: Price promotions (35% of switches), wider width options, brand loyalty
SELECT 
    'Competitive Analysis' as validation_check,
    decision_factor,
    COUNT(*) as frequency,
    STRING_AGG(customer_comment, '; ') as sample_comments
FROM competitive_insights 
WHERE ((primary_brand = 'Nike' AND alternative_brand = 'Adidas') 
    OR (primary_brand = 'Adidas' AND alternative_brand = 'Nike'))
    AND store_id = 101
GROUP BY decision_factor
ORDER BY frequency DESC;

-- Validation 4: Air Max SC vs Air Max 90 Comparison (Demo Question 4)
-- Expected: SC $70 vs 90 $120, positioning as affordable Air Max option
SELECT 
    'Product Comparison' as validation_check,
    product_name,
    price,
    units_sold_6m,
    return_rate,
    CASE 
        WHEN product_name = 'Nike Air Max SC' AND price = 70.00 THEN 'PASS - Correct SC Price'
        WHEN product_name = 'Nike Air Max 90' AND price = 120.00 THEN 'PASS - Correct 90 Price'
        ELSE 'CHECK'
    END as validation_status
FROM product_performance 
WHERE sku IN ('NIKE-AMSC-001', 'NIKE-AM90-001') AND store_id = 101;

-- Validation 5: Customer Feedback Themes
-- Expected: 'Comfortable but runs small' recurring theme, sizing issues
SELECT 
    'Customer Feedback Themes' as validation_check,
    feedback_category,
    COUNT(*) as feedback_count,
    STRING_AGG(review_text, ' | ') as sample_feedback
FROM customer_feedback cf
JOIN product_performance p ON cf.product_id = p.product_id
WHERE p.brand = 'Nike' AND p.product_name LIKE '%Air Max%' AND cf.store_id = 101
GROUP BY feedback_category
ORDER BY feedback_count DESC;

-- Validation 6: Sales Objections and Outcomes
-- Expected: Price objections, sizing concerns, 50% success rate
SELECT 
    'Sales Interactions' as validation_check,
    objection_category,
    COUNT(*) as objection_count,
    SUM(CASE WHEN outcome = 'Sale Completed' THEN 1 ELSE 0 END) as sales_completed,
    SUM(CASE WHEN outcome = 'No Sale' THEN 1 ELSE 0 END) as no_sales,
    ROUND(SUM(CASE WHEN outcome = 'Sale Completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as success_rate_pct
FROM sales_interactions 
WHERE brand = 'Nike' AND store_id = 101
GROUP BY objection_category
ORDER BY objection_count DESC;

-- Overall Data Summary
SELECT 
    'Data Summary' as summary_check,
    (SELECT COUNT(*) FROM customer_brand_profiles WHERE store_id = 101) as total_customers,
    (SELECT COUNT(*) FROM customer_brand_profiles WHERE brand_preference = 'Nike' AND store_id = 101) as nike_customers,
    (SELECT COUNT(*) FROM customer_brand_profiles WHERE brand_preference = 'Adidas' AND store_id = 101) as adidas_customers,
    (SELECT COUNT(*) FROM product_performance WHERE brand = 'Nike' AND store_id = 101) as nike_products,
    (SELECT COUNT(*) FROM customer_feedback WHERE store_id = 101) as total_feedback,
    (SELECT COUNT(*) FROM competitive_insights WHERE store_id = 101) as competitive_records,
    (SELECT COUNT(*) FROM sales_interactions WHERE store_id = 101) as sales_interactions; 