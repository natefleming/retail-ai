USE IDENTIFIER(:database);

-- Customers table for storing customer information
-- Supports manager preparation for customer visits and personal styling appointments
CREATE TABLE IF NOT EXISTS customers (
    -- Customer identification
    customer_id STRING COMMENT 'Unique identifier for the customer',
    customer_name STRING COMMENT 'Full name of the customer',
    preferred_name STRING COMMENT 'Preferred name or nickname for personalized service',
    customer_tier STRING COMMENT 'Customer tier (Premium, Gold, Silver, Standard, etc.)',
    member_since DATE COMMENT 'Date when customer became a member',
    
    -- Contact information
    email_address STRING COMMENT 'Primary email address',
    phone_number STRING COMMENT 'Primary phone number',
    preferred_contact_method STRING COMMENT 'Preferred communication method (email, phone, text)',
    
    -- Personal styling preferences
    preferred_store_id STRING COMMENT 'Preferred store location for shopping',
    preferred_stylist_id STRING COMMENT 'Preferred personal stylist',
    preferred_appointment_time STRING COMMENT 'Preferred time slots for appointments',
    style_preferences STRING COMMENT 'Fashion style preferences and styling notes',
    size_information STRING COMMENT 'Clothing sizes and fit preferences (JSON format)',
    color_preferences STRING COMMENT 'Preferred colors and color dislikes',
    brand_preferences STRING COMMENT 'Preferred and avoided brands',
    budget_range STRING COMMENT 'Typical spending range per visit',
    
    -- Shopping history and behavior
    total_lifetime_spend DOUBLE COMMENT 'Total amount spent as customer',
    average_transaction_value DOUBLE COMMENT 'Average transaction amount',
    last_visit_date DATE COMMENT 'Date of last store visit',
    last_purchase_date DATE COMMENT 'Date of last purchase',
    visit_frequency STRING COMMENT 'How often customer visits (weekly, monthly, seasonal)',
    seasonal_preferences STRING COMMENT 'Seasonal shopping patterns and preferences',
    
    -- Special requirements and notes
    special_occasions STRING COMMENT 'Important dates (birthday, anniversary, etc.) in JSON format',
    dietary_restrictions STRING COMMENT 'Any dietary restrictions for refreshments',
    accessibility_needs STRING COMMENT 'Any accessibility requirements',
    language_preference STRING COMMENT 'Preferred language for service',
    cultural_considerations STRING COMMENT 'Cultural preferences or considerations',
    
    -- Service history
    styling_sessions INT COMMENT 'Number of personal styling sessions completed',
    satisfaction_score DOUBLE COMMENT 'Average satisfaction score from feedback (1.0-5.0)',
    last_feedback STRING COMMENT 'Most recent customer feedback or comments',
    service_notes STRING COMMENT 'Important service notes and preferences',
    
    -- Upcoming appointments
    next_appointment_date TIMESTAMP COMMENT 'Date and time of next scheduled appointment',
    appointment_type STRING COMMENT 'Type of upcoming appointment (personal styling, wardrobe consultation, etc.)',
    appointment_purpose STRING COMMENT 'Purpose or occasion for upcoming appointment',
    preparation_notes STRING COMMENT 'Special preparation notes for upcoming visit',
    
    -- Relationship and family
    family_members STRING COMMENT 'Family member information for gift suggestions (JSON format)',
    gift_history STRING COMMENT 'Previous gifts purchased and occasions (JSON format)',
    referral_source STRING COMMENT 'How customer was referred to the store',
    
    -- Status and flags
    customer_status STRING COMMENT 'Current status (active, inactive, on_hold)',
    priority_level STRING COMMENT 'Service priority level (high, medium, standard)',
    requires_manager_greeting BOOLEAN COMMENT 'Whether manager should personally greet customer',
    customer_alerts STRING COMMENT 'Special alerts or flags for staff attention',
    
    -- Metadata
    created_at TIMESTAMP COMMENT 'Timestamp when record was created',
    updated_at TIMESTAMP COMMENT 'Timestamp when record was last updated',
    created_by STRING COMMENT 'System or user who created the record',
    updated_by STRING COMMENT 'System or user who last updated the record'
)
USING DELTA
COMMENT 'Customer information table for personalized service and appointment preparation'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
);

-- Create view for upcoming customer appointments
CREATE OR REPLACE VIEW upcoming_customer_appointments AS
SELECT 
    c.customer_id,
    c.customer_name,
    c.preferred_name,
    c.customer_tier,
    c.preferred_store_id,
    s.store_name,
    c.preferred_stylist_id,
    e.employee_name as preferred_stylist_name,
    c.next_appointment_date,
    c.appointment_type,
    c.appointment_purpose,
    c.style_preferences,
    c.budget_range,
    c.preparation_notes,
    c.special_occasions,
    c.service_notes,
    c.requires_manager_greeting,
    c.customer_alerts,
    c.satisfaction_score,
    c.total_lifetime_spend,
    c.last_visit_date
FROM customers c
LEFT JOIN dim_stores s ON c.preferred_store_id = s.store_id
LEFT JOIN employee_performance e ON c.preferred_stylist_id = e.employee_id
WHERE c.next_appointment_date IS NOT NULL
    AND c.next_appointment_date >= CURRENT_TIMESTAMP()
    AND c.customer_status = 'active'
ORDER BY c.next_appointment_date ASC;

-- Create view for customer preparation summary
CREATE OR REPLACE VIEW customer_preparation_summary AS
SELECT 
    c.customer_id,
    c.customer_name,
    c.preferred_name,
    c.customer_tier,
    c.preferred_store_id,
    s.store_name,
    c.next_appointment_date,
    c.appointment_type,
    c.appointment_purpose,
    c.style_preferences,
    c.size_information,
    c.color_preferences,
    c.brand_preferences,
    c.budget_range,
    c.preparation_notes,
    c.service_notes,
    c.special_occasions,
    c.dietary_restrictions,
    c.accessibility_needs,
    c.requires_manager_greeting,
    c.customer_alerts,
    c.preferred_stylist_id,
    e.employee_name as preferred_stylist_name,
    e.personal_shopping_sessions as stylist_experience,
    e.customer_satisfaction_score as stylist_rating,
    c.satisfaction_score as customer_satisfaction,
    c.total_lifetime_spend,
    c.average_transaction_value,
    c.last_visit_date,
    c.visit_frequency,
    -- Calculate days since last visit
    DATEDIFF(CURRENT_DATE(), c.last_visit_date) as days_since_last_visit,
    -- Calculate hours until appointment
    ROUND((UNIX_TIMESTAMP(c.next_appointment_date) - UNIX_TIMESTAMP(CURRENT_TIMESTAMP())) / 3600, 1) as hours_until_appointment
FROM customers c
LEFT JOIN dim_stores s ON c.preferred_store_id = s.store_id
LEFT JOIN employee_performance e ON c.preferred_stylist_id = e.employee_id
WHERE c.customer_status = 'active'
ORDER BY c.next_appointment_date ASC; 