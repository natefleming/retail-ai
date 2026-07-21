-- Customer Appointments Table
-- Supports personal styling appointments and customer service scheduling
USE IDENTIFIER(:database);
CREATE TABLE IF NOT EXISTS appointments (
    appointment_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    customer_name VARCHAR(100) NOT NULL,
    appointment_type VARCHAR(50) NOT NULL,
    appointment_date DATE NOT NULL,
    appointment_time TIMESTAMP NOT NULL,
    duration_minutes INTEGER NOT NULL,
    store_id VARCHAR(20) NOT NULL,
    stylist_id VARCHAR(20),
    stylist_name VARCHAR(100),
    status VARCHAR(20) NOT NULL,
    notes STRING,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
); 