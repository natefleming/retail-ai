-- Sample appointments including Victoria Sterling's styling session
USE IDENTIFIER(:database);
INSERT INTO appointments (
    appointment_id, customer_id, customer_name, appointment_type, 
    appointment_date, appointment_time, duration_minutes, store_id, 
    stylist_id, stylist_name, status, notes
) VALUES
-- Victoria Sterling's styling appointment (from demo script)
('APPT-001', 'CUST-005', 'Victoria Sterling', 'Personal Styling', 
 '2024-03-20', '14:00:00', 90, 'STORE-101', 
 'STYLIST-001', 'Maria Rodriguez', 'Confirmed', 
 'Customer interested in sneaker styling. Prefers modern, versatile styles for work and casual wear.'),

-- Additional sample appointments
('APPT-002', 'CUST-001', 'John Smith', 'Personal Styling', 
 '2024-03-21', '10:00:00', 60, 'STORE-101', 
 'STYLIST-002', 'Sarah Johnson', 'Confirmed', 
 'Customer looking for business casual wardrobe refresh.'),

('APPT-003', 'CUST-003', 'Emily Davis', 'Personal Shopping', 
 '2024-03-22', '15:30:00', 120, 'STORE-102', 
 'STYLIST-001', 'Maria Rodriguez', 'Pending', 
 'Special occasion outfit needed for wedding guest.'),

('APPT-004', 'CUST-007', 'Michael Brown', 'Style Consultation', 
 '2024-03-23', '11:00:00', 45, 'STORE-103', 
 'STYLIST-003', 'Alex Chen', 'Confirmed', 
 'Athletic wear styling for fitness enthusiast.'),

('APPT-005', 'CUST-002', 'Jane Johnson', 'Personal Styling', 
 '2024-03-24', '13:00:00', 90, 'STORE-101', 
 'STYLIST-002', 'Sarah Johnson', 'Confirmed', 
 'Seasonal wardrobe update, focus on spring trends.'); 