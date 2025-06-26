-- Data to support Nike Air Max brand training demo scenarios
USE IDENTIFIER(:database);
-- Customer Brand Profiles Data
INSERT INTO customer_brand_profiles VALUES
-- Nike-focused customers (Downtown Market - Store 101)
('CUST-001', '18-25', 'Student', 'Nike', 'High', 'Weekend Shopper', 'Active', 'Monthly', 95.50, 'Weekends, After 5PM', 101, '2024-03-15', 1247.80),
('CUST-002', '26-35', 'Fitness Enthusiast', 'Nike', 'Medium', 'Regular Shopper', 'Very Active', 'Bi-weekly', 112.30, 'Weekends, Mornings', 101, '2024-03-18', 2156.90),
('CUST-003', '18-25', 'Student', 'Nike', 'High', 'Sale Hunter', 'Moderate', 'Quarterly', 67.20, 'Weekends', 101, '2024-02-28', 456.80),
('CUST-004', '26-35', 'Professional', 'Nike', 'Low', 'Convenience Shopper', 'Moderate', 'Monthly', 134.70, 'After 5PM, Weekends', 101, '2024-03-20', 1876.40),
('CUST-005', '18-25', 'Student', 'Nike', 'High', 'Weekend Shopper', 'Active', 'Monthly', 78.90, 'Weekends', 101, '2024-03-10', 623.50),
('CUST-006', '26-35', 'Fitness Enthusiast', 'Nike', 'Medium', 'Regular Shopper', 'Very Active', 'Monthly', 98.60, 'Mornings, Weekends', 101, '2024-03-22', 1543.20),
('CUST-007', '36-45', 'Parent', 'Nike', 'Medium', 'Family Shopper', 'Moderate', 'Bi-monthly', 156.80, 'Weekends', 101, '2024-03-12', 2234.60),
('CUST-008', '18-25', 'Student', 'Nike', 'High', 'Sale Hunter', 'Active', 'Quarterly', 89.40, 'Weekends, After 5PM', 101, '2024-01-15', 267.20),

-- Adidas customers for comparison
('CUST-009', '26-35', 'Professional', 'Adidas', 'High', 'Sale Hunter', 'Moderate', 'Monthly', 76.30, 'Weekends', 101, '2024-03-14', 892.40),
('CUST-010', '18-25', 'Student', 'Adidas', 'High', 'Price Conscious', 'Active', 'Bi-monthly', 65.80, 'Weekends', 101, '2024-03-08', 456.70),
('CUST-011', '26-35', 'Fitness Enthusiast', 'Adidas', 'Medium', 'Brand Loyal', 'Very Active', 'Monthly', 103.20, 'Mornings', 101, '2024-03-19', 1567.80),
('CUST-012', '36-45', 'Parent', 'Adidas', 'Medium', 'Family Shopper', 'Moderate', 'Monthly', 142.50, 'Weekends', 101, '2024-03-16', 1998.30);

-- Product Performance Data
INSERT INTO product_performance VALUES
-- Nike Air Max Products
(2001, 'NIKE-AM90-001', 'Nike Air Max 90', 'Nike', 'Footwear', 'Athletic Sneakers', 120.00, 127, 89, 34, 8.2, 4.3, 156, 101, 'Spring Peak', 6.2, 45.5),
(2002, 'NIKE-AM270-001', 'Nike Air Max 270', 'Nike', 'Footwear', 'Athletic Sneakers', 150.00, 89, 67, 28, 6.8, 4.5, 134, 101, 'Steady', 5.8, 48.2),
(2003, 'NIKE-AMSC-001', 'Nike Air Max SC', 'Nike', 'Footwear', 'Athletic Sneakers', 70.00, 45, 32, 18, 5.2, 4.1, 67, 101, 'Growing', 4.2, 42.8),
(2004, 'NIKE-AMPLUS-001', 'Nike Air Max Plus', 'Nike', 'Footwear', 'Athletic Sneakers', 170.00, 12, 8, 3, 12.5, 3.8, 23, 101, 'Declining', 2.1, 52.3),
(2005, 'NIKE-AM97-001', 'Nike Air Max 97', 'Nike', 'Footwear', 'Athletic Sneakers', 180.00, 34, 28, 12, 9.1, 4.2, 89, 101, 'Stable', 3.8, 51.7),

-- Adidas Products for comparison
(2006, 'ADI-UB22-001', 'Adidas Ultraboost 22', 'Adidas', 'Footwear', 'Athletic Sneakers', 180.00, 67, 45, 23, 7.3, 4.4, 112, 101, 'Steady', 4.9, 47.8),
(2007, 'ADI-GAZ-001', 'Adidas Gazelle', 'Adidas', 'Footwear', 'Lifestyle Sneakers', 90.00, 98, 72, 31, 6.2, 4.2, 145, 101, 'Spring Peak', 7.1, 44.2),
(2008, 'ADI-STAN-001', 'Adidas Stan Smith', 'Adidas', 'Footwear', 'Lifestyle Sneakers', 80.00, 76, 54, 22, 5.8, 4.3, 98, 101, 'Stable', 6.8, 43.5),
(2009, 'ADI-NMD-001', 'Adidas NMD R1', 'Adidas', 'Footwear', 'Athletic Sneakers', 140.00, 43, 31, 15, 8.7, 4.1, 76, 101, 'Declining', 4.2, 46.9);

-- Customer Feedback Data
INSERT INTO customer_feedback VALUES
-- Nike Air Max 90 Feedback
(1001, 'CUST-001', 2001, 'NIKE-AM90-001', 5, 'Love these! Super comfortable for my daily runs. True to size.', 'Comfort', 'Positive', '2024-02-15', '2024-02-20', 101, true, 12),
(1002, 'CUST-002', 2001, 'NIKE-AM90-001', 4, 'Great shoe but runs a bit small. Had to exchange for half size up.', 'Sizing', 'Mixed', '2024-01-28', '2024-02-05', 101, true, 8),
(1003, 'CUST-004', 2001, 'NIKE-AM90-001', 5, 'Perfect for gym and casual wear. Worth the price.', 'Versatility', 'Positive', '2024-03-01', '2024-03-08', 101, true, 15),
(1004, 'CUST-006', 2001, 'NIKE-AM90-001', 4, 'Comfortable but runs small. Size up half a size.', 'Sizing', 'Mixed', '2024-02-22', '2024-03-01', 101, true, 6),

-- Nike Air Max 270 Feedback
(1005, 'CUST-002', 2002, 'NIKE-AM270-001', 5, 'Amazing cushioning! Best running shoe I have owned.', 'Performance', 'Positive', '2024-01-15', '2024-01-22', 101, true, 18),
(1006, 'CUST-007', 2002, 'NIKE-AM270-001', 4, 'Good shoe but a bit pricey. Waiting for sales next time.', 'Price', 'Mixed', '2024-02-10', '2024-02-18', 101, true, 9),

-- Nike Air Max SC Feedback
(1007, 'CUST-003', 2003, 'NIKE-AMSC-001', 4, 'Great value! Looks like Air Max 90 but much more affordable.', 'Value', 'Positive', '2024-02-28', '2024-03-05', 101, true, 11),
(1008, 'CUST-005', 2003, 'NIKE-AMSC-001', 4, 'Good starter Nike shoe. Not as premium as other Air Max but decent quality.', 'Quality', 'Mixed', '2024-03-10', '2024-03-15', 101, true, 7),

-- Nike Air Max Plus Feedback
(1009, 'CUST-004', 2004, 'NIKE-AMPLUS-001', 3, 'Too expensive for what you get. Returned and got Air Max 90 instead.', 'Price', 'Negative', '2024-01-20', '2024-01-25', 101, false, 4),

-- Adidas Feedback for comparison
(1010, 'CUST-009', 2007, 'ADI-GAZ-001', 4, 'Classic style and comes in wide width which Nike doesn not offer.', 'Fit Options', 'Positive', '2024-03-14', '2024-03-18', 101, true, 13),
(1011, 'CUST-010', 2008, 'ADI-STAN-001', 5, 'Better price than Nike and just as comfortable.', 'Value', 'Positive', '2024-03-08', '2024-03-12', 101, true, 16),
(1012, 'CUST-011', 2006, 'ADI-UB22-001', 4, 'Great for running but wish Nike had similar boost technology.', 'Technology', 'Mixed', '2024-03-19', '2024-03-22', 101, true, 10);

-- Competitive Insights Data
INSERT INTO competitive_insights VALUES
(3001, 'CUST-009', 'Nike', 'Adidas', 'Price', 90.00, 'Adidas often has better sales and promotions', 'Nike is overpriced compared to Adidas', 'Chose Adidas', '2024-03-14', 101, 'EMP-001'),
(3002, 'CUST-010', 'Nike', 'Adidas', 'Fit Options', 80.00, 'Adidas offers wide width options', 'Need wide width shoes, Nike does not have them', 'Chose Adidas', '2024-03-08', 101, 'EMP-002'),
(3003, 'CUST-003', 'Adidas', 'Nike', 'Brand Loyalty', 70.00, 'Prefer Nike brand and style', 'Always wanted Air Max look', 'Chose Nike', '2024-02-28', 101, 'EMP-001'),
(3004, 'CUST-011', 'Adidas', 'Nike', 'Performance Features', 150.00, 'Nike Air cushioning vs Boost', 'Curious about Nike Air technology', 'Chose Adidas', '2024-03-19', 101, 'EMP-003'),
(3005, 'CUST-012', 'Nike', 'Adidas', 'Promotion Timing', 120.00, 'Adidas had 30% off sale', 'Nike too expensive without sale', 'Chose Adidas', '2024-03-16', 101, 'EMP-002'),
(3006, 'CUST-006', 'Adidas', 'Nike', 'Durability', 98.60, 'Nike reputation for lasting longer', 'Want shoes that last for running', 'Chose Nike', '2024-03-22', 101, 'EMP-001'),
(3007, 'CUST-007', 'Nike', 'Adidas', 'Family Value', 156.80, 'Adidas family packs and discounts', 'Buying for whole family, need better deals', 'Chose Adidas', '2024-03-12', 101, 'EMP-003');

-- Sales Interactions Data
INSERT INTO sales_interactions VALUES
(4001, 'CUST-001', 'EMP-001', 'Nike Air Max 90', 'Nike', 'Too expensive', 'Price', 'Highlighted durability and versatility', 'Sale Completed', '2024-02-15 14:30:00', 101, 15, false),
(4002, 'CUST-009', 'EMP-002', 'Nike Air Max 270', 'Nike', 'Nike is overpriced', 'Price', 'Showed Adidas comparison, customer still preferred Adidas', 'No Sale', '2024-03-14 16:45:00', 101, 12, true),
(4003, 'CUST-003', 'EMP-001', 'Nike Air Max SC', 'Nike', 'Want Air Max 90 but too expensive', 'Price', 'Positioned SC as affordable Air Max option', 'Sale Completed', '2024-02-28 11:20:00', 101, 18, false),
(4004, 'CUST-010', 'EMP-002', 'Nike Air Max 90', 'Nike', 'Need wide width', 'Fit', 'Explained Nike sizing, customer chose Adidas for width', 'No Sale', '2024-03-08 15:15:00', 101, 10, false),
(4005, 'CUST-011', 'EMP-003', 'Nike Air Max 270', 'Nike', 'How does Air compare to Boost?', 'Technology', 'Explained Air cushioning benefits', 'No Sale', '2024-03-19 13:40:00', 101, 20, true),
(4006, 'CUST-006', 'EMP-001', 'Nike Air Max 90', 'Nike', 'Runs small according to reviews', 'Sizing', 'Recommended sizing up, provided fit guarantee', 'Sale Completed', '2024-03-22 10:30:00', 101, 12, false),
(4007, 'CUST-012', 'EMP-002', 'Nike Air Max Plus', 'Nike', 'Too expensive for family purchase', 'Price', 'Suggested Air Max SC as family-friendly option', 'No Sale', '2024-03-16 17:20:00', 101, 25, true),
(4008, 'CUST-005', 'EMP-003', 'Nike Air Max SC', 'Nike', 'Is this real Air Max?', 'Product Knowledge', 'Explained SC positioning and Air technology', 'Sale Completed', '2024-03-10 12:45:00', 101, 14, false); 