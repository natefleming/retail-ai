USE IDENTIFIER(:database);

-- Synthetic loyalty events: 10K customers (C-00001..C-10000), each with an
-- ENROLL event between 6mo and 5yr ago. ~20% have a TIER_CHANGE event.
-- Scaling knob: change 10000 below to dial total customer count.

INSERT INTO loyalty_events
SELECT
    concat('E-', lpad(cast(id AS STRING), 9, '0')) AS event_id,
    concat('C-', lpad(cast(id AS STRING), 5, '0')) AS customer_id,
    'ENROLL' AS event_type,
    current_timestamp() - make_interval(0, 0, 0, cast(rand(101 + id) * 1800 + 90 AS INT), 0, 0, 0) AS event_ts,
    cast(NULL AS STRING) AS tier_before,
    CASE WHEN rand(102 + id) < 0.05 THEN 'Premium'
         WHEN rand(102 + id) < 0.20 THEN 'Gold'
         WHEN rand(102 + id) < 0.55 THEN 'Silver'
         ELSE 'Standard' END AS tier_after,
    cast(rand(103 + id) * 8 + 101 AS INT) AS store_id,
    NULL AS payload_json
FROM range(10000);

-- Tier change events for ~20% of customers
INSERT INTO loyalty_events
SELECT
    concat('E-T-', lpad(cast(id AS STRING), 9, '0')) AS event_id,
    concat('C-', lpad(cast(id AS STRING), 5, '0')) AS customer_id,
    'TIER_CHANGE' AS event_type,
    current_timestamp() - make_interval(0, 0, 0, cast(rand(201 + id) * 365 + 30 AS INT), 0, 0, 0) AS event_ts,
    'Silver' AS tier_before,
    'Gold' AS tier_after,
    NULL AS store_id,
    '{"reason":"lifetime_spend_threshold"}' AS payload_json
FROM range(10000)
WHERE rand(200 + id) < 0.20;
