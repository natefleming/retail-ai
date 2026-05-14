USE IDENTIFIER(:database);

-- Raw loyalty-program event log. One row per event (enroll, tier change,
-- redemption notification, visit). Source of truth for member lifecycle.
CREATE TABLE IF NOT EXISTS loyalty_events (
    event_id        STRING  COMMENT 'Unique event id',
    customer_id     STRING  COMMENT 'Loyalty customer id',
    event_type      STRING  COMMENT 'ENROLL | TIER_CHANGE | VISIT | REDEMPTION_NOTIFY | OPT_OUT',
    event_ts        TIMESTAMP COMMENT 'When the event occurred',
    tier_before     STRING  COMMENT 'Previous tier (null for ENROLL)',
    tier_after      STRING  COMMENT 'New tier (null for non-tier events)',
    store_id        INT     COMMENT 'Store associated with the event (null for digital)',
    payload_json    STRING  COMMENT 'Optional structured event payload (JSON)'
)
USING DELTA
COMMENT 'Loyalty program event log';
