USE IDENTIFIER(:database);

-- Past offer redemptions. The eval pipeline uses the last 30d of this table
-- as ground truth for the rank-correlation judge.
CREATE TABLE IF NOT EXISTS redemptions (
    redemption_id   STRING  COMMENT 'Unique redemption id',
    customer_id     STRING  COMMENT 'Customer who redeemed',
    offer_id        STRING  COMMENT 'Offer redeemed',
    receipt_id      STRING  COMMENT 'Associated receipt (if redeemed at checkout)',
    redeemed_ts     TIMESTAMP COMMENT 'When the offer was redeemed',
    redemption_value DOUBLE COMMENT 'Dollar value of the discount realized'
)
USING DELTA
COMMENT 'Offer redemption history';
