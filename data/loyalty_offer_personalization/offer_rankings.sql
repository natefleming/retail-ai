USE IDENTIFIER(:database);

-- Output table. Populated by the refresh_offer_rankings UC function
-- (which itself calls rank_offers_for_customer, which calls ai_query
-- on the configured chat-completion endpoint).
CREATE TABLE IF NOT EXISTS offer_rankings (
    customer_id     STRING    COMMENT 'Customer the ranking is for',
    prompt_version  STRING    COMMENT 'Prompt template version used to produce this row',
    model_endpoint  STRING    COMMENT 'Model serving endpoint name used',
    generated_at    TIMESTAMP COMMENT 'When this ranking was produced',
    ranking         ARRAY<STRUCT<
        offer_id: STRING,
        rank: INT,
        score: DOUBLE,
        reason: STRING
    >> COMMENT '10-element ranking of offers from best to worst for this customer, with a per-offer score (0-100) and a one-sentence rationale citing the features used.'
)
USING DELTA
PARTITIONED BY (prompt_version)
COMMENT 'LLM-produced offer rankings per customer per prompt version'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true'
);
