USE IDENTIFIER(:database);

-- Active offer catalog used by the LLM ranker.
-- Each offer carries the marketing-facing description (embedded for vector search)
-- plus the structured attributes the prompt needs (brand, category, discount, margin class).
CREATE TABLE IF NOT EXISTS offer_catalog (
    offer_id            STRING  COMMENT 'Unique offer identifier (e.g., O-0007)',
    name                STRING  COMMENT 'Short marketing name',
    description         STRING  COMMENT 'Customer-facing offer description; embedded into the vector index',
    brand               STRING  COMMENT 'Brand the offer targets (or ALL_BRANDS for catalog-wide)',
    category            STRING  COMMENT 'Product category (Footwear, Activewear, Outerwear, Denim, Apparel-Tops, Apparel-Bottoms, Accessories)',
    discount_kind       STRING  COMMENT 'PERCENT | DOLLAR_OFF | BOGO | FREE_SHIPPING | TIERED',
    discount_pct        DOUBLE  COMMENT 'Normalized discount magnitude (0.0-1.0)',
    margin_class        STRING  COMMENT 'A=protected margin, B=neutral, C=margin-erosive (steep discount)',
    eligibility_json    STRING  COMMENT 'JSON eligibility rules: min tier, min lifetime spend, excluded cohorts',
    valid_from          DATE    COMMENT 'First day the offer is redeemable',
    valid_to            DATE    COMMENT 'Last day the offer is redeemable',
    seasonal_tag        STRING  COMMENT 'SPRING | SUMMER | FALL | WINTER | EVERGREEN'
)
USING DELTA
COMMENT 'Active marketing offers eligible for personalized ranking'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true'
);
