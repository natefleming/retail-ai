USE IDENTIFIER(:database);

-- 25 eval cases covering both personas and all sub-agents.
INSERT INTO evaluation VALUES
-- customer_intelligence agent
('EV-001','CRM',          'Tell me about customer C-00007. Pull their profile and recent purchases.', 'Agent calls get_customer_features and get_recent_receipts in parallel. Response cites top brand and at least one recent SKU.'),
('EV-002','CRM',          'What is customer C-00042 most likely to redeem this week?',                'Agent retrieves the customer profile and at least one ranking source.'),
('EV-003','MERCHANDISER', 'Which 10 customers were most active in Footwear last month?',              'Agent uses the segment_analyst Genie path or a UC function over receipts joined to customer_features.'),

-- offer_catalog agent
('EV-004','CRM',          'Find Nike running shoe offers that are still active and apply to Silver-tier members.', 'Agent uses the offer_catalog vector search; result list includes only currently valid offers and respects eligibility_json min_tier.'),
('EV-005','MERCHANDISER', 'Show me high-discount offers (30% or more off) for outerwear this fall.',  'Agent filters by category=Outerwear and discount_kind/discount_pct; seasonal_tag is honored.'),

-- ranking_explainer agent
('EV-006','CRM',          'Why was offer O-0007 ranked first for customer C-00007?',                  'Agent calls get_offer_ranking and references at least one Customer 360 feature (top_brands, top_categories, price_tolerance, or redemption history).'),
('EV-007','CRM',          'What was the lowest-ranked offer for C-00100 and why?',                    'Agent surfaces stored ranking row and explains in terms of features.'),

-- what_if_ranker agent
('EV-008','CRM',          'Add offer O-0099 to the candidate pool for C-00007 and re-rank.',          'Agent calls rank_offers_for_customer with an extended candidate set; response shows updated ranks and rationale.'),
('EV-009','CRM',          'Re-rank for customer C-00007 using only their currently eligible offers.', 'Agent invokes rank_offers_for_customer with prompt_version=v1; output is a 10-element ranking.'),
('EV-010','MERCHANDISER', 'For a Premium-tier customer with top_brands=[Patagonia] and price_tolerance_score=0.1, which 5 offers would the model push first?', 'Agent runs a what-if with a synthetic customer profile; ranking favors Patagonia and price-insensitive picks.'),

-- segment_analyst agent
('EV-011','MERCHANDISER', 'Which loyalty tier saw the highest redemption rate on offer O-0011?',      'Agent calls top_offers_by_segment or queries the Performance Genie room; result groups by tier.'),
('EV-012','MERCHANDISER', 'Compare offer O-0001 and O-0008 redemption across Silver and Gold tiers.', 'Agent surfaces comparative numbers for the two offers across two tiers.'),
('EV-013','MERCHANDISER', 'Top categories by 90-day redemption volume?',                              'Agent groups by category over redemptions joined to offer_catalog.'),

-- redemption_outcomes agent
('EV-014','CRM',          'Did the new winter coat sale O-0083 lift redemption among lapsed members?', 'Agent joins redemptions to customer_features to find lapsed (days_since_last_visit > 180) and reports redemption count.'),
('EV-015','MERCHANDISER', 'Which offers underperformed last week relative to ranking position?',      'Agent calls top_offers_by_segment, compares to expected from offer_rankings.'),

-- general / supervisor routing
('EV-016','CRM',          'What can you help me do as a loyalty marketer?',                            'general agent responds with the menu of sub-agents.'),
('EV-017','CRM',          'What is the current date and the next loyalty tier review?',               'general agent returns current_time; mentions tier review windows if available.'),

-- multi-step / chained
('EV-018','CRM',          'Pull C-00007''s profile and explain why offer O-0001 should rank in their top 3.', 'Customer_intelligence + ranking_explainer chained; cites at least one feature.'),
('EV-019','CRM',          'Compare offer rankings for C-00007 and C-00042. Where do they overlap and differ?', 'Two ranking_explainer calls; response highlights diffs and shared offers.'),
('EV-020','MERCHANDISER', 'Build me a list of the top 5 customers most likely to redeem outerwear next month.', 'Segment_analyst + ranking_explainer chained; output ranks customers, references offer ranks.'),

-- edge cases
('EV-021','CRM',          'Customer C-99999 — what do we know?',                                       'Agent reports the customer is not in the dataset without hallucinating attributes.'),
('EV-022','CRM',          'Show me the ranking for C-00007 using prompt version v2.',                  'If v2 has not been generated, the agent surfaces that v2 is not yet populated rather than fabricating.'),
('EV-023','MERCHANDISER', 'Which offers are unavailable today?',                                       'Agent filters offer_catalog by valid_to < current_date(); returns expired offers.'),
('EV-024','CRM',          'Did you remember that I usually focus on the Activewear category?',        'Lakebase memory recalls operator preference if set; otherwise gracefully asks.'),
('EV-025','MERCHANDISER', 'What is the average basket value for customers who redeemed Patagonia offers in the last 60 days?', 'Segment_analyst joins redemptions to receipts via customer_features; returns a numeric answer.');
