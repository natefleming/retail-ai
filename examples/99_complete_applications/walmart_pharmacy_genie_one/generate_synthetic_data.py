#!/usr/bin/env python3
"""Generate a synthetic Walmart Pharmacy source table for the Genie One POC.

Builds ``<catalog>.<schema>.market_bus_growth_metric`` — the single wide table the
three metric views (clinical_outcome_snapshot, pharmacy_core_business_growth,
digital_account_snapshot) read — in ONE ``CREATE OR REPLACE TABLE ... AS SELECT``
statement (no row-by-row loop). Rows cover a small store → market → region →
business-unit hierarchy over a short date spine, with row-based metric records for
every ``metric_name`` the growth view aggregates, plus the per-entity YTD / rolling
snapshot columns the clinical and digital views read.

The real Walmart source is ``wmt-hnw-pharmacy-catalog-prod.genie.market_bus_growth_metric``;
this synthesizes a stand-in so the POC is testable in any workspace.

Usage:
    python generate_synthetic_data.py --profile fevm --warehouse <warehouse_id> \
        [--catalog retail_consumer_goods] [--schema walmart_pharmacy_poc]
"""

from __future__ import annotations

import argparse

from databricks.sdk import WorkspaceClient

METRIC_NAMES = [
    "MESSAGING_ADOPTION_RATE", "SYSTEMTIC_REFILL_PERCENTAGE", "DOTCOM_REFILL_PERCENTAGE",
    "AUTO_REFILL_PERCENTAGE", "RXD_SCRIPT_PENETRATION", "FIRST_TIME_ORDER_READY_RATE",
    "RETURN_TO_STOCK_RATE", "OUT_OF_STOCK_RATE", "OUT_OF_STOCK_TRANSFER_RATE", "WALK_IN_QR",
    "NPS_SCORE", "FIVE_STAR_RATE", "VISITS", "SCRIPTS_SALES", "SCRIPTS_SOLD",
    "OUT_OF_STOCK_TRANSFER", "TESTING_AND_TREATEMENTS", "HORMONAL_CONTRACEPTIVES",
    "BASIC_HEALTH_SCREENINGS", "TOTAL_ADMINISTERED", "FLU_ADMINISTERED", "EXPANDED_ADMINISTERED",
    "NEW_DIGITAL_ACCOUNTS", "NEW_DIGITAL_ACCOUNTS_OF_NEW_PATIENTS",
    "NEW_DIGITAL_ACCOUNTS_OF_CONTINUING_PATIENTS", "AVERAGE_ORDERS_SOLD_PER_STORE",
    "AVERAGE_ORDERS_SOLD_PER_STORE_NEW",
]
LEVELS = ["store", "market", "region", "bu", "lob"]


def build_ctas(catalog: str, schema: str) -> str:
    table = f"{catalog}.{schema}.market_bus_growth_metric"
    marr = "array(" + ",".join(f"'{m}'" for m in METRIC_NAMES) + ")"
    # 12 stores across 2 BU / 3 region / 4 market
    stores = [
        (i, 1 + (i - 1) // 4, 1 + (i - 1) // 3, 1 + (i - 1) // 6) for i in range(1, 13)
    ]
    stores_vals = ",".join(f"({s},{r},{m},{b})" for (s, r, m, b) in stores)
    rate_cols = [
        f"{lvl}_{base}"
        for base in ("outcomesone_successful_completion_rate_ytd", "completion_rate_ytd",
                     "validation_rate_ytd")
        for lvl in LEVELS
    ]
    dig_cols = [
        f"{lvl}_{base}"
        for base in ("digital_population", "active_patient_count_with_digital_accs",
                     "active_patient_count_with_new_digital_accs",
                     "active_patient_count_with_continuing_digital_accs")
        for lvl in LEVELS
    ]
    rate_exprs = ",\n  ".join(
        f"CAST(ROUND(60+rand()*38,1) AS STRING) AS {c}" for c in rate_cols
    )
    dig_exprs = ",\n  ".join(
        f"CAST(CAST(1000+rand()*2000000 AS BIGINT) AS STRING) AS {c}" for c in dig_cols
    )
    return f"""CREATE OR REPLACE TABLE {table} AS
WITH d AS (SELECT explode(sequence(date_sub(current_date(),9), current_date(), interval 1 day)) AS calendar_date),
s AS (SELECT * FROM VALUES {stores_vals} AS t(store_number, region_number, market_number, business_unit_i)),
m AS (SELECT explode({marr}) AS metric_name)
SELECT
  d.calendar_date,
  date_sub(d.calendar_date, 364) AS last_year_comparable_calendar_date,
  concat(CAST(year(d.calendar_date) AS STRING), lpad(CAST(weekofyear(d.calendar_date) AS STRING),2,'0')) AS walmart_year_week_number,
  CAST(weekofyear(d.calendar_date) AS INT) AS walmart_week_number,
  CAST(month(d.calendar_date) AS INT) AS walmart_month_number,
  date_format(d.calendar_date,'MMMM') AS walmart_month_name,
  CAST(quarter(d.calendar_date) AS INT) AS walmart_quarter_number,
  CAST(year(d.calendar_date) AS INT) AS walmart_year_number,
  CAST(month(d.calendar_date) AS INT) AS fiscal_month_number,
  date_format(d.calendar_date,'MMMM') AS fiscal_month_name,
  CAST(quarter(d.calendar_date) AS INT) AS fiscal_quarter_nbr,
  CAST(year(d.calendar_date) AS INT) AS fiscal_year_nbr,
  CAST(day(d.calendar_date) AS INT) AS current_calendar_day,
  1 AS week_to_date_flag, 1 AS walmart_month_to_date_flag, 1 AS walmart_quarter_to_date_flag,
  1 AS walmart_year_to_date_flag, 1 AS fiscal_month_to_date_flag, 1 AS fiscal_quarter_to_date_flag, 1 AS fiscal_year_to_date_flag,
  'PHARMACY' AS line_of_business,
  CAST(s.business_unit_i AS STRING) AS business_unit,
  concat('Business Unit ', s.business_unit_i) AS business_unit_name,
  concat('BU-', s.business_unit_i) AS business_unit_display_name,
  s.region_number,
  element_at(array('Northeast','Southeast','Midwest'), s.region_number) AS region_name,
  concat('Region ', s.region_number) AS region_display_name,
  s.market_number,
  concat('Market ', s.market_number) AS market_name,
  concat('MKT-', s.market_number) AS market_display_name,
  s.store_number,
  concat('Pharmacy #', s.store_number) AS store_name,
  concat('Store ', s.store_number) AS store_display_name,
  'OPEN' AS store_status,
  true AS is_active_store,
  (s.store_number % 2 = 0) AS comparable_store_flag,
  concat('SM ', s.store_number) AS store_manager_name,
  concat('MM ', s.market_number) AS market_manager_name,
  concat('RM ', s.region_number) AS region_manager_name,
  concat('BUM ', s.business_unit_i) AS bu_manager_name,
  'WMT' AS op_company_code,
  current_timestamp() AS load_time_stamp,
  'STORE' AS native_summary_grain,
  'market_bus_growth_metric' AS source_table,
  'PHARMACY_GROWTH' AS metric_group,
  m.metric_name,
  'rate' AS metric_type,
  'SUM' AS aggregation_rule,
  ROUND(rand()*1000,2) AS numerator_value_ty,
  ROUND(500+rand()*1000,2) AS denominator_value_ty,
  ROUND(rand()*100,2) AS metric_value_ty,
  ROUND(rand()*1000,2) AS numerator_value_ly,
  ROUND(500+rand()*1000,2) AS denominator_value_ly,
  ROUND(rand()*100,2) AS metric_value_ly,
  'num_rule' AS numerator_rule, 'den_rule' AS denominator_rule,
  'FISCAL_CALENDAR' AS calendar_flag,
  element_at(array('WALKIN','DOTCOM','DELIVERY'), 1+CAST(rand()*3 AS INT)) AS order_channel,
  element_at(array('PICKUP','MAIL','COURIER'), 1+CAST(rand()*3 AS INT)) AS delivery_method,
  '[]' AS neighbouring_store_nbrs, '[]' AS neighbouring_market_nbrs, '[]' AS neighbouring_region_nbrs, '[]' AS neighbouring_subdiv_nbrs,
  CAST(year(current_date()) AS INT) AS current_walmart_year_number,
  CAST(year(current_date()) AS INT) AS current_fiscal_year_number,
  CAST(month(current_date()) AS INT) AS current_walmart_month_number,
  CAST(month(current_date()) AS INT) AS current_fiscal_month_number,
  CAST(quarter(current_date()) AS INT) AS current_walmart_quarter_number,
  CAST(quarter(current_date()) AS INT) AS current_fiscal_quarter_number,
  {rate_exprs},
  {dig_exprs}
FROM d CROSS JOIN s CROSS JOIN m
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True, help="Databricks CLI profile")
    ap.add_argument("--warehouse", required=True, help="SQL warehouse id")
    ap.add_argument("--catalog", default="retail_consumer_goods")
    ap.add_argument("--schema", default="walmart_pharmacy_poc")
    args = ap.parse_args()

    w = WorkspaceClient(profile=args.profile)
    w.schemas.create(name=args.schema, catalog_name=args.catalog)  # idempotent-ish
    ctas = build_ctas(args.catalog, args.schema)
    print(f"Creating {args.catalog}.{args.schema}.market_bus_growth_metric …")
    w.statement_execution.execute_statement(
        warehouse_id=args.warehouse, statement=ctas, wait_timeout="50s"
    )
    res = w.statement_execution.execute_statement(
        warehouse_id=args.warehouse,
        statement=f"SELECT count(*) FROM {args.catalog}.{args.schema}.market_bus_growth_metric",
        wait_timeout="30s",
    )
    print("Row count:", res.result.data_array[0][0])
    print("Done. Next: create the 3 metric views in metric_views/ and their Genie spaces.")


if __name__ == "__main__":
    main()
