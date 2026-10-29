"""Validate the explicit Part A reference checkpoints and saved artifacts."""

import json
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "part_a"
METRICS = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    checks = []
    checks.append(METRICS["raw_rows"]["customers"] == 1200)
    checks.append(METRICS["raw_rows"]["products"] == 40)
    checks.append(METRICS["raw_rows"]["orders"] == 13084)
    checks.append(pd.read_csv(ROOT / "data" / "raw" / "tickets.csv").shape[0] == 164)
    checks.append(METRICS["final_orders"] == 12309)
    checks.append(METRICS["order_status_counts"].get("completed") == 11110)
    checks.append(METRICS["order_status_counts"].get("cancelled") == 1199)
    checks.append(abs(METRICS["completed_revenue_inr"] - 2998668.80) < 0.005)
    checks.append(METRICS["customer_city_counts"] == {
        "Bengaluru": 197, "Chennai": 191, "Delhi": 210,
        "Hyderabad": 191, "Mumbai": 184, "Pune": 227,
    })
    checks.append(sum(METRICS["customer_city_counts"].values()) == 1200)
    checks.append(METRICS["customer_gender_counts"] == {"female": 610, "male": 560, "unknown": 30})
    checks.append(sum(METRICS["customer_gender_counts"].values()) == 1200)
    checks.append(len(METRICS["sql_query_row_counts"]) == 7)
    checks.append(METRICS["sql_query_row_counts"].get("query_1_customer_spending") == 1200)
    checks.append(METRICS["sql_query_row_counts"].get("query_6_customer_spend_rankings") == 1200)
    checks.append(METRICS["sql_query_row_counts"] == {
        "query_1_customer_spending": 1200,
        "query_2_revenue_by_city_and_category": 30,
        "query_3_above-average-priced_products": 11,
        "query_4_customers_without_completed_orders": 25,
        "query_5_above-average_customer_spend_using_a_cte": 561,
        "query_6_customer_spend_rankings": 1200,
        "query_7_daily_cumulative_revenue": 183,
    })
    checks.append(METRICS["top_customer_ranking"][0]["customer_id"] == "C0693")
    checks.append(abs(METRICS["top_customer_ranking"][0]["spend_inr"] - 14766.55) < 0.005)
    checks.append(METRICS["sql_pandas_revenue_reconciled"] is True)
    checks.append(abs(METRICS["sql_pandas_revenue_difference_inr"]) < 0.005)
    checks.append(abs(METRICS["daily_cumulative_final_inr"] - 2998668.80) < 0.005)

    customers = pd.read_csv(ROOT / "data" / "clean" / "customers.csv")
    orders = pd.read_csv(ROOT / "data" / "clean" / "orders.csv")
    checks.append(len(customers) == 1200 and customers["age"].between(10, 100).all())
    checks.append(customers["signup_date"].notna().all())
    checks.append(len(orders) == 12309 and not orders.duplicated().any() and orders["order_id"].is_unique)
    checks.append(orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().all())
    checks.append(not (pd.to_numeric(orders["delivery_minutes"], errors="coerce") < 0).any())
    checks.append((OUT / "raw_audit.json").is_file())
    checks.append((OUT / "descriptive_statistics.csv").is_file() and (OUT / "correlation_matrix.csv").is_file())

    with sqlite3.connect(ROOT / "data" / "instamart.db") as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        checks.append({"customers", "products", "orders"}.issubset(tables))
        timestamp_types = {row[1]: row[2].upper() for row in connection.execute("PRAGMA table_info(orders)")}
        checks.append(timestamp_types.get("order_ts") == "TEXT")
        sql_revenue = connection.execute("SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status='completed'").fetchone()[0]
        checks.append(abs(sql_revenue - 2998668.80) < 0.005)

    if not all(checks):
        failed = [index + 1 for index, passed in enumerate(checks) if not passed]
        raise AssertionError(f"Part A validation failed at checkpoint(s): {failed}")
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
