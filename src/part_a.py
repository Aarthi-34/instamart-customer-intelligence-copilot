"""Reproducible Part A data foundation for the Instamart capstone."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs" / "part_a"
DB = ROOT / "data" / "instamart.db"
SQL_FILE = ROOT / "sql" / "queries.sql"
CITIES = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune"]


def audit_frame(frame: pd.DataFrame, name: str) -> dict:
    missing = frame.isna().sum()
    return {
        "dataset": name,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "shape": list(frame.shape),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing": {
            column: {"count": int(missing[column]), "percent": round(float(missing[column] / len(frame) * 100), 4) if len(frame) else 0.0}
            for column in frame.columns
        },
    }


def read_sql_queries(connection: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    source = SQL_FILE.read_text(encoding="utf-8")
    blocks = re.split(r"(?=-- Query \d+:)", source)
    results = {}
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        match = re.match(r"-- Query (\d+):\s*(.+)\n", block)
        if not match:
            raise ValueError("Each SQL statement must start with '-- Query N: title'")
        number, title = match.groups()
        statement = block[match.end():].strip().rstrip(";")
        results[f"query_{number}_{title.lower().replace(' ', '_')}"] = pd.read_sql_query(statement, connection)
    if len(results) != 7:
        raise AssertionError(f"Expected seven rubric queries, found {len(results)}")
    return results


def main() -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    raw = {name: pd.read_csv(RAW / f"{name}.csv") for name in ("customers", "products", "orders")}
    audits = [audit_frame(frame, name) for name, frame in raw.items()]
    (OUT / "raw_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    for frame, name in zip(raw.values(), raw.keys()):
        summary = pd.DataFrame([
            {"column": c, "dtype": str(frame[c].dtype), "missing_count": int(frame[c].isna().sum()),
             "missing_percent": round(float(frame[c].isna().mean() * 100), 4)}
            for c in frame.columns
        ])
        summary.to_csv(OUT / f"raw_audit_{name}.csv", index=False)

    customers = raw["customers"].copy()
    customers["city"] = customers["city"].astype("string").str.strip().str.title()
    customers["city"] = customers["city"].replace({"Bangalore": "Bengaluru"})
    customers["gender"] = customers["gender"].astype("string").str.strip().str.lower().replace({
        "m": "male", "f": "female", "": pd.NA, "nan": pd.NA,
    }).fillna("unknown")
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    customers.loc[~customers["age"].between(10, 100), "age"] = np.nan
    customer_age_missing_before_fill = int(customers["age"].isna().sum())
    customers["age"] = customers["age"].fillna(customers.groupby("city")["age"].transform("median"))
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")

    products = raw["products"].copy()
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="coerce")

    orders = raw["orders"].copy()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    raw_orders = len(orders)
    exact_duplicates = int(orders.duplicated().sum())
    orders = orders.drop_duplicates().copy()
    after_exact_dedup = len(orders)
    duplicate_order_rows = int(orders.duplicated("order_id", keep=False).sum())
    orders = orders.sort_values(["order_id", "order_ts"], kind="stable").drop_duplicates("order_id", keep="last")
    orders = orders.reset_index(drop=True)

    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="coerce")
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    product_lookup = products.set_index("product_id")
    price = orders["product_id"].map(product_lookup["price_inr"])
    amount_missing = orders["order_amount"].isna()
    amount_imputed = int(amount_missing.sum())
    calculated = (price * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
    orders.loc[amount_missing, "order_amount"] = calculated.loc[amount_missing]
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    negative_delivery = int((orders["delivery_minutes"] < 0).sum())
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    orders["category"] = orders["product_id"].map(product_lookup["category"])
    completed_missing = orders["status"].eq("completed") & orders["delivery_minutes"].isna()
    delivery_imputed = int(completed_missing.sum())
    completed_category_median = orders["delivery_minutes"].where(orders["status"].eq("completed")).groupby(orders["category"]).transform("median")
    orders.loc[completed_missing, "delivery_minutes"] = completed_category_median.loc[completed_missing]
    orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"] = np.nan
    # Keep exact category medians (which may be fractional) instead of rounding them.
    orders = orders.drop(columns="category")

    # Persist timestamps in the requested SQLite text representation and CSV representation.
    customers_csv = customers.copy()
    customers_csv["signup_date"] = customers_csv["signup_date"].dt.strftime("%Y-%m-%d")
    orders_csv = orders.copy()
    orders_csv["order_ts"] = orders_csv["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    customers_csv.to_csv(CLEAN / "customers.csv", index=False)
    products.to_csv(CLEAN / "products.csv", index=False)
    orders_csv.to_csv(CLEAN / "orders.csv", index=False)

    if DB.exists():
        DB.unlink()
    with sqlite3.connect(DB) as connection:
        customers_sql = customers_csv.copy()
        for table, frame in (("customers", customers_sql), ("products", products), ("orders", orders_csv)):
            frame.to_sql(table, connection, index=False, if_exists="replace")

        query_results = read_sql_queries(connection)
        query_row_counts = {}
        for name, result in query_results.items():
            result.to_csv(OUT / f"{name}.csv", index=False)
            query_row_counts[name] = len(result)

        sql_revenue = query_results["query_2_revenue_by_city_and_category"].copy()
        sql_compare = sql_revenue.pivot(index="city", columns="category", values="revenue_inr").fillna(0).sort_index()
        completed = orders.loc[orders["status"].eq("completed")].copy()
        completed = completed.merge(customers[["customer_id", "city"]], on="customer_id", how="left")
        completed = completed.merge(products[["product_id", "category"]], on="product_id", how="left")
        pandas_revenue = pd.pivot_table(completed, values="order_amount", index="city", columns="category",
                                        aggfunc="sum", fill_value=0, margins=True, margins_name="All")
        pandas_revenue.to_csv(OUT / "pandas_city_category_revenue_pivot.csv")
        sql_revenue_matrix = sql_compare.reindex(index=pandas_revenue.index[:-1], columns=pandas_revenue.columns[:-1], fill_value=0)
        reconciliation = bool(np.allclose(sql_revenue_matrix.to_numpy(), pandas_revenue.iloc[:-1, :-1].to_numpy(), atol=0.005))
        if not reconciliation:
            raise AssertionError("SQL and pandas city/category revenue do not reconcile")

        named_aggregation = completed.groupby("city", as_index=False).agg(
            completed_orders=("order_id", "count"), revenue_inr=("order_amount", "sum"),
            average_order_value_inr=("order_amount", "mean"),
        ).sort_values("city")
        named_aggregation.to_csv(OUT / "named_aggregation_by_city.csv", index=False)
        descriptive = completed[["quantity", "discount_pct", "order_amount", "delivery_minutes"]].describe().round(4)
        descriptive.to_csv(OUT / "descriptive_statistics.csv")
        correlation = completed[["quantity", "discount_pct", "order_amount", "delivery_minutes"]].corr(numeric_only=True).round(6)
        correlation.to_csv(OUT / "correlation_matrix.csv")

        customer_totals = completed.groupby("customer_id")["order_amount"].sum()
        city_totals = completed.groupby("city")["order_amount"].sum().sort_values(ascending=False)
        category_totals = completed.groupby("category")["order_amount"].sum().sort_values(ascending=False)
        daily = query_results["query_7_daily_cumulative_revenue"].copy()
        top_daily = daily.sort_values("daily_revenue_inr", ascending=False).iloc[0]
        exploration = [
            f"{city_totals.index[0]} generated the highest completed-order revenue (INR {city_totals.iloc[0]:,.2f}).",
            f"{category_totals.index[0]} led category revenue (INR {category_totals.iloc[0]:,.2f}); {category_totals.index[-1]} was lowest (INR {category_totals.iloc[-1]:,.2f}).",
            f"There were {len(customer_totals):,} customers with completed purchases; their mean spend was INR {customer_totals.mean():,.2f} and median spend was INR {customer_totals.median():,.2f}.",
            f"The highest-revenue day was {top_daily['order_date']} at INR {top_daily['daily_revenue_inr']:,.2f}; cumulative completed revenue ended at INR {daily['cumulative_revenue_inr'].iloc[-1]:,.2f}.",
        ]
        (OUT / "exploratory_answers.md").write_text("\n".join(f"- {line}" for line in exploration) + "\n", encoding="utf-8")

        city_counts = customers["city"].value_counts().reindex(CITIES, fill_value=0).sort_index()
        gender_counts = customers["gender"].value_counts().reindex(["female", "male", "unknown"], fill_value=0)
        status_counts = orders["status"].value_counts().to_dict()
        metrics = {
            "raw_rows": {key: len(value) for key, value in raw.items()},
            "raw_order_duplicates_exact": exact_duplicates,
            "orders_after_exact_dedup": after_exact_dedup,
            "repeated_order_rows_before_order_id_resolution": duplicate_order_rows,
            "final_orders": len(orders),
            "order_status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "completed_revenue_inr": round(float(completed["order_amount"].sum()), 2),
            "customer_age_missing_before_imputation": customer_age_missing_before_fill,
            "customer_age_imputed": customer_age_missing_before_fill,
            "order_amount_imputed": amount_imputed,
            "negative_delivery_values_replaced": negative_delivery,
            "completed_delivery_values_imputed": delivery_imputed,
            "customer_city_counts": {str(k): int(v) for k, v in city_counts.items()},
            "customer_gender_counts": {str(k): int(v) for k, v in gender_counts.items()},
            "sql_query_row_counts": query_row_counts,
            "sql_pandas_revenue_reconciled": reconciliation,
            "sql_pandas_revenue_difference_inr": round(float(pandas_revenue.iloc[-1, -1] - completed["order_amount"].sum()), 2),
            "daily_cumulative_final_inr": round(float(daily["cumulative_revenue_inr"].iloc[-1]), 2),
            "daily_revenue_days": len(daily),
            "top_customer_ranking": query_results["query_6_customer_spend_rankings"].head(10).to_dict(orient="records"),
        }
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    print(f"Part A complete: {len(orders):,} cleaned orders; completed revenue INR {metrics['completed_revenue_inr']:,.2f}")


if __name__ == "__main__":
    main()
