# Overview

This repository contains the Instamart Customer Intelligence & Support Copilot capstone. Part A builds a reproducible data foundation from deterministic, fictional CSV data. Python 3.14.5 was used with pandas 3.1.0 and NumPy 2.5.4; no network service or paid API is used by the Part A implementation.

# Reproduce

Run these commands from the repository root in Windows PowerShell. If PowerShell blocks activation, the process-scoped policy change below applies only to the current PowerShell process; it does not change the system-wide execution policy.

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_data.py
python src/part_a.py
python src/validate_part_a.py
```

`requirements.txt` pins the direct Part A dependencies. The data generator is deterministic and writes its raw CSVs under `data/raw/`; the Part A pipeline reads those files and rebuilds cleaned CSVs, the SQLite database, and analysis outputs. Successful validation prints `ALL CHECKPOINTS PASSED`.

# Part A

## Raw data audit

The audit in `outputs/part_a/raw_audit.json` and the matching per-table CSVs records each raw dataset's shape, pandas dtypes, missing counts, and missing percentages. Raw missing values observed:

| Dataset | Shape | Missing fields |
| --- | ---: | --- |
| Customers | 1,200 × 5 | `age`: 61 (5.0833%); `gender`: 30 (2.5000%) |
| Products | 40 × 4 | None |
| Orders | 13,084 × 9 | `order_amount`: 53 (0.4051%); `delivery_minutes`: 2,541 (19.4207%) |

The other raw fields have zero missing values. Raw inferred dtypes are: customers (`customer_id`, `city`, `signup_date`, `gender` string; `age` float64); products (three string columns and `price_inr` int64); orders (IDs, timestamp, status, and raw amount string; `quantity` and `discount_pct` int64; delivery minutes float64). `order_amount` also contains unparseable text; numeric coercion identifies 112 amount values requiring repair, including those not counted as null by CSV parsing.

## Cleaning rules and row counts

Customer cities are whitespace-trimmed and title-cased into exactly Bengaluru, Chennai, Delhi, Hyderabad, Mumbai, and Pune. Gender variants are normalized to `male`, `female`, or `unknown`. Ages outside 10–100 are set missing; all 67 initially missing or invalid ages are filled with their city median using `groupby` and `transform`. Signup dates are parsed as datetimes.

Orders are processed in the requested order: parse timestamps; remove 119 exact duplicate rows; resolve repeated IDs by keeping the latest timestamp (656 further rows removed); coerce amounts and impute 112 unparsable values from product price × quantity × (1 − discount/100), rounded to two decimals; replace 105 negative delivery durations with missing; fill 774 missing completed-order durations with the completed-order median for that product category. Cancelled delivery durations remain missing.

| Checkpoint | Result |
| --- | ---: |
| Raw customers / products / orders / tickets | 1,200 / 40 / 13,084 / 164 |
| Clean customers / products / orders | 1,200 / 40 / 12,309 |
| Final completed / cancelled orders | 11,110 / 1,199 |
| Completed-order revenue | INR 2,998,668.80 |

Cleaned CSVs are in `data/clean/`. `data/instamart.db` contains `customers`, `products`, and `orders`; SQLite stores `orders.order_ts` as text in `YYYY-MM-DD HH:MM:SS` format. The database is a generated local artifact and is excluded from Git by `.gitignore`.

## Customer counts and analytical summaries

Normalized customer city counts: Bengaluru 197, Chennai 191, Delhi 210, Hyderabad 191, Mumbai 184, Pune 227. Gender counts: female 610, male 560, unknown 30.

The named aggregation of completed orders by city is saved in `outputs/part_a/named_aggregation_by_city.csv`:

| City | Orders | Revenue (INR) | Mean order value (INR) |
| --- | ---: | ---: | ---: |
| Bengaluru | 1,894 | 523,643.40 | 276.47 |
| Chennai | 1,700 | 463,629.15 | 272.72 |
| Delhi | 2,064 | 502,829.80 | 243.62 |
| Hyderabad | 1,855 | 528,589.90 | 284.95 |
| Mumbai | 1,613 | 479,058.50 | 297.00 |
| Pune | 1,984 | 500,918.05 | 252.48 |

Completed-order descriptive statistics (`outputs/part_a/descriptive_statistics.csv`) show quantity mean 1.8867 (median 2), discount mean 12.9581% (median 10%), amount mean INR 269.91 (median INR 135.00), and delivery time mean 27.3743 minutes (median 25). Correlations (`correlation_matrix.csv`) are weak: quantity vs amount 0.087294; discount vs amount −0.095106; delivery time vs amount −0.003123. Full statistics and the full correlation matrix are in the output files.

## SQL findings and reconciliation

`sql/queries.sql` contains seven executable rubric queries: customer spending, city/category revenue, above-average-priced products, customers without completed orders, above-average spend using a CTE, customer spend rankings, and daily cumulative revenue. Each was run through `pandas.read_sql_query`; result CSVs are under `outputs/part_a/`.

| Query | Result rows / finding |
| --- | ---: |
| Customer spending | 1,200 customers |
| Revenue by city and category | 30 city/category groups |
| Products above average price | 11 products |
| No completed orders | 25 customers |
| Spend above customer average | 561 customers |
| Spend rankings | 1,200 customers |
| Daily cumulative revenue | 183 dates |

The spend ranking leaders are C0693 (Mumbai, INR 14,766.55), C0291 (Delhi, INR 13,742.60), and C0488 (Mumbai, INR 13,187.50). Electronics had the highest category revenue (INR 1,219,248.10); Personal Care had the lowest (INR 168,433.95). The top daily revenue was INR 29,969.40 on 2025-06-16. Cumulative daily revenue ends at INR 2,998,668.80.

The pandas city/category pivot includes row and column margins in `pandas_city_category_revenue_pivot.csv`. Its category/city cells reconcile to the SQL result with a difference of INR 0.00, and its grand total equals completed-order revenue.

## Exploratory answers

1. Hyderabad generated the highest completed-order revenue (INR 528,589.90).
2. Electronics led category revenue (INR 1,219,248.10); Personal Care was lowest (INR 168,433.95).
3. Of 1,200 customers, 1,175 had completed purchases; their mean spend was INR 2,552.06 and median spend was INR 2,286.50.
4. The highest-revenue day was 2025-06-16 (INR 29,969.40); cumulative completed revenue ended at INR 2,998,668.80.

These answers contain fewer than 200 words in total. The validation script checks the explicit row-count, revenue, city/gender distribution, SQL row-count, ranking, database, and reconciliation checkpoints.

# Part B

_Placeholder — not implemented yet._

# Part C

_Placeholder — not implemented yet._

# Part D

_Placeholder — not implemented yet._

# Limitations and Next Steps

The generated records are synthetic and deterministic; they do not represent actual Instamart customers or operations. Part A focuses on data quality, SQL analysis, and descriptive exploration. The next recommended step is to review these outputs against the complete capstone rubric, then begin Part B without changing the supplied generator or mock LLM files.
