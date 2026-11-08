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
python src/part_b.py
python src/validate_part_b.py
```

`requirements.txt` pins the direct Part A and Part B dependencies. The data generator is deterministic and writes its raw CSVs under `data/raw/`; the Part A pipeline reads those files and rebuilds cleaned CSVs, the SQLite database, and analysis outputs. Part B reads the cleaned files and produces the segmentation, prediction and reporting artifacts. Successful validations print `ALL CHECKPOINTS PASSED` and `ALL PART B CHECKPOINTS PASSED`, respectively.

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
| Customer spending | 30 customers with qualifying completed orders over INR 500 and customer totals over INR 8,000 |
| Revenue by city and category | 30 city/category groups |
| Products above average price | 11 products |
| No completed orders | 25 customers |
| Spend above customer average | 550 customers with completed orders above the eligible-customer average |
| Spend rankings | 1,175 customers with completed orders; ranks are within city |
| Daily cumulative revenue | 183 dates |

Query 1 returns 30 customers; its top qualifying total is C0693 in Mumbai at INR 13,776.55. The six Query 6 city leaders are Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), and Pune C0426 (INR 10,377.10). In eight rows, `RANK()` differs from `ROW_NUMBER()` because customers tie on city spend. Electronics had the highest category revenue (INR 1,219,248.10); Personal Care had the lowest (INR 168,433.95). The top daily revenue was INR 29,969.40 on 2025-06-16. Cumulative daily revenue ends at INR 2,998,668.80.

The pandas city/category pivot includes row and column margins in `pandas_city_category_revenue_pivot.csv`. Its category/city cells reconcile to the SQL result with a difference of INR 0.00, and its grand total equals completed-order revenue.

## Exploratory answers

1. Hyderabad generated the highest completed-order revenue (INR 528,589.90).
2. Electronics led category revenue (INR 1,219,248.10); Personal Care was lowest (INR 168,433.95).
3. Of 1,200 customers, 1,175 had completed purchases; their mean spend was INR 2,552.06 and median spend was INR 2,286.50.
4. The highest-revenue day was 2025-06-16 (INR 29,969.40); cumulative completed revenue ended at INR 2,998,668.80.

These answers contain fewer than 200 words in total. The validation script checks the explicit row-count, revenue, city/gender distribution, SQL row-count, ranking, database, and reconciliation checkpoints.

# Part B

## RFM analysis and customer segments

RFM includes the 1,175 customers with at least one completed order. At the snapshot of 2025-10-01 00:00:00, recency is the whole elapsed days since the latest completed order; frequency is the count of distinct completed order IDs; monetary is the sum of completed order amounts. The calculated checks are median/min/max recency 19/0/180 days, mean/max frequency 9.46/33 orders, and mean/max monetary INR 2,552.06/14,766.55.

The three RFM features were standardized before KMeans. Each K uses `n_init=10` and `random_state=42`; silhouette was computed on the same scaled matrix.

| K | Inertia | Silhouette |
| ---: | ---: | ---: |
| 2 | 1,820.761 | 0.4650 |
| 3 | 1,268.235 | 0.4452 |
| 4 | 743.071 | 0.5255 |
| 5 | 617.367 | 0.4805 |
| 6 | 503.171 | 0.4819 |
| 7 | 408.460 | 0.4163 |
| 8 | 366.503 | 0.4123 |

K=4 is selected: the inertia drop through K=4 marks a useful elbow, K=4 has the strongest silhouette, and four profiles remain practical to describe and act on. K=8 has lower inertia because adding clusters can only reduce or maintain within-cluster squared distances. That alone does not make it better: at K=8 the silhouette is lower and customers are split into more, less useful groups.

| Segment | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Suggested action |
| --- | ---: | ---: | ---: | ---: | --- |
| Core Customers | 436 | 29.86 | 3.77 | 1,057.62 | Use low-cost personalized nudges to encourage another purchase. |
| Premium | 138 | 25.12 | 6.77 | 7,129.70 | Offer tailored recognition and early access; protect the high-value relationship. |
| Frequent Loyalists | 419 | 9.77 | 19.40 | 3,410.02 | Reinforce the habit with convenience benefits and complementary recommendations. |
| At-Risk Customers | 182 | 110.04 | 2.20 | 685.99 | Prioritize a relevant win-back offer and monitor for renewed activity. |

Premium is assigned to the segment with the highest measured mean monetary value. KMeans IDs are arbitrary labels, not business rankings. The saved `outputs/customer_segments.csv` contains one row per eligible customer and the boolean `is_premium` flag. PCA on the scaled RFM features explains 0.6343 and 0.2180 of variance in its first two components, 0.8523 total.

## Lapse prediction

The model cohort has 1,156 customers with at least one completed order strictly before 2025-09-01 00:00:00. A customer is labeled lapsed when they have no completed order during September 1–30; 443 customers are lapsed (38.32%). Features use only pre-cutoff order history plus the customer table: recency, distinct completed-order frequency, completed spend, average discount, final-status cancellation rate, average completed delivery time, most frequent product category, city, age, gender and tenure. The latest order timestamp used for any feature was 2025-08-31 22:35:45. The label, whole-window segment ID and Premium flag are excluded from model features.

The split is stratified with `test_size=0.2` and `random_state=42`: 924 training rows and 232 untouched test rows. Each model uses a pipeline with numeric `StandardScaler` and `OneHotEncoder(drop="first", handle_unknown="ignore")` for city, gender and most frequent category. Preprocessing is fit within training folds. Five-fold F1 GridSearchCV selected logistic regression C=100 (best CV F1 0.651) and random forest max_depth=8/min_samples_leaf=10 (best CV F1 0.681; 45 fits).

| Model / result | Accuracy | Lapsed precision | Lapsed recall | Lapsed F1 | Confusion matrix `[[TN, FP], [FN, TP]]` |
| --- | ---: | ---: | ---: | ---: | --- |
| All-negative baseline | 0.616 | 0.000 | 0.000 | 0.000 | `[[143, 0], [89, 0]]` |
| Logistic regression, tuned | 0.728 | 0.644 | 0.652 | 0.648 | `[[111, 32], [31, 58]]` |
| Random forest, tuned | 0.737 | 0.659 | 0.652 | 0.655 | `[[113, 30], [31, 58]]` |

The random forest is selected because its training-only cross-validation F1 (0.681) exceeds logistic regression (0.651); the held-out test results are reported after that selection. Missing a likely lapse means losing an opportunity to retain that customer, so recall matters. However, false positives spend campaign budget and discounts on customers who would have stayed, so precision also matters. The selected model gives a modest precision and accuracy improvement while retaining the same recall as logistic regression, making it the better measured balance here. It still misses 31 of 89 lapsed test customers and flags 30 non-lapsed customers, so campaign thresholds and costs should be reviewed before deployment.

## Oversampling experiment

The training split initially contains 570 non-lapsed and 354 lapsed customers. All 924 original rows were kept and 216 lapsed rows were duplicated with replacement (`random_state=42`), yielding 570 per class. The best logistic-regression configuration was refit on that balanced training set and evaluated on the same untouched test set.

| Logistic regression | Precision | Recall | F1 | Confusion matrix `[[TN, FP], [FN, TP]]` |
| --- | ---: | ---: | ---: | --- |
| Original training split | 0.644 | 0.652 | 0.648 | `[[111, 32], [31, 58]]` |
| Oversampled training split | 0.589 | 0.854 | 0.697 | `[[90, 53], [13, 76]]` |

Recall changed most, increasing by about 0.202; the model catches more lapsed customers but generates more false positives (53 instead of 32). Precision fell by about 0.055, so a retention campaign should use a lower-cost message or a capacity-aware probability threshold for this higher-recall list.

## Feature interpretation and saved artifacts

The five leading random-forest importances are `numeric__frequency` (0.280), `numeric__monetary` (0.271), `numeric__recency_days` (0.115), `numeric__average_delivery_minutes` (0.061), and `numeric__cancellation_rate` (0.052). These are predictive associations, not causal effects.

The three largest-magnitude logistic coefficients are: frequency −1.698 (more prior completed orders associate with lower predicted lapse odds); Electronics −1.137 relative to the dropped Dairy category; and Grocery −0.449 relative to Dairy. Numeric coefficients use standardized features. One-hot coefficients compare each encoded level with its dropped reference; signs describe model associations and do not establish causation.

The selected random-forest pipeline is saved as `outputs/lapse_model.joblib`; it was reloaded and produced identical predictions on the test rows. `outputs/retention_list.csv` contains the top 10 Premium customers in descending predicted lapse probability, plus `actually_lapsed` for retrospective reporting. Nine of the top 10 were actually lapsed. The list uses the trained model’s scores; the actual label is not a model input.

**Post-hoc selection limitation:** Premium membership comes from whole-window RFM through September 30, which overlaps the lapse-label window. Therefore, using this flag to select the retention list is post-hoc and is not leakage-free evaluation of a Premium-only retention strategy. It is illustrative only.

# Part C

_Placeholder — not implemented yet._

# Part D

_Placeholder — not implemented yet._

# Limitations and Next Steps

The generated records are synthetic and deterministic; they do not represent actual Instamart customers or operations. The lapse model is evaluated on a stratified holdout from one synthetic time window and may not generalize to future customer behavior. The Premium retention selection is post-hoc because whole-window RFM includes the September label period; it must not be interpreted as leakage-free evaluation. Review the campaign cost trade-off and validate on a later time-based holdout before any operational use. Parts C and D are not yet implemented.
