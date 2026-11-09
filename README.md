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

## Support ticket triage with guardrails

Run `python src/part_c.py` from the repository root to perform all offline classification runs and save the artifacts. The final few-shot triage is written to `outputs/part_c/triage_results.csv`; the zero-shot budget demonstration is written to `outputs/triage_budget_demo.csv`. The implementation calls only the supplied `mock_llm.py`; each complete run resets its deterministic mock state. Run `python src/validate_part_c.py` to check the saved results against the raw ticket and segment data.

The audit used `keep_default_na=False` and found 164 tickets: 4 empty tickets were marked `skipped_empty`, 4 tickets over 400 estimated tokens were marked `flagged_long`, and 156 were eligible for classification. Email, phone, and order ID redactions were applied in the specified order; the audit counted 55 email, 91 phone, and 132 order ID matches. Neither empty nor long tickets were sent to the model. Each model user message was exactly the redacted ticket text. Prompt SHA-256 fingerprints are saved with run results so validation can independently confirm that property without saving raw ticket text.

The zero-shot and few-shot comparison runs each made calls for all 156 eligible tickets. Both produced 135 valid JSON classifications, 12 valid keyword fallbacks, and 9 outputs requiring human review. The zero-shot accuracy on its 147 valid classifications was 85.71% (126 correct); few-shot accuracy was 100% (147/147). Mean token use was 70.31 per valid zero-shot classification and 124.44 per valid few-shot classification. The zero-shot confusion matrix is saved in `outputs/part_c/zero_shot_confusion_matrix.csv`; its true-category rows and predicted-category columns are billing (38 correct), delivery (36 correct), technical (29 correct and 10 predicted billing), and fraud (23 correct and 11 predicted billing). Invalid outputs are excluded from that matrix and from accuracy, with their count and coverage reported separately in `metrics.json`.

The mock produced 19 transient failures in each full comparison run. All 19 retries recovered in both runs; the retry log records one row and the actual exponential-jitter delay for each retry. Invalid responses were never guessed: the 12 single-category keyword fallbacks were marked `classified_keyword`, and the remaining 9 malformed or unsupported responses were marked `flagged_invalid_output` with a blank category.

The zero-shot 6,000-token demonstration in `outputs/triage_budget_demo.csv` used 6,036 tokens from completed calls, classified 91 tickets (86 valid and 5 invalid), then marked the remaining 65 eligible tickets `budget_exhausted` without further calls. Its measured accuracy on valid classifications was 83.72%. The final few-shot run in `outputs/part_c/triage_results.csv` used the required 100,000-token budget and completed without exhaustion: 18,292 tokens total, or 124.44 per valid classification, with 147 valid classifications and 9 invalid outputs routed for review.

Few-shot classification used 18,292 tokens compared with 10,336 for zero-shot, an increase of approximately 77%. However, accuracy improved from 85.71% to 100% on valid classifications. For this support-triage experiment, the additional token cost is justified when incorrect ticket routing, especially for fraud, has a high operational cost. In a production system, this trade-off should be evaluated against actual model pricing and human-review costs.

Final routing from the 164-row run contained 17 `human_review` cases (4 empty, 4 long and 9 invalid outputs), 34 `security_team` cases for valid fraud classifications, 11 `premium_desk` cases for valid non-fraud Premium customers, and 102 `support_queue` cases. Priorities were HIGH for the 34 fraud and 11 Premium cases, and NORMAL for the other 119. Missing segment records are treated as non-Premium. The detailed per-ticket audit, independent comparison runs, retry log, final classifications, and structured metrics are saved under `outputs/part_c/`; the budget demonstration is at `outputs/triage_budget_demo.csv`.

The Part C validator passed all privacy, guardrail, retry, response validation, accuracy reconciliation, budget, and routing checks. Accuracy is measured on valid classifications; the invalid-output count and coverage make the excluded human-review cases explicit. These results describe the deterministic offline mock, not a hosted model.

# Part D

## Policy RAG and refund agent

Run `python src/part_d.py` from the repository root to build the index, evaluate policy Q&A, and run the seven offline refund scenarios. Run `python src/validate_part_d.py` to validate the saved artifacts against the supplied policy evaluation questions and scenario expectations. No external API or network is used.

Policy files are split deterministically into one sentence per chunk, in filename and sentence order. TF-IDF uses unigrams and bigrams with English stop words removed; retrieval returns the top three chunks. The minimum retrieval score is zero, so questions with no lexical match abstain before calling the mock assistant. On the 14 supplied questions, top-1 document accuracy and top-3 document recall were both 100% on the 12 answerable questions. Both `NONE` questions were abstained correctly. The mock policy assistant returned the expected fact for all 12 answerable questions, using 1,703 tokens total.

The refund agent uses local read-only order lookup and allow-listed policy, calculation, issuance, and escalation tools. The seven scenarios matched all expected outcomes: S1 and S2 were refunded (INR 128.25 and INR 135.00); S3 paused for human approval at INR 1,999; S4 and S5 were denied; S6 was not found; and S7 safely stopped on an ungrounded policy-search call and fell back for human review. The agent made 19 tool calls across the scenarios. It will not issue a refund above INR 1,000 without an explicit approval.

S3 is saved in `outputs/part_d/pending_escalations.json`. After a human decision, resume it with `python src/part_d.py --resume-scenario S3 --decision approve --approved-by "support lead"` or use `--decision reject`. Approval records the approver and passes issuance through the same guarded, idempotent `issue_refund()` function; rejection records that a customer callback is required. The validator checks both resume paths without taking a decision for the saved scenario. The scenario fixture has no request timestamp, so the offline run uses the verified delivery timestamp as a fixture request time; live callers must supply a trusted timestamp. This limitation is recorded in `outputs/part_d/metrics.json`.

Artifacts are under `outputs/part_d/`: policy chunks, retrieval ranks and evaluation, RAG answers, agent outcomes, tool traces, pending escalations, the idempotent `refund_ledger.json`, and structured metrics. The Part D validator prints `ALL PART D CHECKPOINTS PASSED` only after the saved outputs reconcile to the input files and safety checks.

# Limitations and Next Steps

The generated records are synthetic and deterministic; they do not represent actual Instamart customers or operations. The lapse model is evaluated on a stratified holdout from one synthetic time window and may not generalize to future customer behavior. The Premium retention selection is post-hoc because whole-window RFM includes the September label period; it must not be interpreted as leakage-free evaluation. Review the campaign cost trade-off and validate on a later time-based holdout before any operational use. Parts C and D are offline prototypes, not production integrations.
