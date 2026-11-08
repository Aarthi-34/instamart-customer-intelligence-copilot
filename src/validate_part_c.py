"""Validate Part C results against ticket, segment, and saved run artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part_c import CATEGORIES, FEW_SYSTEM, REDACTIONS, ZERO_SYSTEM, audit_tickets, load_inputs  # noqa: E402
import mock_llm  # noqa: E402

OUT = ROOT / "outputs"
PC = OUT / "part_c"


def check(condition: bool, description: str) -> None:
    if not condition:
        raise AssertionError(description)


def main() -> None:
    check("CLASSIFY" in ZERO_SYSTEM and "CLASSIFY" in FEW_SYSTEM, "classification system prompt marker missing")
    check("EXAMPLES:" not in ZERO_SYSTEM, "zero-shot prompt must not contain examples")
    check("EXAMPLES:" in FEW_SYSTEM and FEW_SYSTEM.lower().count("ticket:") >= 2
          and '"category":"fraud"' in FEW_SYSTEM, "few-shot prompt examples are incomplete")
    check('{"category":"...","reason":"..."}' in ZERO_SYSTEM
          and '{"category":"...","reason":"..."}' in FEW_SYSTEM, "JSON response format missing from system prompt")
    tickets, segments = load_inputs()
    audit = pd.read_csv(PC / "ticket_audit.csv", keep_default_na=False)
    expected_audit, prepared = audit_tickets(tickets)
    check(len(tickets) == 164, f"raw ticket count is {len(tickets)}, expected 164")
    check(len(audit) == 164 and audit.ticket_id.is_unique, "ticket audit must have one row per raw ticket")
    check(audit.equals(expected_audit), "saved audit does not match raw empty/long/token/redaction calculations")
    check((audit.status == "skipped_empty").sum() == (tickets.text.str.strip() == "").sum(), "empty ticket handling mismatch")
    check((audit.status == "flagged_long").sum() == (audit.estimated_tokens > 400).sum(), "long ticket handling mismatch")
    check(not ((audit.status == "skipped_empty") & (audit.estimated_tokens > 0)).any(), "empty tickets have token estimates")

    # Verify ordered PII redaction and privacy fingerprints for every actual prompt.
    prepared_by_id = {item["ticket_id"]: item for item in prepared}
    email_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    phone_re = re.compile(r"\b[6-9]\d{9}\b")
    order_re = re.compile(r"ORD-\d{6}")
    for run_name, file_path in [("zero_shot", PC / "zero_shot_results.csv"),
                                ("few_shot", PC / "few_shot_results.csv"),
                                ("zero_shot_budget_demo", OUT / "triage_budget_demo.csv"),
                                ("final_few_shot", PC / "final_few_shot_classifications.csv")]:
        results = pd.read_csv(file_path, keep_default_na=False)
        for row in results.to_dict("records"):
            if row["prompt_sha256"]:
                prepared_ticket = prepared_by_id[row["ticket_id"]]
                expected_prompt = prepared_ticket["redacted_text"]
                digest = hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
                check(row["prompt_sha256"] == digest, f"{run_name}: prompt is not exactly the ordered-redacted ticket text")
                check(not email_re.search(expected_prompt) and not phone_re.search(expected_prompt)
                      and not order_re.search(expected_prompt), f"{run_name}: PII remained in a model prompt")
            elif row["status"] in {"classified_json", "classified_keyword"}:
                raise AssertionError(f"{run_name}: model-classified ticket missing prompt hash")
            if row["status"] in {"skipped_empty", "flagged_long", "budget_exhausted"}:
                check(not row["prompt_sha256"], f"{run_name}: gated ticket was sent to model")

    retry_log = pd.read_csv(PC / "retry_logs.csv", keep_default_na=False)
    check(set(retry_log.columns) == {"run", "ticket_id", "retry_number", "delay_seconds", "event"}, "retry log schema mismatch")
    if len(retry_log):
        counts = retry_log.groupby(["run", "ticket_id"]).size()
        check(int(counts.max()) <= 3, "ticket exceeded three retries")
        for row in retry_log.to_dict("records"):
            n = int(row["retry_number"])
            check(1 <= n <= 3, "retry number outside allowed range")
            expected_low = 0.05 * (2 ** (n - 1)) * 0.9
            expected_high = 0.05 * (2 ** (n - 1)) * 1.15
            check(expected_low <= float(row["delay_seconds"]) <= expected_high,
                  "retry delay does not use exponential backoff and jitter")
            check(row["event"].startswith("retry:"), "unexpected retry log entry")

    metrics = json.loads((PC / "metrics.json").read_text(encoding="utf-8"))
    for run_name in ("zero_shot", "few_shot", "zero_shot_budget_demo", "final_few_shot"):
        file_path = {"zero_shot": PC / "zero_shot_results.csv",
                     "few_shot": PC / "few_shot_results.csv",
                     "zero_shot_budget_demo": OUT / "triage_budget_demo.csv",
                     "final_few_shot": PC / "final_few_shot_classifications.csv"}[run_name]
        results = pd.read_csv(file_path, keep_default_na=False)
        m = metrics["runs"][run_name]
        valid = results[results.status.isin(["classified_json", "classified_keyword"])]
        correct = int((valid.category == valid.true_category).sum())
        check(m["valid_classifications"] == len(valid), f"{run_name}: valid classification metric mismatch")
        check(m["invalid_outputs"] == int((results.status == "flagged_invalid_output").sum()), f"{run_name}: invalid-output metric mismatch")
        check(m["tokens_total"] == int(results.tokens_used.sum()) == m["run_total_tokens_from_usage"], f"{run_name}: token usage mismatch")
        expected_accuracy = correct / len(valid) if len(valid) else None
        check(m["accuracy_on_valid_classifications"] == expected_accuracy, f"{run_name}: accuracy not computed from saved outputs")
        check(m["valid_classifications"] > 0 and expected_accuracy is not None, f"{run_name}: no classified tickets")
        check(m["json_classifications"] == int((results.status == "classified_json").sum()), f"{run_name}: JSON response status mismatch")
        check(m["keyword_classifications"] == int((results.status == "classified_keyword").sum()), f"{run_name}: keyword fallback status mismatch")
        check((results.loc[results.status == "flagged_invalid_output", "category"] == "").all(),
              f"{run_name}: invalid category was guessed")
        check(float(m["tokens_per_valid_classification"]) == float(results.tokens_used.sum() / len(valid)),
              f"{run_name}: per-classification token metric mismatch")
        check(0.0 <= expected_accuracy <= 1.0, f"{run_name}: accuracy outside valid range")
        retried_ids = retry_log.loc[retry_log.run == run_name, "ticket_id"].unique().tolist()
        if retried_ids:
            recovered = results[results.ticket_id.isin(retried_ids)]
            check(len(recovered) == len(retried_ids)
                  and not recovered.status.isin(["budget_exhausted", "skipped_empty", "flagged_long"]).any(),
                  f"{run_name}: retry did not recover into a model response")

    zero = metrics["runs"]["zero_shot"]
    few = metrics["runs"]["few_shot"]
    check(zero["valid_classifications"] == few["valid_classifications"], "comparison runs are not full independent runs")
    matrix = pd.read_csv(PC / "zero_shot_confusion_matrix.csv")
    check(matrix.columns.tolist() == ["true_category", *CATEGORIES], "confusion matrix schema mismatch")
    check(int(matrix[CATEGORIES].to_numpy().sum()) == zero["valid_classifications"], "confusion matrix does not reconcile to zero-shot classifications")
    check(int(retry_log[retry_log.run.isin(["zero_shot", "few_shot", "zero_shot_budget_demo", "final_few_shot"])].shape[0]) > 0,
          "mock transient failures did not produce retry logs")
    check(len(retry_log[retry_log.run == "zero_shot"]) > 0 and len(retry_log[retry_log.run == "few_shot"]) > 0,
          "retry recovery was not observed in both comparison runs")

    demo = pd.read_csv(OUT / "triage_budget_demo.csv", keep_default_na=False)
    demo_metrics = metrics["runs"]["zero_shot_budget_demo"]
    check(demo_metrics["token_budget"] == 6000, "zero-shot demonstration budget was changed")
    check(demo_metrics["budget_exhausted_tickets"] > 0, "6000-token run did not demonstrate budget exhaustion")
    check(demo_metrics["tokens_total"] >= 6000, "budget exhaustion occurred before reaching budget")
    exhausted_positions = demo.index[demo.status == "budget_exhausted"].tolist()
    check(bool(exhausted_positions), "budget exhaustion status is missing")
    first_exhausted = exhausted_positions[0]
    after_budget = demo.loc[first_exhausted:]
    check(not after_budget.status.isin(["classified_json", "classified_keyword", "flagged_invalid_output"]).any(),
          "model calls continued after budget exhaustion")
    check((after_budget.loc[after_budget.estimated_tokens.between(1, 400), "status"] == "budget_exhausted").all(),
          "eligible tickets after budget exhaustion were not routed to human review")
    final_metrics = metrics["runs"]["final_few_shot"]
    check(final_metrics["token_budget"] == 100000 and final_metrics["budget_exhausted_tickets"] == 0,
          "final few-shot run did not use the required 100000-token budget")

    final_classifications = pd.read_csv(PC / "final_few_shot_classifications.csv", keep_default_na=False)
    triage = pd.read_csv(PC / "triage_results.csv", keep_default_na=False)
    check(len(triage) == 164 and triage.ticket_id.is_unique, "final triage must have one row per ticket")
    check(triage.columns.tolist() == ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"], "final triage columns mismatch")
    joined = triage.merge(final_classifications, on=["ticket_id", "customer_id"], suffixes=("_triage", "_class"), validate="one_to_one")
    segment_map = segments.drop_duplicates("customer_id").set_index("customer_id").segment_name.to_dict()
    for row in joined.to_dict("records"):
        check(row["status_triage"] == row["status_class"] and row["category_triage"] == row["category_class"], "final triage does not match final classification")
        check(row["segment_name"] == segment_map.get(row["customer_id"], ""), "segment lookup mismatch")
        status, category, segment = row["status_triage"], row["category_triage"], row["segment_name"]
        if status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}:
            expected = ("NORMAL", "human_review")
        elif category == "fraud":
            expected = ("HIGH", "security_team")
        elif segment == "Premium":
            expected = ("HIGH", "premium_desk")
        else:
            expected = ("NORMAL", "support_queue")
        check((row["priority"], row["route"]) == expected, f"routing mismatch for {row['ticket_id']}")
        check((not category) if status == "flagged_invalid_output" else True, "invalid output category must stay blank")

    # Validate redaction counters using the supplied ordered patterns, without exposing raw text in artifacts.
    totals = {name: 0 for name, _, _ in REDACTIONS}
    for raw in tickets.text.astype(str):
        current = raw
        for name, pattern, replacement in REDACTIONS:
            current, count = pattern.subn(replacement, current)
            totals[name] += count
    saved_stats = metrics["ticket_audit_and_redaction"]
    check(saved_stats["tickets"] == 164, "ticket audit metrics ticket count mismatch")
    for name in totals:
        key = f"{name}_redactions"
        check(saved_stats[key] == totals[name], f"redaction count mismatch for {name}")

    print("ALL PART C CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
