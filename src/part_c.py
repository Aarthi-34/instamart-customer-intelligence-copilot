"""Part C: offline support ticket triage with privacy and routing guardrails."""

from __future__ import annotations

import json
import hashlib
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mock_llm  # noqa: E402

RAW_TICKETS = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS = ROOT / "outputs" / "customer_segments.csv"
OUT = ROOT / "outputs"
PART_C_OUT = OUT / "part_c"
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
TOKEN_BUDGET = 6000
FINAL_BUDGET = 100000
BASE_DELAY = 0.05
MAX_RETRIES = 3
RANDOM_SEED = 42
REDACTIONS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO_SYSTEM = (
    "CLASSIFY each support ticket into exactly one of these categories: billing, delivery, "
    "technical, fraud. Return JSON only, with exactly this shape: "
    '{"category":"...","reason":"..."}. Do not add markdown or other text.'
)
FEW_SYSTEM = (
    "CLASSIFY each support ticket into exactly one of these categories: billing, delivery, "
    "technical, fraud. Return JSON only, with exactly this shape: "
    '{"category":"...","reason":"..."}. Do not add markdown or other text.\n'
    "EXAMPLES:\n"
    'Ticket: "I was charged twice and need a refund." -> {"category":"billing","reason":"duplicate charge"}\n'
    'Ticket: "The rider says delivered but I did not get it." -> {"category":"delivery","reason":"missing delivery"}\n'
    'Ticket: "There is an unauthorized transaction on my account." -> {"category":"fraud","reason":"unauthorized transaction"}'
)


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    current = text
    for name, pattern, replacement in REDACTIONS:
        current, n = pattern.subn(replacement, current)
        counts[name] = n
    return current, counts


def classify_response(text: str | None) -> tuple[str, str, str]:
    """Return (status, category, reason), rejecting malformed/unknown categories."""
    if text is None:
        return "flagged_invalid_output", "", ""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", text.lower())
        found = [word for word in words if word in CATEGORIES]
        if len(found) == 1:
            return "classified_keyword", found[0], ""
        return "flagged_invalid_output", "", ""
    if isinstance(parsed, dict) and parsed.get("category") in CATEGORIES:
        reason = parsed.get("reason", "")
        return "classified_json", parsed["category"], str(reason)
    return "flagged_invalid_output", "", ""


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    tickets = pd.read_csv(RAW_TICKETS, keep_default_na=False, dtype=str)
    segments = pd.read_csv(SEGMENTS, keep_default_na=False)
    required_tickets = {"ticket_id", "customer_id", "order_id", "text", "true_category"}
    required_segments = {"customer_id", "segment_name"}
    if not required_tickets.issubset(tickets.columns):
        raise ValueError(f"Ticket file missing columns: {sorted(required_tickets - set(tickets.columns))}")
    if not required_segments.issubset(segments.columns):
        raise ValueError(f"Customer segments missing columns: {sorted(required_segments - set(segments.columns))}")
    return tickets, segments


def audit_tickets(tickets: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    audit_rows = []
    prepared = []
    for row in tickets.to_dict("records"):
        raw = str(row["text"])
        estimate = mock_llm.est_tokens(raw)
        redacted, counts = redact(raw)
        status = "skipped_empty" if not raw.strip() else (
            "flagged_long" if estimate > 400 else "ready"
        )
        audit_rows.append({
            "ticket_id": row["ticket_id"],
            "estimated_tokens": estimate,
            "status": status,
            "email_redactions": counts["email"],
            "phone_redactions": counts["phone"],
            "order_id_redactions": counts["order_id"],
        })
        prepared.append({**row, "estimated_tokens": estimate, "initial_status": status,
                         "redacted_text": redacted, **counts})
    return pd.DataFrame(audit_rows), prepared


def run_classification(prepared: list[dict], system_prompt: str, run_name: str,
                       token_budget: int | None) -> tuple[pd.DataFrame, list[dict], int]:
    # Reset directly before each independent complete classification run.
    mock_llm.reset()
    rng = random.Random(RANDOM_SEED)
    used_tokens = 0
    results: list[dict] = []
    retry_log: list[dict] = []
    for ticket in prepared:
        base = {"ticket_id": ticket["ticket_id"], "customer_id": ticket["customer_id"],
                "true_category": ticket["true_category"], "estimated_tokens": ticket["estimated_tokens"],
                "tokens_used": 0, "status": ticket["initial_status"], "category": "", "reason": "",
                "prompt_sha256": ""}
        if ticket["initial_status"] != "ready":
            results.append(base)
            continue
        if token_budget is not None and used_tokens >= token_budget:
            base["status"] = "budget_exhausted"
            results.append(base)
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            # The user prompt contains the redacted ticket text and nothing else.
            {"role": "user", "content": ticket["redacted_text"]},
        ]
        base["prompt_sha256"] = hashlib.sha256(messages[-1]["content"].encode("utf-8")).hexdigest()
        response = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = mock_llm.chat(messages)
                break
            except mock_llm.TransientLLMError as exc:
                if attempt >= MAX_RETRIES:
                    base["status"] = "flagged_invalid_output"
                    retry_log.append({"run": run_name, "ticket_id": ticket["ticket_id"],
                                      "retry_number": attempt + 1, "delay_seconds": 0.0,
                                      "event": f"retry_limit_exceeded: {exc}"})
                    break
                delay = BASE_DELAY * (2 ** attempt) * rng.uniform(0.9, 1.15)
                retry_log.append({"run": run_name, "ticket_id": ticket["ticket_id"],
                                  "retry_number": attempt + 1, "delay_seconds": delay,
                                  "event": f"retry: {exc}"})
                time.sleep(delay)
        if response is not None:
            usage = response.usage
            call_tokens = int(usage.total_tokens)
            used_tokens += call_tokens
            base["tokens_used"] = call_tokens
            status, category, reason = classify_response(response.choices[0].message.content)
            base.update(status=status, category=category, reason=reason)
        results.append(base)
    return pd.DataFrame(results), retry_log, used_tokens


def evaluate(results: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    eligible = results[results["status"].isin(["classified_json", "classified_keyword", "flagged_invalid_output"])]
    valid = results[results["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((valid["category"] == valid["true_category"]).sum())
    matrix = pd.crosstab(
        pd.Categorical(valid["true_category"], categories=CATEGORIES),
        pd.Categorical(valid["category"], categories=CATEGORIES),
        dropna=False,
    ).reindex(index=CATEGORIES, columns=CATEGORIES, fill_value=0)
    matrix.index.name = "true_category"
    matrix.columns.name = "predicted_category"
    metrics = {
        "eligible_tickets": int((results["status"] != "skipped_empty").sum()),
        "model_eligible_tickets": int((results["estimated_tokens"].le(400) & results["estimated_tokens"].gt(0)).sum()),
        "valid_classifications": int(len(valid)),
        "json_classifications": int((results["status"] == "classified_json").sum()),
        "keyword_classifications": int((results["status"] == "classified_keyword").sum()),
        "invalid_outputs": int((results["status"] == "flagged_invalid_output").sum()),
        "accuracy_on_valid_classifications": (correct / len(valid)) if len(valid) else None,
        "valid_classification_coverage_of_model_eligible": (len(valid) / max(1, int(results["estimated_tokens"].le(400).sum() - (results["status"] == "skipped_empty").sum()))),
        "tokens_total": int(results["tokens_used"].sum()),
        "tokens_per_valid_classification": (float(results["tokens_used"].sum() / len(valid)) if len(valid) else None),
        "confusion_matrix_valid_outputs_only": matrix.values.astype(int).tolist(),
        "confusion_matrix_categories": CATEGORIES,
    }
    return metrics, matrix.reset_index()


def route_final(results: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    segment_map = segments.drop_duplicates("customer_id").set_index("customer_id")["segment_name"].to_dict()
    final_rows = []
    for row in results.to_dict("records"):
        segment = segment_map.get(row["customer_id"], "")
        if row["status"] in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}:
            priority, route = "NORMAL", "human_review"
        elif row["category"] == "fraud":
            priority, route = "HIGH", "security_team"
        elif segment == "Premium":
            priority, route = "HIGH", "premium_desk"
        else:
            priority, route = "NORMAL", "support_queue"
        final_rows.append({
            "ticket_id": row["ticket_id"], "customer_id": row["customer_id"],
            "status": row["status"], "category": row["category"], "segment_name": segment,
            "priority": priority, "route": route, "tokens_used": row["tokens_used"],
        })
    return pd.DataFrame(final_rows, columns=["ticket_id", "customer_id", "status", "category",
                                             "segment_name", "priority", "route", "tokens_used"])


def main() -> None:
    PART_C_OUT.mkdir(parents=True, exist_ok=True)
    tickets, segments = load_inputs()
    audit, prepared = audit_tickets(tickets)
    audit.to_csv(PART_C_OUT / "ticket_audit.csv", index=False)
    redaction_stats = {
        "tickets": int(len(tickets)),
        "empty_tickets": int((audit["status"] == "skipped_empty").sum()),
        "long_tickets": int((audit["status"] == "flagged_long").sum()),
        "model_eligible_tickets": int((audit["status"] == "ready").sum()),
        "email_redactions": int(audit["email_redactions"].sum()),
        "phone_redactions": int(audit["phone_redactions"].sum()),
        "order_id_redactions": int(audit["order_id_redactions"].sum()),
    }
    all_retries: list[dict] = []
    all_metrics = {"ticket_audit_and_redaction": redaction_stats, "runs": {}}
    run_specs = [
        ("zero_shot", ZERO_SYSTEM, None),
        ("few_shot", FEW_SYSTEM, None),
        ("zero_shot_budget_demo", ZERO_SYSTEM, TOKEN_BUDGET),
        ("final_few_shot", FEW_SYSTEM, FINAL_BUDGET),
    ]
    saved: dict[str, pd.DataFrame] = {}
    for run_name, prompt, budget in run_specs:
        result, retries, tokens = run_classification(prepared, prompt, run_name, budget)
        saved[run_name] = result
        all_retries.extend(retries)
        metrics, matrix = evaluate(result)
        metrics.update({"token_budget": budget, "budget_exhausted_tickets": int((result["status"] == "budget_exhausted").sum()),
                        "run_total_tokens_from_usage": tokens})
        all_metrics["runs"][run_name] = metrics
        if run_name in {"zero_shot", "few_shot"}:
            result.to_csv(PART_C_OUT / f"{run_name}_results.csv", index=False)
        if run_name == "zero_shot":
            matrix.to_csv(PART_C_OUT / "zero_shot_confusion_matrix.csv", index=False)
        if run_name == "zero_shot_budget_demo":
            result.to_csv(OUT / "triage_budget_demo.csv", index=False)
        if run_name == "final_few_shot":
            result.to_csv(PART_C_OUT / "final_few_shot_classifications.csv", index=False)
            route_final(result, segments).to_csv(PART_C_OUT / "triage_results.csv", index=False)
    pd.DataFrame(all_retries, columns=["run", "ticket_id", "retry_number", "delay_seconds", "event"]
                 ).to_csv(PART_C_OUT / "retry_logs.csv", index=False)
    pd.DataFrame([{ "run": name, **{k: v for k, v in metrics.items() if k != "confusion_matrix_valid_outputs_only" and k != "confusion_matrix_categories"}}
                  for name, metrics in all_metrics["runs"].items()]).to_csv(PART_C_OUT / "comparison_metrics.csv", index=False)
    with (PART_C_OUT / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(json.dumps({"outputs": str(PART_C_OUT), "triage": str(PART_C_OUT / "triage_results.csv"),
                      "budget_demo": str(OUT / "triage_budget_demo.csv"),
                      "redaction": redaction_stats,
                      "runs": {name: {"accuracy": m["accuracy_on_valid_classifications"],
                                      "valid": m["valid_classifications"], "invalid": m["invalid_outputs"],
                                      "budget_exhausted": m["budget_exhausted_tickets"],
                                      "tokens": m["tokens_total"]}
                               for name, m in all_metrics["runs"].items()}}, indent=2))


if __name__ == "__main__":
    main()
