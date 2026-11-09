"""Part D: deterministic policy RAG and offline refund-agent orchestration."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mock_llm  # noqa: E402

POLICY_DIR = ROOT / "data" / "policies"
POLICY_EVAL = ROOT / "data" / "raw" / "policy_eval.csv"
SCENARIOS = ROOT / "data" / "raw" / "agent_scenarios.csv"
ORDERS = ROOT / "data" / "clean" / "orders.csv"
OUT = ROOT / "outputs" / "part_d"
TOP_K = 3
MIN_RETRIEVAL_SCORE = 0.0
MAX_AGENT_STEPS = 6

POLICY_SYSTEM = (
    "You are a POLICY ASSISTANT. Answer only from the supplied policy context. "
    "If the context does not contain the answer, say you cannot find it in the policy. "
    "Do not invent policy details."
)
AGENT_SYSTEM = (
    "You are an AGENT handling refund requests. Use the available tools in order, "
    "follow policy and tool results, and never claim a refund was issued unless the "
    "issue_refund tool confirms it. Escalate refunds above INR 1000 to a human."
)

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_order", "description": "Look up the customer's order by ID.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "search_policy", "description": "Retrieve refund policy passages.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculate_refund", "description": "Calculate an eligible completed-order late-delivery refund.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, "required": ["order_amount", "delivery_minutes"]}}},
    {"type": "function", "function": {"name": "issue_refund", "description": "Issue a refund no greater than INR 1000, or an explicitly approved larger refund.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}}},
    {"type": "function", "function": {"name": "escalate_to_human", "description": "Pause a refund above INR 1000 for human approval.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}}},
]


def load_policy_documents(policy_dir: Path = POLICY_DIR) -> dict[str, str]:
    """Load policy text without applying a chunking strategy."""
    return {path.name: path.read_text(encoding="utf-8").strip()
            for path in sorted(policy_dir.glob("*_policy.txt"))}


def chunk_policy_documents(documents: dict[str, str]) -> pd.DataFrame:
    """Split loaded policies into the existing one-sentence, attributed chunks."""
    rows = []
    for filename, text in sorted(documents.items()):
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        for index, sentence in enumerate(sentences):
            rows.append({"chunk_id": f"{Path(filename).stem}_{index:02d}",
                         "source_file": filename, "chunk_index": index,
                         "text": sentence})
    if not rows:
        raise ValueError("No policy text was supplied for chunking")
    return pd.DataFrame(rows)


def load_policy_chunks(policy_dir: Path = POLICY_DIR) -> pd.DataFrame:
    """Compatibility wrapper for callers that need the complete policy corpus."""
    documents = load_policy_documents(policy_dir)
    if not documents:
        raise FileNotFoundError(f"No policy files found in {policy_dir}")
    return chunk_policy_documents(documents)


def build_index(chunks: pd.DataFrame) -> tuple[TfidfVectorizer, Any]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", lowercase=True)
    matrix = vectorizer.fit_transform(chunks["text"].tolist())
    return vectorizer, matrix


def retrieve(query: str, chunks: pd.DataFrame, vectorizer: TfidfVectorizer,
             matrix: Any, top_k: int = TOP_K) -> list[dict]:
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]),
                    str(chunks.iloc[i]["source_file"]), int(chunks.iloc[i]["chunk_index"])))
    return [{**chunks.iloc[i].to_dict(), "score": float(scores[i]), "rank": rank + 1}
            for rank, i in enumerate(ranked[:top_k])]


def evaluate_retrieval(eval_rows: pd.DataFrame, chunks: pd.DataFrame,
                       vectorizer: TfidfVectorizer, matrix: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    details = []
    for row in eval_rows.to_dict("records"):
        matches = retrieve(row["question"], chunks, vectorizer, matrix)
        answerable = row["expected_doc"] != "NONE"
        top_doc = matches[0]["source_file"] if matches else ""
        best_score = matches[0]["score"] if matches else 0.0
        abstained = best_score <= MIN_RETRIEVAL_SCORE
        summaries.append({
            "question": row["question"], "expected_doc": row["expected_doc"],
            "expected_fact": row["expected_fact"], "top1_doc": top_doc,
            "top1_score": best_score,
            "expected_doc_top1": bool(answerable and top_doc == row["expected_doc"]),
            "expected_doc_top3": bool(answerable and any(m["source_file"] == row["expected_doc"] for m in matches)),
            "is_unanswerable": not answerable, "abstained": bool(abstained),
            "abstention_correct": bool((not answerable) == abstained),
        })
        for match in matches:
            details.append({"question": row["question"], "expected_doc": row["expected_doc"],
                            **match})
    return pd.DataFrame(summaries), pd.DataFrame(details)


def answer_policy_questions(eval_rows: pd.DataFrame, retrieved_chunks: pd.DataFrame) -> pd.DataFrame:
    """Generate answers from already-ranked context; retrieval is not repeated here."""
    mock_llm.reset()
    output = []
    by_question = {question: group.sort_values("rank")
                   for question, group in retrieved_chunks.groupby("question")}
    for row in eval_rows.to_dict("records"):
        group = by_question.get(row["question"], pd.DataFrame())
        ranked = group.to_dict("records") if len(group) else []
        top_score = ranked[0]["score"] if ranked else 0.0
        if top_score <= MIN_RETRIEVAL_SCORE:
            answer = "I could not find that information in the policy documents."
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            status = "abstained_no_retrieval_match"
        else:
            context = "\n".join(f"[{hit['source_file']}] {hit['text']}" for hit in ranked)
            messages = [
                {"role": "system", "content": POLICY_SYSTEM},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION: {row['question']}"},
            ]
            response = mock_llm.chat(messages)
            answer = response.choices[0].message.content or ""
            usage = {"prompt_tokens": int(response.usage.prompt_tokens),
                     "completion_tokens": int(response.usage.completion_tokens),
                     "total_tokens": int(response.usage.total_tokens)}
            status = "answered_from_retrieved_context"
        expected_fact = str(row["expected_fact"])
        expected_doc = str(row["expected_doc"])
        relevant = by_question.get(row["question"], pd.DataFrame())
        sources = sorted(set(relevant["source_file"].tolist())) if len(relevant) else []
        fact_found = bool(expected_fact and expected_fact.lower() in answer.lower())
        output.append({
            "question": row["question"], "expected_doc": expected_doc,
            "expected_fact": expected_fact, "answer": answer,
            "status": status, "retrieved_sources": "|".join(sources),
            "expected_doc_retrieved": bool(expected_doc != "NONE" and expected_doc in sources),
            "expected_fact_found": fact_found if expected_doc != "NONE" else False,
            "unanswerable_correctly_abstained": bool(expected_doc == "NONE" and status.startswith("abstained")),
            **usage,
        })
    return pd.DataFrame(output)


def load_orders() -> pd.DataFrame:
    if not ORDERS.exists():
        raise FileNotFoundError(f"Clean orders are required for Part D: {ORDERS}")
    frame = pd.read_csv(ORDERS, keep_default_na=False)
    required = {"order_id", "customer_id", "status", "order_amount", "delivery_minutes", "order_ts"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Clean orders are missing columns: {sorted(missing)}")
    return frame


def get_order(order_id: str, customer_id: str, orders: pd.DataFrame) -> dict:
    """Return a verified order only when it belongs to the requesting customer."""
    found = orders.loc[orders.order_id.astype(str).eq(str(order_id))]
    if found.empty:
        return {"error": "order_not_found", "order_id": str(order_id)}
    order = found.iloc[0].to_dict()
    if str(order["customer_id"]) != str(customer_id):
        return {"error": "customer_order_mismatch", "order_id": str(order_id)}
    amount = pd.to_numeric(pd.Series([order["order_amount"]]), errors="coerce").iloc[0]
    minutes = pd.to_numeric(pd.Series([order["delivery_minutes"]]), errors="coerce").iloc[0]
    order_ts = pd.to_datetime(order["order_ts"], errors="coerce")
    delivery_ts = None
    if pd.notna(order_ts) and pd.notna(minutes):
        delivery_ts = order_ts + pd.to_timedelta(float(minutes), unit="m")
    return {
        "verified": True,
        "order_id": str(order["order_id"]),
        "customer_id": str(order["customer_id"]),
        "status": str(order["status"]),
        "order_amount": float(amount) if pd.notna(amount) else None,
        "delivery_minutes": float(minutes) if pd.notna(minutes) else None,
        "order_ts": order_ts.isoformat() if pd.notna(order_ts) else None,
        "delivery_timestamp": delivery_ts.isoformat() if delivery_ts is not None else None,
    }


def search_policy(query: str, chunks: pd.DataFrame, vectorizer: TfidfVectorizer,
                  matrix: Any) -> dict:
    """Return top refund-policy passages for a tool query."""
    hits = retrieve(query, chunks, vectorizer, matrix, top_k=TOP_K)
    return {"matches": [{"source_file": hit["source_file"], "text": hit["text"], "score": hit["score"]}
                         for hit in hits if hit["score"] > MIN_RETRIEVAL_SCORE]}


def calculate_refund(arguments: dict, active_order: dict | None,
                     request_received_at: str | None) -> dict:
    """Compute eligibility solely from verified order data and trusted request time."""
    if not active_order or not active_order.get("verified"):
        return {"error": "verified_order_required", "refund_inr": 0.0}
    requested_order_id = arguments.get("order_id")
    order_id = str(active_order["order_id"])
    if requested_order_id is not None and str(requested_order_id) != order_id:
        return {"error": "order_id_mismatch", "order_id": order_id, "refund_inr": 0.0}
    if active_order.get("status") != "completed":
        return {"order_id": order_id, "eligible": False, "refund_inr": 0.0,
                "reason": "only_completed_orders_are_eligible"}
    amount = active_order.get("order_amount")
    minutes = active_order.get("delivery_minutes")
    delivery_value = active_order.get("delivery_timestamp")
    if amount is None or minutes is None or delivery_value is None:
        return {"order_id": order_id, "eligible": False, "refund_inr": 0.0,
                "reason": "verified_delivery_data_missing"}
    delivery_ts = pd.Timestamp(delivery_value)
    if request_received_at is None:
        return {"order_id": order_id, "eligible": False, "refund_inr": 0.0,
                "reason": "trusted_request_timestamp_missing"}
    request_ts = pd.Timestamp(request_received_at)
    elapsed = request_ts - delivery_ts
    if elapsed < pd.Timedelta(0) or elapsed > pd.Timedelta(days=7):
        return {"order_id": order_id, "eligible": False, "refund_inr": 0.0,
                "delivery_timestamp": delivery_ts.isoformat(), "request_received_at": request_ts.isoformat(),
                "reason": "request_outside_seven_day_window"}
    if float(minutes) <= 45:
        return {"order_id": order_id, "eligible": False, "refund_inr": 0.0,
                "delivery_timestamp": delivery_ts.isoformat(), "request_received_at": request_ts.isoformat(),
                "reason": "delivery_not_over_45_minutes"}
    refund = round(float(amount), 2)
    return {"order_id": order_id, "eligible": True, "refund_inr": refund,
            "delivery_timestamp": delivery_ts.isoformat(), "request_received_at": request_ts.isoformat(),
            "requires_approval": refund > 1000,
            "reason": "completed_order_delivered_over_45_minutes_within_seven_days"}


def issue_refund(order_id: str, requested_amount: float, active_order: dict | None,
                 calculation: dict | None, ledger: dict[str, dict],
                 approval: dict | None = None) -> dict:
    """Authorize and idempotently record issuance; never trust model refund amounts."""
    order_id = str(order_id)
    if not active_order or not active_order.get("verified"):
        return {"error": "verified_order_required", "refund_inr": 0.0}
    if order_id != str(active_order.get("order_id")):
        return {"error": "order_id_mismatch", "refund_inr": 0.0}
    if not calculation or not calculation.get("eligible") or calculation.get("order_id") != order_id:
        return {"error": "eligible_calculation_required", "refund_inr": 0.0}
    trusted_amount = round(float(calculation["refund_inr"]), 2)
    try:
        supplied_amount = round(float(requested_amount), 2)
    except (TypeError, ValueError):
        return {"error": "invalid_refund_amount", "refund_inr": 0.0}
    if supplied_amount != trusted_amount:
        return {"error": "refund_amount_mismatch", "refund_inr": 0.0}
    if active_order.get("status") != "completed" or trusted_amount <= 0:
        return {"error": "refund_not_eligible", "refund_inr": 0.0}
    if trusted_amount > float(active_order.get("order_amount") or 0.0):
        return {"error": "refund_exceeds_order_amount", "refund_inr": 0.0}
    approved_amount = None
    if approval:
        if approval.get("decision") != "approve":
            return {"error": "refund_not_approved", "refund_inr": 0.0}
        try:
            approved_amount = round(float(approval.get("refund_inr")), 2)
        except (TypeError, ValueError):
            return {"error": "invalid_approval_amount", "refund_inr": 0.0}
        if str(approval.get("order_id")) != order_id or approved_amount != trusted_amount:
            return {"error": "approval_does_not_match_refund", "refund_inr": 0.0}
    if trusted_amount > 1000:
        if (not approval or approval.get("decision") != "approve"
                or not str(approval.get("approved_by", "")).strip()
                or not str(approval.get("approval_id", "")).strip()
                or approved_amount != trusted_amount
                or approval.get("scenario_id") != calculation.get("scenario_id")):
            return {"error": "explicit_matching_human_approval_required", "refund_inr": 0.0}

    existing = ledger.get(order_id)
    if existing:
        if (existing.get("status") == "issued"
                and round(float(existing.get("refund_inr", -1)), 2) == trusted_amount
                and (trusted_amount <= 1000 or existing.get("approval"))):
            return dict(existing)
        return {"error": "refund_already_issued_for_order", "refund_inr": 0.0}
    entry = {
        "order_id": order_id, "refund_inr": trusted_amount, "status": "issued",
        "scenario_id": str(calculation.get("scenario_id", "")),
        "approval": dict(approval) if trusted_amount > 1000 else None,
    }
    ledger[order_id] = entry
    return dict(entry)


def escalate_to_human(order_id: str, requested_amount: float, active_order: dict | None,
                      calculation: dict | None, scenario: dict, pending: list[dict]) -> dict:
    """Persist a verified high-value request as pending without issuing funds."""
    order_id = str(order_id)
    if not active_order or not active_order.get("verified"):
        return {"error": "verified_order_required"}
    if order_id != str(active_order.get("order_id")):
        return {"error": "order_id_mismatch"}
    if not calculation or not calculation.get("eligible") or calculation.get("order_id") != order_id:
        return {"error": "eligible_calculation_required"}
    refund = round(float(calculation["refund_inr"]), 2)
    if round(float(requested_amount), 2) != refund or refund <= 1000:
        return {"error": "human_escalation_amount_mismatch_or_not_required"}
    existing = next((item for item in pending if item["order_id"] == order_id), None)
    if existing:
        if round(float(existing["refund_inr"]), 2) != refund:
            return {"error": "conflicting_pending_refund_for_order"}
        return {"status": "pending_human_approval", "order_id": order_id,
                "refund_inr": refund, "scenario_id": existing["scenario_id"]}
    state = {
        "scenario_id": str(scenario["scenario_id"]), "customer_id": str(scenario["customer_id"]),
        "order_id": order_id, "order_amount": float(active_order["order_amount"]),
        "refund_inr": refund, "delivery_timestamp": calculation["delivery_timestamp"],
        "request_received_at": calculation["request_received_at"],
        "status": "pending_human_approval",
        "policy": "A refund above INR 1000 requires human support lead approval.",
    }
    pending.append(state)
    return {"status": "pending_human_approval", "order_id": order_id,
            "refund_inr": refund, "scenario_id": state["scenario_id"]}


@dataclass
class AgentContext:
    """Verified data and mutable state shared across one scenario's tool calls."""

    scenario: dict
    orders: pd.DataFrame
    chunks: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: Any
    refund_ledger: dict[str, dict] = field(default_factory=dict)
    pending: list[dict] = field(default_factory=list)
    request_received_at: str | None = None
    active_order: dict | None = None
    calculation: dict | None = None
    last_issue_result: dict | None = None


def tool_result(name: str, arguments: dict, context: AgentContext) -> dict:
    """Compatibility dispatcher for mock_llm tool calls."""
    if name == "get_order":
        result = get_order(str(arguments.get("order_id", "")),
                           str(context.scenario["customer_id"]), context.orders)
        context.active_order = result if result.get("verified") else None
        context.calculation = None
        if context.active_order and context.request_received_at is None:
            # The supplied offline scenarios have no received-at timestamp; treat the fixture event
            # as occurring at verified delivery time. Real callers must pass an explicit timestamp.
            context.request_received_at = context.active_order["delivery_timestamp"]
        return result
    if name == "search_policy":
        return search_policy(str(arguments.get("query", "refund")), context.chunks,
                             context.vectorizer, context.matrix)
    if name == "calculate_refund":
        result = calculate_refund(arguments, context.active_order, context.request_received_at)
        result["scenario_id"] = str(context.scenario["scenario_id"])
        context.calculation = result
        return result
    if name == "issue_refund":
        result = issue_refund(str(arguments.get("order_id", "")), arguments.get("refund_inr"),
                              context.active_order, context.calculation, context.refund_ledger)
        context.last_issue_result = result
        return result
    if name == "escalate_to_human":
        return escalate_to_human(str(arguments.get("order_id", "")), arguments.get("refund_inr", 0),
                                 context.active_order, context.calculation, context.scenario, context.pending)
    return {"error": "unknown_tool", "tool": name}


def run_agent_scenario(scenario: dict, orders: pd.DataFrame, chunks: pd.DataFrame,
                       vectorizer: TfidfVectorizer, matrix: Any,
                       refund_ledger: dict[str, dict] | None = None,
                       request_received_at: str | None = None) -> tuple[dict, list[dict], list[dict]]:
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": str(scenario["message"])},
    ]
    traces: list[dict] = []
    context = AgentContext(
        scenario=scenario, orders=orders, chunks=chunks, vectorizer=vectorizer, matrix=matrix,
        refund_ledger=refund_ledger if refund_ledger is not None else {},
        request_received_at=request_received_at or scenario.get("request_received_at") or None,
    )
    seen_calls: set[tuple[str, str]] = set()
    total_tokens = 0
    final_text = ""
    status = "running"
    stop_reason = ""
    outcome = "FALLBACK"
    refund = 0.0
    step_count = 0
    mock_llm.reset()

    for step in range(1, MAX_AGENT_STEPS + 1):
        step_count = step
        response = mock_llm.chat(messages, tools=TOOL_SCHEMAS)
        total_tokens += int(response.usage.total_tokens)
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            final_text = message.content or ""
            if final_text.startswith("REFUND:"):
                amount_match = re.search(r"INR\s+([0-9]+(?:\.[0-9]+)?)", final_text)
                claimed_amount = round(float(amount_match.group(1)), 2) if amount_match else 0.0
                issuance = context.last_issue_result or {}
                if issuance.get("status") == "issued" and round(float(issuance.get("refund_inr", 0)), 2) == claimed_amount:
                    outcome, refund, status = "REFUND", claimed_amount, "completed"
                else:
                    outcome, refund, status = "DENIED", 0.0, "issuance_not_confirmed"
                    final_text = "Refund was not reported as issued because the issuance tool did not confirm it."
            elif final_text.startswith("NOT_FOUND:"):
                outcome, status = "NOT_FOUND", "not_found"
            elif final_text.startswith("DENIED:"):
                outcome, status = "DENIED", "denied"
            elif final_text.startswith("ESCALATE:"):
                outcome, status = "ESCALATE", "pending_human_approval"
            else:
                outcome, status = "FALLBACK", "human_review"
            break

        call = tool_calls[0]
        name = str(call.name)
        arguments = call.arguments if isinstance(call.arguments, dict) else json.loads(call.arguments)
        signature = (name, json.dumps(arguments, sort_keys=True))
        if signature in seen_calls:
            status, outcome, stop_reason = "safe_stopped", "FALLBACK", "repeated_tool_call"
            traces.append({"scenario_id": scenario["scenario_id"], "step": step, "tool": name,
                           "arguments": arguments, "result": {"error": "repeated_tool_call"}, "safe_stop": True})
            break
        seen_calls.add(signature)

        # The mock's loop sentinel requests policy search before establishing an order.
        # Stop that ungrounded call safely, before executing any tool.
        if name == "search_policy" and context.active_order is None:
            status, outcome, stop_reason = "safe_stopped", "FALLBACK", "policy_search_without_order_lookup"
            traces.append({"scenario_id": scenario["scenario_id"], "step": step, "tool": name,
                           "arguments": arguments, "result": {"error": "order_lookup_required_first"}, "safe_stop": True})
            break
        if name not in {"get_order", "search_policy", "calculate_refund", "issue_refund", "escalate_to_human"}:
            status, outcome, stop_reason = "safe_stopped", "FALLBACK", "unknown_tool"
            traces.append({"scenario_id": scenario["scenario_id"], "step": step, "tool": name,
                           "arguments": arguments, "result": {"error": "unknown_tool"}, "safe_stop": True})
            break

        result = tool_result(name, arguments, context)
        traces.append({"scenario_id": scenario["scenario_id"], "step": step, "tool": name,
                       "arguments": arguments, "result": result, "safe_stop": False})
        messages.append({"role": "assistant", "content": message.content,
                         "tool_calls": [{"id": call.id, "name": name, "arguments": arguments}]})
        messages.append({"role": "tool", "tool_call_id": call.id, "name": name,
                         "content": json.dumps(result, sort_keys=True)})

        if name == "escalate_to_human" and result.get("status") == "pending_human_approval":
            status, outcome, refund = "pending_human_approval", "ESCALATE", 0.0
            stop_reason = "human_approval_required"
            break
    else:
        status, outcome, stop_reason = "safe_stopped", "FALLBACK", "maximum_agent_steps_reached"

    if status == "running":
        status, outcome, stop_reason = "safe_stopped", "FALLBACK", "agent_did_not_finish"
    message_order = re.search(r"ORD-\d{6}", str(scenario["message"]))
    scenario_order_id = (str(context.active_order["order_id"]) if context.active_order
                         else (message_order.group(0) if message_order else ""))
    handoff_summary = ""
    if outcome == "ESCALATE":
        handoff_summary = (f"Scenario {scenario['scenario_id']}: refund request for order {scenario_order_id} "
                           f"of INR {context.pending[-1]['refund_inr']:.2f} requires support lead approval.")
    elif status == "safe_stopped" or outcome == "FALLBACK":
        handoff_summary = (f"Scenario {scenario['scenario_id']} was safely stopped: {stop_reason}. "
                           "Review the request and tool trace before taking action.")
    if stop_reason == "maximum_agent_steps_reached":
        traces.append({
            "scenario_id": str(scenario["scenario_id"]), "step": MAX_AGENT_STEPS,
            "tool": "human_handoff", "arguments": {},
            "result": {"status": "human_review", "reason": stop_reason,
                       "summary": handoff_summary},
            "safe_stop": True, "event": "handoff",
        })
    result_row = {
        "scenario_id": str(scenario["scenario_id"]), "customer_id": str(scenario["customer_id"]),
        "order_id": scenario_order_id,
        "expected_outcome": str(scenario["expected_outcome"]), "outcome": outcome,
        "expected_refund_inr": round(float(scenario["expected_refund_inr"]), 2),
        "requested_refund_inr": (round(float(context.pending[-1]["refund_inr"]), 2)
                                  if context.pending else round(refund, 2)),
        "refund_inr": round(refund, 2), "status": status,
        "priority": "HIGH" if outcome == "ESCALATE" else "NORMAL",
        "route": "human_review" if outcome == "FALLBACK" else ("support_lead" if outcome == "ESCALATE" else "support_queue"),
        "human_review_required": bool(outcome == "ESCALATE" or outcome == "FALLBACK"),
        "callback_required": False, "steps": step_count, "tokens_used": total_tokens,
        "stop_reason": stop_reason, "handoff_summary": handoff_summary, "final_text": final_text,
    }
    return result_row, traces, context.pending


def resume_escalation(state: dict, decision: str, orders: pd.DataFrame,
                      refund_ledger: dict[str, dict], approved_by: str = "explicit_cli_operator") -> dict:
    """Re-verify a pending case and pass approved issuance through issue_refund()."""
    if state.get("status") != "pending_human_approval":
        raise ValueError("Only a pending human-approval case can be resumed")
    if decision not in {"approve", "reject"}:
        raise ValueError("Human decision must be 'approve' or 'reject'")
    already_issued = refund_ledger.get(str(state.get("order_id")))
    if already_issued and decision == "reject":
        raise ValueError("An issued refund cannot be rejected retroactively")
    active_order = get_order(str(state["order_id"]), str(state["customer_id"]), orders)
    if not active_order.get("verified"):
        raise ValueError("Pending order could not be re-verified")
    calculation = calculate_refund({"order_id": state["order_id"]}, active_order,
                                    state.get("request_received_at"))
    if (not calculation.get("eligible")
            or round(float(calculation["refund_inr"]), 2) != round(float(state["refund_inr"]), 2)):
        raise ValueError("Pending refund is no longer eligible at its saved amount")
    amount = round(float(calculation["refund_inr"]), 2)
    if amount <= 1000:
        raise ValueError("Resume approval path is reserved for refunds above INR 1000")
    if decision == "approve":
        approval = {
            "decision": "approve", "approved_by": str(approved_by).strip(),
            "approval_id": f"approval:{state['scenario_id']}:{state['order_id']}:{amount:.2f}",
            "scenario_id": str(state["scenario_id"]), "order_id": str(state["order_id"]),
            "refund_inr": amount,
        }
        if not approval["approved_by"]:
            raise ValueError("Approval requires a non-empty approver identity")
        calculation["scenario_id"] = str(state["scenario_id"])
        issued = issue_refund(str(state["order_id"]), amount, active_order,
                              calculation, refund_ledger, approval=approval)
        if issued.get("status") != "issued":
            return {"scenario_id": state["scenario_id"], "order_id": state["order_id"],
                    "decision": decision, "outcome": "ESCALATE", "refund_inr": 0.0,
                    "status": "issuance_not_confirmed", "callback_required": False,
                    "issuance_result": issued}
        return {"scenario_id": state["scenario_id"], "order_id": state["order_id"],
                "decision": decision, "outcome": "REFUND", "refund_inr": issued["refund_inr"],
                "status": "issued_after_human_approval", "callback_required": False,
                "issuance_result": issued}
    return {"scenario_id": state["scenario_id"], "order_id": state["order_id"],
            "decision": decision, "outcome": "ESCALATE", "refund_inr": 0.0,
            "status": "rejected_by_human", "callback_required": True}


def run_pipeline() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = load_policy_chunks()
    vectorizer, matrix = build_index(chunks)
    eval_rows = pd.read_csv(POLICY_EVAL, keep_default_na=False)
    retrieval, topk = evaluate_retrieval(eval_rows, chunks, vectorizer, matrix)
    answers = answer_policy_questions(eval_rows, topk)
    orders = load_orders()
    scenarios = pd.read_csv(SCENARIOS, keep_default_na=False)
    ledger_path = OUT / "refund_ledger.json"
    refund_ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    agent_rows = []
    all_traces = []
    all_pending = []
    for scenario in scenarios.to_dict("records"):
        result, traces, pending = run_agent_scenario(
            scenario, orders, chunks, vectorizer, matrix, refund_ledger=refund_ledger,
        )
        agent_rows.append(result)
        all_traces.extend(traces)
        all_pending.extend(pending)

    chunks.to_csv(OUT / "policy_chunks.csv", index=False)
    retrieval.to_csv(OUT / "retrieval_evaluation.csv", index=False)
    topk.to_csv(OUT / "retrieval_top_k.csv", index=False)
    answers.to_csv(OUT / "rag_answers.csv", index=False)
    agents = pd.DataFrame(agent_rows)
    agents.to_csv(OUT / "agent_scenarios.csv", index=False)
    with (OUT / "tool_traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in all_traces:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
    (OUT / "pending_escalations.json").write_text(json.dumps(all_pending, indent=2), encoding="utf-8")
    ledger_path.write_text(json.dumps(refund_ledger, indent=2, sort_keys=True), encoding="utf-8")

    answerable = retrieval[retrieval.expected_doc != "NONE"]
    unanswerable = retrieval[retrieval.expected_doc == "NONE"]
    rag_answerable = answers[answers.expected_doc != "NONE"]
    rag_unanswerable = answers[answers.expected_doc == "NONE"]
    metrics = {
        "retrieval": {
            "policy_documents": int(chunks.source_file.nunique()), "chunks": int(len(chunks)),
            "chunking": "one sentence per chunk, stable filename and sentence order",
            "tfidf_ngram_range": [1, 2], "tfidf_stop_words": "english", "top_k": TOP_K,
            "minimum_retrieval_score": MIN_RETRIEVAL_SCORE,
            "eval_questions": int(len(retrieval)),
            "answerable_questions": int(len(answerable)), "unanswerable_questions": int(len(unanswerable)),
            "top1_document_accuracy": float(answerable.expected_doc_top1.mean()) if len(answerable) else None,
            "top3_document_recall": float(answerable.expected_doc_top3.mean()) if len(answerable) else None,
            "unanswerable_abstention_accuracy": float(unanswerable.abstention_correct.mean()) if len(unanswerable) else None,
        },
        "rag": {
            "answerable_questions": int(len(rag_answerable)),
            "expected_document_retrieved": int(rag_answerable.expected_doc_retrieved.sum()),
            "expected_fact_matches": int(rag_answerable.expected_fact_found.sum()),
            "expected_fact_match_rate": float(rag_answerable.expected_fact_found.mean()) if len(rag_answerable) else None,
            "unanswerable_questions": int(len(rag_unanswerable)),
            "correct_abstentions": int(rag_unanswerable.unanswerable_correctly_abstained.sum()),
            "tokens_used": int(answers.total_tokens.sum()),
        },
        "refund_agent": {
            "scenarios": int(len(agents)),
            "outcome_matches": int((agents.outcome == agents.expected_outcome).sum()),
            "requested_refund_amount_matches": int((agents.requested_refund_inr.round(2) == agents.expected_refund_inr.round(2)).sum()),
            "issued_refund_amount_matches": int((agents.loc[agents.expected_outcome == "REFUND", "refund_inr"].round(2) ==
                                                  agents.loc[agents.expected_outcome == "REFUND", "expected_refund_inr"].round(2)).sum()),
            "safe_stops": int((agents.status == "safe_stopped").sum()),
            "pending_human_approvals": int((agents.status == "pending_human_approval").sum()),
            "tool_calls": int(sum(trace.get("event") != "handoff" for trace in all_traces)),
            "max_steps": MAX_AGENT_STEPS,
            "resume_handling": "pending cases require explicit approve/reject; rejected high-value request requires customer callback",
        },
        "limitations": [
            "The scenario CSV has no request timestamp; offline scenarios use the verified delivery timestamp as their request-time fixture. Live callers must provide a trusted request timestamp.",
            "The mock agent fixture does not resolve a human escalation itself; escalation is persisted and resumed only with an explicit human decision.",
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": str(OUT), "metrics": metrics}, indent=2))
    return metrics


def resume_from_file(scenario_id: str, decision: str, approved_by: str = "explicit_cli_operator") -> dict:
    state_path = OUT / "pending_escalations.json"
    if not state_path.exists():
        raise FileNotFoundError("No pending escalations found; run Part D first")
    pending = json.loads(state_path.read_text(encoding="utf-8"))
    matches = [state for state in pending if state.get("scenario_id") == scenario_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one pending case for {scenario_id}; found {len(matches)}")
    orders = load_orders()
    ledger_path = OUT / "refund_ledger.json"
    refund_ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    resumed = resume_escalation(matches[0], decision, orders, refund_ledger, approved_by=approved_by)
    ledger_path.write_text(json.dumps(refund_ledger, indent=2, sort_keys=True), encoding="utf-8")
    (OUT / f"resume_{scenario_id.lower()}_{decision}.json").write_text(
        json.dumps(resumed, indent=2), encoding="utf-8")
    print(json.dumps(resumed, indent=2))
    return resumed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-scenario", help="Resume a persisted human-approval case, for example S3")
    parser.add_argument("--decision", choices=["approve", "reject"], help="Explicit human decision for --resume-scenario")
    parser.add_argument("--approved-by", default="explicit_cli_operator",
                        help="Approver identity recorded for an explicit approval decision")
    args = parser.parse_args()
    if args.resume_scenario:
        if not args.decision:
            parser.error("--decision is required with --resume-scenario")
        resume_from_file(args.resume_scenario, args.decision, approved_by=args.approved_by)
    elif args.decision:
        parser.error("--decision requires --resume-scenario")
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
