"""Validate Part D retrieval, RAG, refund tools, safe stops, and resume state."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part_d import (  # noqa: E402
    MAX_AGENT_STEPS,
    OUT,
    POLICY_EVAL,
    SCENARIOS,
    build_index,
    calculate_refund,
    evaluate_retrieval,
    get_order,
    issue_refund,
    load_policy_chunks,
    load_orders,
    resume_escalation,
    run_agent_scenario,
)
import mock_llm  # noqa: E402


def check(condition: bool, description: str) -> None:
    if not condition:
        raise AssertionError(description)


def test_six_step_handoff(orders: pd.DataFrame, chunks: pd.DataFrame,
                          vectorizer, matrix) -> dict:
    """Drive six unresolved mock turns through the real orchestration loop."""
    state = {"calls": 0}

    def synthetic_chat(messages, tools=None):
        state["calls"] += 1
        step = state["calls"]
        if step > MAX_AGENT_STEPS:
            raise AssertionError("agent made a seventh model call after the six-step limit")
        if step == 1:
            name, arguments = "get_order", {"order_id": "ORD-100001"}
        else:
            name, arguments = "search_policy", {"query": f"synthetic unresolved step {step}"}
        tool_call = SimpleNamespace(id=f"synthetic_call_{step}", name=name, arguments=arguments)
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=SimpleNamespace(total_tokens=1),
        )

    scenario = {
        "scenario_id": "SYNTHETIC_SIX_STEP",
        "customer_id": "C0001",
        "message": "Please review order ORD-100001.",
        "expected_outcome": "FALLBACK",
        "expected_refund_inr": 0.0,
    }
    original_chat = mock_llm.chat
    ledger: dict[str, dict] = {}
    try:
        mock_llm.chat = synthetic_chat
        result, trace, pending = run_agent_scenario(
            scenario, orders, chunks, vectorizer, matrix, refund_ledger=ledger,
        )
    finally:
        mock_llm.chat = original_chat

    handoffs = [event for event in trace if event.get("event") == "handoff"]
    check(state["calls"] == MAX_AGENT_STEPS and result["steps"] == MAX_AGENT_STEPS,
          "synthetic agent did not execute exactly six unresolved steps")
    check(result["stop_reason"] == "maximum_agent_steps_reached"
          and result["status"] == "safe_stopped"
          and result["route"] == "human_review"
          and result["human_review_required"],
          "six-step limit did not trigger a human handoff")
    check(bool(result["handoff_summary"].strip()), "six-step handoff summary is empty")
    check(len(handoffs) == 1 and handoffs[0]["step"] == MAX_AGENT_STEPS
          and handoffs[0]["result"]["status"] == "human_review"
          and handoffs[0]["result"]["reason"] == "maximum_agent_steps_reached"
          and handoffs[0]["result"]["summary"] == result["handoff_summary"],
          "six-step execution trace does not record the handoff")
    check(not any(event["tool"] == "issue_refund" for event in trace)
          and not pending and not ledger and result["refund_inr"] == 0.0,
          "synthetic handoff issued a refund without authorization")
    print("SIX_STEP_HANDOFF_TEST: " + json.dumps({
        "model_calls": state["calls"], "steps_executed": result["steps"],
        "handoff_status": handoffs[0]["result"]["status"],
        "stop_reason": result["stop_reason"],
        "handoff_summary": result["handoff_summary"],
        "refund_issued": bool(ledger), "refund_inr": result["refund_inr"],
    }))
    return result


def main() -> None:
    policy_paths = sorted((ROOT / "data" / "policies").glob("*_policy.txt"))
    check(len(policy_paths) == 5, "expected the five supplied policy documents")
    chunks = load_policy_chunks()
    saved_chunks = pd.read_csv(OUT / "policy_chunks.csv", keep_default_na=False)
    check(saved_chunks.equals(chunks), "saved chunks differ from the deterministic policy chunking")
    check(chunks.chunk_id.is_unique and chunks.text.str.strip().ne("").all(), "policy chunks must be non-empty and uniquely identified")

    eval_rows = pd.read_csv(POLICY_EVAL, keep_default_na=False)
    vectorizer, matrix = build_index(chunks)
    expected_retrieval, expected_topk = evaluate_retrieval(eval_rows, chunks, vectorizer, matrix)
    retrieval = pd.read_csv(OUT / "retrieval_evaluation.csv", keep_default_na=False)
    topk = pd.read_csv(OUT / "retrieval_top_k.csv", keep_default_na=False)
    check(retrieval.equals(expected_retrieval), "saved retrieval metrics do not reproduce from the input questions")
    check(topk.equals(expected_topk), "saved ranked passages do not reproduce from the TF-IDF index")
    check(len(retrieval) == len(eval_rows) == 14, "all policy evaluation questions must be evaluated")
    answerable = retrieval[retrieval.expected_doc != "NONE"]
    unanswerable = retrieval[retrieval.expected_doc == "NONE"]
    check(len(answerable) == 12 and len(unanswerable) == 2, "expected 12 answerable and 2 unanswerable policy questions")
    check(answerable.expected_doc_top1.all(), "a policy question retrieved the wrong top document")
    check(answerable.expected_doc_top3.all(), "a policy question's expected document is absent from top-k")
    check(unanswerable.abstention_correct.all(), "unanswerable retrieval questions were not abstained")

    answers = pd.read_csv(OUT / "rag_answers.csv", keep_default_na=False)
    check(len(answers) == 14 and answers.question.is_unique, "RAG output must contain one row per evaluation question")
    rag_answerable = answers[answers.expected_doc != "NONE"]
    rag_none = answers[answers.expected_doc == "NONE"]
    check(rag_answerable.expected_doc_retrieved.all(), "RAG answer lacks its expected source document in retrieved context")
    check(rag_answerable.expected_fact_found.all(), "RAG answer missed an expected policy fact")
    check((rag_none.status == "abstained_no_retrieval_match").all(), "RAG answered an unanswerable question")
    check((rag_none.total_tokens == 0).all(), "unanswerable questions should not invoke the mock assistant")
    retrieved_by_q = topk.groupby("question").text.apply(set).to_dict()
    check(all(row["answer"] in retrieved_by_q[row["question"]]
              for row in rag_answerable.to_dict("records")), "RAG answer is not a retrieved policy passage")

    # Adversarial tool-argument tests: all eligibility facts come from the verified order.
    orders = load_orders()
    test_six_step_handoff(orders, chunks, vectorizer, matrix)
    trusted_s1 = get_order("ORD-100048", "C0007", orders)
    s1_delivery = trusted_s1["delivery_timestamp"]
    manipulated = calculate_refund(
        {"order_id": "ORD-100048", "order_amount": 99999999, "delivery_minutes": 1,
         "status": "cancelled", "eligible": False},
        trusted_s1, s1_delivery,
    )
    check(manipulated.get("eligible") and manipulated["refund_inr"] == 128.25,
          "model-supplied amount/status/delivery values overrode the verified order")
    trusted_s4 = get_order("ORD-100001", "C0001", orders)
    manipulated_late = calculate_refund(
        {"order_id": "ORD-100001", "order_amount": 5000, "delivery_minutes": 999},
        trusted_s4, trusted_s4["delivery_timestamp"],
    )
    check(not manipulated_late.get("eligible") and manipulated_late["refund_inr"] == 0.0,
          "model-supplied delivery time made an on-time order refund-eligible")
    check(calculate_refund({"order_id": "ORD-OTHER"}, trusted_s1, s1_delivery).get("error") == "order_id_mismatch",
          "calculation accepted a mismatched order ID")
    check(calculate_refund({}, None, s1_delivery).get("error") == "verified_order_required",
          "calculation accepted a missing verified order")
    check(calculate_refund({}, trusted_s1, None).get("reason") == "trusted_request_timestamp_missing",
          "calculation accepted a missing trusted request timestamp")
    outside_window = pd.Timestamp(s1_delivery) + pd.Timedelta(days=7, seconds=1)
    expired = calculate_refund({}, trusted_s1, outside_window.isoformat())
    check(not expired.get("eligible") and expired["reason"] == "request_outside_seven_day_window",
          "calculation did not enforce the seven-day request window")
    at_deadline = calculate_refund({}, trusted_s1,
                                   (pd.Timestamp(s1_delivery) + pd.Timedelta(days=7)).isoformat())
    check(at_deadline.get("eligible"), "request exactly seven days after delivery should remain in-window")

    scenarios = pd.read_csv(SCENARIOS, keep_default_na=False)
    agent = pd.read_csv(OUT / "agent_scenarios.csv", keep_default_na=False)
    check(len(agent) == len(scenarios) == 7 and agent.scenario_id.is_unique, "all seven refund scenarios must be run")
    expected = scenarios.set_index("scenario_id")
    check((agent.set_index("scenario_id").outcome == expected.expected_outcome).all(), "agent outcomes differ from scenario expectations")
    expected_refunds = expected.expected_refund_inr.astype(float).round(2)
    requested = agent.set_index("scenario_id").requested_refund_inr.astype(float).round(2)
    check((requested == expected_refunds).all(), "requested or escalated refund amount mismatch")
    issued = agent[agent.expected_outcome == "REFUND"]
    check((issued.refund_inr.astype(float).round(2) == issued.expected_refund_inr.astype(float).round(2)).all(),
          "issued refund amount mismatch")
    check((agent.steps.astype(int) <= MAX_AGENT_STEPS).all(), "agent exceeded the six-step safety limit")
    check((agent.loc[agent.expected_outcome == "ESCALATE", "status"] == "pending_human_approval").all(),
          "high-value request did not pause for human approval")
    check((agent.loc[agent.expected_outcome == "ESCALATE", "refund_inr"].astype(float) == 0).all(),
          "high-value refund was issued before human approval")
    check((agent.loc[agent.expected_outcome == "ESCALATE", "route"] == "support_lead").all(),
          "high-value request was not routed to a support lead")
    check((agent.loc[agent.expected_outcome == "FALLBACK", "status"] == "safe_stopped").all(),
          "loop scenario did not stop safely")
    check((agent.loc[agent.expected_outcome == "FALLBACK", "route"] == "human_review").all(),
          "safe-stop fallback was not routed to human review")
    check(agent.loc[agent.expected_outcome.isin(["ESCALATE", "FALLBACK"]), "handoff_summary"].astype(str).str.strip().ne("").all(),
          "safe stop or human escalation is missing its handoff summary")

    traces = [json.loads(line) for line in (OUT / "tool_traces.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    allowed_tools = {"get_order", "search_policy", "calculate_refund", "issue_refund", "escalate_to_human"}
    check(all(row["tool"] in allowed_tools for row in traces), "agent invoked a tool outside the allow-list")
    for row in traces:
        if row["tool"] == "issue_refund":
            check(float(row["arguments"].get("refund_inr", 0)) <= 1000,
                  "agent tried to issue a greater-than-INR-1000 refund without approval")
        if row["scenario_id"] == "S7" and row.get("safe_stop"):
            check(row["tool"] == "search_policy" and row["result"].get("error") == "order_lookup_required_first",
                  "loop-safe-stop did not block policy access before order lookup")
    check({row["tool"] for row in traces if not row.get("safe_stop")} == allowed_tools,
          "scenario workflow did not exercise every dedicated refund-agent tool")
    for row in agent[agent.outcome == "REFUND"].to_dict("records"):
        success = any(trace["scenario_id"] == row["scenario_id"] and trace["tool"] == "issue_refund"
                      and trace["result"].get("status") == "issued"
                      and round(float(trace["result"].get("refund_inr", 0)), 2) == round(float(row["refund_inr"]), 2)
                      for trace in traces)
        check(success, f"{row['scenario_id']} reported a refund without issuance confirmation")

    pending = json.loads((OUT / "pending_escalations.json").read_text(encoding="utf-8"))
    check(len(pending) == 1 and pending[0]["scenario_id"] == "S3", "expected S3 to persist as the pending human escalation")
    approval_ledger = {}
    approved = resume_escalation(pending[0], "approve", orders, approval_ledger, approved_by="validator operator")
    check(approved["status"] == "issued_after_human_approval" and approved["refund_inr"] == 1999.0,
          "approved human decision did not resume the refund safely")
    check(approved["issuance_result"].get("status") == "issued"
          and approval_ledger["ORD-100145"]["approval"]["approved_by"] == "validator operator",
          "approval was not recorded and confirmed by issue_refund")
    check(resume_escalation(pending[0], "approve", orders, approval_ledger,
                            approved_by="validator operator")["status"] == "issued_after_human_approval"
          and len(approval_ledger) == 1,
          "repeated approval caused duplicate issuance")
    try:
        resume_escalation(pending[0], "reject", orders, approval_ledger)
    except ValueError:
        pass
    else:
        raise AssertionError("an already issued refund was allowed to be rejected retroactively")

    rejection_ledger = {}
    rejected = resume_escalation(pending[0], "reject", orders, rejection_ledger)
    check(rejected["status"] == "rejected_by_human" and rejected["refund_inr"] == 0.0
          and rejected["callback_required"] and not rejection_ledger,
          "rejected high-value request issued or failed to require callback")

    s3_order = get_order("ORD-100145", "C0016", orders)
    s3_calculation = calculate_refund({}, s3_order, pending[0]["request_received_at"])
    s3_calculation["scenario_id"] = "S3"
    unapproved_ledger = {}
    unapproved = issue_refund("ORD-100145", 1999.0, s3_order, s3_calculation, unapproved_ledger)
    check(unapproved.get("error") == "explicit_matching_human_approval_required" and not unapproved_ledger,
          "pending high-value approval issued a refund")
    pending_approval = {
        "decision": "pending", "approved_by": "validator operator", "approval_id": "pending-approval",
        "scenario_id": "S3", "order_id": "ORD-100145", "refund_inr": 1999.0,
    }
    rejected_approval = {**pending_approval, "decision": "reject"}
    check(issue_refund("ORD-100145", 1999.0, s3_order, s3_calculation,
                       unapproved_ledger, approval=pending_approval).get("error") == "refund_not_approved",
          "pending approval issued a high-value refund")
    check(issue_refund("ORD-100145", 1999.0, s3_order, s3_calculation,
                       unapproved_ledger, approval=rejected_approval).get("error") == "refund_not_approved"
          and not unapproved_ledger, "rejected approval issued a high-value refund")

    # Direct repeated issuance of the same automatic refund is idempotent.
    low_calculation = calculate_refund({}, trusted_s1, s1_delivery)
    low_calculation["scenario_id"] = "S1"
    duplicate_ledger = {}
    first_issue = issue_refund("ORD-100048", 128.25, trusted_s1, low_calculation, duplicate_ledger)
    second_issue = issue_refund("ORD-100048", 128.25, trusted_s1, low_calculation, duplicate_ledger)
    check(first_issue.get("status") == second_issue.get("status") == "issued"
          and first_issue == second_issue and len(duplicate_ledger) == 1,
          "duplicate refund issuance was not prevented idempotently")
    tampered_issue = issue_refund("ORD-100048", 999999.0, trusted_s1,
                                  low_calculation, duplicate_ledger)
    check(tampered_issue.get("error") == "refund_amount_mismatch",
          "issuance accepted a model-supplied amount different from the trusted calculation")

    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    check(metrics["retrieval"]["top1_document_accuracy"] == float(answerable.expected_doc_top1.mean()),
          "reported top-1 retrieval accuracy does not match saved evaluations")
    check(metrics["retrieval"]["top3_document_recall"] == float(answerable.expected_doc_top3.mean()),
          "reported top-3 recall does not match saved evaluations")
    check(metrics["rag"]["expected_fact_matches"] == int(rag_answerable.expected_fact_found.sum()),
          "reported RAG fact matches do not match saved answers")
    check(metrics["refund_agent"]["outcome_matches"] == int((agent.outcome == agent.expected_outcome).sum()),
          "reported agent outcomes do not match saved scenarios")
    check(metrics["refund_agent"]["requested_refund_amount_matches"] == 7,
          "reported requested refund matches do not reconcile")
    saved_ledger = json.loads((OUT / "refund_ledger.json").read_text(encoding="utf-8"))
    check({"ORD-100048", "ORD-100067"}.issubset(saved_ledger)
          and set(saved_ledger).issubset({"ORD-100048", "ORD-100067", "ORD-100145"})
          and all(record.get("status") == "issued" for record in saved_ledger.values()),
          "saved issuance ledger contains an unexpected or unconfirmed refund")
    if "ORD-100145" in saved_ledger:
        approval = saved_ledger["ORD-100145"].get("approval") or {}
        check(approval.get("decision") == "approve" and approval.get("approved_by")
              and float(saved_ledger["ORD-100145"]["refund_inr"]) == 1999.0,
              "persisted S3 issuance lacks a matching explicit human approval")
    print("ALL PART D CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
