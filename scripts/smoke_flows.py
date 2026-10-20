"""Thorough offline (+ optional live) flow smoke for demo paths.

Usage:
  python scripts/smoke_flows.py
  python scripts/smoke_flows.py --live
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Also hit Gemini (lite)")
    args = parser.parse_args()

    results: list[tuple[str, str, str]] = []

    def ok(name: str, detail: str = "") -> None:
        results.append(("PASS", name, detail))
        print(f"PASS  {name}" + (f" — {detail}" if detail else ""))

    def fail(name: str, err: object) -> None:
        results.append(("FAIL", name, str(err)))
        print(f"FAIL  {name} — {err}")

    def run(name: str, fn: Callable[[], str]) -> None:
        try:
            ok(name, fn())
        except Exception as e:
            fail(name, f"{e}\n{traceback.format_exc()[-600:]}")

    # --- Samples A–E ---
    def samples_flow() -> str:
        from app.approvals import build_queue, record_decision
        from app.demo_packages import SAMPLE_CATALOG, load_sample

        lines = []
        for sid in SAMPLE_CATALOG:
            pkg = load_sample(sid)
            assert pkg.findings is not None
            assert pkg.knowledge_groundings, f"no knowledge_groundings on {sid}"
            q = build_queue(pkg, contract_id=sid)
            for t in q.tasks:
                assert t.knowledge_citations or t.knowledge_abstained
            open_tasks = q.open_tasks()
            if open_tasks:
                t0 = open_tasks[0]
                role = t0.required_teams[0] if t0.required_teams else "procurement"
                try:
                    q = record_decision(
                        q,
                        t0.task_id,
                        reviewer="flow.tester",
                        role=role,
                        decision="need_information",
                        reason="flow smoke",
                    )
                except Exception:
                    if "legal" in t0.required_teams:
                        q = record_decision(
                            q,
                            t0.task_id,
                            reviewer="flow.tester",
                            role="legal",
                            decision="need_information",
                            reason="flow smoke",
                        )
                    else:
                        raise
            lines.append(
                f"{sid}: findings={len(pkg.findings)} tasks={len(q.tasks)} "
                f"groundings={len(pkg.knowledge_groundings)} route={pkg.route.route}"
            )
        return "; ".join(lines)

    run("samples::A-E+approvals+rag", samples_flow)

    # --- RAG ---
    def rag_flow() -> str:
        from app.rag import ask_knowledge_base, build_default_retriever, retrieve_for_negotiation

        retriever, status, _ = build_default_retriever()
        r1 = ask_knowledge_base(
            "What does our policy require for auto-renewal notice?", retriever
        )
        assert not r1.abstained and r1.evidence
        r2 = ask_knowledge_base(
            "What is our policy for lunar mineral royalties?",
            retriever,
            min_top_score=5.0,
        )
        assert r2.abstained
        ev, excerpts, audit = retrieve_for_negotiation()
        assert ev and excerpts
        return f"chunks={status.chunk_count} neg_ev={len(ev)} mode={audit.mode}"

    run("rag::ask+negotiation", rag_flow)

    # --- Negotiation scripted ---
    def neg_scripted() -> str:
        from app.negotiation import load_scenario, new_run
        from app.negotiation_agents import (
            ScriptedNegotiationBackend,
            analyze_supplier_reply,
            draft_buyer_email,
            generate_strategy,
            human_approve_send,
            plan_next_move,
            run_scripted_live_demo,
            simulate_supplier_reply,
        )

        state = run_scripted_live_demo(max_cycles=2)
        assert len(state.thread) >= 2
        assert state.status in (
            "AGREEMENT_IN_PRINCIPLE",
            "ESCALATION_REQUIRED",
            "NEXT_MOVE_READY",
            "IMPASSE",
        )
        sc = load_scenario()
        be = ScriptedNegotiationBackend()
        s = generate_strategy(new_run(sc), sc, backend=be)
        assert s.knowledge_excerpts_used
        s = draft_buyer_email(s, sc, backend=be)
        s = human_approve_send(s)
        s = simulate_supplier_reply(s, sc, backend=be)
        s = analyze_supplier_reply(s, sc, backend=be)
        s = plan_next_move(s, sc, backend=be)
        return (
            f"demo={state.status} step={s.status} "
            f"rag_excerpts={len(s.knowledge_excerpts_used)}"
        )

    run("negotiation::scripted", neg_scripted)

    # --- Procurement ---
    def procurement_flow() -> str:
        import tempfile
        from pathlib import Path

        from app import procurement_requests as pr

        store = Path(tempfile.mkdtemp()) / "pr_store"
        store.mkdir()
        pr.STORE = store
        req = pr.create_request(
            {
                "request_id": "PR-FLOW-1",
                "title": "Flow smoke SaaS",
                "requesting_team": "IT",
                "business_owner": "BO",
                "business_justification": "Need tool for flow smoke",
                "category": "saas_software",
                "new_purchase_or_renewal": "new_purchase",
                "number_of_users": 10,
                "desired_start_date": "2026-12-01",
                "budget_amount": 100000,
                "budget_period": "annual",
                "budget_confirmed": True,
                "requested_contract_term_months": 12,
                "required_sla": "99.9%",
                "data_sensitivity": "internal",
            }
        )
        out = pr.submit_request("PR-FLOW-1")
        return f"id={req.request_id} draft={req.status} after_submit={out.status}"

    run("procurement::create+submit", procurement_flow)

    # --- Commercial ---
    def commercial_flow() -> str:
        from app.commercial import compare_demo_quotations

        cmp_ = compare_demo_quotations(include_baseline=True)
        assert cmp_.rows
        return f"suppliers={len(cmp_.rows)} budget={cmp_.budget_inr}"

    run("commercial::demo", commercial_flow)

    if args.live:
        def live_slice() -> str:
            from app.schemas import ReviewRequest
            from app.slice import run_clause_slice

            out = run_clause_slice(
                ReviewRequest(
                    pdf_path="data/contracts/contract_a.pdf",
                    playbook_version="demo-slice-v1",
                    ask_gate=False,
                )
            )
            return (
                f"status={out.status} route={out.routing_decision} "
                f"findings={len(out.findings)} meta_known="
                f"{15 - len(out.metadata_unknown_fields)}/15"
            )

        run("live::clause_slice_A", live_slice)

        def live_full_review() -> str:
            from app.runner import run_review
            from app.schemas import ReviewRequest
            from app.approvals import build_queue

            pkg = run_review(
                ReviewRequest(
                    pdf_path="data/contracts/contract_a.pdf",
                    playbook_version="demo-v1",
                    ask_gate=False,
                )
            )
            assert pkg.status in ("complete", "manual_review")
            assert pkg.knowledge_groundings or pkg.stage_status.get("knowledge")
            q = build_queue(pkg, contract_id="live-a")
            return (
                f"status={pkg.status} route={pkg.route.route} "
                f"findings={len(pkg.findings)} tasks={len(q.tasks)} "
                f"groundings={len(pkg.knowledge_groundings)} "
                f"model_stage={pkg.stage_status}"
            )

        run("live::full_review_A", live_full_review)

        def live_negotiation() -> str:
            from app.negotiation import load_scenario, new_run
            from app.negotiation_agents import (
                GeminiNegotiationBackend,
                analyze_supplier_reply,
                draft_buyer_email,
                generate_strategy,
                human_approve_send,
                plan_next_move,
                simulate_supplier_reply,
            )

            sc = load_scenario()
            be = GeminiNegotiationBackend()
            s = generate_strategy(new_run(sc), sc, backend=be)
            assert s.knowledge_excerpts_used
            s = draft_buyer_email(s, sc, backend=be)
            assert s.pending_draft is not None
            s = human_approve_send(s)
            s = simulate_supplier_reply(s, sc, backend=be)
            s = analyze_supplier_reply(s, sc, backend=be)
            s = plan_next_move(s, sc, backend=be)
            return (
                f"status={s.status} excerpts={len(s.knowledge_excerpts_used)} "
                f"thread={len(s.thread)} action={s.pending_action.action_type if s.pending_action else None}"
            )

        run("live::negotiation_cycle", live_negotiation)

    passed = sum(1 for r in results if r[0] == "PASS")
    failed = sum(1 for r in results if r[0] == "FAIL")
    print(f"\n=== FLOWS: {passed} passed, {failed} failed / {len(results)} ===")
    for st, name, detail in results:
        if st == "FAIL":
            print("---", name)
            print(detail[:1200])
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
