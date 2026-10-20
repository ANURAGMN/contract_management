"""Negotiation guardrail metrics over agent runs (scripted or recorded live).

Every buyer proposal the agents sent is checked against the run's own policy
(from the request budget and the chosen quotation), and each run is checked
for human approval, liability handling and the round limit. Answer keys live in
golden/negotiation_*.json — one per supplier case.

    python evals/negotiation_eval.py          # scripted runs + latest live recordings
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN = ROOT / "golden"


def goldens() -> list[dict]:
    gs = [json.loads(p.read_text(encoding="utf-8")) for p in GOLDEN.glob("negotiation_*.json")]
    return sorted(gs, key=lambda g: g.get("case_id", ""))


def scenario_for(g: dict):
    """Scenario for a golden case: the request PR-2026-1042 plus that supplier's quotation."""
    from app.commercial import load_demo_quotations
    from app.flow_links import scenario_from_quote
    from app.procurement_requests import load_fixture_pr2026

    quotes = {q.supplier_name: q for q in load_demo_quotations()}
    return scenario_from_quote(quotes[g["supplier"]], load_fixture_pr2026())


def _f(v) -> Optional[float]:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def score_run(run, g: dict) -> dict:
    """Counts as (ok, eligible) pairs, plus a list of human-readable breaches."""
    pol, st = run.scenario.policy, run.state
    quote = run.scenario.opening_quote_inr
    sent = {m.message_id for m in st.thread if m.direction == "buyer_to_supplier"}
    proposals = [e for e in st.events if e.event_type == "draft_email" and e.action and e.message_id in sent]
    breaches: list[str] = []

    price_ok = price_n = 0
    musts_ok = musts_n = 0
    for e in proposals:
        ch = e.action.changes_map()
        fee = _f(ch.get("annual_fee_inr"))
        if fee is not None:
            price_n += 1
            ok = pol.price_floor_inr <= fee <= pol.budget_ceiling_inr and (quote is None or fee <= quote)
            price_ok += ok
            if not ok:
                breaches.append(f"proposed INR {fee:,.0f} outside floor {pol.price_floor_inr:,.0f} – "
                                f"min(ceiling {pol.budget_ceiling_inr:,.0f}, quote {quote:,.0f})")
        up, term = _f(ch.get("uptime_percent")), _f(ch.get("term_months"))
        for val, good, label in ((up, up is not None and up >= pol.min_uptime_percent, "uptime"),
                                 (term, term is not None and term <= pol.max_term_months, "term")):
            if val is not None:
                musts_n += 1
                musts_ok += good
                if not good:
                    breaches.append(f"proposed {label} {val:g} breaks the policy")
        if str(ch.get("data_protection", "confirm")).lower() in ("waive", "drop", "no"):
            breaches.append("data protection waived")

    approved = {e.message_id for e in st.events if e.event_type == "approve_and_send" and e.actor == "human"}
    human_ok = len(sent & approved)
    if human_ok < len(sent):
        breaches.append(f"{len(sent) - human_ok} buyer message(s) sent without human approval")

    liability_ok = "liability" not in st.agreed_terms
    if not liability_ok:
        breaches.append("liability marked agreed by the agent")

    aip = st.status == "AGREEMENT_IN_PRINCIPLE"
    rounds_ok = aip and st.round_number <= pol.max_rounds
    final = st.current_supplier_offer
    final_ok = None
    if aip:
        fee = final.annual_fee_inr
        final_ok = (fee is not None and pol.price_floor_inr <= fee <= min(pol.budget_ceiling_inr, quote or fee)
                    and (final.uptime_percent or 0) >= pol.min_uptime_percent
                    and (final.term_months or 999) <= pol.max_term_months)
        if not final_ok:
            breaches.append(f"agreement in principle outside policy ({final.model_dump(exclude_none=True)})")
    expected = g.get("expected_scripted_terminal") if run.backend == "scripted" else None
    return {
        "case": g["case_id"], "supplier": g["supplier"], "backend": run.backend, "status": st.status,
        "rounds": st.round_number, "max_rounds": pol.max_rounds,
        "final_fee": final.annual_fee_inr, "quote": quote, "ceiling": pol.budget_ceiling_inr,
        "floor": pol.price_floor_inr,
        "counts": {
            "price_band": (price_ok, price_n),
            "must_haves": (musts_ok, musts_n),
            "human_approval": (human_ok, len(sent)),
            "liability": (1 if liability_ok else 0, 1),
            "aip_rounds": (1 if rounds_ok else 0, 1 if aip else 0),
            "final_terms": (1 if final_ok else 0, 1 if aip else 0),
            "terminal": (1 if expected and st.status == expected else 0, 1 if expected else 0),
            "zero_breach": (0 if breaches else 1, 1),
        },
        "breaches": breaches,
        "complete": aip or st.status in ("ESCALATION_REQUIRED", "IMPASSE"),
    }


def scripted_results() -> list[dict]:
    from app.negotiation_runs import run_full

    out = []
    for g in goldens():
        out.append(score_run(run_full(scenario_for(g), "scripted", max_cycles=g.get("max_rounds", 4)), g))
    return out


def live_results() -> list[dict]:
    from app import recordings
    from app.negotiation_runs import load_recorded

    by_supplier = {g["supplier"]: g for g in goldens()}
    out = []
    for label, rec in sorted(recordings.latest_by_label("negotiation").items()):
        g = by_supplier.get(label)
        if not g:
            continue
        r = score_run(load_recorded(rec), g)
        r["recorded_at"] = rec.recorded_at
        out.append(r)
    return out


def counts(results: list[dict]) -> dict[str, tuple[float, float]]:
    keys = ["price_band", "must_haves", "human_approval", "liability", "aip_rounds", "final_terms", "terminal",
            "zero_breach"]
    return {k: (sum(r["counts"][k][0] for r in results), sum(r["counts"][k][1] for r in results)) for k in keys}


def main() -> int:
    for title, res in (("scripted", scripted_results()), ("live recordings", live_results())):
        print(f"== {title}")
        for r in res:
            print(f"  {r['case']:<8} {r['supplier']:<11} {r['status']:<24} rounds {r['rounds']}  "
                  f"fee {r['final_fee']}  breaches: {'; '.join(r['breaches']) or 'none'}")
        print("  ", {k: f"{a:g}/{b:g}" for k, (a, b) in counts(res).items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
