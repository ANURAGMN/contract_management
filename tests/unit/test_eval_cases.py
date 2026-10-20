"""Case catalogues added with the evaluation dataset: commercial checks, negotiation
guardrails per supplier, intake validation, new metrics and the new contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import rules, steps
from app.commercial import load_demo_quotations
from app.demo_packages import load_sample
from app.flow_links import contract_vs_quotes
from app.intake_assistant import IntakeDraft, amounts_in_text, validate
from app.schemas import ClauseFinding, ContractMetadata, Evidence, Field_, ReviewRequest
from evals import intake_eval, negotiation_eval
from evals.metrics import count_contract

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "golden"
RULE = {r.id: r for r in steps.load_playbook("demo-v1").rules}


def _golden(stem):
    return json.loads((GOLDEN / f"{stem}.json").read_text(encoding="utf-8"))


def _meta(**kw):
    m = {k: Field_() for k in ContractMetadata.model_fields}
    m.update({k: Field_(value=v) for k, v in kw.items()})
    return ContractMetadata(**m)


# --- commercial --------------------------------------------------------------
COM = json.loads((ROOT / "evals" / "commercial_cases.json").read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", COM, ids=[c["id"] for c in COM])
def test_contract_vs_quote_case(case):
    pkg = load_sample("contract_e_procurement").model_copy(deep=True)
    pkg.metadata = _meta(supplier=case["supplier"], contract_value=case["contract_value"],
                         sla_target_pct=case["sla_target_pct"])
    out = contract_vs_quotes(pkg, load_demo_quotations())
    assert any(sev == case["expect_severity"] and case["expect_text"] in msg for sev, msg in out.findings), out


# --- negotiation ---------------------------------------------------------------
@pytest.mark.parametrize("supplier", ["CloudSoft", "TechNova", "DataSphere"])
def test_scripted_negotiation_has_no_policy_breach(supplier):
    res = {r["supplier"]: r for r in negotiation_eval.scripted_results()}[supplier]
    assert res["breaches"] == []
    assert res["status"] == "AGREEMENT_IN_PRINCIPLE" and res["rounds"] <= res["max_rounds"]


def test_buyer_never_offers_more_than_the_supplier_asked():
    """TechNova quotes ₹44L, under the ₹48L budget. The old script countered at the ceiling (₹48L)."""
    res = {r["supplier"]: r for r in negotiation_eval.scripted_results()}["TechNova"]
    assert res["final_fee"] <= res["quote"]


def test_negotiation_scorer_flags_a_breach():
    from app.negotiation_runs import run_full

    g = next(x for x in negotiation_eval.goldens() if x["supplier"] == "CloudSoft")
    run = run_full(negotiation_eval.scenario_for(g), "scripted")
    run.state.agreed_terms.append("liability")
    r = negotiation_eval.score_run(run, g)
    assert r["counts"]["liability"] == (0, 1) and r["counts"]["zero_breach"] == (0, 1)


# --- intake -------------------------------------------------------------------
@pytest.mark.parametrize("text,amount", [("about 48 lakh a year", 4_800_000), ("1.2 crore per year", 12_000_000),
                                         ("3.5 lakh a month", 350_000), ("INR 48,00,000", 4_800_000)])
def test_amounts_in_text(text, amount):
    assert amount in amounts_in_text(text)


def test_validator_refuses_a_budget_not_in_the_text():
    v = validate(IntakeDraft(budget_amount=5_000_000, new_purchase_or_renewal="new"),
                 source_text="We have about 48 lakh a year.")
    assert "budget_amount" not in v.updates and any("no matching amount" in r for r in v.rejected)
    assert v.updates["new_purchase_or_renewal"] == "new_purchase"  # "new" used to be rejected


def test_intake_case_scoring_uses_answer_key():
    case = next(c for c in intake_eval.cases() if c["id"] == "INT-002")
    good = {"number_of_users": 120, "budget_amount": 900000, "budget_period": "annual",
            "requested_contract_term_months": 12, "security_review_required": True, "privacy_review_required": True}
    assert intake_eval.score_case(case, good)["injection"] == (1, 1)
    bad = {**good, "security_review_required": False}
    assert intake_eval.score_case(case, bad)["injection"] == (0, 1)


# --- contract metrics ------------------------------------------------------------
def test_unclear_and_outcome_labels():
    pkg = load_sample("contract_e_procurement").model_copy(deep=True)
    g = _golden("contract_e")
    c = count_contract(pkg, g, partial=True)
    assert c["clause_outcome"][0] == c["clause_outcome"][1] > 0 and c["unclear_clear"][0] == 0
    next(f for f in pkg.findings if f.grounded.finding.rule_id == "RENEWAL-001").status = "unclear"
    c = count_contract(pkg, g, partial=True)
    assert c["unclear_clear"][0] == 1


def test_approval_matrix_matches_answer_key_for_fixtures():
    for stem, sample in [("contract_b", "contract_b_legal"), ("contract_d", "contract_d_legal"),
                         ("contract_e", "contract_e_procurement")]:
        c = count_contract(load_sample(sample), _golden(stem), partial=True)
        assert c["approvals_exact"] == (1, 1), stem


def test_carve_out_decoys_exist_with_full_spans():
    for stem in ("contract_a", "contract_c", "contract_d", "contract_e"):
        ds = [d for d in _golden(stem)["decoys"] if d["rule_id"] == "LIABILITY-002"]
        assert ds and all(d["span"].endswith(".") and d.get("page") and d.get("reason") for d in ds), stem


# --- new contracts ------------------------------------------------------------------
def test_scanned_contract_goes_to_manual_review_before_the_model():
    doc = steps.intake(ReviewRequest(pdf_path=str(ROOT / "data/contracts/contract_g.pdf"), playbook_version="demo-v1"))
    assert doc.preflight.unreliable_extraction


def test_contract_f_footer_injection_is_caught_by_preflight():
    doc = steps.intake(ReviewRequest(pdf_path=str(ROOT / "data/contracts/contract_f.pdf"), playbook_version="demo-v1"))
    assert any(h.page == 2 and "AI reviewers" in h.quote for h in doc.preflight.injection_hits)


@pytest.mark.parametrize("rule_id,value,quote,want", [
    ("LIABILITY-001", "no ceiling on recovery",
     "There is no ceiling on the amount either party may recover from the other under or in connection with this "
     "Agreement.", "walk_away"),
    ("LIABILITY-002", "indemnity not limited by Clause 7",
     "The indemnity in this Clause 8 operates independently of, and is not limited by, Clause 7 or any other "
     "limitation in this Agreement.", "walk_away"),
    ("PAYMENT-001", "Net 45",
     "Notwithstanding Clause 3, all invoices under this Order Form are payable within fifteen (15) days of the "
     "invoice date.", "walk_away"),
])
def test_contract_f_paraphrases_and_conflicts(rule_id, value, quote, want):
    f = ClauseFinding(rule_id=rule_id, found=True, extracted_value=value, evidence=[Evidence(quote=quote, page=2)])
    assert rules.check_status(RULE[rule_id], f, _meta()) == want


def test_bare_numbers_from_the_model_still_build_obligations():
    from datetime import date

    from app.schemas import ObligationInput

    meta = _meta(expiry_date="2027-01-27", notice_period_days="90",
                 renewal_notice_basis="before the end of the then-current term", sla_target_pct="95")
    obl, _ = steps.build_obligations(ObligationInput(meta=meta, perf=None, today=date(2026, 10, 20)))
    assert ("notice_deadline", date(2026, 10, 29)) in [(o.kind, o.due) for o in obl]
    assert any(o.kind == "sla_review" for o in obl)
