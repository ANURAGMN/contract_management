"""Intake assistant gate, request→quote→negotiation links, live scorecard."""

from __future__ import annotations

import pytest

from app import recordings
from app.demo_packages import load_sample
from app.flow_links import comparison_for_request, contract_vs_quotes, required_uptime, scenario_from_quote
from app.commercial import load_demo_quotations
from app.intake_assistant import IntakeDraft, validate
from app.negotiation import load_scenario
from app.negotiation_runs import load_recorded, record, run_full
from app.procurement_requests import load_fixture_pr2026


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(recordings, "RECORDINGS", tmp_path / "recordings")


# --- intake assistant ------------------------------------------------------ #
def test_intake_validation_keeps_good_fields_and_rejects_bad_ones():
    draft = IntakeDraft(
        title="Collaboration suite", category="saas_software", number_of_users=500,
        budget_amount=4_800_000, budget_period="annual", data_sensitivity="top_secret",
        requested_contract_term_months=999, desired_start_date="2026-11-01",
        open_questions=["Which SSO provider?"],
    )
    v = validate(draft)
    assert v.updates["category"] == "saas_software"
    assert v.updates["budget_amount"] == 4_800_000
    assert v.updates["estimated_recurring_cost"] == 4_800_000
    assert "data_sensitivity" not in v.updates and "requested_contract_term_months" not in v.updates
    assert any(r.startswith("data_sensitivity") for r in v.rejected)
    assert v.updates["desired_start_date"].isoformat() == "2026-11-01"
    assert v.open_questions == ["Which SSO provider?"]


def test_intake_never_sets_status():
    v = validate(IntakeDraft(title="x"))
    assert "status" not in v.updates


# --- commercial linked to the request ------------------------------------- #
def test_comparison_uses_request_numbers():
    req = load_fixture_pr2026().model_copy(update={"budget_amount": 4_500_000, "required_sla": "99.5% uptime"})
    cmp = comparison_for_request(req)
    assert cmp.budget_inr == 4_500_000
    assert cmp.required_uptime_percent == 99.5
    assert required_uptime(None) == 99.9


def test_contract_from_unquoted_supplier_is_flagged():
    check = contract_vs_quotes(load_sample("contract_e_procurement"), load_demo_quotations())
    assert check.matched_quote is None
    assert check.findings[0][0] == "high"


def test_scenario_from_quote_takes_limits_from_request():
    req = load_fixture_pr2026()
    tech = next(q for q in load_demo_quotations() if q.supplier_name == "TechNova")
    sc = scenario_from_quote(tech, req)
    assert sc.supplier.name == "TechNova"
    assert sc.opening_quote_inr == tech.annual_recurring_fee
    assert sc.policy.budget_ceiling_inr == req.budget_amount
    assert sc.policy.price_floor_inr <= sc.policy.target_price_inr <= sc.policy.budget_ceiling_inr


# --- negotiation provenance ----------------------------------------------- #
def test_scripted_run_attributes_every_message_and_round_trips():
    run = run_full(load_scenario(), "scripted")
    assert run.state.thread
    for msg in run.state.thread:
        assert run.origins[msg.message_id].source == "scripted"
    rec = record(run)
    again = load_recorded(recordings.load("negotiation", rec.recording_id))
    assert again.state.status == run.state.status
    assert set(again.origins) == set(run.origins)


# --- live scorecard -------------------------------------------------------- #
def test_live_scorecard_scores_latest_recording_per_contract():
    import evals.score_live as score_live

    pkg = load_sample("contract_b_legal")
    pkg.provenance = "live"
    recordings.save("review", "contract_b.pdf", pkg, [{"agent": "ClauseFindings", "model": "m", "started_at": "t",
                                                       "seconds": 2.0, "ok": True}])
    out = score_live.build("live")
    scored = {r["contract"]: r for r in out["per_contract"]}
    assert set(scored) == {"contract_b"}
    assert scored["contract_b"]["counts"]["critical_recall"] == [2, 2]
    assert "contract_a" in out["missing"]
    assert out["overall"] == "incomplete"  # gates pass so far, but not every contract has a live run
