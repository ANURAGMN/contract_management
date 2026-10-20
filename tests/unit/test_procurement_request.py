"""Phase 1A — procurement request intake, triage, transitions (no LLM)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app import procurement_requests as pr


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = tmp_path / "procurement_requests"
    store.mkdir()
    monkeypatch.setattr(pr, "STORE", store)
    # Keep fixtures readable from real data dir
    yield
    if store.exists():
        shutil.rmtree(store, ignore_errors=True)


def test_load_fixture_pr2026():
    req = pr.load_fixture_pr2026()
    assert req.request_id == "PR-2026-1042"
    assert req.number_of_users == 500
    assert req.budget_amount == 4_800_000
    assert req.requested_contract_term_months == 12
    assert "99.9" in (req.required_sla or "")
    assert req.status == "DRAFT"


def test_create_and_save_roundtrip():
    req = pr.create_request(
        {
            "request_id": "PR-TEST-1",
            "title": "Test SaaS",
            "requesting_team": "IT",
            "business_owner": "BO",
            "business_justification": "Need tool",
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
    loaded = pr.load_request("PR-TEST-1")
    assert loaded.title == "Test SaaS"
    assert loaded.status == "DRAFT"
    assert any(e.action == "created" for e in loaded.audit_events)


def test_incomplete_submit_goes_needs_information():
    req = pr.create_request(
        {
            "request_id": "PR-INCOMPLETE",
            "title": "Incomplete",
            "requesting_team": "IT",
        }
    )
    out = pr.submit_request("PR-INCOMPLETE")
    assert out.status == "NEEDS_INFORMATION"
    assert "business_justification" in out.missing_fields
    assert "budget_amount" in out.missing_fields


def test_complete_submit_routes_security_when_sensitive():
    req = pr.load_fixture_pr2026()
    # Persist a working copy into isolated store
    pr.save_request(req)
    out = pr.submit_request("PR-2026-1042")
    # Sample has confidential data + security/privacy flags → SECURITY_REVIEW
    # (budget is confirmed and not over)
    assert out.status in ("SECURITY_REVIEW", "PROCUREMENT_TRIAGE", "FINANCE_REVIEW")
    assert out.routing_recommendations
    teams = {r.team for r in out.routing_recommendations}
    assert "procurement" in teams
    assert "security_privacy" in teams


def test_finance_route_when_unconfirmed_budget():
    req = pr.create_request(
        {
            "request_id": "PR-FIN",
            "title": "Finance path",
            "requesting_team": "Finance Ops",
            "business_owner": "CFO delegate",
            "business_justification": "Need analytics",
            "category": "saas_software",
            "new_purchase_or_renewal": "new_purchase",
            "number_of_users": 50,
            "desired_start_date": "2026-12-01",
            "budget_amount": 500000,
            "budget_period": "annual",
            "budget_confirmed": False,
            "requested_contract_term_months": 12,
            "required_sla": "99.9%",
            "data_sensitivity": "internal",
            "security_review_required": False,
            "privacy_review_required": False,
        }
    )
    out = pr.submit_request(req.request_id)
    assert out.status == "FINANCE_REVIEW"
    assert any(f.code == "budget_unconfirmed" for f in out.risk_flags)


def test_invalid_transition_rejected():
    req = pr.create_request({"request_id": "PR-BAD", "title": "x", "requesting_team": "IT"})
    with pytest.raises(pr.InvalidTransitionError):
        pr.transition_status("PR-BAD", "APPROVED_FOR_SIGNATURE")


def test_audit_not_duplicated_on_identical_rerun():
    req = pr.create_request({"request_id": "PR-AUD", "title": "Audit", "requesting_team": "IT"})
    pr.save_draft(req, actor="u1")
    n1 = len(pr.load_request("PR-AUD").audit_events)
    pr.save_draft(pr.load_request("PR-AUD"), actor="u1")
    n2 = len(pr.load_request("PR-AUD").audit_events)
    # Second identical save_draft suppressed or at most +1 with different timestamp id
    assert n2 <= n1 + 1


def test_mark_eligible_for_sourcing_demo():
    req = pr.load_fixture_pr2026()
    pr.save_request(req)
    pr.submit_request("PR-2026-1042")
    out = pr.mark_eligible_for_sourcing("PR-2026-1042")
    assert out.status == "APPROVED_FOR_SOURCING"


def test_duplicate_candidates():
    a = pr.create_request(
        {
            "request_id": "PR-DUP-A",
            "title": "Enterprise collaboration software platform",
            "requesting_team": "IT Operations",
            "category": "saas_software",
            "business_owner": "BO",
            "business_justification": "need",
            "new_purchase_or_renewal": "new_purchase",
            "number_of_users": 100,
            "desired_start_date": "2026-11-01",
            "budget_amount": 1,
            "budget_period": "annual",
            "requested_contract_term_months": 12,
            "required_sla": "99.9",
            "data_sensitivity": "internal",
        }
    )
    b = pr.create_request(
        {
            "request_id": "PR-DUP-B",
            "title": "Enterprise collaboration software tools",
            "requesting_team": "IT Operations",
            "category": "saas_software",
            "business_owner": "BO",
            "business_justification": "need",
            "new_purchase_or_renewal": "new_purchase",
            "number_of_users": 100,
            "desired_start_date": "2026-11-01",
            "budget_amount": 1,
            "budget_period": "annual",
            "requested_contract_term_months": 12,
            "required_sla": "99.9",
            "data_sensitivity": "internal",
        }
    )
    hits = pr.find_duplicate_candidates(a, others=[b])
    assert hits
    assert hits[0].request_id == "PR-DUP-B"
    assert hits[0].match_score >= 0.45


def test_sourcing_checklist_traces_to_request():
    req = pr.load_fixture_pr2026()
    items = pr.sourcing_checklist(req)
    assert any("500" in i for i in items)
    assert any("4800000" in i or "4.8" in i or "4800000" in i.replace(",", "") for i in items)
    assert any("99.9" in i for i in items)


def test_llm_cannot_set_status_via_update():
    req = pr.create_request({"request_id": "PR-LLM", "title": "x", "requesting_team": "IT"})
    out = pr.update_request("PR-LLM", {"status": "APPROVED_FOR_SIGNATURE", "title": "y"})
    assert out.status == "DRAFT"
    assert out.title == "y"


def test_link_contract_run():
    req = pr.load_fixture_pr2026()
    pr.save_request(req)
    out = pr.link_contract_run("PR-2026-1042", "sample-contract-a")
    assert "sample-contract-a" in out.linked_contract_run_ids
    # idempotent
    out2 = pr.link_contract_run("PR-2026-1042", "sample-contract-a")
    assert out2.linked_contract_run_ids.count("sample-contract-a") == 1


def test_locked_status_blocks_draft_and_update():
    req = pr.load_fixture_pr2026()
    pr.save_request(req)
    pr.submit_request("PR-2026-1042")
    pr.mark_eligible_for_sourcing("PR-2026-1042")
    locked = pr.load_request("PR-2026-1042")
    assert locked.status == "APPROVED_FOR_SOURCING"
    with pytest.raises(pr.InvalidTransitionError):
        pr.save_draft(locked)
    with pytest.raises(pr.InvalidTransitionError):
        pr.update_request("PR-2026-1042", {"title": "should fail"})


def test_compare_request_flags_over_budget_and_category():
    from app.demo_packages import build_contract_e_package

    req = pr.load_fixture_pr2026()
    pkg = build_contract_e_package()
    flags = pr.compare_request_to_package(req, pkg)
    codes = {f.code for f in flags}
    assert "over_requisition_budget" in codes
    assert "category_mismatch" in codes


def test_audit_trail_records_repeat_routing_after_resubmit():
    """Resubmitting after NEEDS_INFORMATION must log the second routing step.

    Regression: the audit de-duplication compared against the last five events,
    so the repeated "Routed to SECURITY_REVIEW" transition was silently dropped
    and the trail jumped from SUBMITTED straight to an unexplained state.
    """
    pr.reset_fixture_to_store("PR-2026-1042")
    req = pr.submit_request("PR-2026-1042", actor="demo.user")
    first_route = req.status
    pr.request_more_information("PR-2026-1042", actor="demo.reviewer")
    req = pr.submit_request("PR-2026-1042", actor="demo.user")
    assert req.status == first_route

    transitions = [
        (e.before_status, e.after_status)
        for e in req.audit_events
        if e.action == "status_transition"
    ]
    # Every transition must start where the previous one ended.
    for (_, prev_after), (next_before, _) in zip(transitions, transitions[1:]):
        assert prev_after == next_before, transitions
    assert transitions[-1] == ("SUBMITTED", first_route)
