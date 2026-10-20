"""Negotiation state machine + policy + replay (no LLM)."""

from __future__ import annotations

import pytest

from app.negotiation import (
    NegotiationError,
    check_offer_against_policy,
    check_price_against_policy,
    load_replay,
    load_scenario,
    new_run,
    run_replay,
    transition,
    validate_proposed_action,
)
from app.negotiation_schemas import CurrentOffer, NegotiationAction


def test_scenario_loads():
    s = load_scenario()
    assert s.scenario_id == "cloudsoft_pr2026"
    assert s.opening_quote_inr == 5000000
    assert s.policy.budget_ceiling_inr == 4800000


def test_illegal_transition_blocked():
    state = new_run(load_scenario())
    with pytest.raises(NegotiationError) as exc:
        transition(state, "EMAIL_SENT_TO_SIMULATOR", actor="system", reason="skip")
    assert exc.value.code == "illegal_transition"


def test_price_over_budget_flagged():
    policy = load_scenario().policy
    reasons = check_price_against_policy(5000000, policy)
    assert any("budget ceiling" in r for r in reasons)


def test_price_below_floor_requires_escalation():
    policy = load_scenario().policy
    reasons = check_price_against_policy(4500000, policy)
    assert any("floor" in r for r in reasons)


def test_uptime_cannot_be_conceded():
    scenario = load_scenario()
    state = new_run(scenario)
    state = transition(state, "STRATEGY_READY", actor="system", reason="t")
    action = NegotiationAction(
        action_type="counteroffer",
        rationale="bad",
        proposed_term_changes={"uptime_percent": "99.5"},
    )
    blocks = validate_proposed_action(state, scenario, action)
    assert any("uptime" in b.lower() for b in blocks)


def test_term_over_max_blocked_without_escalate():
    scenario = load_scenario()
    state = new_run(scenario)
    state = transition(state, "STRATEGY_READY", actor="system", reason="t")
    action = NegotiationAction(
        action_type="counteroffer",
        rationale="bad",
        proposed_term_changes={"term_months": "24"},
    )
    blocks = validate_proposed_action(state, scenario, action)
    assert any("Term" in b for b in blocks)


def test_data_protection_cannot_be_waived():
    scenario = load_scenario()
    state = new_run(scenario)
    state = transition(state, "STRATEGY_READY", actor="system", reason="t")
    action = NegotiationAction(
        action_type="accept",
        rationale="bad",
        proposed_term_changes={"data_protection": "waive"},
    )
    blocks = validate_proposed_action(state, scenario, action)
    assert any("data-protection" in b.lower() for b in blocks)


def test_liability_accept_requires_escalation_flag():
    scenario = load_scenario()
    state = new_run(scenario)
    state = transition(state, "STRATEGY_READY", actor="system", reason="t")
    action = NegotiationAction(
        action_type="accept",
        rationale="accept liability",
        proposed_term_changes={"liability": "supplier_standard"},
        requires_escalation=False,
    )
    blocks = validate_proposed_action(state, scenario, action)
    assert any("Liability" in b for b in blocks)


def test_offer_sla_check():
    policy = load_scenario().policy
    reasons = check_offer_against_policy(
        CurrentOffer(annual_fee_inr=4800000, uptime_percent=99.5, term_months=12),
        policy,
    )
    assert any("Uptime" in r for r in reasons)


def test_full_replay_reaches_aip():
    fixture = load_replay()
    state = run_replay()
    assert state.status == "AGREEMENT_IN_PRINCIPLE"
    assert state.status == fixture.expected_final_status
    assert len(state.thread) >= 4  # 2 buyer + 2 supplier
    assert any(e.event_type == "approve_and_send" for e in state.events)
    assert state.current_supplier_offer.annual_fee_inr == 4800000
    # Every outbound must have been approved
    buyer_msgs = [m for m in state.thread if m.direction == "buyer_to_supplier"]
    assert buyer_msgs
    assert all(m.approval_status == "simulated_sent" for m in buyer_msgs)
    assert "liability" in state.open_issues
    assert "annual_fee_inr" in state.agreed_terms
