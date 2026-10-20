"""Deterministic next-move resolution for agentic negotiation."""

from __future__ import annotations

from app.negotiation import load_scenario, new_run
from app.negotiation_policy import (
    commercial_within_policy,
    resolve_next_move,
    sanitize_midflow_action,
)
from app.negotiation_schemas import CurrentOffer, NegotiationAction, TermChange


def test_commercial_within_policy_respects_ceiling_and_floor():
    sc = load_scenario()
    pol = sc.policy
    assert commercial_within_policy(
        CurrentOffer(annual_fee_inr=pol.budget_ceiling_inr, term_months=12, uptime_percent=99.9),
        pol,
    )
    assert not commercial_within_policy(
        CurrentOffer(annual_fee_inr=pol.budget_ceiling_inr + 1, term_months=12, uptime_percent=99.9),
        pol,
    )
    assert not commercial_within_policy(
        CurrentOffer(annual_fee_inr=pol.price_floor_inr - 1, term_months=12, uptime_percent=99.9),
        pol,
    )


def test_premature_escalate_while_fee_high_becomes_continue():
    sc = load_scenario()
    state = new_run(sc).model_copy(
        update={
            "round_number": 1,
            "current_supplier_offer": CurrentOffer(
                annual_fee_inr=4_900_000, term_months=12, uptime_percent=99.9
            ),
            "open_issues": ["annual_fee_inr", "liability"],
        }
    )
    bad = NegotiationAction(
        action_type="escalate",
        rationale="liability open",
        requires_escalation=True,
        proposed_term_changes=[TermChange(key="payment_terms", value="Net 45")],
    )
    hint, action = resolve_next_move(state, sc, bad, "escalate")
    assert hint == "continue"
    assert action.action_type == "counteroffer"
    assert action.changes_map()["annual_fee_inr"] == str(int(sc.policy.budget_ceiling_inr))


def test_commercial_ok_with_liability_open_is_agreement_not_hard_stop():
    sc = load_scenario()
    state = new_run(sc).model_copy(
        update={
            "round_number": 2,
            "current_supplier_offer": CurrentOffer(
                annual_fee_inr=sc.policy.budget_ceiling_inr,
                term_months=12,
                uptime_percent=99.9,
            ),
            "open_issues": ["liability"],
            "agreed_terms": ["uptime_percent", "term_months"],
        }
    )
    bad = NegotiationAction(
        action_type="escalate",
        rationale="send to legal",
        requires_escalation=True,
    )
    hint, action = resolve_next_move(state, sc, bad, "escalate")
    assert hint == "agreement"
    assert action.requires_escalation is True
    assert "liability" in (action.uncertainty_note or "").lower() or "Legal" in action.rationale


def test_sanitize_rewrites_analyze_escalate_when_fee_high():
    sc = load_scenario()
    state = new_run(sc).model_copy(
        update={
            "current_supplier_offer": CurrentOffer(annual_fee_inr=4_900_000),
        }
    )
    action = sanitize_midflow_action(
        state,
        sc,
        NegotiationAction(action_type="escalate", rationale="liability"),
    )
    assert action.action_type == "counteroffer"


def test_scripted_full_still_reaches_aip_or_next():
    from app.negotiation_agents import run_scripted_live_demo

    state = run_scripted_live_demo(max_cycles=2)
    assert state.status in (
        "AGREEMENT_IN_PRINCIPLE",
        "NEXT_MOVE_READY",
        "ESCALATION_REQUIRED",
    )
    # With policy resolver, expected happy path is AIP
    assert state.status == "AGREEMENT_IN_PRINCIPLE"
