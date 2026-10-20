"""Milestone F — scripted negotiation agents (no Gemini)."""

from __future__ import annotations

import pytest

from app.negotiation import load_scenario, new_run, validate_proposed_action
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
from app.negotiation_schemas import NegotiationAction


def test_scripted_strategy_respects_policy():
    scenario = load_scenario()
    state = new_run(scenario)
    backend = ScriptedNegotiationBackend()
    state = generate_strategy(state, scenario, backend=backend)
    assert state.status == "STRATEGY_READY"
    assert state.strategy_summary
    assert state.pending_action is not None
    blocks = validate_proposed_action(state, scenario, state.pending_action)
    assert blocks == []


def test_strategy_auto_injects_rag_excerpts():
    scenario = load_scenario()
    state = generate_strategy(
        new_run(scenario), scenario, backend=ScriptedNegotiationBackend()
    )
    assert state.knowledge_excerpts_used, "strategist should auto-retrieve demo policy"
    assert state.knowledge_evidence_ids
    assert not state.knowledge_abstained
    assert "knowledge" in state.strategy_summary.lower() or "demo" in state.strategy_summary.lower()


def test_strategy_honors_explicit_excerpts():
    scenario = load_scenario()
    custom = ["[ev-test | Demo Policy | v1 | approved]\nBudget ceiling requires CFO approval."]
    state = generate_strategy(
        new_run(scenario),
        scenario,
        backend=ScriptedNegotiationBackend(),
        knowledge_excerpts=custom,
    )
    assert state.knowledge_excerpts_used == custom
    assert "1 approved demo knowledge" in state.strategy_summary.lower() or "knowledge" in state.strategy_summary.lower()


def test_draft_requires_human_approval_gate():
    scenario = load_scenario()
    backend = ScriptedNegotiationBackend()
    state = generate_strategy(new_run(scenario), scenario, backend=backend)
    state = draft_buyer_email(state, scenario, backend=backend)
    assert state.status == "AWAITING_USER_APPROVAL"
    assert state.pending_draft is not None
    assert state.pending_draft.approval_status == "draft"
    assert "4800000" in state.pending_draft.body.replace(",", "") or "48" in state.pending_draft.body


def test_full_scripted_live_reaches_agreement_or_escalation():
    state = run_scripted_live_demo(max_cycles=2)
    assert state.status in (
        "AGREEMENT_IN_PRINCIPLE",
        "ESCALATION_REQUIRED",
        "NEXT_MOVE_READY",
        "IMPASSE",
    )
    assert len(state.thread) >= 2  # at least buyer + supplier
    assert any(e.event_type == "approve_and_send" for e in state.events)
    # Human gate was used
    assert any(e.actor == "human" for e in state.events)


def test_policy_blocks_bad_uptime_concession_from_agent_action():
    scenario = load_scenario()
    state = new_run(scenario)
    state = generate_strategy(state, scenario, backend=ScriptedNegotiationBackend())
    bad = NegotiationAction(
        action_type="counteroffer",
        rationale="bad",
        proposed_term_changes={"uptime_percent": "99.5"},
    )
    state = state.model_copy(update={"pending_action": bad})
    with pytest.raises(Exception) as exc:
        draft_buyer_email(state, scenario, backend=ScriptedNegotiationBackend())
    assert "policy" in str(exc.value).lower() or getattr(exc.value, "code", "") == "policy_block"


def test_step_order_enforced():
    scenario = load_scenario()
    state = new_run(scenario)
    with pytest.raises(Exception):
        draft_buyer_email(state, scenario, backend=ScriptedNegotiationBackend())


def test_one_cycle_manual_steps():
    scenario = load_scenario()
    backend = ScriptedNegotiationBackend()
    state = new_run(scenario)
    state = generate_strategy(state, scenario, backend=backend)
    state = draft_buyer_email(state, scenario, backend=backend)
    state = human_approve_send(state)
    assert state.status == "EMAIL_SENT_TO_SIMULATOR"
    state = simulate_supplier_reply(state, scenario, backend=backend)
    assert state.status == "SUPPLIER_REPLY_RECEIVED"
    assert state.current_supplier_offer.annual_fee_inr == 4_900_000
    state = analyze_supplier_reply(state, scenario, backend=backend)
    assert state.status == "RESPONSE_ANALYZED"
    state = plan_next_move(state, scenario, backend=backend)
    assert state.status == "NEXT_MOVE_READY"
