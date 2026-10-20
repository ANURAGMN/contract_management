"""Deterministic negotiation state machine + concession policy (Phase A/B).

No LLM calls. Transitions are validated in code. Replay fixtures drive the
Streamlit Negotiation Simulator without Gemini.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.negotiation_schemas import (
    ConcessionPolicy,
    CurrentOffer,
    NegotiationAction,
    NegotiationEmail,
    NegotiationEvent,
    NegotiationReplayFixture,
    NegotiationScenario,
    NegotiationState,
    NegotiationStatus,
)

ROOT = Path(__file__).resolve().parent.parent
NEG_DIR = ROOT / "data" / "negotiation"

# Allowed edges for the demo state machine (spec §16.3).
ALLOWED_TRANSITIONS: dict[NegotiationStatus, set[NegotiationStatus]] = {
    "NOT_STARTED": {"STRATEGY_READY", "CANCELLED"},
    "STRATEGY_READY": {"DRAFT_READY", "CANCELLED"},
    "DRAFT_READY": {"AWAITING_USER_APPROVAL", "CANCELLED"},
    "AWAITING_USER_APPROVAL": {
        "EMAIL_SENT_TO_SIMULATOR",
        "DRAFT_READY",  # edit / reject draft
        "CANCELLED",
    },
    "EMAIL_SENT_TO_SIMULATOR": {"SUPPLIER_REPLY_RECEIVED", "CANCELLED"},
    "SUPPLIER_REPLY_RECEIVED": {"RESPONSE_ANALYZED", "CANCELLED"},
    "RESPONSE_ANALYZED": {"NEXT_MOVE_READY", "ESCALATION_REQUIRED", "IMPASSE", "CANCELLED"},
    "NEXT_MOVE_READY": {
        "DRAFT_READY",
        "AGREEMENT_IN_PRINCIPLE",
        "ESCALATION_REQUIRED",
        "IMPASSE",
        "CANCELLED",
    },
    "AGREEMENT_IN_PRINCIPLE": set(),
    "ESCALATION_REQUIRED": set(),
    "IMPASSE": set(),
    "CANCELLED": set(),
}


class NegotiationError(ValueError):
    def __init__(self, message: str, *, code: str = "negotiation_error") -> None:
        super().__init__(message)
        self.code = code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _eid(prefix: str = "evt") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def load_scenario(path: str | Path | None = None) -> NegotiationScenario:
    p = Path(path) if path else NEG_DIR / "cloudsoft_pr2026.json"
    return NegotiationScenario.model_validate_json(p.read_text(encoding="utf-8"))


def load_replay(path: str | Path | None = None) -> NegotiationReplayFixture:
    p = Path(path) if path else NEG_DIR / "cloudsoft_replay.json"
    return NegotiationReplayFixture.model_validate_json(p.read_text(encoding="utf-8"))


def new_run(scenario: NegotiationScenario) -> NegotiationState:
    offer = CurrentOffer(
        annual_fee_inr=scenario.opening_quote_inr,
        term_months=12,
        uptime_percent=scenario.policy.min_uptime_percent,
        notes=["Opening supplier quote"],
    )
    open_issues = [
        t.key for t in scenario.terms if t.priority in ("must_have", "high") and not t.resolved
    ]
    return NegotiationState(
        run_id=f"neg-{datetime.now().strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:4]}",
        scenario_id=scenario.scenario_id,
        status="NOT_STARTED",
        round_number=0,
        current_supplier_offer=offer,
        open_issues=open_issues,
    )


def can_transition(current: NegotiationStatus, nxt: NegotiationStatus) -> bool:
    return nxt in ALLOWED_TRANSITIONS.get(current, set())


def transition(
    state: NegotiationState,
    nxt: NegotiationStatus,
    *,
    actor: str,
    reason: str,
    event_type: str = "transition",
    message_id: Optional[str] = None,
    action: Optional[NegotiationAction] = None,
) -> NegotiationState:
    if not can_transition(state.status, nxt):
        raise NegotiationError(
            f"Illegal transition {state.status} -> {nxt}",
            code="illegal_transition",
        )
    before = state.status
    events = list(state.events)
    events.append(
        NegotiationEvent(
            event_id=_eid(),
            actor=actor,  # type: ignore[arg-type]
            event_type=event_type,
            timestamp=_now(),
            state_before=before,
            state_after=nxt,
            message_id=message_id,
            action=action,
            reason=reason,
        )
    )
    return state.model_copy(update={"status": nxt, "events": events})


# --------------------------------------------------------------------------- #
# Concession / policy checks (deterministic)
# --------------------------------------------------------------------------- #
def check_price_against_policy(
    fee_inr: float, policy: ConcessionPolicy
) -> list[str]:
    """Return escalation reasons; empty list means within policy."""
    reasons: list[str] = []
    if fee_inr > policy.budget_ceiling_inr:
        reasons.append(
            f"Price INR {fee_inr:,.0f} exceeds budget ceiling "
            f"INR {policy.budget_ceiling_inr:,.0f}"
        )
    if fee_inr < policy.price_floor_inr:
        reasons.append(
            f"Price INR {fee_inr:,.0f} is below configured floor "
            f"INR {policy.price_floor_inr:,.0f} — escalate before accepting"
        )
    return reasons


def check_offer_against_policy(
    offer: CurrentOffer, policy: ConcessionPolicy
) -> list[str]:
    reasons: list[str] = []
    if offer.annual_fee_inr is not None:
        reasons.extend(check_price_against_policy(offer.annual_fee_inr, policy))
    if offer.uptime_percent is not None and offer.uptime_percent < policy.min_uptime_percent:
        reasons.append(
            f"Uptime {offer.uptime_percent}% below required "
            f"{policy.min_uptime_percent}%"
        )
    if offer.term_months is not None and offer.term_months > policy.max_term_months:
        reasons.append(
            f"Term {offer.term_months} months exceeds max "
            f"{policy.max_term_months} without human exception"
        )
    return reasons


def validate_proposed_action(
    state: NegotiationState,
    scenario: NegotiationScenario,
    action: NegotiationAction,
) -> list[str]:
    """Block forbidden concessions. Returns list of blocking reasons."""
    blocks: list[str] = []
    policy = scenario.policy

    if state.round_number >= policy.max_rounds and action.action_type not in (
        "end",
        "escalate",
        "accept",
    ):
        blocks.append(f"Round limit ({policy.max_rounds}) reached")

    changes = action.changes_map()
    if "annual_fee_inr" in changes:
        try:
            fee = float(changes["annual_fee_inr"])
        except ValueError:
            blocks.append("annual_fee_inr is not numeric")
        else:
            blocks.extend(check_price_against_policy(fee, policy))

    if "uptime_percent" in changes:
        try:
            up = float(changes["uptime_percent"])
        except ValueError:
            blocks.append("uptime_percent is not numeric")
        else:
            if up < policy.min_uptime_percent:
                blocks.append("Cannot concede uptime below policy minimum")

    if "term_months" in changes:
        try:
            term = int(float(changes["term_months"]))
        except ValueError:
            blocks.append("term_months is not numeric")
        else:
            if term > policy.max_term_months and action.action_type != "escalate":
                blocks.append(
                    "Term above max requires human-approved exception / escalate"
                )

    if policy.never_waive_data_protection and changes.get("data_protection") in (
        "waive",
        "none",
        "deferred",
    ):
        blocks.append("Cannot waive data-protection requirements")

    if policy.liability_requires_legal and "liability" in changes:
        if action.action_type == "accept" and not action.requires_escalation:
            blocks.append("Liability wording requires Legal escalation")

    for key in changes:
        term = next((t for t in scenario.terms if t.key == key), None)
        if term and term.priority == "must_have" and action.action_type == "accept":
            # accepting a must_have change that worsens vs target needs care
            pass
        if (
            term
            and term.priority == "low"
            and key not in policy.tradable_low_priority_keys
            and action.action_type == "counteroffer"
        ):
            blocks.append(f"Low-priority key '{key}' is not on the tradable list")

    return blocks


def require_approval_before_send(state: NegotiationState) -> None:
    if state.status != "AWAITING_USER_APPROVAL":
        raise NegotiationError(
            "Outbound simulator send requires AWAITING_USER_APPROVAL",
            code="approval_required",
        )
    if state.pending_draft is None:
        raise NegotiationError("No pending draft to approve", code="no_draft")
    if state.pending_draft.approval_status not in ("draft", "approved"):
        raise NegotiationError(
            f"Draft status {state.pending_draft.approval_status} cannot be sent",
            code="bad_draft_status",
        )


def approve_and_send_to_simulator(state: NegotiationState) -> NegotiationState:
    """Human gate: mark draft approved, append to thread, advance state."""
    require_approval_before_send(state)
    assert state.pending_draft is not None
    draft = state.pending_draft.model_copy(
        update={"approval_status": "simulated_sent", "timestamp": _now()}
    )
    thread = list(state.thread) + [draft]
    state = state.model_copy(
        update={
            "pending_draft": None,
            "thread": thread,
            "round_number": state.round_number + 1,
        }
    )
    return transition(
        state,
        "EMAIL_SENT_TO_SIMULATOR",
        actor="human",
        reason="User approved outbound message to simulator (no real email sent)",
        event_type="approve_and_send",
        message_id=draft.message_id,
    )


# --------------------------------------------------------------------------- #
# Replay driver (Phase B)
# --------------------------------------------------------------------------- #
def apply_replay_step(
    state: NegotiationState,
    scenario: NegotiationScenario,
    step,  # NegotiationReplayStep
) -> NegotiationState:
    """Apply one fixture step. Validates transition legality."""
    if state.status != step.from_status:
        raise NegotiationError(
            f"Replay desync: state is {state.status}, step expects {step.from_status}",
            code="replay_desync",
        )

    updates: dict = {}
    if step.strategy_summary is not None:
        updates["strategy_summary"] = step.strategy_summary
    if step.action is not None:
        blocks = validate_proposed_action(state, scenario, step.action)
        if blocks and step.to_status not in ("ESCALATION_REQUIRED", "IMPASSE"):
            raise NegotiationError(
                "Replay action blocked by policy: " + "; ".join(blocks),
                code="policy_block",
            )
        updates["pending_action"] = step.action
    if step.email is not None and not step.approve_pending_draft:
        if step.email.direction == "buyer_to_supplier":
            updates["pending_draft"] = step.email
        else:
            updates["thread"] = list(state.thread) + [step.email]
    if step.supplier_offer_update is not None:
        updates["current_supplier_offer"] = step.supplier_offer_update
    if step.buyer_offer_update is not None:
        updates["buyer_last_approved_offer"] = step.buyer_offer_update
    if step.open_issues is not None:
        updates["open_issues"] = step.open_issues
    if step.agreed_terms is not None:
        updates["agreed_terms"] = step.agreed_terms
    if step.concessions_buyer is not None:
        updates["concessions_buyer"] = step.concessions_buyer
    if step.concessions_supplier is not None:
        updates["concessions_supplier"] = step.concessions_supplier

    state = state.model_copy(update=updates)

    if step.approve_pending_draft:
        # pending draft should already be set by a prior DRAFT_READY step
        if step.email is not None:
            state = state.model_copy(update={"pending_draft": step.email})
        state = approve_and_send_to_simulator(state)
        # approve_and_send already transitioned; allow fixture to_status match
        if state.status != step.to_status and can_transition(state.status, step.to_status):
            state = transition(
                state,
                step.to_status,
                actor=step.actor,
                reason=step.reason,
                event_type="replay_step",
            )
        elif state.status != step.to_status:
            # Fixture expects EMAIL_SENT_TO_SIMULATOR after approve — already there
            if not (step.to_status == "EMAIL_SENT_TO_SIMULATOR" and state.status == "EMAIL_SENT_TO_SIMULATOR"):
                raise NegotiationError(
                    f"After approve, status={state.status}, expected {step.to_status}",
                    code="replay_desync",
                )
        return state

    return transition(
        state,
        step.to_status,
        actor=step.actor,
        reason=step.reason,
        event_type="replay_step",
        message_id=step.email.message_id if step.email else None,
        action=step.action,
    )


def run_replay(
    scenario: Optional[NegotiationScenario] = None,
    fixture: Optional[NegotiationReplayFixture] = None,
) -> NegotiationState:
    scenario = scenario or load_scenario()
    fixture = fixture or load_replay()
    if fixture.scenario_id != scenario.scenario_id:
        raise NegotiationError("Replay fixture scenario_id mismatch", code="scenario_mismatch")
    state = new_run(scenario)
    for step in fixture.steps:
        state = apply_replay_step(state, scenario, step)
    if state.status != fixture.expected_final_status:
        raise NegotiationError(
            f"Final status {state.status} != expected {fixture.expected_final_status}",
            code="final_status_mismatch",
        )
    return state


def write_state(state: NegotiationState, path: str | Path) -> None:
    Path(path).write_text(state.model_dump_json(indent=2), encoding="utf-8")


def negotiation_brief(scenario: NegotiationScenario, state: NegotiationState) -> dict:
    """Compact briefing card for the UI left panel."""
    policy = scenario.policy
    offer = state.current_supplier_offer
    return {
        "scenario_id": scenario.scenario_id,
        "requisition_id": scenario.requisition_id,
        "status": state.status,
        "round": state.round_number,
        "budget_ceiling_inr": policy.budget_ceiling_inr,
        "target_price_inr": policy.target_price_inr,
        "price_floor_inr": policy.price_floor_inr,
        "current_supplier_fee_inr": offer.annual_fee_inr,
        "max_rounds": policy.max_rounds,
        "must_haves": [
            {"key": t.key, "label": t.label, "target": t.target_value}
            for t in scenario.terms
            if t.priority == "must_have"
        ],
        "open_issues": state.open_issues,
        "agreed_terms": state.agreed_terms,
        "strategy_summary": state.strategy_summary,
        "simulation_mode": scenario.simulation_mode,
        "non_binding_notice": state.non_binding_notice,
        "knowledge_excerpts_used": getattr(state, "knowledge_excerpts_used", []) or [],
        "knowledge_evidence_ids": getattr(state, "knowledge_evidence_ids", []) or [],
        "knowledge_abstained": bool(getattr(state, "knowledge_abstained", False)),
    }
