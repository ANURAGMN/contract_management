"""Deterministic commercial / escalation resolution for negotiation agents.

LLM agents propose; this module decides whether the flow should continue,
reach agreement-in-principle (with Legal still open), hard-escalate, or
impasse — so a chatty model cannot abort the demo on round 1.
"""

from __future__ import annotations

from typing import Optional

from app.negotiation import validate_proposed_action
from app.negotiation_schemas import (
    ConcessionPolicy,
    CurrentOffer,
    NegotiationAction,
    NegotiationScenario,
    NegotiationState,
    TermChange,
)


def commercial_within_policy(offer: CurrentOffer, policy: ConcessionPolicy) -> bool:
    if offer.annual_fee_inr is None:
        return False
    if offer.annual_fee_inr > policy.budget_ceiling_inr:
        return False
    if offer.annual_fee_inr < policy.price_floor_inr:
        return False
    if (
        offer.uptime_percent is not None
        and offer.uptime_percent < policy.min_uptime_percent
    ):
        return False
    if offer.term_months is not None and offer.term_months > policy.max_term_months:
        return False
    return True


def liability_still_open(state: NegotiationState) -> bool:
    """Liability is never auto-agreed in the demo — open unless explicitly agreed."""
    return "liability" not in state.agreed_terms


def buyer_price_ask(scenario: NegotiationScenario, supplier_fee: Optional[float] = None) -> float:
    """The fee the buyer proposes. Never above the budget ceiling, never above what the supplier
    is already asking (a counter above the supplier's own price would overpay), and never below
    the floor. With the supplier above budget this is the ceiling (the CloudSoft demo path)."""
    pol = scenario.policy
    ask_now = supplier_fee if supplier_fee is not None else scenario.opening_quote_inr
    if ask_now is None or ask_now > pol.budget_ceiling_inr:
        return float(pol.budget_ceiling_inr)
    target = pol.target_price_inr or pol.budget_ceiling_inr
    return float(max(pol.price_floor_inr, min(target, ask_now)))


def budget_counter_action(scenario: NegotiationScenario, *, rationale: str,
                          supplier_fee: Optional[float] = None) -> NegotiationAction:
    pol = scenario.policy
    return NegotiationAction(
        action_type="counteroffer",
        rationale=rationale,
        proposed_term_changes=[
            TermChange(key="annual_fee_inr", value=str(int(buyer_price_ask(scenario, supplier_fee)))),
            TermChange(key="uptime_percent", value=str(pol.min_uptime_percent)),
            TermChange(key="term_months", value=str(pol.max_term_months)),
            TermChange(key="data_protection", value="confirm"),
        ],
        requires_escalation=False,
    )


def agreement_with_legal_action(
    action: NegotiationAction,
    *,
    rationale: Optional[str] = None,
) -> NegotiationAction:
    note = action.uncertainty_note or ""
    if "liability" not in note.lower():
        note = (note + "; " if note else "") + "Liability remains open for Legal review"
    return action.model_copy(
        update={
            "action_type": "escalate",
            "requires_escalation": True,
            "uncertainty_note": note,
            "rationale": rationale
            or action.rationale
            or "Commercial terms within policy; escalate liability to Legal (non-binding AIP).",
        }
    )


def resolve_next_move(
    state: NegotiationState,
    scenario: NegotiationScenario,
    action: NegotiationAction,
    llm_hint: Optional[str],
) -> tuple[str, NegotiationAction]:
    """Return (terminal_hint, action) after applying demo policy.

    terminal_hint ∈ agreement | escalate | impasse | continue
    """
    pol = scenario.policy
    offer = state.current_supplier_offer
    hint = (llm_hint or "continue").strip().lower()
    if hint not in ("agreement", "escalate", "impasse", "continue"):
        hint = "continue"

    fee = offer.annual_fee_inr
    commercial_ok = commercial_within_policy(offer, pol)
    legal_open = liability_still_open(state)
    at_round_limit = state.round_number >= pol.max_rounds

    # Round limit: AIP if commercial ok, else impasse
    if at_round_limit:
        if commercial_ok:
            return "agreement", agreement_with_legal_action(
                action,
                rationale="Round limit reached with commercial terms in policy — "
                "agreement in principle; Legal reviews liability.",
            )
        return "impasse", action.model_copy(
            update={
                "action_type": "end",
                "rationale": action.rationale or "Round limit without commercial agreement",
            }
        )

    # Commercial settled → AIP (Legal may still be open). Never hard-stop here.
    if commercial_ok:
        if legal_open or action.requires_escalation or action.action_type == "escalate":
            return "agreement", agreement_with_legal_action(action)
        return "agreement", action.model_copy(
            update={"action_type": "accept", "requires_escalation": False}
        )

    # Fee still above budget — keep negotiating; rewrite premature escalate
    if fee is not None and fee > pol.budget_ceiling_inr:
        if action.action_type in ("escalate", "end", "accept") or hint in (
            "escalate",
            "agreement",
            "impasse",
        ):
            action = budget_counter_action(
                scenario, supplier_fee=fee,
                rationale=(
                    f"Supplier at INR {fee:,.0f} is still above ceiling "
                    f"INR {pol.budget_ceiling_inr:,.0f} — continue counteroffer; "
                    "defer Legal escalation until commercial AIP."
                ),
            )
        return "continue", action

    # Fee below floor — must escalate (cannot accept)
    if fee is not None and fee < pol.price_floor_inr:
        return "escalate", action.model_copy(
            update={
                "action_type": "escalate",
                "requires_escalation": True,
                "uncertainty_note": (
                    f"Fee INR {fee:,.0f} is below floor INR {pol.price_floor_inr:,.0f}"
                ),
            }
        )

    # Policy blocks on the proposed action → escalate
    blocks = validate_proposed_action(state, scenario, action)
    if blocks and action.action_type not in ("escalate", "end"):
        return "escalate", action.model_copy(
            update={
                "action_type": "escalate",
                "requires_escalation": True,
                "uncertainty_note": "; ".join(blocks),
            }
        )

    if hint == "impasse":
        return "impasse", action.model_copy(update={"action_type": "end"})

    # Mid-flow "escalate" from the LLM without a hard block → continue
    if action.action_type == "escalate" and hint == "escalate":
        return "continue", budget_counter_action(
            scenario, supplier_fee=offer.annual_fee_inr,
            rationale="Model requested escalate mid-negotiation; continuing "
            "commercial counter instead (Legal only at AIP).",
        )

    return "continue", action


def sanitize_midflow_action(
    state: NegotiationState,
    scenario: NegotiationScenario,
    action: NegotiationAction,
) -> NegotiationAction:
    """After analyze: do not leave a terminal escalate while fee is still high."""
    pol = scenario.policy
    fee = state.current_supplier_offer.annual_fee_inr
    if fee is not None and fee > pol.budget_ceiling_inr and action.action_type in (
        "escalate",
        "accept",
        "end",
    ):
        return budget_counter_action(
            scenario, supplier_fee=fee,
            rationale=(
                action.rationale
                + " [rewritten: commercial gap remains — counter toward budget]"
            ),
        )
    if commercial_within_policy(state.current_supplier_offer, pol):
        # Prefer continue/accept path; plan_next_move will turn into AIP
        if action.action_type == "escalate" and not action.requires_escalation:
            return action.model_copy(update={"requires_escalation": True})
    return action
