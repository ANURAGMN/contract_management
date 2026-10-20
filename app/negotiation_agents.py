"""Milestone F — live negotiation agents (strategist, drafter, supplier, analyzer).

LLM path uses the same google-genai schema-out pattern as ``app.agents``.
A **scripted** backend (no API) powers unit tests and offline demos; Streamlit
can choose Live Gemini or Scripted demo.

Deterministic ``app.negotiation`` still owns transitions and policy blocks.
"""

from __future__ import annotations

import uuid
from typing import Optional, Protocol

from pydantic import BaseModel

from app.negotiation import (
    NegotiationError,
    approve_and_send_to_simulator,
    transition,
    validate_proposed_action,
)
from app.negotiation_agent_schemas import (
    AnalyzeReplyAgentInput,
    AnalyzeReplyAgentOutput,
    EmailDraftAgentInput,
    EmailDraftAgentOutput,
    NextMoveAgentInput,
    NextMoveAgentOutput,
    StrategyAgentInput,
    StrategyAgentOutput,
    SupplierSimAgentInput,
    SupplierSimAgentOutput,
)
from app.negotiation_schemas import (
    CurrentOffer,
    NegotiationAction,
    NegotiationEmail,
    NegotiationScenario,
    NegotiationState,
)

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
_NEG_PREAMBLE = """You are a procurement negotiation component inside a demo system.
You do NOT send real email. You do NOT bind the company to terms.
You never change numeric policy limits (budget, floor, uptime, term max).
Output must match the JSON schema exactly. Synthetic demo scenario only.

Action rules:
- While annual_fee is ABOVE budget_ceiling: use ask or counteroffer toward the ceiling.
  Do NOT use action_type=escalate just because liability is open.
- When commercial terms are within policy: you may set terminal_hint=agreement and
  requires_escalation=true so Legal reviews liability (non-binding AIP).
- action_type=escalate as a HARD STOP only if accepting would breach floor/uptime/term
  or human approval is mandatory mid-deal — not for routine liability parking.
- proposed_term_changes is a list of {key, value} objects (not a map)."""

STRATEGY_PROMPT = (
    _NEG_PREAMBLE
    + """
Task: Produce a negotiation strategy for the buyer from the scenario and state.
Prefer bringing annual fee within budget_ceiling while holding must-have uptime,
term, and data protection. Liability requires Legal later — do not auto-accept it.
Opening move: ask or counteroffer (not escalate)."""
)

DRAFT_PROMPT = (
    _NEG_PREAMBLE
    + """
Task: Draft a concise professional buyer email to the supplier implementing the
approved NegotiationAction. No threats. Reference only terms in the scenario.
Subject line should mention the requisition id when known.
Mention that liability wording will be handled separately with Legal."""
)

SUPPLIER_PROMPT = (
    _NEG_PREAMBLE
    + """
Task: You simulate the supplier persona (price-focused; firm on liability;
flexible on price/payment). Reply to the buyer's last message with a realistic
counter. Update offer_update with your new annual_fee_inr / term / uptime.
Move price toward the buyer's budget over rounds (e.g. 49L then 48L).
Never claim to waive legal review. This is a simulation only."""
)

ANALYZE_PROMPT = (
    _NEG_PREAMBLE
    + """
Task: Extract structured offers, objections, and commitments from the supplier
reply. Populate interpreted_* fields when numbers are explicit.
Recommend the buyer's next NegotiationAction:
- If fee still above budget_ceiling → counteroffer (not escalate).
- If fee within budget but liability open → counteroffer or accept with
  requires_escalation=true (Legal later), not a hard escalate stop.
Do not invent numbers not present in the reply or scenario."""
)

NEXT_MOVE_PROMPT = (
    _NEG_PREAMBLE
    + """
Task: Given the analysis, recommend the next buyer action and an updated
strategy_summary. Set terminal_hint to one of:
- continue — commercial gap remains (keep negotiating)
- agreement — commercial within policy; Legal may still review liability
- escalate — hard stop needing a human now (policy breach / cannot proceed)
- impasse — no path forward
Prefer continue or agreement over escalate when the fee is still above ceiling
or when the only open item is liability."""
)


# --------------------------------------------------------------------------- #
# Agent backend protocol
# --------------------------------------------------------------------------- #
class NegotiationAgentBackend(Protocol):
    def strategize(self, inp: StrategyAgentInput) -> StrategyAgentOutput: ...

    def draft_email(self, inp: EmailDraftAgentInput) -> EmailDraftAgentOutput: ...

    def simulate_supplier(self, inp: SupplierSimAgentInput) -> SupplierSimAgentOutput: ...

    def analyze_reply(self, inp: AnalyzeReplyAgentInput) -> AnalyzeReplyAgentOutput: ...

    def plan_next_move(self, inp: NextMoveAgentInput) -> NextMoveAgentOutput: ...


def _mid(prefix: str = "msg") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# Scripted backend (offline / tests) — mirrors CloudSoft demo arc
# --------------------------------------------------------------------------- #
class ScriptedNegotiationBackend:
    """Deterministic persona — no network. Safe for pytest and Sample demo."""

    # The persona is driven by the scenario (the chosen quotation and the request's limits),
    # so TechNova / DataSphere get their own numbers instead of CloudSoft's script.
    @staticmethod
    def _quoted(scenario) -> tuple[float, float, int]:
        terms = {t.key: t.current_value for t in scenario.terms}

        def num(key, default):
            try:
                return float(terms.get(key) or default)
            except (TypeError, ValueError):
                return float(default)

        pol = scenario.policy
        return (float(scenario.opening_quote_inr), num("uptime_percent", pol.min_uptime_percent),
                int(num("term_months", pol.max_term_months)))

    def strategize(self, inp: StrategyAgentInput) -> StrategyAgentOutput:
        from app.negotiation_policy import buyer_price_ask

        pol = inp.scenario.policy
        target = pol.target_price_inr or pol.budget_ceiling_inr
        ask = buyer_price_ask(inp.scenario)
        rag_note = ""
        if inp.knowledge_excerpts:
            rag_note = (
                f" Grounded on {len(inp.knowledge_excerpts)} approved demo knowledge "
                "excerpts (budget/SLA/liability/DPA)."
            )
        return StrategyAgentOutput(
            strategy_summary=(
                f"Open by asking {inp.scenario.supplier.name} to meet INR {ask:,.0f} "
                f"(budget ceiling {pol.budget_ceiling_inr:,.0f}, target ~{target:,.0f}), hold "
                f"{pol.min_uptime_percent}% uptime and a {pol.max_term_months}-month term, "
                f"confirm DPA; escalate liability to Legal — do not auto-accept."
                f"{rag_note}"
            ),
            recommended_action=NegotiationAction(
                action_type="ask",
                rationale="Clarify commercial fit to budget while locking must-haves.",
                proposed_term_changes={
                    "annual_fee_inr": str(int(ask)),
                    "uptime_percent": str(pol.min_uptime_percent),
                    "term_months": str(pol.max_term_months),
                    "data_protection": "confirm",
                },
            ),
            non_negotiables=[
                f"uptime >= {pol.min_uptime_percent}%",
                f"term <= {pol.max_term_months} months",
                "data protection / DPA",
                "no uncapped liability without Legal",
            ],
            tradable_concessions=list(pol.tradable_low_priority_keys),
        )

    def draft_email(self, inp: EmailDraftAgentInput) -> EmailDraftAgentOutput:
        from app.negotiation_policy import buyer_price_ask

        s = inp.scenario
        pol = s.policy
        fee = inp.action.changes_map().get("annual_fee_inr", str(int(buyer_price_ask(s))))
        return EmailDraftAgentOutput(
            subject=f"{s.requisition_id} — commercial and contractual clarifications",
            body=(
                f"Dear {s.supplier.name} team,\n\n"
                f"Thank you for the proposal at INR {s.opening_quote_inr:,.0f}/year. "
                f"{s.buyer_name} is targeting a ceiling of INR {pol.budget_ceiling_inr:,.0f}. "
                f"Can you confirm whether you can meet approximately INR {fee} while retaining "
                f"{pol.min_uptime_percent}% uptime, a {pol.max_term_months}-month initial term, "
                f"service credits, and data-protection terms?\n\n"
                f"Liability wording will need Legal review separately.\n\n"
                f"Kind regards,\n{s.buyer_name} Procurement"
            ),
            linked_term_keys=list(inp.action.changes_map().keys()),
        )

    def simulate_supplier(self, inp: SupplierSimAgentInput) -> SupplierSimAgentOutput:
        from app.negotiation_policy import buyer_price_ask

        round_n = inp.state.round_number
        sc = inp.scenario
        pol = sc.policy
        quote, q_uptime, q_term = self._quoted(sc)
        ask = buyer_price_ask(sc)
        if round_n <= 1:
            # First reply: a small price move (≈2%), quoted SLA and term kept.
            fee = max(ask, round(quote * 0.98, -4))
            uptime, term = q_uptime, q_term
            concessions = ["price_move"] + (["term"] if term <= pol.max_term_months else []) + (
                ["uptime"] if uptime >= pol.min_uptime_percent else [])
            body = (
                f"We can move to INR {fee:,.0f}/year with a {term}-month term and {uptime}% uptime. "
                "We ask for 60-day renewal notice. Liability wording remains as proposed — "
                "we cannot expand the cap further without a separate discussion."
            )
        else:
            # Second reply: meet the buyer's ask and must-haves; liability stays firm.
            fee = ask
            uptime, term = max(q_uptime, pol.min_uptime_percent), min(q_term, pol.max_term_months)
            concessions = ["price_to_budget", "dpa_confirm"]
            body = (
                f"We can meet INR {fee:,.0f}/year with {term} months, {uptime}% uptime, and DPA "
                f"confirmation. Liability remains subject to our standard wording and "
                f"would need bilateral Legal review — we are not accepting uncapped exposure."
            )
        return SupplierSimAgentOutput(
            subject=f"Re: {sc.requisition_id} — {sc.supplier.name} response",
            body=body,
            offer_update=CurrentOffer(
                annual_fee_inr=fee,
                term_months=term,
                uptime_percent=uptime,
                notes=[f"Scripted supplier reply round {round_n}"],
            ),
            concessions_hint=concessions,
            firm_positions=["liability"],
        )

    def analyze_reply(self, inp: AnalyzeReplyAgentInput) -> AnalyzeReplyAgentOutput:
        from app.negotiation_policy import buyer_price_ask, commercial_within_policy

        offer = inp.state.current_supplier_offer
        fee = offer.annual_fee_inr
        pol = inp.scenario.policy
        if commercial_within_policy(offer, pol):
            action = NegotiationAction(
                action_type="accept",
                rationale="Commercial must-haves appear met; liability still for Legal.",
                proposed_term_changes={
                    "annual_fee_inr": str(int(fee or 0)),
                    "uptime_percent": str(offer.uptime_percent or pol.min_uptime_percent),
                    "term_months": str(offer.term_months or pol.max_term_months),
                    "data_protection": "confirm",
                },
                requires_escalation=True,
                uncertainty_note="Liability not accepted — escalate to Legal.",
            )
            agreed = ["annual_fee_inr", "uptime_percent", "term_months", "data_protection"]
            open_issues = ["liability"]
        else:
            action = NegotiationAction(
                action_type="counteroffer",
                rationale="Push to the policy price; hold must-haves.",
                proposed_term_changes={
                    "annual_fee_inr": str(int(buyer_price_ask(inp.scenario, fee))),
                    "uptime_percent": str(pol.min_uptime_percent),
                    "term_months": str(pol.max_term_months),
                },
            )
            agreed = [k for k, ok in (
                ("uptime_percent", offer.uptime_percent is not None and offer.uptime_percent >= pol.min_uptime_percent),
                ("term_months", offer.term_months is not None and offer.term_months <= pol.max_term_months),
            ) if ok]
            open_issues = [k for k in ("annual_fee_inr", "uptime_percent", "term_months") if k not in agreed] + [
                "liability"]
            if fee is not None and fee <= pol.budget_ceiling_inr and "annual_fee_inr" in open_issues:
                open_issues.remove("annual_fee_inr")
        return AnalyzeReplyAgentOutput(
            interpreted_fee_inr=fee,
            interpreted_term_months=offer.term_months,
            interpreted_uptime_percent=offer.uptime_percent,
            objections=["liability"],
            commitments=agreed,
            open_issues=open_issues,
            agreed_term_keys=agreed,
            recommended_action=action,
        )

    def plan_next_move(self, inp: NextMoveAgentInput) -> NextMoveAgentOutput:
        a = inp.analysis
        pol = inp.scenario.policy
        fee = a.interpreted_fee_inr
        from app.negotiation_policy import commercial_within_policy

        if commercial_within_policy(inp.state.current_supplier_offer, pol) and "liability" in a.open_issues:
            return NextMoveAgentOutput(
                recommended_action=a.recommended_action.model_copy(
                    update={"action_type": "escalate", "requires_escalation": True}
                ),
                strategy_summary=(
                    "Commercial agreement in principle on fee/term/SLA/DPA; "
                    "escalate liability to Legal before any binding commitment."
                ),
                terminal_hint="agreement",
            )
        if fee is not None and fee > pol.budget_ceiling_inr:
            return NextMoveAgentOutput(
                recommended_action=a.recommended_action,
                strategy_summary="Continue counter toward budget ceiling; do not waive must-haves.",
                terminal_hint="continue",
            )
        return NextMoveAgentOutput(
            recommended_action=a.recommended_action,
            strategy_summary=inp.state.strategy_summary or "Continue within policy guardrails.",
            terminal_hint="continue",
        )


# --------------------------------------------------------------------------- #
# Live Gemini backend — compact payloads (full state/events confuse Lite)
# --------------------------------------------------------------------------- #
class _AgentPayload(BaseModel):
    """Minimal JSON envelope for Gemini (avoids dumping event history)."""

    task_context: dict


def _state_snapshot(state: NegotiationState) -> dict:
    return {
        "status": state.status,
        "round_number": state.round_number,
        "current_supplier_offer": state.current_supplier_offer.model_dump(mode="json"),
        "open_issues": list(state.open_issues),
        "agreed_terms": list(state.agreed_terms),
        "strategy_summary": state.strategy_summary,
        "concessions_supplier": list(state.concessions_supplier)[-6:],
        "recent_thread": [
            {
                "direction": m.direction,
                "subject": m.subject,
                "body": (m.body or "")[:900],
                "round_number": m.round_number,
            }
            for m in state.thread[-4:]
        ],
    }


def _scenario_snapshot(scenario: NegotiationScenario) -> dict:
    return {
        "scenario_id": scenario.scenario_id,
        "requisition_id": scenario.requisition_id,
        "buyer_name": scenario.buyer_name,
        "supplier": scenario.supplier.model_dump(mode="json"),
        "opening_quote_inr": scenario.opening_quote_inr,
        "policy": scenario.policy.model_dump(mode="json"),
        "terms": [
            {
                "key": t.key,
                "label": t.label,
                "priority": t.priority,
                "target_value": t.target_value,
                "minimum_acceptable": t.minimum_acceptable,
            }
            for t in scenario.terms
        ],
        "description": scenario.description,
    }


class GeminiNegotiationBackend:
    def _run(self, prompt: str, context: dict, schema, temp: float = 0.2):
        from app.agents import FLASH, _run

        return _run(FLASH, prompt, _AgentPayload(task_context=context), schema, temp)

    def strategize(self, inp: StrategyAgentInput) -> StrategyAgentOutput:
        return self._run(
            STRATEGY_PROMPT,
            {
                "scenario": _scenario_snapshot(inp.scenario),
                "state": _state_snapshot(inp.state),
                "knowledge_excerpts": list(inp.knowledge_excerpts)[:8],
            },
            StrategyAgentOutput,
            0.2,
        )

    def draft_email(self, inp: EmailDraftAgentInput) -> EmailDraftAgentOutput:
        return self._run(
            DRAFT_PROMPT,
            {
                "scenario": _scenario_snapshot(inp.scenario),
                "state": _state_snapshot(inp.state),
                "action": inp.action.model_dump(mode="json"),
            },
            EmailDraftAgentOutput,
            0.3,
        )

    def simulate_supplier(self, inp: SupplierSimAgentInput) -> SupplierSimAgentOutput:
        return self._run(
            SUPPLIER_PROMPT,
            {
                "scenario": _scenario_snapshot(inp.scenario),
                "state": _state_snapshot(inp.state),
                "last_buyer_message": (inp.last_buyer_message or "")[:1200],
            },
            SupplierSimAgentOutput,
            0.4,
        )

    def analyze_reply(self, inp: AnalyzeReplyAgentInput) -> AnalyzeReplyAgentOutput:
        return self._run(
            ANALYZE_PROMPT,
            {
                "scenario": _scenario_snapshot(inp.scenario),
                "state": _state_snapshot(inp.state),
                "supplier_message": (inp.supplier_message or "")[:1200],
            },
            AnalyzeReplyAgentOutput,
            0.1,
        )

    def plan_next_move(self, inp: NextMoveAgentInput) -> NextMoveAgentOutput:
        return self._run(
            NEXT_MOVE_PROMPT,
            {
                "scenario": _scenario_snapshot(inp.scenario),
                "state": _state_snapshot(inp.state),
                "analysis": inp.analysis.model_dump(mode="json"),
            },
            NextMoveAgentOutput,
            0.2,
        )


def get_backend(mode: str = "scripted") -> NegotiationAgentBackend:
    if mode == "live":
        return GeminiNegotiationBackend()
    return ScriptedNegotiationBackend()


# --------------------------------------------------------------------------- #
# Live orchestration steps (policy-gated)
# --------------------------------------------------------------------------- #
def generate_strategy(
    state: NegotiationState,
    scenario: NegotiationScenario,
    *,
    backend: Optional[NegotiationAgentBackend] = None,
    knowledge_excerpts: Optional[list[str]] = None,
) -> NegotiationState:
    if state.status not in ("NOT_STARTED", "NEXT_MOVE_READY"):
        # Allow from NOT_STARTED primarily; NEXT_MOVE may refresh strategy via plan_next_move
        if state.status != "NOT_STARTED":
            raise NegotiationError(
                f"generate_strategy not allowed from {state.status}",
                code="bad_status",
            )
    backend = backend or get_backend("scripted")
    excerpts = list(knowledge_excerpts or [])
    evidence_ids: list[str] = []
    abstained = False
    if not excerpts:
        try:
            from app.rag import retrieve_for_negotiation, validate_citations

            evidence, excerpts, _audit = retrieve_for_negotiation()
            evidence_ids = [e.evidence_id for e in evidence]
            # Drop any hallucinated IDs if an agent later echoes them
            valid, _invalid = validate_citations(evidence_ids, evidence)
            evidence_ids = [e.evidence_id for e in valid]
            abstained = len(excerpts) == 0
        except Exception:
            excerpts = []
            abstained = True
    out = backend.strategize(
        StrategyAgentInput(
            scenario=scenario,
            state=state,
            knowledge_excerpts=excerpts,
        )
    )
    # Prefer strategy that acknowledges retrieved policy when present
    summary = out.strategy_summary
    if excerpts and "demo knowledge" not in summary.lower() and "policy" not in summary.lower():
        summary = (
            summary
            + " Strategy is constrained by retrieved approved demo policy excerpts "
            + f"({len(excerpts)} passages)."
        )
    blocks = validate_proposed_action(state, scenario, out.recommended_action)
    action = out.recommended_action
    if blocks:
        action = action.model_copy(
            update={
                "requires_escalation": True,
                "action_type": "escalate",
                "uncertainty_note": "; ".join(blocks),
            }
        )
    state = state.model_copy(
        update={
            "strategy_summary": summary,
            "pending_action": action,
            "knowledge_excerpts_used": excerpts,
            "knowledge_evidence_ids": evidence_ids,
            "knowledge_abstained": abstained,
        }
    )
    if state.status == "NOT_STARTED":
        state = transition(
            state,
            "STRATEGY_READY",
            actor="buyer_agent",
            reason="Strategy generated",
            event_type="generate_strategy",
            action=action,
        )
    return state


def draft_buyer_email(
    state: NegotiationState,
    scenario: NegotiationScenario,
    *,
    backend: Optional[NegotiationAgentBackend] = None,
) -> NegotiationState:
    if state.status not in ("STRATEGY_READY", "NEXT_MOVE_READY"):
        raise NegotiationError(
            f"draft_buyer_email not allowed from {state.status}",
            code="bad_status",
        )
    action = state.pending_action
    if action is None:
        raise NegotiationError("No pending_action to draft from", code="no_action")
    blocks = validate_proposed_action(state, scenario, action)
    if blocks and action.action_type not in ("escalate", "end"):
        raise NegotiationError(
            "Draft blocked by policy: " + "; ".join(blocks),
            code="policy_block",
        )
    backend = backend or get_backend("scripted")
    drafted = backend.draft_email(
        EmailDraftAgentInput(scenario=scenario, state=state, action=action)
    )
    email = NegotiationEmail(
        message_id=_mid("buyer"),
        thread_id=state.run_id,
        direction="buyer_to_supplier",
        subject=drafted.subject,
        body=drafted.body,
        timestamp=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        round_number=state.round_number + 1,
        simulated=True,
        approval_status="draft",
        linked_term_keys=drafted.linked_term_keys,
    )
    state = state.model_copy(update={"pending_draft": email})
    if state.status == "STRATEGY_READY":
        state = transition(
            state, "DRAFT_READY", actor="buyer_agent", reason="Email drafted",
            event_type="draft_email", message_id=email.message_id, action=action,
        )
    elif state.status == "NEXT_MOVE_READY":
        state = transition(
            state, "DRAFT_READY", actor="buyer_agent", reason="Follow-up email drafted",
            event_type="draft_email", message_id=email.message_id, action=action,
        )
    state = transition(
        state,
        "AWAITING_USER_APPROVAL",
        actor="system",
        reason="Draft ready for human approval (no real send)",
        event_type="await_approval",
        message_id=email.message_id,
    )
    return state


def human_approve_send(state: NegotiationState) -> NegotiationState:
    return approve_and_send_to_simulator(state)


def simulate_supplier_reply(
    state: NegotiationState,
    scenario: NegotiationScenario,
    *,
    backend: Optional[NegotiationAgentBackend] = None,
) -> NegotiationState:
    if state.status != "EMAIL_SENT_TO_SIMULATOR":
        raise NegotiationError(
            f"simulate_supplier_reply not allowed from {state.status}",
            code="bad_status",
        )
    last_buyer = ""
    for m in reversed(state.thread):
        if m.direction == "buyer_to_supplier":
            last_buyer = m.body
            break
    backend = backend or get_backend("scripted")
    out = backend.simulate_supplier(
        SupplierSimAgentInput(
            scenario=scenario, state=state, last_buyer_message=last_buyer
        )
    )
    # Policy-check supplier offer numbers (informational — supplier can propose out of policy)
    email = NegotiationEmail(
        message_id=_mid("supplier"),
        thread_id=state.run_id,
        direction="supplier_to_buyer",
        subject=out.subject,
        body=out.body,
        timestamp=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        round_number=state.round_number,
        simulated=True,
        approval_status="simulated_sent",
        linked_term_keys=[],
    )
    concessions = list(state.concessions_supplier) + list(out.concessions_hint)
    state = state.model_copy(
        update={
            "thread": list(state.thread) + [email],
            "current_supplier_offer": out.offer_update,
            "concessions_supplier": concessions,
        }
    )
    return transition(
        state,
        "SUPPLIER_REPLY_RECEIVED",
        actor="supplier_sim",
        reason="Simulated supplier reply (no real mailbox)",
        event_type="supplier_reply",
        message_id=email.message_id,
    )


def analyze_supplier_reply(
    state: NegotiationState,
    scenario: NegotiationScenario,
    *,
    backend: Optional[NegotiationAgentBackend] = None,
) -> NegotiationState:
    if state.status != "SUPPLIER_REPLY_RECEIVED":
        raise NegotiationError(
            f"analyze_supplier_reply not allowed from {state.status}",
            code="bad_status",
        )
    supplier_msg = ""
    for m in reversed(state.thread):
        if m.direction == "supplier_to_buyer":
            supplier_msg = m.body
            break
    backend = backend or get_backend("scripted")
    out = backend.analyze_reply(
        AnalyzeReplyAgentInput(
            scenario=scenario, state=state, supplier_message=supplier_msg
        )
    )
    from app.negotiation_policy import sanitize_midflow_action

    action = out.recommended_action
    blocks = validate_proposed_action(state, scenario, action)
    if blocks and action.action_type not in ("escalate", "end"):
        # Soft-flag; sanitize_midflow_action may still rewrite escalate→counter
        action = action.model_copy(
            update={
                "requires_escalation": True,
                "uncertainty_note": "; ".join(blocks),
            }
        )
    # Apply interpreted offer onto a working copy before sanitize (needs fee)
    offer = state.current_supplier_offer.model_copy(
        update={
            "annual_fee_inr": out.interpreted_fee_inr
            if out.interpreted_fee_inr is not None
            else state.current_supplier_offer.annual_fee_inr,
            "term_months": out.interpreted_term_months
            if out.interpreted_term_months is not None
            else state.current_supplier_offer.term_months,
            "uptime_percent": out.interpreted_uptime_percent
            if out.interpreted_uptime_percent is not None
            else state.current_supplier_offer.uptime_percent,
        }
    )
    working = state.model_copy(update={"current_supplier_offer": offer})
    action = sanitize_midflow_action(working, scenario, action)
    # Keep liability visible as an open issue for AIP / Legal
    open_issues = list(out.open_issues or state.open_issues)
    if "liability" not in open_issues and "liability" not in (
        out.agreed_term_keys or []
    ):
        open_issues.append("liability")
    state = state.model_copy(
        update={
            "pending_action": action,
            "open_issues": open_issues,
            "agreed_terms": list(
                dict.fromkeys(list(state.agreed_terms) + out.agreed_term_keys)
            ),
            "current_supplier_offer": offer,
        }
    )
    return transition(
        state,
        "RESPONSE_ANALYZED",
        actor="buyer_agent",
        reason="Supplier reply analyzed",
        event_type="analyze_reply",
        action=action,
    )


def plan_next_move(
    state: NegotiationState,
    scenario: NegotiationScenario,
    *,
    backend: Optional[NegotiationAgentBackend] = None,
) -> NegotiationState:
    if state.status != "RESPONSE_ANALYZED":
        raise NegotiationError(
            f"plan_next_move not allowed from {state.status}",
            code="bad_status",
        )
    backend = backend or get_backend("scripted")
    # Reconstruct a minimal analysis view from state for the planner
    analysis = AnalyzeReplyAgentOutput(
        interpreted_fee_inr=state.current_supplier_offer.annual_fee_inr,
        interpreted_term_months=state.current_supplier_offer.term_months,
        interpreted_uptime_percent=state.current_supplier_offer.uptime_percent,
        open_issues=list(state.open_issues),
        agreed_term_keys=list(state.agreed_terms),
        recommended_action=state.pending_action
        or NegotiationAction(action_type="clarify", rationale="No prior action"),
    )
    out = backend.plan_next_move(
        NextMoveAgentInput(scenario=scenario, state=state, analysis=analysis)
    )
    from app.negotiation_policy import resolve_next_move

    # Deterministic policy wins over a premature LLM escalate/AIP
    hint, action = resolve_next_move(
        state, scenario, out.recommended_action, out.terminal_hint
    )

    state = state.model_copy(
        update={
            "pending_action": action,
            "strategy_summary": out.strategy_summary
            or state.strategy_summary
            or action.rationale,
        }
    )

    # RESPONSE_ANALYZED → NEXT_MOVE_READY / ESCALATION / IMPASSE.
    # AGREEMENT_IN_PRINCIPLE only from NEXT_MOVE_READY.
    if hint == "escalate":
        return transition(
            state,
            "ESCALATION_REQUIRED",
            actor="buyer_agent",
            reason=action.uncertainty_note or action.rationale or "Escalation required",
            event_type="next_move",
            action=action,
        )
    if hint == "impasse" or action.action_type == "end":
        return transition(
            state,
            "IMPASSE",
            actor="buyer_agent",
            reason=action.rationale or "Impasse",
            event_type="next_move",
            action=action,
        )

    state = transition(
        state,
        "NEXT_MOVE_READY",
        actor="buyer_agent",
        reason="Next move planned — draft follow-up when ready",
        event_type="next_move",
        action=action,
    )
    if hint == "agreement":
        # Commercial AIP; liability may still be open for Legal (non-binding).
        state = transition(
            state,
            "AGREEMENT_IN_PRINCIPLE",
            actor="buyer_agent",
            reason="Agreement in principle (non-binding); "
            + (action.uncertainty_note or "Legal items may remain open"),
            event_type="agreement_in_principle",
            action=action,
        )
    return state


def run_scripted_live_demo(
    scenario: Optional[NegotiationScenario] = None,
    *,
    max_cycles: int = 2,
) -> NegotiationState:
    """Full offline agent loop (scripted backend) for demos/tests."""
    from app.negotiation import load_scenario, new_run

    scenario = scenario or load_scenario()
    backend = ScriptedNegotiationBackend()
    state = new_run(scenario)
    state = generate_strategy(state, scenario, backend=backend)
    state = draft_buyer_email(state, scenario, backend=backend)
    state = human_approve_send(state)
    state = simulate_supplier_reply(state, scenario, backend=backend)
    state = analyze_supplier_reply(state, scenario, backend=backend)
    state = plan_next_move(state, scenario, backend=backend)

    cycles = 1
    while state.status == "NEXT_MOVE_READY" and cycles < max_cycles:
        state = draft_buyer_email(state, scenario, backend=backend)
        state = human_approve_send(state)
        state = simulate_supplier_reply(state, scenario, backend=backend)
        state = analyze_supplier_reply(state, scenario, backend=backend)
        state = plan_next_move(state, scenario, backend=backend)
        cycles += 1
    return state
