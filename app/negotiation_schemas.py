"""Negotiation simulator domain models (updated Final spec §16).

State transitions are validated in Python — never by free-form LLM output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

NegotiationStatus = Literal[
    "NOT_STARTED",
    "STRATEGY_READY",
    "DRAFT_READY",
    "AWAITING_USER_APPROVAL",
    "EMAIL_SENT_TO_SIMULATOR",
    "SUPPLIER_REPLY_RECEIVED",
    "RESPONSE_ANALYZED",
    "NEXT_MOVE_READY",
    "AGREEMENT_IN_PRINCIPLE",
    "ESCALATION_REQUIRED",
    "IMPASSE",
    "CANCELLED",
]

TermPriority = Literal["must_have", "high", "medium", "low"]
EmailDirection = Literal["buyer_to_supplier", "supplier_to_buyer"]
ApprovalStatus = Literal["draft", "approved", "rejected", "simulated_sent"]
ActionType = Literal[
    "ask", "counteroffer", "accept", "reject", "clarify", "escalate", "end"
]


class NegotiationTerm(BaseModel):
    key: str
    label: str
    current_value: Optional[str] = None
    target_value: Optional[str] = None
    minimum_acceptable: Optional[str] = None
    priority: TermPriority = "medium"
    unit: Optional[str] = None  # e.g. INR, percent, months
    evidence_source: Optional[str] = None
    approval_owner: Optional[str] = None  # procurement | legal | finance
    resolved: bool = False


class ConcessionPolicy(BaseModel):
    """Deterministic limits the model cannot override."""

    budget_ceiling_inr: float
    price_floor_inr: float  # never go below without escalation
    target_price_inr: Optional[float] = None
    min_uptime_percent: float = 99.9
    max_term_months: int = 12
    never_waive_data_protection: bool = True
    liability_requires_legal: bool = True
    tradable_low_priority_keys: list[str] = Field(default_factory=list)
    max_rounds: int = 4


class SupplierPersona(BaseModel):
    name: str = "CloudSoft"
    style: str = "price_focused"  # illustrative persona label
    firm_on: list[str] = Field(default_factory=list)
    flexible_on: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class NegotiationScenario(BaseModel):
    scenario_id: str
    requisition_id: str = "PR-2026-1042"
    buyer_name: str = "ABC Technologies"
    supplier: SupplierPersona
    opening_quote_inr: float
    policy: ConcessionPolicy
    terms: list[NegotiationTerm]
    simulation_mode: bool = True
    description: str = ""


class TermChange(BaseModel):
    """Single proposed term update (key/value list — Gemini-safe structured output)."""

    key: str
    value: str


class NegotiationAction(BaseModel):
    action_type: ActionType
    rationale: str
    # list[TermChange] (not bare dict) so response_schema works on Gemini Developer API
    proposed_term_changes: list[TermChange] = Field(default_factory=list)
    referenced_finding_ids: list[str] = Field(default_factory=list)
    requires_escalation: bool = False
    uncertainty_note: Optional[str] = None

    @field_validator("proposed_term_changes", mode="before")
    @classmethod
    def _coerce_term_changes(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, dict):
            return [{"key": str(k), "value": str(val)} for k, val in v.items()]
        return v

    def changes_map(self) -> dict[str, str]:
        return {c.key: c.value for c in self.proposed_term_changes}


class NegotiationEmail(BaseModel):
    message_id: str
    thread_id: str
    direction: EmailDirection
    subject: str
    body: str
    timestamp: str  # ISO-8601
    round_number: int
    simulated: bool = True
    approval_status: ApprovalStatus = "draft"
    linked_term_keys: list[str] = Field(default_factory=list)
    linked_finding_ids: list[str] = Field(default_factory=list)


class NegotiationEvent(BaseModel):
    event_id: str
    actor: Literal["buyer_agent", "supplier_sim", "human", "system"]
    event_type: str
    timestamp: str
    state_before: NegotiationStatus
    state_after: NegotiationStatus
    message_id: Optional[str] = None
    action: Optional[NegotiationAction] = None
    reason: str = ""


class CurrentOffer(BaseModel):
    annual_fee_inr: Optional[float] = None
    term_months: Optional[int] = None
    uptime_percent: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


class NegotiationState(BaseModel):
    run_id: str
    scenario_id: str
    status: NegotiationStatus = "NOT_STARTED"
    round_number: int = 0
    current_supplier_offer: CurrentOffer = Field(default_factory=CurrentOffer)
    buyer_last_approved_offer: Optional[CurrentOffer] = None
    open_issues: list[str] = Field(default_factory=list)
    agreed_terms: list[str] = Field(default_factory=list)
    rejected_terms: list[str] = Field(default_factory=list)
    concessions_buyer: list[str] = Field(default_factory=list)
    concessions_supplier: list[str] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)
    strategy_summary: Optional[str] = None
    pending_draft: Optional[NegotiationEmail] = None
    pending_action: Optional[NegotiationAction] = None
    thread: list[NegotiationEmail] = Field(default_factory=list)
    events: list[NegotiationEvent] = Field(default_factory=list)
    # RAG grounding (Phase 3) — citations fed to the strategist
    knowledge_excerpts_used: list[str] = Field(default_factory=list)
    knowledge_evidence_ids: list[str] = Field(default_factory=list)
    knowledge_abstained: bool = False
    non_binding_notice: str = (
        "Agreement in principle is non-binding and subject to human approval "
        "and final contract review."
    )


class NegotiationReplayStep(BaseModel):
    """One deterministic step in an offline replay fixture."""

    label: str
    from_status: NegotiationStatus
    to_status: NegotiationStatus
    actor: Literal["buyer_agent", "supplier_sim", "human", "system"]
    reason: str
    email: Optional[NegotiationEmail] = None
    action: Optional[NegotiationAction] = None
    supplier_offer_update: Optional[CurrentOffer] = None
    buyer_offer_update: Optional[CurrentOffer] = None
    open_issues: Optional[list[str]] = None
    agreed_terms: Optional[list[str]] = None
    concessions_buyer: Optional[list[str]] = None
    concessions_supplier: Optional[list[str]] = None
    strategy_summary: Optional[str] = None
    approve_pending_draft: bool = False


class NegotiationReplayFixture(BaseModel):
    fixture_id: str
    scenario_id: str
    description: str
    expected_final_status: NegotiationStatus
    steps: list[NegotiationReplayStep]
