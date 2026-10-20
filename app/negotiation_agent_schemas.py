"""I/O schemas for Milestone F negotiation LLM agents.

Agents propose; deterministic code in ``app.negotiation`` validates and
transitions state. The model never sets NegotiationState.status directly.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.negotiation_schemas import (
    CurrentOffer,
    NegotiationAction,
    NegotiationScenario,
    NegotiationState,
)


class StrategyAgentInput(BaseModel):
    scenario: NegotiationScenario
    state: NegotiationState
    knowledge_excerpts: list[str] = Field(default_factory=list)


class StrategyAgentOutput(BaseModel):
    strategy_summary: str = Field(
        description="2-4 sentences: opening position, must-haves, what not to concede."
    )
    recommended_action: NegotiationAction
    non_negotiables: list[str] = Field(default_factory=list)
    tradable_concessions: list[str] = Field(default_factory=list)
    uncertainty_note: Optional[str] = None


class EmailDraftAgentInput(BaseModel):
    scenario: NegotiationScenario
    state: NegotiationState
    action: NegotiationAction


class EmailDraftAgentOutput(BaseModel):
    subject: str
    body: str
    linked_term_keys: list[str] = Field(default_factory=list)


class SupplierSimAgentInput(BaseModel):
    scenario: NegotiationScenario
    state: NegotiationState
    last_buyer_message: str


class SupplierSimAgentOutput(BaseModel):
    subject: str
    body: str
    offer_update: CurrentOffer
    concessions_hint: list[str] = Field(default_factory=list)
    firm_positions: list[str] = Field(default_factory=list)


class AnalyzeReplyAgentInput(BaseModel):
    scenario: NegotiationScenario
    state: NegotiationState
    supplier_message: str


class AnalyzeReplyAgentOutput(BaseModel):
    interpreted_fee_inr: Optional[float] = None
    interpreted_term_months: Optional[int] = None
    interpreted_uptime_percent: Optional[float] = None
    objections: list[str] = Field(default_factory=list)
    commitments: list[str] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)
    agreed_term_keys: list[str] = Field(default_factory=list)
    recommended_action: NegotiationAction
    uncertainty_note: Optional[str] = None


class NextMoveAgentInput(BaseModel):
    scenario: NegotiationScenario
    state: NegotiationState
    analysis: AnalyzeReplyAgentOutput


class NextMoveAgentOutput(BaseModel):
    recommended_action: NegotiationAction
    strategy_summary: str
    terminal_hint: Optional[str] = Field(
        default=None,
        description="agreement | escalate | impasse | continue",
    )
