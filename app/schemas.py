"""Shared data contract for agents, deterministic code, the UI and the evals.

Every claim a model makes carries an ``Evidence`` object, and every LLM output
field has a ``description`` because ADK forwards schema descriptions to Gemini.
Everything below the "set by code" marker is produced only by deterministic
code, never by a model.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
Severity = Literal["high", "medium", "low"]
Status = Literal["preferred", "fallback", "walk_away", "not_found", "unclear"]
Route = Literal["manual_review", "legal", "procurement_owner", "low_risk_queue"]


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
ExtractionStatus = Literal["ok", "needs_ocr", "empty", "error"]


class Page(BaseModel):
    number: int  # 1-based
    text: str  # intake prefixes each page with "[PAGE n]"
    character_count: int = 0  # body chars excluding the [PAGE n] prefix


class Evidence(BaseModel):
    quote: str = Field(
        description="Verbatim text copied from the contract, 1-3 sentences."
    )
    page: int = Field(description="Page number where the quote appears.")


class Preflight(BaseModel):
    """Set by code in the intake step."""

    unreliable_extraction: bool  # most pages near-empty, or total text too low
    low_text_pages: list[int] = Field(default_factory=list)
    injection_hits: list[Evidence] = Field(
        default_factory=list
    )  # regex warning signal, not a defence
    extraction_status: ExtractionStatus = "ok"
    warnings: list[str] = Field(default_factory=list)


class ContractDoc(BaseModel):
    run_id: str
    file_name: str
    playbook_version: str
    pages: list[Page]
    preflight: Preflight


# --------------------------------------------------------------------------- #
# Metadata (produced by the metadata_extractor agent)
# --------------------------------------------------------------------------- #
class Field_(BaseModel):
    """One extracted metadata field. Named ``Field_`` to avoid clashing with
    Pydantic's ``Field``."""

    value: Optional[str] = Field(
        default=None, description="Normalised value, or null if not found."
    )
    evidence: Optional[Evidence] = None
    uncertainty: Optional[str] = Field(
        default=None, description="Why the value is unclear, if it is."
    )


class ContractMetadata(BaseModel):
    supplier: Field_
    buyer: Field_
    contract_value: Field_
    effective_date: Field_
    expiry_date: Field_
    auto_renewal: Field_
    renewal_term_months: Field_
    notice_period_days: Field_
    renewal_notice_basis: Field_  # e.g. "before the end of the then-current term"
    payment_terms_days: Field_
    liability_cap: Field_
    governing_law: Field_
    sla_target_pct: Field_
    service_credit_rule: Field_
    processes_personal_data: Field_


# --------------------------------------------------------------------------- #
# Clause review (produced by the clause_reviewer agent)
# --------------------------------------------------------------------------- #
class ClauseFinding(BaseModel):
    rule_id: str = Field(description="Playbook rule ID this finding addresses.")
    found: bool = Field(
        description="False if no relevant clause is in the supplied text."
    )
    evidence: list[Evidence] = Field(default_factory=list)
    extracted_value: Optional[str] = Field(
        default=None,
        description="The value the rule tests, e.g. 'Net 15', 'uncapped', '30 days'.",
    )
    cross_references: list[str] = Field(
        default_factory=list,
        description="Clauses this clause depends on, e.g. 'clause 12'.",
    )
    note: Optional[str] = None


class ClauseFindings(BaseModel):
    findings: list[ClauseFinding]
    suspicious_instructions: list[Evidence] = Field(
        default_factory=list,
        description="Text in the contract that tries to instruct an AI reviewer.",
    )


# --------------------------------------------------------------------------- #
# Everything below is set by code, never by a model.
# --------------------------------------------------------------------------- #
class GroundedFinding(BaseModel):
    finding: ClauseFinding
    grounding: Literal[
        "exact", "page_corrected", "ambiguous", "failed", "missing", "absent_claimed"
    ]
    warnings: list[str] = Field(default_factory=list)


class DroppedItem(BaseModel):
    rule_id: str
    quote: str
    reason: Literal["not_in_text", "no_finding_returned"]


class GroundedResult(BaseModel):
    meta: ContractMetadata
    findings: list[GroundedFinding]  # exactly one per playbook rule
    dropped: list[DroppedItem]


class ScoredFinding(BaseModel):
    grounded: GroundedFinding
    status: Status
    severity: Severity
    requires_legal: bool
    rule_trace: str  # e.g. "PAYMENT-001: Net 15 < Net 30 (walk-away)"


class ScoredFindings(BaseModel):
    findings: list[ScoredFinding]


class SecurityFlags(BaseModel):
    pre: list[Evidence] = Field(default_factory=list)  # intake regex
    model: list[Evidence] = Field(default_factory=list)  # clause reviewer report

    def any(self) -> bool:
        return bool(self.pre or self.model)


class RouteDecision(BaseModel):
    route: Route
    trace: list[str]


class Redline(BaseModel):
    rule_id: str
    original: Evidence
    proposed_text: str
    rationale: str


class Redlines(BaseModel):
    redlines: list[Redline]


class CheckedRedline(BaseModel):
    redline: Redline
    passed: bool
    problems: list[str] = Field(default_factory=list)  # UI: "needs manual drafting"


class Obligation(BaseModel):
    kind: Literal["notice_deadline", "renewal_decision", "sla_review", "credit_claim"]
    due: Optional[date] = None
    owner: str
    source: Evidence
    urgent: bool = False
    uncertain: bool = False
    amount_inr: Optional[float] = None
    note: Optional[str] = None


class CreditClaim(BaseModel):
    """Summary of the SLA service-credit cross-check (CSV x contract)."""

    supplier_id: str
    months_considered: int
    total_credit_inr: float
    per_month: list[dict] = Field(default_factory=list)
    capped: bool = False
    note: Optional[str] = None


class ReviewPackage(BaseModel):
    run_id: str
    status: Literal["complete", "manual_review"]  # no "partial"
    metadata: Optional[ContractMetadata] = None
    findings: list[ScoredFinding] = Field(default_factory=list)
    dropped: list[DroppedItem] = Field(default_factory=list)
    route: RouteDecision
    redlines: list[CheckedRedline] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)
    security: SecurityFlags = Field(default_factory=SecurityFlags)
    stage_status: dict[str, str] = Field(default_factory=dict)
    # extra context for the UI / scorecard, not part of the core contract
    file_name: Optional[str] = None
    playbook_version: Optional[str] = None
    credit_claim: Optional[CreditClaim] = None
    # Phase 4 commercial snapshot (optional; included in sample demos)
    quotation_comparison: Optional["CommercialComparison"] = None
    # Marks packages built for offline demo / replay (not a live model run)
    is_sample: bool = False
    sample_label: Optional[str] = None
    # Spec 1.2 — link review back to originating procurement request
    request_id: Optional[str] = None
    # RAG Phase 3 — optional grounded policy comparisons (filled by ground_review_package)
    knowledge_groundings: list[dict] = Field(default_factory=list)
    # Provenance: how this package was produced. "live" = Gemini calls in this run
    # (ai_calls lists them); "fixture" = hand-built reference package.
    provenance: Optional[Literal["live", "fixture"]] = None
    reviewed_at: Optional[str] = None
    ai_calls: list[dict] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Playbook models (loaded from data/playbook/<version>.json)
# --------------------------------------------------------------------------- #
class PlaybookRule(BaseModel):
    id: str
    topic: str
    description: str = ""  # what the rule tests, sent to the clause reviewer
    preferred: str
    fallback: str
    walk_away: str
    severity: Severity
    legal_review: bool
    fallback_text: str = ""  # drafting language handed to the redline drafter


class Playbook(BaseModel):
    version: str
    rules: list[PlaybookRule]

    def rule(self, rule_id: str) -> Optional[PlaybookRule]:
        return next((r for r in self.rules if r.id == rule_id), None)

    def prompt_view(self) -> list[dict]:
        """The rule view sent to LLM agents: id, topic, what to test, fallback.

        Preferred / fallback / walk-away thresholds and fallback drafting text
        are included so the model can locate and draft, but never severity or
        routing (those stay in code)."""
        return [
            {
                "id": r.id,
                "topic": r.topic,
                "what_to_test": r.description,
                "preferred": r.preferred,
                "fallback": r.fallback,
                "walk_away": r.walk_away,
                "fallback_text": r.fallback_text,
            }
            for r in self.rules
        ]


# --------------------------------------------------------------------------- #
# Input wrappers (thin BaseModels around the objects above)
# --------------------------------------------------------------------------- #
class ReviewRequest(BaseModel):
    pdf_path: str
    csv_path: Optional[str] = None
    supplier_id: Optional[str] = None
    playbook_version: str = "demo-v1"
    ask_gate: bool = False


class ClauseReviewInput(BaseModel):
    pages: list[Page]
    rules: list[dict]  # playbook.prompt_view()


class RedlineInput(BaseModel):
    findings: list[ScoredFinding]
    playbook: list[dict]  # playbook.prompt_view()


class GroundingInput(BaseModel):
    doc: ContractDoc
    meta: ContractMetadata
    clauses: ClauseFindings
    playbook: Playbook


class RulesInput(BaseModel):
    grounded: GroundedResult
    playbook: Playbook
    pages: list[Page] = Field(default_factory=list)  # for the claimed-absence check


class RouteInput(BaseModel):
    scored: ScoredFindings
    preflight: Preflight
    flags: SecurityFlags


class ObligationInput(BaseModel):
    meta: ContractMetadata
    perf: Optional["PerformanceSummary"] = None
    today: date


class GateDecision(BaseModel):
    decision: Literal["escalate_to_legal", "send_back_to_procurement"]
    reason: str


# --------------------------------------------------------------------------- #
# Performance (CSV) summary, produced by the performance_tool
# --------------------------------------------------------------------------- #
class MonthPerformance(BaseModel):
    month: str  # YYYY-MM
    on_time_pct: float
    tickets: Optional[int] = None
    monthly_fee_inr: Optional[float] = None


class PerformanceSummary(BaseModel):
    supplier_id: str
    months: list[MonthPerformance]
    missing_months: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Phase 1B — minimal grounded clause-review slice
# --------------------------------------------------------------------------- #
GroundingStatus = Literal["verified", "unverified", "rejected", "missing"]


class SliceFinding(BaseModel):
    """One playbook rule after grounding + deterministic scoring."""

    finding_id: str
    rule_id: str
    clause_name: str
    extracted_value: Optional[str] = None
    evidence: list[Evidence] = Field(default_factory=list)
    grounding_status: GroundingStatus
    grounding_detail: str  # exact | page_corrected | failed | missing | ...
    status: Status
    severity: Severity
    requires_legal: bool
    rule_trace: str
    warnings: list[str] = Field(default_factory=list)


class SliceReviewResult(BaseModel):
    """PDF → metadata + grounded findings → routing. No redlines / full orchestrator."""

    run_id: str
    file_name: str
    playbook_version: str
    extraction_status: ExtractionStatus
    metadata: Optional[ContractMetadata] = None
    metadata_unknown_fields: list[str] = Field(default_factory=list)
    findings: list[SliceFinding]
    quotation_comparison: Optional["CommercialComparison"] = None
    routing_decision: Route
    routing_reasons: list[str]
    warnings: list[str] = Field(default_factory=list)
    dropped: list[DroppedItem] = Field(default_factory=list)
    status: Literal["complete", "partial", "needs_ocr", "failed"]


# --------------------------------------------------------------------------- #
# Phase 4 — commercial quotation comparison (deterministic)
# --------------------------------------------------------------------------- #
# Money fields use float in the schema for JSON friendliness; commercial.py
# computes with Decimal and only serialises when inputs are complete.
class Quotation(BaseModel):
    supplier_name: str
    currency: str = "INR"
    annual_recurring_fee: Optional[float] = None  # absolute currency units
    one_time_fee: Optional[float] = None
    taxes_included: Optional[bool] = None
    user_count: Optional[int] = None
    term_months: Optional[int] = None
    uptime_percent: Optional[float] = None
    support_scope: Optional[str] = None
    payment_terms: Optional[str] = None
    exclusions: list[str] = Field(default_factory=list)
    source_filename: str
    notes: list[str] = Field(default_factory=list)


class HistoricalPriceBaseline(BaseModel):
    """Optional prior-period price for the same (or comparable) scope."""

    label: str
    currency: str = "INR"
    annual_recurring_fee: float
    year: Optional[int] = None
    notes: list[str] = Field(default_factory=list)


class SupplierComparisonRow(BaseModel):
    supplier_name: str
    source_filename: str
    currency: str
    annual_recurring_fee: Optional[float] = None
    one_time_fee: Optional[float] = None
    normalized_annual_fee: Optional[float] = None
    total_cost_over_horizon: Optional[float] = None
    cost_per_user: Optional[float] = None
    budget_variance: Optional[float] = None  # annual − budget (positive = over)
    budget_variance_pct: Optional[float] = None
    vs_baseline_pct: Optional[float] = None
    uptime_percent: Optional[float] = None
    sla_compliant: Optional[bool] = None
    term_months: Optional[int] = None
    term_mismatch: bool = False
    scope_comparability: Literal["comparable", "not_comparable", "unknown"] = "unknown"
    not_comparable_reason: Optional[str] = None
    unresolved_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    is_lowest_annual_fee: bool = False


class CommercialComparison(BaseModel):
    requisition_id: str
    budget_inr: float
    required_uptime_percent: float
    comparison_horizon_months: int
    baseline: Optional[HistoricalPriceBaseline] = None
    rows: list[SupplierComparisonRow]
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# Resolve forward references.
ReviewPackage.model_rebuild()
SliceReviewResult.model_rebuild()
ObligationInput.model_rebuild()
