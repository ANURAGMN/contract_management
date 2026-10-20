"""Phase 1A (spec 1.2) — procurement request intake and triage.

Deterministic create/load/update, completeness checks, status transitions,
duplicate warnings, and routing recommendations. No LLM sets status.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "procurement_requests"
STORE = ROOT / "runs" / "procurement_requests"

RequestStatus = Literal[
    "DRAFT",
    "SUBMITTED",
    "NEEDS_INFORMATION",
    "PROCUREMENT_TRIAGE",
    "FINANCE_REVIEW",
    "SECURITY_REVIEW",
    "LEGAL_REVIEW",
    "APPROVED_FOR_SOURCING",
    "SOURCING_IN_PROGRESS",
    "CONTRACT_REVIEW",
    "APPROVED_FOR_SIGNATURE",
    "REJECTED",
    "CANCELLED",
]

Category = Literal[
    "saas_software",
    "services",
    "hardware",
    "consulting",
    "other",
]

PurchaseType = Literal["new_purchase", "renewal"]

DataSensitivity = Literal[
    "public",
    "internal",
    "confidential",
    "personal_data",
    "regulated",
    "unknown",
]

SourcingMethod = Literal[
    "to_be_decided",
    "rfp",
    "rfq",
    "single_source",
    "renewal",
]

Team = Literal[
    "procurement",
    "finance",
    "security_privacy",
    "legal",
    "business_owner",
]

# Allowed transitions (from -> set of to). LLM cannot bypass this map.
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "DRAFT": {"SUBMITTED", "CANCELLED"},
    "SUBMITTED": {
        "NEEDS_INFORMATION",
        "PROCUREMENT_TRIAGE",
        "FINANCE_REVIEW",
        "SECURITY_REVIEW",
        "CANCELLED",
    },
    "NEEDS_INFORMATION": {"DRAFT", "SUBMITTED", "CANCELLED"},
    "PROCUREMENT_TRIAGE": {
        "FINANCE_REVIEW",
        "SECURITY_REVIEW",
        "LEGAL_REVIEW",
        "APPROVED_FOR_SOURCING",
        "NEEDS_INFORMATION",
        "REJECTED",
        "CANCELLED",
    },
    "FINANCE_REVIEW": {
        "PROCUREMENT_TRIAGE",
        "SECURITY_REVIEW",
        "APPROVED_FOR_SOURCING",
        "NEEDS_INFORMATION",
        "REJECTED",
        "CANCELLED",
    },
    "SECURITY_REVIEW": {
        "PROCUREMENT_TRIAGE",
        "LEGAL_REVIEW",
        "APPROVED_FOR_SOURCING",
        "NEEDS_INFORMATION",
        "REJECTED",
        "CANCELLED",
    },
    "LEGAL_REVIEW": {
        "PROCUREMENT_TRIAGE",
        "APPROVED_FOR_SOURCING",
        "NEEDS_INFORMATION",
        "REJECTED",
        "CANCELLED",
    },
    "APPROVED_FOR_SOURCING": {
        "SOURCING_IN_PROGRESS",
        "CANCELLED",
    },
    "SOURCING_IN_PROGRESS": {
        "CONTRACT_REVIEW",
        "CANCELLED",
    },
    "CONTRACT_REVIEW": {
        "APPROVED_FOR_SIGNATURE",
        "SOURCING_IN_PROGRESS",
        "REJECTED",
        "CANCELLED",
    },
    "APPROVED_FOR_SIGNATURE": set(),
    "REJECTED": set(),
    "CANCELLED": set(),
}

# Fields required before SUBMITTED can succeed without NEEDS_INFORMATION
REQUIRED_FOR_SUBMIT = [
    "title",
    "requesting_team",
    "business_owner",
    "business_justification",
    "category",
    "new_purchase_or_renewal",
    "number_of_users",
    "desired_start_date",
    "budget_amount",
    "budget_period",
    "currency",
    "requested_contract_term_months",
    "required_sla",
    "data_sensitivity",
]


class AuditEvent(BaseModel):
    event_id: str
    timestamp: datetime
    actor: str
    action: str
    detail: str = ""
    before_status: Optional[str] = None
    after_status: Optional[str] = None


class RoutingRecommendation(BaseModel):
    team: Team
    reason_code: str
    reason: str
    mandatory: bool = True


class RiskFlag(BaseModel):
    code: str
    message: str
    severity: Literal["low", "medium", "high"] = "medium"


class DuplicateCandidate(BaseModel):
    request_id: str
    title: str
    category: str
    requesting_team: str
    status: str
    match_score: float
    match_reasons: list[str] = Field(default_factory=list)


class ProcurementRequest(BaseModel):
    request_id: str
    title: str = ""
    requesting_team: str = ""
    requester_role: str = ""
    business_owner: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    business_justification: str = ""
    problem_statement: str = ""
    expected_outcomes: list[str] = Field(default_factory=list)
    requested_capabilities: list[str] = Field(default_factory=list)
    number_of_users: Optional[int] = None
    category: Optional[Category] = None
    new_purchase_or_renewal: Optional[PurchaseType] = None
    existing_supplier: Optional[str] = None
    desired_start_date: Optional[date] = None
    estimated_one_time_cost: Optional[float] = None
    estimated_recurring_cost: Optional[float] = None
    currency: str = "INR"
    budget_amount: Optional[float] = None
    budget_period: Optional[str] = None  # annual | monthly | project
    cost_center: Optional[str] = None
    budget_confirmed: bool = False
    requested_contract_term_months: Optional[int] = None
    required_sla: Optional[str] = None
    data_types: list[str] = Field(default_factory=list)
    data_sensitivity: Optional[DataSensitivity] = None
    security_review_required: bool = False
    privacy_review_required: bool = False
    integrations: list[str] = Field(default_factory=list)
    business_continuity_requirements: Optional[str] = None
    preferred_supplier: Optional[str] = None
    sourcing_method: SourcingMethod = "to_be_decided"
    attachments: list[str] = Field(default_factory=list)
    status: RequestStatus = "DRAFT"
    missing_fields: list[str] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    routing_recommendations: list[RoutingRecommendation] = Field(default_factory=list)
    audit_events: list[AuditEvent] = Field(default_factory=list)
    version: int = 1
    label: str = "Synthetic demo data — not a real purchase requisition"
    linked_contract_run_ids: list[str] = Field(default_factory=list)
    linked_quotation_ids: list[str] = Field(default_factory=list)


class InvalidTransitionError(ValueError):
    pass


class UnknownRequestError(KeyError):
    pass


# --------------------------------------------------------------------------- #
# IDs / audit helpers
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _event_id(request_id: str, action: str, ts: datetime, detail: str = "") -> str:
    raw = f"{request_id}|{action}|{ts.isoformat()}|{detail}"
    return "praud-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _append_audit(
    req: ProcurementRequest,
    *,
    actor: str,
    action: str,
    detail: str = "",
    before: Optional[str] = None,
    after: Optional[str] = None,
    ts: Optional[datetime] = None,
) -> None:
    """Append audit event; skip exact duplicates (same action+detail+statuses)."""
    ts = ts or _now()
    eid = _event_id(req.request_id, action, ts, detail)
    if any(e.event_id == eid for e in req.audit_events):
        return
    # Suppress only an immediate repeat of the previous event (double-click /
    # Streamlit rerun). Comparing against older events hid legitimate repeats,
    # e.g. a resubmitted request routed to SECURITY_REVIEW a second time.
    if req.audit_events:
        e = req.audit_events[-1]
        if (
            e.action == action
            and e.detail == detail
            and e.before_status == before
            and e.after_status == after
        ):
            return
    req.audit_events.append(
        AuditEvent(
            event_id=eid,
            timestamp=ts,
            actor=actor,
            action=action,
            detail=detail,
            before_status=before,
            after_status=after,
        )
    )


def new_request_id(prefix: str = "PR-2026") -> str:
    """Generate a demo request id (stable only when persisted)."""
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    return f"{prefix}-{stamp}"


# --------------------------------------------------------------------------- #
# Completeness, risks, routing
# --------------------------------------------------------------------------- #
def compute_missing_fields(req: ProcurementRequest) -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_FOR_SUBMIT:
        val = getattr(req, name, None)
        if val is None or val == "" or val == []:
            missing.append(name)
    if req.new_purchase_or_renewal == "renewal" and not req.existing_supplier:
        missing.append("existing_supplier")
    return missing


def compute_risk_flags(req: ProcurementRequest) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    if not req.budget_confirmed:
        flags.append(
            RiskFlag(
                code="budget_unconfirmed",
                message="Budget is not confirmed — Finance/Procurement review recommended.",
                severity="high",
            )
        )
    if (
        req.budget_amount is not None
        and req.estimated_recurring_cost is not None
        and req.estimated_recurring_cost > req.budget_amount
    ):
        flags.append(
            RiskFlag(
                code="over_budget",
                message=(
                    f"Estimated recurring cost ({req.estimated_recurring_cost}) exceeds "
                    f"budget ({req.budget_amount} {req.currency})."
                ),
                severity="high",
            )
        )
    sens = req.data_sensitivity or "unknown"
    if sens in ("confidential", "personal_data", "regulated") or req.security_review_required:
        flags.append(
            RiskFlag(
                code="security_review",
                message="Sensitive/hosted data or explicit security review flag.",
                severity="high",
            )
        )
    if sens in ("personal_data", "regulated") or req.privacy_review_required:
        flags.append(
            RiskFlag(
                code="privacy_review",
                message="Personal/regulated data or privacy review required.",
                severity="high",
            )
        )
    if req.integrations:
        flags.append(
            RiskFlag(
                code="integrations",
                message="Integrations declared — include in security triage.",
                severity="medium",
            )
        )
    if not req.required_sla:
        flags.append(
            RiskFlag(
                code="sla_unspecified",
                message="Required SLA not specified.",
                severity="medium",
            )
        )
    if not req.requested_contract_term_months:
        flags.append(
            RiskFlag(
                code="term_unspecified",
                message="Requested contract term not specified.",
                severity="medium",
            )
        )
    if req.new_purchase_or_renewal == "renewal":
        flags.append(
            RiskFlag(
                code="renewal",
                message="Renewal request — check incumbent terms and notice windows.",
                severity="low",
            )
        )
    return flags


def compute_routing(req: ProcurementRequest) -> list[RoutingRecommendation]:
    """Deterministic routing from request facts (not LLM)."""
    recs: list[RoutingRecommendation] = []
    seen: set[str] = set()

    def add(team: Team, code: str, reason: str, mandatory: bool = True) -> None:
        key = f"{team}|{code}"
        if key in seen:
            return
        seen.add(key)
        recs.append(
            RoutingRecommendation(
                team=team, reason_code=code, reason=reason, mandatory=mandatory
            )
        )

    add("procurement", "intake_triage", "All submitted requests require Procurement triage.")
    add("business_owner", "business_need", "Business Owner confirms need and scope.")

    if not req.budget_confirmed or any(f.code == "over_budget" for f in req.risk_flags):
        add(
            "finance",
            "budget",
            "Unconfirmed budget or estimated cost above budget.",
        )

    if any(f.code in ("security_review", "integrations") for f in req.risk_flags) or req.security_review_required:
        add(
            "security_privacy",
            "security",
            "Security review triggered by data sensitivity, integrations, or flag.",
        )

    if any(f.code == "privacy_review" for f in req.risk_flags) or req.privacy_review_required:
        add(
            "security_privacy",
            "privacy",
            "Privacy review triggered by personal/regulated data or flag.",
        )
        add(
            "legal",
            "dpa_likely",
            "Personal/regulated data may require contractual DPA language.",
            mandatory=False,
        )

    if req.category == "saas_software" and req.requested_contract_term_months and req.requested_contract_term_months > 12:
        add(
            "legal",
            "multi_year_term",
            "Term longer than 12 months — Legal review of lock-in recommended.",
            mandatory=False,
        )

    return recs


def refresh_derived(req: ProcurementRequest) -> ProcurementRequest:
    req.missing_fields = compute_missing_fields(req)
    req.risk_flags = compute_risk_flags(req)
    req.routing_recommendations = compute_routing(req)
    req.updated_at = _now()
    return req


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _ensure_store() -> Path:
    STORE.mkdir(parents=True, exist_ok=True)
    return STORE


def request_path(request_id: str, *, fixture_ok: bool = True) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", request_id)
    run_path = _ensure_store() / f"{safe}.json"
    if run_path.exists():
        return run_path
    if fixture_ok:
        fix = FIXTURES / f"{safe}.json"
        if fix.exists():
            return fix
    return run_path


def save_request(req: ProcurementRequest) -> Path:
    path = _ensure_store() / f"{req.request_id}.json"
    path.write_text(req.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_request(request_id: str) -> ProcurementRequest:
    path = request_path(request_id)
    if not path.exists():
        raise UnknownRequestError(request_id)
    return ProcurementRequest.model_validate_json(path.read_text(encoding="utf-8"))


def list_request_ids() -> list[str]:
    ids: set[str] = set()
    if FIXTURES.exists():
        for p in FIXTURES.glob("*.json"):
            ids.add(p.stem)
    if STORE.exists():
        for p in STORE.glob("*.json"):
            ids.add(p.stem)
    return sorted(ids)


def load_fixture_pr2026() -> ProcurementRequest:
    return load_request("PR-2026-1042")


def reset_fixture_to_store(request_id: str = "PR-2026-1042") -> ProcurementRequest:
    """Copy the committed fixture into the run store as a fresh DRAFT (demo reset)."""
    fix = FIXTURES / f"{request_id}.json"
    if not fix.exists():
        raise UnknownRequestError(request_id)
    req = ProcurementRequest.model_validate_json(fix.read_text(encoding="utf-8"))
    req.status = "DRAFT"
    req.audit_events = []
    req.linked_contract_run_ids = []
    refresh_derived(req)
    _append_audit(req, actor="system", action="fixture_reset", detail="Loaded fixture as DRAFT", after="DRAFT")
    save_request(req)
    return req


# --------------------------------------------------------------------------- #
# CRUD + transitions
# --------------------------------------------------------------------------- #
def create_request(
    data: dict | ProcurementRequest,
    *,
    actor: str = "demo.user",
    request_id: Optional[str] = None,
) -> ProcurementRequest:
    if isinstance(data, ProcurementRequest):
        req = data.model_copy(deep=True)
    else:
        payload = dict(data)
        if request_id:
            payload["request_id"] = request_id
        if "request_id" not in payload or not payload["request_id"]:
            payload["request_id"] = new_request_id()
        req = ProcurementRequest.model_validate(payload)
    req.status = "DRAFT"
    req.version = max(1, req.version)
    refresh_derived(req)
    _append_audit(req, actor=actor, action="created", detail="Request created", after="DRAFT")
    save_request(req)
    return req


# Statuses where the form may still edit fields / save draft / submit
EDITABLE_STATUSES: set[str] = {"DRAFT", "NEEDS_INFORMATION"}
# Statuses where sourcing has been approved — no draft saves or field edits
LOCKED_STATUSES: set[str] = {
    "APPROVED_FOR_SOURCING",
    "SOURCING_IN_PROGRESS",
    "CONTRACT_REVIEW",
    "APPROVED_FOR_SIGNATURE",
    "REJECTED",
    "CANCELLED",
}


def update_request(
    request_id: str,
    updates: dict,
    *,
    actor: str = "demo.user",
    bump_version_on_material_change: bool = True,
) -> ProcurementRequest:
    req = load_request(request_id)
    if req.status in LOCKED_STATUSES:
        raise InvalidTransitionError(
            f"Cannot update request while status is {req.status}"
        )
    if req.status not in EDITABLE_STATUSES:
        raise InvalidTransitionError(
            f"Field edits only allowed in DRAFT or NEEDS_INFORMATION (now {req.status})"
        )
    # If loading from fixture, ensure we write to store copy
    material_keys = {
        "budget_amount",
        "estimated_recurring_cost",
        "data_sensitivity",
        "requested_contract_term_months",
        "required_sla",
        "business_justification",
        "number_of_users",
        "security_review_required",
        "privacy_review_required",
    }
    before = req.model_dump()
    for k, v in updates.items():
        if k in ("request_id", "audit_events", "status"):
            continue  # status via transition API only
        if hasattr(req, k):
            setattr(req, k, v)
    if bump_version_on_material_change and any(
        before.get(k) != getattr(req, k) for k in material_keys if k in updates
    ):
        req.version += 1
        _append_audit(
            req,
            actor=actor,
            action="version_bumped",
            detail=f"Material change → version {req.version}",
        )
    refresh_derived(req)
    _append_audit(req, actor=actor, action="updated", detail="Fields updated")
    save_request(req)
    return req


def transition_status(
    request_id: str,
    new_status: RequestStatus,
    *,
    actor: str = "demo.user",
    reason: str = "",
) -> ProcurementRequest:
    req = load_request(request_id)
    # Always work on store copy
    allowed = ALLOWED_TRANSITIONS.get(req.status, set())
    if new_status not in allowed:
        raise InvalidTransitionError(
            f"Cannot transition {req.status} → {new_status}"
        )
    before = req.status
    req.status = new_status
    refresh_derived(req)
    _append_audit(
        req,
        actor=actor,
        action="status_transition",
        detail=reason or f"{before} → {new_status}",
        before=before,
        after=new_status,
    )
    save_request(req)
    return req


def save_draft(req: ProcurementRequest, *, actor: str = "demo.user") -> ProcurementRequest:
    if req.status in LOCKED_STATUSES:
        raise InvalidTransitionError(
            f"Cannot save draft while status is {req.status}"
        )
    if req.status not in EDITABLE_STATUSES and req.status not in (
        "SUBMITTED",
        "PROCUREMENT_TRIAGE",
        "FINANCE_REVIEW",
        "SECURITY_REVIEW",
        "LEGAL_REVIEW",
    ):
        raise InvalidTransitionError(f"Cannot save draft from {req.status}")
    # Edits after triage are allowed only via update_request with version bump;
    # save_draft itself is for DRAFT / NEEDS_INFORMATION.
    if req.status not in EDITABLE_STATUSES:
        raise InvalidTransitionError(
            f"Save draft is only allowed in DRAFT or NEEDS_INFORMATION (now {req.status})"
        )
    refresh_derived(req)
    _append_audit(req, actor=actor, action="save_draft", detail="Draft saved", after=req.status)
    save_request(req)
    return req


def submit_request(
    request_id: str,
    *,
    actor: str = "demo.user",
) -> ProcurementRequest:
    """Submit draft: NEEDS_INFORMATION if incomplete, else PROCUREMENT_TRIAGE path."""
    req = load_request(request_id)
    if req.status not in ("DRAFT", "NEEDS_INFORMATION"):
        raise InvalidTransitionError(f"Cannot submit from {req.status}")
    refresh_derived(req)
    missing = req.missing_fields
    before = req.status

    if missing:
        # DRAFT → SUBMITTED → NEEDS_INFORMATION (two-step via allowed graph)
        if before == "DRAFT":
            req.status = "SUBMITTED"
            _append_audit(
                req,
                actor=actor,
                action="status_transition",
                detail="Submitted (incomplete)",
                before="DRAFT",
                after="SUBMITTED",
            )
            before = "SUBMITTED"
        req.status = "NEEDS_INFORMATION"
        _append_audit(
            req,
            actor=actor,
            action="status_transition",
            detail="Missing: " + ", ".join(missing),
            before=before,
            after="NEEDS_INFORMATION",
        )
        refresh_derived(req)
        save_request(req)
        return req

    # Complete: DRAFT → SUBMITTED → PROCUREMENT_TRIAGE (+ parallel review signals)
    if before == "DRAFT":
        req.status = "SUBMITTED"
        _append_audit(
            req,
            actor=actor,
            action="status_transition",
            detail="Submitted",
            before="DRAFT",
            after="SUBMITTED",
        )
        before = "SUBMITTED"
    elif before == "NEEDS_INFORMATION":
        req.status = "SUBMITTED"
        _append_audit(
            req,
            actor=actor,
            action="status_transition",
            detail="Resubmitted after information provided",
            before="NEEDS_INFORMATION",
            after="SUBMITTED",
        )
        before = "SUBMITTED"

    # Prefer finance/security queue if triggered, else procurement triage
    next_status: RequestStatus = "PROCUREMENT_TRIAGE"
    if any(f.code in ("over_budget", "budget_unconfirmed") for f in req.risk_flags):
        next_status = "FINANCE_REVIEW"
    elif any(f.code in ("security_review", "privacy_review") for f in req.risk_flags):
        next_status = "SECURITY_REVIEW"

    req.status = next_status
    _append_audit(
        req,
        actor=actor,
        action="status_transition",
        detail=f"Routed to {next_status}",
        before=before,
        after=next_status,
    )
    refresh_derived(req)
    save_request(req)
    return req


def request_more_information(
    request_id: str,
    *,
    actor: str = "demo.reviewer",
    reason: str = "Additional information required",
) -> ProcurementRequest:
    req = load_request(request_id)
    if req.status in ("DRAFT", "CANCELLED", "REJECTED", "APPROVED_FOR_SIGNATURE"):
        raise InvalidTransitionError(f"Cannot request information from {req.status}")
    return transition_status(
        request_id, "NEEDS_INFORMATION", actor=actor, reason=reason
    )


def mark_eligible_for_sourcing(
    request_id: str,
    *,
    actor: str = "demo.procurement",
) -> ProcurementRequest:
    """Demo-only happy path: jump to APPROVED_FOR_SOURCING when transitions allow."""
    req = load_request(request_id)
    # Walk allowed path if needed
    path_hints = [
        "PROCUREMENT_TRIAGE",
        "FINANCE_REVIEW",
        "SECURITY_REVIEW",
        "LEGAL_REVIEW",
        "APPROVED_FOR_SOURCING",
    ]
    if req.status == "APPROVED_FOR_SOURCING":
        return req
    if req.status in ("DRAFT", "NEEDS_INFORMATION"):
        req = submit_request(request_id, actor=actor)
    # From review states, transition directly if allowed
    if "APPROVED_FOR_SOURCING" in ALLOWED_TRANSITIONS.get(req.status, set()):
        return transition_status(
            request_id,
            "APPROVED_FOR_SOURCING",
            actor=actor,
            reason="Demo: mark eligible for sourcing",
        )
    # Try moving to PROCUREMENT_TRIAGE first
    if "PROCUREMENT_TRIAGE" in ALLOWED_TRANSITIONS.get(req.status, set()):
        req = transition_status(
            request_id, "PROCUREMENT_TRIAGE", actor=actor, reason="Triage before sourcing"
        )
    if "APPROVED_FOR_SOURCING" in ALLOWED_TRANSITIONS.get(req.status, set()):
        return transition_status(
            request_id,
            "APPROVED_FOR_SOURCING",
            actor=actor,
            reason="Demo: mark eligible for sourcing",
        )
    raise InvalidTransitionError(
        f"Cannot mark eligible for sourcing from {req.status}"
    )


def link_contract_run(request_id: str, run_id: str, *, actor: str = "system") -> ProcurementRequest:
    req = load_request(request_id)
    if run_id not in req.linked_contract_run_ids:
        req.linked_contract_run_ids.append(run_id)
        _append_audit(
            req,
            actor=actor,
            action="link_contract",
            detail=f"Linked run_id={run_id}",
        )
        save_request(req)
    return req


# --------------------------------------------------------------------------- #
# Duplicate detection (simple metadata match)
# --------------------------------------------------------------------------- #
def find_duplicate_candidates(
    req: ProcurementRequest,
    *,
    others: Optional[list[ProcurementRequest]] = None,
) -> list[DuplicateCandidate]:
    if others is None:
        others = []
        for rid in list_request_ids():
            if rid == req.request_id:
                continue
            try:
                others.append(load_request(rid))
            except Exception:
                continue

    active = {
        "DRAFT",
        "SUBMITTED",
        "NEEDS_INFORMATION",
        "PROCUREMENT_TRIAGE",
        "FINANCE_REVIEW",
        "SECURITY_REVIEW",
        "LEGAL_REVIEW",
        "APPROVED_FOR_SOURCING",
        "SOURCING_IN_PROGRESS",
        "CONTRACT_REVIEW",
    }
    hits: list[DuplicateCandidate] = []
    title_tokens = set(re.findall(r"[a-z0-9]+", (req.title or "").lower()))
    for o in others:
        if o.status not in active:
            continue
        reasons: list[str] = []
        score = 0.0
        if req.category and o.category and req.category == o.category:
            score += 0.35
            reasons.append("same category")
        if req.requesting_team and o.requesting_team and req.requesting_team == o.requesting_team:
            score += 0.25
            reasons.append("same requesting team")
        o_tokens = set(re.findall(r"[a-z0-9]+", (o.title or "").lower()))
        if title_tokens and o_tokens:
            overlap = len(title_tokens & o_tokens) / max(1, len(title_tokens | o_tokens))
            if overlap >= 0.4:
                score += 0.4 * overlap
                reasons.append(f"title overlap {overlap:.0%}")
        if score >= 0.45 and reasons:
            hits.append(
                DuplicateCandidate(
                    request_id=o.request_id,
                    title=o.title,
                    category=o.category or "",
                    requesting_team=o.requesting_team,
                    status=o.status,
                    match_score=round(score, 3),
                    match_reasons=reasons,
                )
            )
    hits.sort(key=lambda h: -h.match_score)
    return hits


def compare_request_to_package(
    req: ProcurementRequest,
    pkg,
) -> list[RiskFlag]:
    """Deterministic PR ↔ contract/commercial consistency checks."""
    flags: list[RiskFlag] = []
    # Budget vs contract value
    meta = getattr(pkg, "metadata", None)
    raw_val = None
    if meta and meta.contract_value and meta.contract_value.value:
        try:
            raw_val = float(str(meta.contract_value.value).replace(",", ""))
        except ValueError:
            raw_val = None
    if (
        raw_val is not None
        and req.budget_amount is not None
        and raw_val > req.budget_amount
    ):
        flags.append(
            RiskFlag(
                code="over_requisition_budget",
                message=(
                    f"Contract value ({raw_val:,.0f} {req.currency}) exceeds "
                    f"requisition budget ({req.budget_amount:,.0f} {req.currency})."
                ),
                severity="high",
            )
        )

    supplier = (meta.supplier.value if meta and meta.supplier else "") or ""
    title = (req.title or "").lower()
    cat = req.category or ""
    # Category / scope mismatch heuristics for the demo
    bpo_like = any(w in supplier.lower() for w in ("bpo", "support services", "talkbridge"))
    saas_request = cat == "saas_software" or "saas" in title or "collaboration" in title
    if bpo_like and saas_request:
        flags.append(
            RiskFlag(
                code="category_mismatch",
                message=(
                    f"Requisition looks like collaboration SaaS, but linked contract "
                    f"supplier is '{supplier}' (BPO/support). Confirm before sourcing."
                ),
                severity="high",
            )
        )

    pkg_req = getattr(pkg, "request_id", None)
    if pkg_req and pkg_req != req.request_id:
        flags.append(
            RiskFlag(
                code="request_id_mismatch",
                message=f"Package request_id `{pkg_req}` ≠ active requisition `{req.request_id}`.",
                severity="medium",
            )
        )
    return flags


def sourcing_checklist(req: ProcurementRequest) -> list[str]:
    """Requirements to include in RFP/RFQ — traced to request fields."""
    items: list[str] = []
    if req.number_of_users:
        items.append(f"Support at least {req.number_of_users} users (from request).")
    if req.budget_amount is not None:
        items.append(
            f"Stay within {req.budget_amount:,.0f} {req.currency} "
            f"{req.budget_period or 'period'} budget (from request)."
        )
    if req.requested_contract_term_months:
        items.append(
            f"Initial term {req.requested_contract_term_months} months (from request)."
        )
    if req.required_sla:
        items.append(f"Meet SLA: {req.required_sla} (from request).")
    if req.data_sensitivity and req.data_sensitivity != "public":
        items.append(
            f"Address data sensitivity '{req.data_sensitivity}' and provide DPA if personal data (from request)."
        )
    if req.security_review_required or req.privacy_review_required:
        items.append("Complete security/privacy questionnaire before award (from request flags).")
    for cap in req.requested_capabilities:
        items.append(f"Capability: {cap} (from request).")
    for integ in req.integrations:
        items.append(f"Integration: {integ} (from request).")
    if not items:
        items.append("Request incomplete — finalize requirements before sourcing.")
    return items
