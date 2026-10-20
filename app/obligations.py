"""Milestone B (spec §20) — contract obligations, renewals and deadline tracking.

This module adds a richer, traceable *obligation register* on top of the light
``Obligation`` list already produced by the review pipeline. Everything here is
deterministic Python — no LLM. An LLM may *propose* obligation candidates
upstream, but every date, recurrence and routing decision below is computed by
code and is fully unit-testable with an injected ``today``.

Design rules enforced here:

* The renewal/termination **notice deadline** is NOT the contract expiry date —
  it is ``term_end - notice_period`` and is only computed when the anchor date is
  grounded and the wording is unambiguous.
* Ambiguous wording (e.g. "business days"), a missing effective/term-end date, or
  a conflicting term renders the obligation ``needs_human_confirmation=True`` with
  ``due_date=None`` — never a guessed date.
* Obligations carry a **stable id** derived from (contract, type, source quote),
  so reprocessing the same contract updates rows in place instead of duplicating.
* A confirmed obligation is preserved across re-runs (its status and owner are
  not silently overwritten by a fresh extraction).
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app import rules
from app.schemas import ContractMetadata

_ONE_DAY = timedelta(days=1)

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
ObligationType = Literal[
    "payment",
    "renewal_notice",
    "termination_notice",
    "sla",
    "security",
    "privacy",
    "reporting",
    "implementation",
    "insurance",
    "audit",
    "other",
]
ResponsibleParty = Literal["buyer", "supplier", "both", "unknown"]
ObligationStatus = Literal["proposed", "confirmed", "complete", "overdue", "not_applicable"]
Criticality = Literal["low", "medium", "high", "critical"]
TriggerType = Literal["fixed_date", "relative_to_anchor", "recurring", "on_event", "unknown"]
AnchorType = Literal[
    "effective_date",
    "signature_date",
    "service_start",
    "service_end",
    "invoice_date",
    "renewal_date",
    "term_end",
]
Confidence = Literal["high", "medium", "low", "unknown"]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ContractDateAnchor(BaseModel):
    anchor_type: AnchorType
    value: Optional[date] = None
    confidence: Confidence = "unknown"
    source_page: Optional[int] = None
    source_quote: Optional[str] = None
    confirmed: bool = False


class ContractObligation(BaseModel):
    obligation_id: str
    contract_id: str
    title: str
    description: str = ""
    obligation_type: ObligationType
    responsible_party: ResponsibleParty = "unknown"
    owner_role: Optional[str] = None
    owner_user: Optional[str] = None
    trigger_type: TriggerType = "unknown"
    trigger_date: Optional[date] = None
    due_date: Optional[date] = None
    recurrence: Optional[str] = None  # "monthly" | "annual" | ...
    notice_period_days: Optional[int] = None
    status: ObligationStatus = "proposed"
    criticality: Criticality = "medium"
    completion_evidence_required: Optional[str] = None
    source_evidence_ids: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    source_quotes: list[str] = Field(default_factory=list)
    date_calculation_explanation: Optional[str] = None
    needs_human_confirmation: bool = False
    created_at: str = ""
    updated_at: str = ""


class ReminderCandidate(BaseModel):
    obligation_id: str
    reminder_date: date
    reason: str
    channel: Literal["in_app_demo"] = "in_app_demo"
    delivery_status: Literal["pending", "shown", "dismissed"] = "pending"


class ObligationRegister(BaseModel):
    contract_id: str
    generated_at: str
    today: date
    obligations: list[ContractObligation] = Field(default_factory=list)
    anchors: list[ContractDateAnchor] = Field(default_factory=list)
    reminders: list[ReminderCandidate] = Field(default_factory=list)
    audit: list[dict] = Field(default_factory=list)

    def by_id(self, obligation_id: str) -> Optional[ContractObligation]:
        return next((o for o in self.obligations if o.obligation_id == obligation_id), None)

    def upcoming(self, within_days: int = 90) -> list[ContractObligation]:
        out = []
        for o in self.obligations:
            if o.due_date is None:
                continue
            delta = (o.due_date - self.today).days
            if 0 <= delta <= within_days:
                out.append(o)
        return sorted(out, key=lambda o: o.due_date)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Deterministic date calculator
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_date(value: Optional[str]) -> Optional[date]:
    """Parse the normalised date strings the metadata agent emits."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def add_months(d: date, months: int) -> date:
    """Add (or subtract) whole months, clamping to the last valid day.

    Leap years are handled by the clamp: 2028-02-29 + 12 months -> 2029-02-28.
    """
    total = (d.year * 12 + (d.month - 1)) + months
    year, month = divmod(total, 12)
    month += 1
    # Last day of the target month.
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    last_day = (next_month_first - _ONE_DAY).day
    return date(year, month, min(d.day, last_day))


def term_end_from_anchor(
    effective: Optional[date], term_months: Optional[int]
) -> Optional[date]:
    """Compute the end of the initial term from an effective date + term length."""
    if effective is None or not term_months:
        return None
    return add_months(effective, term_months) - _ONE_DAY


def compute_notice_deadline(
    term_end: Optional[date],
    notice_days: Optional[int],
    *,
    business_days_wording: bool = False,
) -> tuple[Optional[date], str]:
    """Return (deadline, explanation).

    The notice deadline is ``term_end - notice_days``. It is only returned when
    the term-end date is known and the notice is expressed in calendar days.
    """
    if term_end is None:
        return None, "Term-end date is not grounded; cannot compute a notice deadline."
    if not notice_days:
        return None, "Notice period is unknown; cannot compute a notice deadline."
    if business_days_wording:
        return (
            None,
            "Notice is expressed in business/working days; calendar conversion "
            "needs human confirmation.",
        )
    deadline = term_end - timedelta(days=notice_days)
    return (
        deadline,
        f"term_end {term_end.isoformat()} minus {notice_days} calendar days "
        f"= {deadline.isoformat()}.",
    )


def stable_obligation_id(contract_id: str, obligation_type: str, source_quote: str) -> str:
    key = f"{contract_id}|{obligation_type}|{(source_quote or '').strip().lower()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{obligation_type}-{digest}"


# --------------------------------------------------------------------------- #
# Register builder
# --------------------------------------------------------------------------- #
def _field(meta: ContractMetadata, name: str):
    return getattr(meta, name, None)


def _quote_page(field) -> tuple[Optional[str], Optional[int]]:
    if field is not None and field.evidence is not None:
        return field.evidence.quote, field.evidence.page
    return None, None


def _as_days(value: Optional[str]) -> Optional[int]:
    """Parse a day count from a normalised field: a bare int, 'Net 30', '60 days'."""
    if not value:
        return None
    t = str(value).strip()
    if t.isdigit():
        return int(t)
    return rules.parse_net_days(t) or rules.parse_days(t)


def _looks_numeric(value: Optional[str]) -> bool:
    """True if a normalised field holds a number ('99.0', '99.9%', '95')."""
    if not value:
        return False
    try:
        float(str(value).strip().rstrip("%"))
        return True
    except ValueError:
        return rules.parse_percent(value) is not None


def _business_days_wording(meta: ContractMetadata) -> bool:
    basis = (meta.renewal_notice_basis.value or "").lower()
    return "business day" in basis or "working day" in basis


def build_register(
    meta: ContractMetadata,
    *,
    contract_id: str,
    today: date,
    existing: Optional[ObligationRegister] = None,
) -> ObligationRegister:
    """Build (or refresh) the obligation register for one contract.

    When ``existing`` is supplied, obligations are merged by stable id so that a
    re-run updates rows in place and preserves human confirmations.
    """
    gen_at = _now_iso()
    anchors: list[ContractDateAnchor] = []
    obligations: list[ContractObligation] = []
    audit: list[dict] = []

    # --- Date anchors ----------------------------------------------------- #
    effective = parse_date(meta.effective_date.value)
    eff_q, eff_p = _quote_page(meta.effective_date)
    anchors.append(
        ContractDateAnchor(
            anchor_type="effective_date",
            value=effective,
            confidence="high" if effective else "unknown",
            source_page=eff_p,
            source_quote=eff_q,
        )
    )

    expiry = parse_date(meta.expiry_date.value)
    exp_q, exp_p = _quote_page(meta.expiry_date)
    term_months = None
    try:
        tm = getattr(meta, "renewal_term_months", None)
        term_months = int(tm.value) if tm and tm.value and str(tm.value).isdigit() else None
    except (ValueError, AttributeError):
        term_months = None

    # Prefer a grounded expiry; otherwise derive the term end from effective + term.
    term_end = expiry or term_end_from_anchor(effective, term_months)
    term_end_confidence: Confidence = (
        "high" if expiry else ("medium" if (effective and term_months) else "unknown")
    )
    anchors.append(
        ContractDateAnchor(
            anchor_type="term_end",
            value=term_end,
            confidence=term_end_confidence,
            source_page=exp_p,
            source_quote=exp_q or (eff_q if term_end else None),
        )
    )

    # --- Renewal / termination notice ------------------------------------ #
    notice_days = _as_days(meta.notice_period_days.value)
    biz = _business_days_wording(meta)
    deadline, explanation = compute_notice_deadline(
        term_end, notice_days, business_days_wording=biz
    )
    n_quote, n_page = _quote_page(meta.notice_period_days)
    needs_conf = deadline is None
    renewal = ContractObligation(
        obligation_id=stable_obligation_id(contract_id, "renewal_notice", n_quote or "notice"),
        contract_id=contract_id,
        title="Give renewal / non-renewal notice",
        description=(
            "Deadline by which the buyer must give notice to prevent automatic "
            "renewal or to exit at term end."
        ),
        obligation_type="renewal_notice",
        responsible_party="buyer",
        owner_role="procurement",
        trigger_type="relative_to_anchor",
        trigger_date=term_end,
        due_date=deadline,
        notice_period_days=notice_days,
        status="proposed",
        criticality="high",
        completion_evidence_required="Dated notice letter / email acknowledgement",
        source_pages=[p for p in [n_page] if p],
        source_quotes=[q for q in [n_quote] if q],
        date_calculation_explanation=explanation,
        needs_human_confirmation=needs_conf,
        created_at=gen_at,
        updated_at=gen_at,
    )
    if deadline is not None and (deadline - today).days <= 90:
        renewal.criticality = "critical"
    obligations.append(renewal)

    # --- Payment (recurring) --------------------------------------------- #
    pay_days = _as_days(meta.payment_terms_days.value)
    if pay_days:
        p_quote, p_page = _quote_page(meta.payment_terms_days)
        obligations.append(
            ContractObligation(
                obligation_id=stable_obligation_id(contract_id, "payment", p_quote or "payment"),
                contract_id=contract_id,
                title=f"Pay undisputed invoices within {pay_days} days",
                obligation_type="payment",
                responsible_party="buyer",
                owner_role="accounts_payable",
                trigger_type="recurring",
                recurrence="per_invoice",
                notice_period_days=pay_days,
                status="proposed",
                criticality="medium",
                completion_evidence_required="Payment confirmation per invoice",
                source_pages=[p for p in [p_page] if p],
                source_quotes=[q for q in [p_quote] if q],
                date_calculation_explanation=f"Net {pay_days} from each valid invoice date.",
                created_at=gen_at,
                updated_at=gen_at,
            )
        )

    # --- SLA review (recurring monthly) ---------------------------------- #
    if meta.sla_target_pct.value and _looks_numeric(meta.sla_target_pct.value):
        s_quote, s_page = _quote_page(meta.sla_target_pct)
        obligations.append(
            ContractObligation(
                obligation_id=stable_obligation_id(contract_id, "sla", s_quote or "sla"),
                contract_id=contract_id,
                title=f"Review supplier SLA ({meta.sla_target_pct.value})",
                obligation_type="sla",
                responsible_party="buyer",
                owner_role="service_manager",
                trigger_type="recurring",
                recurrence="monthly",
                status="proposed",
                criticality="medium",
                completion_evidence_required="Verified monthly service-level report",
                source_pages=[p for p in [s_page] if p],
                source_quotes=[q for q in [s_quote] if q],
                created_at=gen_at,
                updated_at=gen_at,
            )
        )

    # --- Data protection (one-off, evidence) ----------------------------- #
    dp = meta.processes_personal_data.value
    if dp and str(dp).lower() in ("yes", "true"):
        d_quote, d_page = _quote_page(meta.processes_personal_data)
        obligations.append(
            ContractObligation(
                obligation_id=stable_obligation_id(contract_id, "privacy", d_quote or "dpa"),
                contract_id=contract_id,
                title="Maintain Data Processing Agreement evidence",
                obligation_type="privacy",
                responsible_party="both",
                owner_role="security_privacy",
                trigger_type="on_event",
                status="proposed",
                criticality="high",
                completion_evidence_required="Signed DPA / processing records",
                source_pages=[p for p in [d_page] if p],
                source_quotes=[q for q in [d_quote] if q],
                created_at=gen_at,
                updated_at=gen_at,
            )
        )

    # --- Overdue marking -------------------------------------------------- #
    for o in obligations:
        if o.due_date is not None and o.due_date < today and o.status == "proposed":
            o.status = "overdue"

    register = ObligationRegister(
        contract_id=contract_id,
        generated_at=gen_at,
        today=today,
        obligations=obligations,
        anchors=anchors,
        reminders=build_reminders(obligations, today=today),
        audit=audit,
    )

    if existing is not None:
        register = merge_registers(existing, register)
    else:
        for o in register.obligations:
            register.audit.append(
                {"event": "created", "obligation_id": o.obligation_id, "at": gen_at}
            )
    return register


def build_reminders(
    obligations: list[ContractObligation], *, today: date, offsets=(90, 60, 30)
) -> list[ReminderCandidate]:
    """In-app reminder candidates ahead of each dated notice deadline."""
    out: list[ReminderCandidate] = []
    for o in obligations:
        if o.obligation_type not in ("renewal_notice", "termination_notice"):
            continue
        if o.due_date is None:
            continue
        for days in offsets:
            out.append(
                ReminderCandidate(
                    obligation_id=o.obligation_id,
                    reminder_date=o.due_date - timedelta(days=days),
                    reason=f"{days} days before notice deadline {o.due_date.isoformat()}",
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Merge / dedup — stable identity across reprocessing
# --------------------------------------------------------------------------- #
# Fields a human owns; a fresh extraction must never silently overwrite them.
_PRESERVED_ON_CONFIRM = (
    "status",
    "owner_user",
    "owner_role",
    "responsible_party",
    "due_date",
    "needs_human_confirmation",
)


def merge_registers(old: ObligationRegister, new: ObligationRegister) -> ObligationRegister:
    """Merge a fresh register into an existing one, keyed by obligation id.

    Confirmed obligations keep their human-owned fields; everything else is
    refreshed. New obligations are added; obligations no longer present are kept
    but marked ``not_applicable`` (preserved for audit, never dropped silently).
    """
    now = _now_iso()
    old_by_id = {o.obligation_id: o for o in old.obligations}
    merged: list[ContractObligation] = []
    audit = list(old.audit)
    seen: set[str] = set()

    for fresh in new.obligations:
        seen.add(fresh.obligation_id)
        prior = old_by_id.get(fresh.obligation_id)
        if prior is None:
            fresh.created_at = fresh.created_at or now
            fresh.updated_at = now
            merged.append(fresh)
            audit.append({"event": "created", "obligation_id": fresh.obligation_id, "at": now})
            continue
        updated = fresh.model_copy()
        updated.created_at = prior.created_at or fresh.created_at
        updated.updated_at = now
        if prior.status in ("confirmed", "complete", "not_applicable"):
            for field_name in _PRESERVED_ON_CONFIRM:
                setattr(updated, field_name, getattr(prior, field_name))
        merged.append(updated)
        if prior.model_dump() != updated.model_dump():
            audit.append({"event": "updated", "obligation_id": updated.obligation_id, "at": now})

    # Obligations that vanished from the fresh extraction: keep, flag.
    for oid, prior in old_by_id.items():
        if oid in seen:
            continue
        if prior.status != "not_applicable":
            prior = prior.model_copy(update={"status": "not_applicable", "updated_at": now})
            audit.append({"event": "retired", "obligation_id": oid, "at": now})
        merged.append(prior)

    return ObligationRegister(
        contract_id=new.contract_id,
        generated_at=new.generated_at,
        today=new.today,
        obligations=merged,
        anchors=new.anchors,
        reminders=build_reminders(merged, today=new.today),
        audit=audit,
    )


def confirm_obligation(
    register: ObligationRegister,
    obligation_id: str,
    *,
    owner_user: Optional[str] = None,
    due_date: Optional[date] = None,
) -> ObligationRegister:
    """Record a human confirmation of dates/owner for one obligation."""
    now = _now_iso()
    for o in register.obligations:
        if o.obligation_id == obligation_id:
            o.status = "confirmed"
            o.needs_human_confirmation = False
            if owner_user:
                o.owner_user = owner_user
            if due_date is not None:
                o.due_date = due_date
            o.updated_at = now
            register.audit.append(
                {"event": "confirmed", "obligation_id": obligation_id, "at": now}
            )
            break
    register.reminders = build_reminders(register.obligations, today=register.today)
    return register


# --------------------------------------------------------------------------- #
# Persistence (JSON)
# --------------------------------------------------------------------------- #
def save_register(register: ObligationRegister, path) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(register.model_dump_json(indent=2), encoding="utf-8")


def load_register(path) -> Optional[ObligationRegister]:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    return ObligationRegister.model_validate_json(p.read_text(encoding="utf-8"))
