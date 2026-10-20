"""Typed rule checkers: one per playbook rule.

Each checker turns the model's ``extracted_value`` (plus, where needed, a little
metadata context) into a :data:`~app.schemas.Status`. All numbers and dates are
parsed here, in code, so severity and status are reproducible and auditable.
Checkers return ``"unclear"`` whenever they cannot parse a value rather than
guessing. ``apply_rules`` layers the remaining overrides (missing/failed
grounding, ambiguous-on-high, unresolved cross-references, claimed-absence).
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from app.schemas import ClauseFinding, ContractMetadata, PlaybookRule, Status

# --------------------------------------------------------------------------- #
# Parse helpers
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def _first_number(text: str) -> Optional[float]:
    m = _NUM.search(text or "")
    return float(m.group(1)) if m else None


def parse_net_days(v: Optional[str]) -> Optional[int]:
    """'Net 15' / '15 days from invoice' / 'net-30' -> 15 / 15 / 30."""
    if not v:
        return None
    t = v.lower()
    m = re.search(r"net[\s\-]*?(\d+)", t)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\)?\s*(?:calendar\s+|business\s+)?days", t)
    if m:
        return int(m.group(1))
    return None


def parse_bare_number(v: Optional[str]) -> Optional[float]:
    """A value that is only a number ("90", "95.0") — models often drop the unit."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", v or "")
    return float(m.group(1)) if m else None


def parse_percent(v: Optional[str]) -> Optional[float]:
    if not v:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", v)
    return float(m.group(1)) if m else None


def parse_months(v: Optional[str]) -> Optional[int]:
    if not v:
        return None
    m = re.search(r"(\d+)\)?\s*(?:-\s*)?months?", v.lower())
    return int(m.group(1)) if m else None


def parse_days(v: Optional[str]) -> Optional[int]:
    if not v:
        return None
    # Handles "30 days", "30-day" and the legal style "thirty (30) days".
    m = re.search(r"(\d+)\)?\s*(?:-\s*)?(?:calendar\s+|business\s+)?days?", v.lower())
    return int(m.group(1)) if m else None


def _has(v: Optional[str], *words: str) -> bool:
    t = (v or "").lower()
    return any(w in t for w in words)


# --------------------------------------------------------------------------- #
# Per-rule checkers. Signature: (value, finding, meta) -> Status
# --------------------------------------------------------------------------- #
def check_liability_cap(v, finding, meta) -> Status:
    t = (v or "").lower()
    if _has(t, "uncapped", "unlimited", "no limit", "not subject to any limit", "without limit",
            "no ceiling", "no cap on", "no cap ", "no financial cap"):
        return "walk_away"
    # The cap is switched off for some obligations ("the foregoing limitation shall not apply to
    # the Supplier's obligations except as set out in Clause 12") — the cap does not hold.
    if _has(t, "carved out", "carve-out", "limitation shall not apply", "cap shall not apply",
            "shall not apply to the supplier"):
        return "walk_away"
    # Preferred: fees paid in the prior / preceding 12 months (trailing-12).
    if "fee" in t and _has(t, "prior", "preceding", "last 12", "twelve", "12 month", "prior 12"):
        return "preferred"
    # Fallback: 1x annual fees (or bare "1x" without trailing-12 language).
    if _has(t, "annual fee", "annual fees"):
        return "fallback"
    if _has(t, "1x", "1 x", "one time", "one year"):
        return "fallback"
    return "unclear"


def check_indemnity(v, finding: ClauseFinding, meta) -> Status:
    """LIABILITY-002. Preferred: indemnity within the cap, except IP and data-breach carve-outs.

    Walk-away only when the *general* indemnity escapes the cap. An indemnity that is
    subject to the cap with only IP-infringement / data-breach claims left uncapped is the
    playbook's preferred position, not a deviation (it used to be scored walk-away because
    any "uncapped" + "indemnif" text matched, which then escalated LIABILITY-001 too).
    """
    joined = " ".join(e.quote.lower() for e in finding.evidence) + " " + (v or "").lower()
    general_outside_cap = _has(
        joined,
        "not be subject to any limitation",
        "shall not be subject to any limit",
        "notwithstanding any limitation",
        "shall not apply to this clause",
        "uncapped general",
        "general indemnity is uncapped",
        "outside the cap",
        "not limited by",
        "operates independently of",
        "without regard to the cap",
    )
    mentions_uncapped = _has(joined, "uncapped", "without limit", "unlimited")
    carve_out = _has(joined, "intellectual property", "ip infringement", "ip claims", "data breach",
                     "breach of data protection")
    within_cap = _has(joined, "subject to the cap", "subject to the limitation", "within the cap",
                      "subject to the limit", "capped at", "subject to clause")
    if general_outside_cap:
        return "walk_away"
    if mentions_uncapped and not carve_out:
        return "walk_away"
    if _has(v, "separate cap", "2x", "two times", "2 x"):
        return "fallback"
    if within_cap or (mentions_uncapped and carve_out):
        return "preferred"
    return "unclear"


def check_payment(v, finding, meta) -> Status:
    days = parse_net_days(v)
    # Conflicting terms: an overriding clause the model quoted ("Notwithstanding Clause 3,
    # invoices are payable within fifteen (15) days") governs over the main-terms value.
    for e in getattr(finding, "evidence", None) or []:
        q = e.quote.lower()
        if _has(q, "notwithstanding", "prevails", "shall prevail") and parse_net_days(q) is not None:
            days = parse_net_days(q)
    if days is None:
        return "unclear"
    if days >= 45:
        return "preferred"
    if days >= 30:
        return "fallback"
    return "walk_away"


def check_renewal(v, finding, meta: ContractMetadata) -> Status:
    text = (v or "")
    if meta is not None:
        text += " " + " ".join(
            x for x in [
                meta.auto_renewal.value or "",
                meta.renewal_term_months.value or "",
                meta.notice_period_days.value or "",
            ]
        )
    auto = _has(text, "auto-renew", "auto renew", "automatically renew", "renews")
    no_auto = _has(text, "no auto", "does not renew", "no automatic")
    notice = parse_days(text)
    term = parse_months(text)
    if term is not None and term > 12:
        return "walk_away"
    if notice is not None and notice < 30:
        return "walk_away"
    if no_auto:
        return "preferred"
    # Preferred notice is 60 days. Longer notice is a process burden (fallback).
    if notice is not None and notice > 60:
        return "fallback"
    if notice is not None and notice == 60:
        return "preferred"
    if notice is not None and notice >= 30:
        return "fallback"
    if auto and notice is None:
        return "unclear"
    return "unclear"


def check_termination(v, finding, meta) -> Status:
    if _has(v, "no termination for convenience", "no buyer", "may not terminate", "not terminate for convenience"):
        return "walk_away"
    days = parse_days(v)
    convenience = _has(v, "convenience", "terminate")
    if not convenience and days is None:
        return "unclear"
    if days is None:
        return "unclear"
    if days <= 30:
        return "preferred"
    if days <= 60:
        return "fallback"
    return "fallback"  # longer notice is still a buyer right, treated as fallback


def check_data(v, finding: ClauseFinding, meta: ContractMetadata) -> Status:
    # Only applies where personal data is processed (handled in apply_rules too).
    quotes = " ".join(e.quote for e in (getattr(finding, "evidence", None) or []))
    joined = f"{v or ''} {quotes}"
    if _has(joined, "dpa", "data processing agreement", "data protection addendum"):
        if _has(joined, "before go-live", "will sign", "commit", "to be signed", "prior to"):
            return "fallback"
        return "preferred"
    if not finding.found:
        return "not_found"
    meta_pd = ((meta.processes_personal_data.value or "") if meta is not None and
               getattr(meta, "processes_personal_data", None) is not None else "").strip().lower()
    no_pd = _has(joined, "no personal data", "do not involve the processing of personal data",
                 "does not involve the processing of personal data", "not process personal data") \
        or meta_pd in ("no", "false", "n")
    if no_pd:
        return "preferred"  # rule not triggered: no personal data processed
    processes = _has(joined, "personal data", "personal information") or meta_pd in ("yes", "true", "y")
    if processes:
        return "not_found"  # personal data processed with no DPA language
    return "unclear"


def check_sla(v, finding, meta) -> Status:
    """Score platform uptime SLAs. Ticket-only response % is not an uptime SLA."""
    meta_v = ""
    if meta is not None and getattr(meta, "sla_target_pct", None) is not None:
        meta_v = meta.sla_target_pct.value or ""
    quotes = " ".join(e.quote for e in (getattr(finding, "evidence", None) or []))
    joined = f"{v or ''} {meta_v} {quotes}".lower()

    if _has(
        joined,
        "no measurable",
        "no sla",
        "does not commit",
        "no specific uptime",
        "no uptime",
    ):
        return "walk_away"

    has_uptime = _has(joined, "uptime", "availability", "platform availability")
    ticket_only = _has(
        joined, "ticket", "tickets", "support ticket", "answered on time"
    ) and not has_uptime
    if ticket_only:
        return "walk_away"

    pct = parse_percent(v) or parse_percent(meta_v)
    if pct is None:
        # Accept bare numeric extracts such as "99.0" / "95.0"
        pct = _first_number(v or "") or _first_number(meta_v)

    if pct is None:
        if not has_uptime:
            return "walk_away"
        return "unclear"

    # Below 99.0% platform uptime is walk-away; 99.0–99.89 fallback; >=99.9 preferred w/ credits
    if not has_uptime and pct is not None:
        # Percentage without uptime/availability context (ambiguous) — treat as walk-away
        return "walk_away"
    if pct < 99.0:
        return "walk_away"
    if pct >= 99.9:
        if _has(joined, "credit", "service credit"):
            return "preferred"
        return "fallback"
    # 99.0 <= pct < 99.9
    if _has(joined, "negotiable", "to be agreed"):
        return "fallback"
    return "fallback"


def check_price(v, finding, meta) -> Status:
    if _has(v, "fixed for the term", "fixed for term", "no increase", "price fixed", "fixed for the initial term",
            "not subject to increase", "fees are fixed"):
        return "preferred"
    if _has(v, "unilateral", "at supplier's discretion", "sole discretion"):
        return "walk_away"
    pct = parse_percent(v)
    if pct is not None:
        # Any annual increase above 5% is walk-away (per the implementation guide).
        return "fallback" if pct <= 5 else "walk_away"
    if _has(v, "cpi", "consumer price index"):
        return "fallback"
    return "unclear"


def check_law(v, finding, meta) -> Status:
    if _has(v, "bengaluru", "bangalore") and _has(v, "india"):
        return "preferred"
    if _has(v, "neutral", "mutually agreed", "arbitration"):
        return "fallback"
    if _has(v, "exclusive", "foreign", "courts of") and not _has(v, "india"):
        return "walk_away"
    if _has(v, "india"):
        return "preferred"
    return "unclear"


def check_assign(v, finding, meta) -> Status:
    t = (v or "").lower()
    free = _has(t, "freely assign", "free assignment", "may assign", "assign to any", "novate or subcontract")
    consent_required = _has(
        t, "prior written consent", "not assign without", "no assignment without",
        "without the prior", "with consent", "requires consent",
    )
    # Free assignment is walk-away unless it is limited to affiliates (fallback).
    if free and not consent_required:
        return "fallback" if "affiliate" in t else "walk_away"
    if consent_required:
        return "preferred"
    if "affiliate" in t:
        return "fallback"
    return "unclear"


CHECKERS: dict[str, Callable] = {
    "LIABILITY-001": check_liability_cap,
    "LIABILITY-002": check_indemnity,
    "PAYMENT-001": check_payment,
    "RENEWAL-001": check_renewal,
    "TERMINATION-001": check_termination,
    "DATA-001": check_data,
    "SLA-001": check_sla,
    "PRICE-001": check_price,
    "LAW-001": check_law,
    "ASSIGN-001": check_assign,
}


def check_status(rule: PlaybookRule, finding: ClauseFinding, meta: ContractMetadata) -> Status:
    checker = CHECKERS.get(rule.id)
    if checker is None:
        return "unclear"
    status = checker(finding.extracted_value, finding, meta)
    if status == "unclear" and finding.evidence:
        # Short model extracts ("Singapore", "thirty (30) days") often omit the words the
        # checker needs; the verified quote has them. Only resolves "unclear" — never
        # overrides a definite outcome derived from the extracted value.
        quotes = " ".join(e.quote for e in finding.evidence)
        status = checker(f"{finding.extracted_value or ''} {quotes}", finding, meta)
    return status


# --------------------------------------------------------------------------- #
# Service-credit arithmetic (SLA cross-check)
# --------------------------------------------------------------------------- #
def check_credit_claim(
    target: float, actual: float, rate: float, cap: float, fee: float
) -> Optional[float]:
    """Return the credit amount for one month, or None if none is owed.

    Example from the spec: 8 points short x 5%/pt = 40%, capped at 20% -> payout
    is 20% of the monthly fee.
    """
    if actual >= target:
        return None
    pct = min((target - actual) * rate, cap)  # percentage points of the monthly fee
    return round(pct / 100 * fee, 2)
