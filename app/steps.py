"""Deterministic workflow nodes.

Everything that must be exact or reproducible lives here as plain functions:
intake, performance CSV, grounding, rule application, routing, obligations,
service credits, redline checking and final assembly. Each is unit-tested
before it is wired into the ADK workflow, and none of them calls a model.
"""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from app import rules
from app.errors import IntakeError
from app.schemas import (
    CheckedRedline,
    ClauseFindings,
    ContractDoc,
    ContractMetadata,
    CreditClaim,
    DroppedItem,
    Evidence,
    ExtractionStatus,
    GroundedFinding,
    GroundedResult,
    GroundingInput,
    MonthPerformance,
    Obligation,
    ObligationInput,
    Page,
    PerformanceSummary,
    Playbook,
    Preflight,
    Redline,
    Redlines,
    ReviewRequest,
    RouteDecision,
    RouteInput,
    RulesInput,
    ReviewPackage,
    ScoredFinding,
    ScoredFindings,
    SecurityFlags,
)

# --------------------------------------------------------------------------- #
# Repo-relative paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def today() -> date:
    return date.today()


def load_playbook(version: str) -> Playbook:
    path = DATA / "playbook" / f"{version}.json"
    return Playbook.model_validate_json(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. Intake
# --------------------------------------------------------------------------- #
# Lines in a contract that are addressed at an AI reviewer are a warning signal
# (surfaced as injection_hits), never a defence on their own.
_INJECTION = re.compile(
    r"(ai\s+review|ai\s+reviewer|language\s+model|llm|assistant|classify\s+this|"
    r"mark\s+this\s+as|treat\s+this\s+as\s+low\s+risk|ignore\s+(?:all\s+)?previous|"
    r"you\s+are\s+an?\s+ai|as\s+an?\s+ai)",
    re.I,
)

# Demo upload ceiling from the Final implementation spec (15 MB).
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
_LOW_PAGE_CHARS = 200
_MIN_TOTAL_CHARS = 1500


def _page_body(text: str) -> str:
    return re.sub(r"^\[PAGE \d+\]\s*", "", text).strip()


def _make_page(number: int, raw_text: str) -> Page:
    body = (raw_text or "").strip()
    return Page(
        number=number,
        text=f"[PAGE {number}]\n{body}",
        character_count=len(body),
    )


def _pages_from_reader(reader) -> list[Page]:
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append(_make_page(i, page.extract_text() or ""))
    return pages


def _validate_pdf_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path)
    if not path.exists():
        raise IntakeError(f"File not found: {path}", code="not_found")
    if path.suffix.lower() != ".pdf":
        raise IntakeError(
            f"Unsupported file type '{path.suffix or '(none)'}'; only .pdf is accepted.",
            code="unsupported_type",
        )
    size = path.stat().st_size
    if size == 0:
        raise IntakeError("Empty file rejected.", code="empty_file")
    if size > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise IntakeError(
            f"File exceeds the demo upload limit of {mb:.0f} MB "
            f"({size} bytes).",
            code="too_large",
        )
    return path


def _validate_pdf_bytes(data: bytes, filename: str) -> None:
    name = Path(filename).name or "upload.pdf"
    if Path(name).suffix.lower() != ".pdf":
        raise IntakeError(
            f"Unsupported file type '{Path(name).suffix or '(none)'}'; only .pdf is accepted.",
            code="unsupported_type",
        )
    if not data:
        raise IntakeError("Empty file rejected.", code="empty_file")
    if len(data) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise IntakeError(
            f"File exceeds the demo upload limit of {mb:.0f} MB "
            f"({len(data)} bytes).",
            code="too_large",
        )


def build_preflight(
    pages: list[Page],
    *,
    extraction_status: Optional[ExtractionStatus] = None,
    extra_warnings: Optional[list[str]] = None,
) -> Preflight:
    """Compute extraction quality + injection warnings from page text.

    A signature or appendix page alone must not trip ``unreliable_extraction``:
    the test is that more than half the pages are under 200 chars, or the total
    body text is under 1,500 chars.
    """
    warnings = list(extra_warnings or [])
    bodies = [_page_body(p.text) for p in pages]
    total_chars = sum(len(b) for b in bodies)
    low_text_pages = [
        p.number for p, b in zip(pages, bodies) if len(b) < _LOW_PAGE_CHARS
    ]
    n = max(len(pages), 1)
    unreliable = (len(low_text_pages) > n / 2) or (total_chars < _MIN_TOTAL_CHARS)

    if extraction_status is None:
        if not pages or total_chars == 0:
            extraction_status = "empty"
            unreliable = True
            warnings.append("No extractable text found in the PDF.")
        elif unreliable:
            extraction_status = "needs_ocr"
            warnings.append(
                "Little or no extractable text; OCR may be required "
                "(not available in this demo slice)."
            )
        else:
            extraction_status = "ok"

    if extraction_status in ("needs_ocr", "empty", "error"):
        unreliable = True

    injection_hits: list[Evidence] = []
    for p, b in zip(pages, bodies):
        for line in b.splitlines():
            if _INJECTION.search(line):
                injection_hits.append(
                    Evidence(quote=line.strip()[:300], page=p.number)
                )

    return Preflight(
        unreliable_extraction=unreliable,
        low_text_pages=low_text_pages,
        injection_hits=injection_hits,
        extraction_status=extraction_status,
        warnings=warnings,
    )


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M-") + uuid.uuid4().hex[:4]


def _contract_doc(
    *,
    file_name: str,
    playbook_version: str,
    pages: list[Page],
    preflight: Preflight,
) -> ContractDoc:
    return ContractDoc(
        run_id=_new_run_id(),
        file_name=file_name,
        playbook_version=playbook_version,
        pages=pages,
        preflight=preflight,
    )


def extract_pdf_bytes(
    data: bytes,
    *,
    filename: str,
    playbook_version: str = "demo-v1",
) -> ContractDoc:
    """Page-aware PDF intake from in-memory bytes (Streamlit / upload path)."""
    _validate_pdf_bytes(data, filename)
    from io import BytesIO

    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(BytesIO(data))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001 — surface as classified error
                return _contract_doc(
                    file_name=Path(filename).name,
                    playbook_version=playbook_version,
                    pages=[],
                    preflight=build_preflight(
                        [],
                        extraction_status="error",
                        extra_warnings=[
                            f"Encrypted or unreadable PDF: {exc}",
                        ],
                    ),
                )
        pages = _pages_from_reader(reader)
    except PdfReadError as exc:
        return _contract_doc(
            file_name=Path(filename).name,
            playbook_version=playbook_version,
            pages=[],
            preflight=build_preflight(
                [],
                extraction_status="error",
                extra_warnings=[f"Corrupt or unreadable PDF: {exc}"],
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return _contract_doc(
            file_name=Path(filename).name,
            playbook_version=playbook_version,
            pages=[],
            preflight=build_preflight(
                [],
                extraction_status="error",
                extra_warnings=[f"PDF intake failed: {exc}"],
            ),
        )

    return _contract_doc(
        file_name=Path(filename).name,
        playbook_version=playbook_version,
        pages=pages,
        preflight=build_preflight(pages),
    )


def intake(req: ReviewRequest) -> ContractDoc:
    """Page-aware PDF intake from a filesystem path."""
    path = _validate_pdf_path(req.pdf_path)
    data = path.read_bytes()
    # Re-validate size after read (guards race / symlink surprises).
    if len(data) > MAX_UPLOAD_BYTES:
        raise IntakeError(
            f"File exceeds the demo upload limit of "
            f"{MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB.",
            code="too_large",
        )
    return extract_pdf_bytes(
        data, filename=path.name, playbook_version=req.playbook_version
    )


# --------------------------------------------------------------------------- #
# 2c. Performance tool (CSV)
# --------------------------------------------------------------------------- #
def performance_tool(req: ReviewRequest) -> Optional[PerformanceSummary]:
    if not req.csv_path:
        return None
    import pandas as pd

    df = pd.read_csv(req.csv_path)
    if "supplier_id" not in df.columns:
        return None
    supplier = req.supplier_id or str(df["supplier_id"].iloc[0])
    df = df[df["supplier_id"].astype(str) == str(supplier)].copy()
    if df.empty:
        return PerformanceSummary(supplier_id=supplier, months=[], missing_months=[])

    df = df.sort_values("month").tail(3)
    months = [
        MonthPerformance(
            month=str(row["month"]),
            on_time_pct=float(row["on_time_pct"]),
            tickets=int(row["tickets"]) if "tickets" in df.columns and not pd.isna(row.get("tickets")) else None,
            monthly_fee_inr=float(row["monthly_fee_inr"]) if "monthly_fee_inr" in df.columns else None,
        )
        for _, row in df.iterrows()
    ]
    return PerformanceSummary(supplier_id=supplier, months=months, missing_months=[])


# --------------------------------------------------------------------------- #
# 3. Grounding
# --------------------------------------------------------------------------- #
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"-\s*\n\s*", "", s)  # de-hyphenate line breaks
    return re.sub(r"\s+", " ", s).strip().lower()


def find_pages(quote: str, pages: list[Page]) -> list[int]:
    q = norm(quote)
    if not q:
        return []
    by_no = {p.number: norm(p.text) for p in pages}
    hits = [n for n, t in by_no.items() if q in t]
    if not hits:  # quote split across a page break: join page n + n+1
        hits = [n for n in by_no if q in by_no[n] + " " + by_no.get(n + 1, "")]
    return hits


def check_meta(meta: ContractMetadata, doc: ContractDoc) -> ContractMetadata:
    """Drop metadata values whose evidence quote is not in the page text.

    Unknown is better than invented: if the supporting quote cannot be grounded,
    clear both the value and the evidence and record the uncertainty.
    """
    updated = {}
    for name, field in meta.__dict__.items():
        if field is not None and field.evidence is not None:
            if not find_pages(field.evidence.quote, doc.pages):
                note = (field.uncertainty or "").strip()
                if "quote not found" not in note:
                    note = (note + " (quote not found in text)").strip()
                field = field.model_copy(
                    update={
                        "value": None,
                        "evidence": None,
                        "uncertainty": note or "quote not found in text",
                    }
                )
        updated[name] = field
    return ContractMetadata(**updated)


def ground(inp: GroundingInput) -> GroundedResult:
    out: list[GroundedFinding] = []
    dropped: list[DroppedItem] = []
    by_rule = {f.rule_id: f for f in inp.clauses.findings}

    for rule in inp.playbook.rules:  # every rule gets exactly one result
        f = by_rule.get(rule.id)
        if f is None:
            dropped.append(
                DroppedItem(rule_id=rule.id, quote="", reason="no_finding_returned")
            )
            from app.schemas import ClauseFinding

            out.append(
                GroundedFinding(
                    finding=ClauseFinding(rule_id=rule.id, found=False),
                    grounding="missing",
                )
            )
            continue

        if not f.found:
            out.append(GroundedFinding(finding=f, grounding="absent_claimed"))
            continue

        kept: list[Evidence] = []
        warnings: list[str] = []
        level = "exact"
        for ev in f.evidence:
            hits = find_pages(ev.quote, inp.doc.pages)
            if not hits:
                dropped.append(
                    DroppedItem(rule_id=f.rule_id, quote=ev.quote, reason="not_in_text")
                )
                continue
            if len(hits) > 1:
                level = "ambiguous"
                warnings.append(f"quote appears on pages {hits}")
            elif ev.page not in hits:
                level = "page_corrected" if level == "exact" else level
                warnings.append(f"page {ev.page} corrected to {hits[0]}")
                ev = ev.model_copy(update={"page": hits[0]})
            kept.append(ev)

        out.append(
            GroundedFinding(
                finding=f.model_copy(update={"evidence": kept}),
                grounding=level if kept else "failed",
                warnings=warnings,
            )
        )

    return GroundedResult(
        meta=check_meta(inp.meta, inp.doc), findings=out, dropped=dropped
    )


# --------------------------------------------------------------------------- #
# Cross-reference check (feeds apply_rules)
# --------------------------------------------------------------------------- #
# Only CARVE-OUT language matters for the "unresolved cross-reference" check:
# a clause that is *subject to* the cap is benign, but one that is excepted from
# it ("except as set out in clause 12", "notwithstanding clause 5") can hide an
# obligation outside the cap and must be resolved before the rule can pass.
XREF = re.compile(
    r"(except as (?:set out|provided) in|save as (?:set out|provided) in|"
    r"notwithstanding)\s+(?:clause|section)\s+(\d+(?:\.\d+)*)",
    re.I,
)


def _clause_number(ref: str) -> str:
    """Extract the clause number from a free-form reference, e.g.
    'Clause 12 (Indemnity)' -> '12', 'clause 5.2' -> '5.2'."""
    m = re.search(r"\d+(?:\.\d+)*", ref or "")
    return m.group(0) if m else (ref or "").strip()


def unresolved_xrefs(g: GroundedFinding) -> list[str]:
    refs = {m.group(2) for ev in g.finding.evidence for m in XREF.finditer(ev.quote)}
    reviewed = {_clause_number(r) for r in g.finding.cross_references}
    return sorted(refs - reviewed)


# --------------------------------------------------------------------------- #
# 4. Apply rules
# --------------------------------------------------------------------------- #
# Topic keywords used for the "claimed absence while topic appears" check.
_TOPIC_KEYWORDS = {
    "LIABILITY-001": ["liability", "limitation of liability", "liable"],
    "LIABILITY-002": ["indemnif", "indemnity", "indemnification"],
    "PAYMENT-001": ["payment", "net ", "invoice", "pay"],
    "RENEWAL-001": ["renew", "term", "notice"],
    "TERMINATION-001": ["terminat"],
    "DATA-001": ["personal data", "data protection", "dpa", "privacy"],
    "SLA-001": ["service level", "sla", "on-time", "uptime", "credit"],
    "PRICE-001": ["price", "fee", "increase", "charges"],
    "LAW-001": ["governing law", "jurisdiction", "courts"],
    "ASSIGN-001": ["assign"],
}


def _topic_in_pages(rule_id: str, pages: list[Page]) -> bool:
    kws = _TOPIC_KEYWORDS.get(rule_id, [])
    blob = norm(" ".join(p.text for p in pages))
    return any(k in blob for k in kws)


def _processes_personal_data(meta: ContractMetadata) -> bool:
    v = (meta.processes_personal_data.value or "").lower()
    return v in ("yes", "true", "1") or "personal data" in v or v == "y"


def apply_rules(inp: RulesInput) -> ScoredFindings:
    pages = inp.pages or []
    scored: list[ScoredFinding] = []
    for g in inp.grounded.findings:
        rule = inp.playbook.rule(g.finding.rule_id)
        if rule is None:
            continue

        status = rules.check_status(rule, g.finding, inp.grounded.meta)

        # Grounding-driven overrides.
        if g.grounding in ("failed", "missing"):
            status = "unclear"
        elif g.grounding == "ambiguous" and (rule.severity == "high" or rule.legal_review):
            status = "unclear"

        # Claimed absence while the topic clearly appears in the text.
        if g.grounding == "absent_claimed":
            if rule.id == "DATA-001":
                status = "not_found" if _processes_personal_data(inp.grounded.meta) else "preferred"
            elif _topic_in_pages(rule.id, pages):
                status = "unclear"
            else:
                status = "not_found"

        # Unresolved cross-reference -> unclear.
        if g.grounding not in ("missing", "absent_claimed") and unresolved_xrefs(g):
            status = "unclear"

        trace = _rule_trace(rule, g, status)
        scored.append(
            ScoredFinding(
                grounded=g,
                status=status,
                severity=rule.severity,
                requires_legal=rule.legal_review,
                rule_trace=trace,
            )
        )

    _apply_liability_interaction(scored)
    return ScoredFindings(findings=scored)


def _apply_liability_interaction(scored: list[ScoredFinding]) -> None:
    """Liability-indemnity interaction (the documented Contract B fix).

    If the general indemnity (LIABILITY-002) is uncapped, the stated liability
    cap (LIABILITY-001) does not actually hold, so a passing cap is escalated to
    walk-away. This catches the cross-referenced carve-out that a reader of the
    cap clause alone would miss.
    """
    by_id = {f.grounded.finding.rule_id: f for f in scored}
    l2 = by_id.get("LIABILITY-002")
    l1 = by_id.get("LIABILITY-001")
    if l2 and l2.status == "walk_away" and l1 and l1.status in ("preferred", "fallback", "unclear"):
        l1.status = "walk_away"
        l1.rule_trace += " | escalated: indemnity (LIABILITY-002) sits outside the cap"


def _rule_trace(rule, g: GroundedFinding, status: str) -> str:
    value = g.finding.extracted_value or "no value"
    target = {
        "preferred": rule.preferred,
        "fallback": rule.fallback,
        "walk_away": rule.walk_away,
    }
    label = status.replace("_", "-")
    if status in target:
        return f"{rule.id}: {value} -> {label} (vs preferred: {rule.preferred})"
    return f"{rule.id}: {value} -> {label}"


# --------------------------------------------------------------------------- #
# 5. Router (ordered rules, fail closed, first match wins, full trace)
# --------------------------------------------------------------------------- #
def route(inp: RouteInput) -> RouteDecision:
    s = inp.scored.findings
    flags = inp.flags

    # R1: fail closed.
    if inp.preflight.unreliable_extraction:
        return RouteDecision(route="manual_review", trace=["R1: extraction unreliable"])
    unclear = [
        f for f in s
        if f.status == "unclear" and (f.severity == "high" or f.requires_legal)
    ]
    if unclear:
        return RouteDecision(
            route="manual_review", trace=[f"R1: {f.rule_trace}" for f in unclear]
        )

    # R2: Legal.
    legal = [
        f for f in s
        if (f.severity == "high" and f.status == "walk_away")
        or (f.requires_legal and f.status in ("walk_away", "not_found"))
    ]
    if legal:
        return RouteDecision(route="legal", trace=[f"R2: {f.rule_trace}" for f in legal])

    # R3: procurement owner = any other walk-away, or any security flag.
    walk = [f for f in s if f.status == "walk_away"]
    if walk or flags.any():
        trace = [f"R3: {f.rule_trace}" for f in walk]
        if flags.any():
            trace.append("R3: text aimed at an AI reviewer found in contract")
        return RouteDecision(route="procurement_owner", trace=trace)

    # R4: only preferred or fallback terms.
    notes = [f"R4 note: {f.rule_trace}" for f in s if f.status == "fallback"]
    return RouteDecision(
        route="low_risk_queue", trace=notes or ["R4: all terms at preferred"]
    )


# --------------------------------------------------------------------------- #
# 6b. Obligations and service credits
# --------------------------------------------------------------------------- #
def _parse_date(v: Optional[str]) -> Optional[date]:
    if not v:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(v.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _ev(field) -> Evidence:
    if field is not None and field.evidence is not None:
        return field.evidence
    return Evidence(quote="(derived)", page=0)


def build_obligations(inp: ObligationInput) -> tuple[list[Obligation], Optional[CreditClaim]]:
    meta = inp.meta
    out: list[Obligation] = []

    expiry = _parse_date(meta.expiry_date.value)
    notice_days = rules.parse_days(meta.notice_period_days.value) or rules.parse_net_days(
        meta.notice_period_days.value
    )
    if notice_days is None and rules.parse_bare_number(meta.notice_period_days.value) is not None:
        notice_days = int(rules.parse_bare_number(meta.notice_period_days.value))  # "90" from the model
    basis = (meta.renewal_notice_basis.value or "").lower()
    business_days = "business day" in basis or "working day" in basis

    notice_deadline: Optional[date] = None
    if expiry and notice_days and "then-current term" in basis and not business_days:
        notice_deadline = expiry - timedelta(days=notice_days)
        urgent = (notice_deadline - inp.today).days <= 30
        out.append(
            Obligation(
                kind="notice_deadline",
                due=notice_deadline,
                owner="procurement",
                source=_ev(meta.notice_period_days),
                urgent=urgent,
            )
        )
        out.append(
            Obligation(
                kind="renewal_decision",
                due=notice_deadline - timedelta(days=14),
                owner="procurement",
                source=_ev(meta.notice_period_days),
                urgent=urgent,
            )
        )
    elif expiry and notice_days:
        out.append(
            Obligation(
                kind="notice_deadline",
                due=None,
                owner="procurement",
                source=_ev(meta.renewal_notice_basis),
                uncertain=True,
            )
        )

    # SLA review (monthly) when a measurable SLA exists.
    if meta.sla_target_pct.value and (rules.parse_percent(meta.sla_target_pct.value)
                                      or rules.parse_bare_number(meta.sla_target_pct.value)):
        out.append(
            Obligation(
                kind="sla_review",
                due=None,
                owner="procurement",
                source=_ev(meta.sla_target_pct),
            )
        )

    # Service-credit cross-check (CSV x contract).
    credit = _credit_claim(meta, inp.perf)
    if credit and credit.total_credit_inr > 0:
        out.append(
            Obligation(
                kind="credit_claim",
                due=None,
                owner="procurement",
                source=_ev(meta.service_credit_rule),
                amount_inr=credit.total_credit_inr,
            )
        )
    return out, credit


def _credit_claim(meta: ContractMetadata, perf: Optional[PerformanceSummary]) -> Optional[CreditClaim]:
    if perf is None or not perf.months:
        return None
    target = rules.parse_percent(meta.sla_target_pct.value) or rules.parse_bare_number(meta.sla_target_pct.value)
    rule_text = meta.service_credit_rule.value or ""
    rate = rules.parse_percent(rule_text)  # credit % per point missed
    # The cap is the second percentage in the rule text, if present.
    pcts = re.findall(r"(\d+(?:\.\d+)?)\s*%", rule_text)
    cap = float(pcts[1]) if len(pcts) >= 2 else (float(pcts[0]) if pcts else None)
    if target is None or rate is None:
        return None
    if cap is None:
        cap = 100.0

    per_month = []
    total = 0.0
    capped_any = False
    for m in perf.months:
        fee = m.monthly_fee_inr or 0.0
        amount = rules.check_credit_claim(target, m.on_time_pct, rate, cap, fee)
        if amount is None:
            continue
        raw_pct = (target - m.on_time_pct) * rate
        if raw_pct > cap:
            capped_any = True
        per_month.append(
            {
                "month": m.month,
                "on_time_pct": m.on_time_pct,
                "credit_pct": min(raw_pct, cap),
                "amount_inr": amount,
            }
        )
        total += amount

    if not per_month:
        return CreditClaim(
            supplier_id=perf.supplier_id,
            months_considered=len(perf.months),
            total_credit_inr=0.0,
            per_month=[],
            note="Actual performance met or exceeded the SLA target.",
        )
    return CreditClaim(
        supplier_id=perf.supplier_id,
        months_considered=len(perf.months),
        total_credit_inr=round(total, 2),
        per_month=per_month,
        capped=capped_any,
    )


# --------------------------------------------------------------------------- #
# Redline check (plain function)
# --------------------------------------------------------------------------- #
NUM = re.compile(r"\d+(?:\.\d+)?%?")
TERM = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b")  # multi-word capitalised terms
MODAL = re.compile(r"\b(shall|must)\b", re.I)


def check_redline(r: Redline, fallback: str) -> CheckedRedline:
    p, o = r.proposed_text, r.original.quote
    problems: list[str] = []
    if p.startswith("ESCALATE"):
        problems.append("drafter escalated")
    if set(NUM.findall(p)) - set(NUM.findall(o)) - set(NUM.findall(fallback)):
        problems.append("new number not in original or fallback")
    if set(TERM.findall(o)) - set(TERM.findall(p)):
        problems.append("defined term removed")
    if len(MODAL.findall(p)) > len(MODAL.findall(o)):
        problems.append("new obligation added")
    if len(p) > 1.6 * len(o):
        problems.append("much longer than original")
    return CheckedRedline(redline=r, passed=not problems, problems=problems)


def check_redlines(
    redlines: Redlines, deviating: list[ScoredFinding], playbook: Playbook
) -> list[CheckedRedline]:
    fallback_by_rule = {r.id: r.fallback_text for r in playbook.rules}
    out: list[CheckedRedline] = []
    for r in redlines.redlines:
        out.append(check_redline(r, fallback_by_rule.get(r.rule_id, "")))
    return out


# --------------------------------------------------------------------------- #
# 8. Assemble / manual-review fallback
# --------------------------------------------------------------------------- #
def attach_knowledge_groundings(pkg: ReviewPackage) -> ReviewPackage:
    """RAG Phase 3 — persist contract-vs-policy comparisons on the package."""
    try:
        from app.rag import ground_review_package

        comps = ground_review_package(pkg)
        return pkg.model_copy(
            update={
                "knowledge_groundings": [c.model_dump(mode="json") for c in comps],
                "stage_status": {**pkg.stage_status, "knowledge": "ok"},
            }
        )
    except Exception as exc:
        return pkg.model_copy(
            update={"stage_status": {**pkg.stage_status, "knowledge": f"skipped: {exc}"}}
        )


def assemble(
    doc: ContractDoc,
    grounded: GroundedResult,
    scored: ScoredFindings,
    route_decision: RouteDecision,
    redlines: list[CheckedRedline],
    obligations: list[Obligation],
    flags: SecurityFlags,
    credit: Optional[CreditClaim] = None,
    gate=None,
) -> ReviewPackage:
    status = "complete" if route_decision.route != "manual_review" else "manual_review"
    stage_status = {
        "intake": "ok",
        "grounding": "ok",
        "rules": "ok",
        "route": route_decision.route,
    }
    if gate is not None:
        stage_status["legal_gate"] = getattr(gate, "decision", "n/a")
    pkg = ReviewPackage(
        run_id=doc.run_id,
        status=status,
        metadata=grounded.meta,
        findings=scored.findings,
        dropped=grounded.dropped,
        route=route_decision,
        redlines=redlines,
        obligations=obligations,
        security=flags,
        stage_status=stage_status,
        file_name=doc.file_name,
        playbook_version=doc.playbook_version,
        credit_claim=credit,
    )
    return attach_knowledge_groundings(pkg)


def manual_review_package(doc_or_req, reason: str) -> ReviewPackage:
    """Fail-closed package. Accepts either a ContractDoc or a ReviewRequest."""
    run_id = getattr(doc_or_req, "run_id", None) or datetime.now().strftime(
        "%Y%m%d-%H%M-"
    ) + uuid.uuid4().hex[:4]
    file_name = getattr(doc_or_req, "file_name", None) or Path(
        getattr(doc_or_req, "pdf_path", "unknown.pdf")
    ).name
    return ReviewPackage(
        run_id=run_id,
        status="manual_review",
        metadata=None,
        findings=[],
        dropped=[],
        route=RouteDecision(route="manual_review", trace=[f"manual review: {reason}"]),
        redlines=[],
        obligations=[],
        security=SecurityFlags(),
        stage_status={"error": reason},
        file_name=file_name,
        # Keep the audit trail complete even when the run stops before the model.
        playbook_version=getattr(doc_or_req, "playbook_version", None),
    )
