"""Phase 1B + Phase 2 — grounded clause-review slice with metadata.

Pipeline (explicit order, no LLM coordinator):

    PDF intake → metadata extractor (injectable) + clause reviewer (injectable)
              → evidence grounding (meta + findings)
              → deterministic rules → routing → ``SliceReviewResult``

Unit tests inject fake agents so the suite runs without an API key.
Live path uses ``app.agents.metadata_extractor`` / ``clause_reviewer``.
"""

from __future__ import annotations

from typing import Callable, Optional

from pydantic import ValidationError

from app import agents, steps
from app.errors import IntakeError
from app.schemas import (
    ClauseFindings,
    ClauseReviewInput,
    ContractDoc,
    ContractMetadata,
    Field_,
    GroundedFinding,
    GroundingInput,
    GroundingStatus,
    ReviewRequest,
    RouteInput,
    RulesInput,
    ScoredFinding,
    ScoredFindings,
    SecurityFlags,
    SliceFinding,
    SliceReviewResult,
)

ClauseReviewerFn = Callable[[ClauseReviewInput], ClauseFindings]
MetadataExtractorFn = Callable[[ContractDoc], ContractMetadata]


def _empty_meta() -> ContractMetadata:
    return ContractMetadata(**{name: Field_() for name in ContractMetadata.model_fields})


def _unknown_fields(meta: ContractMetadata) -> list[str]:
    """Fields with no value after grounding (unknown stays unknown)."""
    unknown: list[str] = []
    for name in ContractMetadata.model_fields:
        field = getattr(meta, name)
        if field is None or field.value is None:
            unknown.append(name)
    return unknown


def _map_grounding(g: GroundedFinding) -> GroundingStatus:
    """Only text-matched evidence may be marked verified."""
    if g.grounding in ("exact", "page_corrected") and g.finding.evidence:
        return "verified"
    if g.grounding == "ambiguous" and g.finding.evidence:
        return "unverified"
    if g.grounding == "failed":
        return "rejected"
    if g.grounding in ("missing", "absent_claimed"):
        return "missing"
    return "unverified"


def _to_slice_finding(scored: ScoredFinding, clause_name: str) -> SliceFinding:
    g = scored.grounded
    return SliceFinding(
        finding_id=f"{g.finding.rule_id}-F1",
        rule_id=g.finding.rule_id,
        clause_name=clause_name,
        extracted_value=g.finding.extracted_value,
        evidence=list(g.finding.evidence),
        grounding_status=_map_grounding(g),
        grounding_detail=g.grounding,
        status=scored.status,
        severity=scored.severity,
        requires_legal=scored.requires_legal,
        rule_trace=scored.rule_trace,
        warnings=list(g.warnings),
    )


def _package_status(doc: ContractDoc, agent_failed: bool) -> str:
    if doc.preflight.extraction_status == "error":
        return "failed"
    if doc.preflight.extraction_status in ("needs_ocr", "empty"):
        return "needs_ocr"
    if agent_failed or doc.preflight.unreliable_extraction:
        return "partial"
    return "complete"


def _manual_route(doc: ContractDoc, extra_reasons: list[str] | None = None):
    route = steps.route(
        RouteInput(
            scored=ScoredFindings(findings=[]),
            preflight=doc.preflight,
            flags=SecurityFlags(pre=doc.preflight.injection_hits),
        )
    )
    reasons = list(route.trace)
    if extra_reasons:
        reasons.extend(extra_reasons)
    return route.route, reasons


def run_clause_slice(
    req: ReviewRequest,
    *,
    clause_reviewer: Optional[ClauseReviewerFn] = None,
    metadata_extractor: Optional[MetadataExtractorFn] = None,
    doc: Optional[ContractDoc] = None,
    skip_metadata: bool = False,
    include_commercial: bool = False,
    budget_inr: float | None = None,
) -> SliceReviewResult:
    """Run the Phase 1B/2 vertical slice.

    Parameters
    ----------
    req:
        Review request (pdf path + playbook version). Prefer
        ``playbook_version='demo-slice-v1'`` for the thin demo ruleset.
    clause_reviewer / metadata_extractor:
        Optional injectable agents. Default to live Gemini agents.
    doc:
        Optional pre-built ``ContractDoc`` (tests may skip PDF I/O).
    skip_metadata:
        If True, skip the metadata agent (Phase 1B-only behaviour).
    include_commercial:
        If True, attach demo quotation comparison (Phase 4, no LLM).
    budget_inr:
        Annual budget for commercial comparison (default ₹48 lakh = 4_800_000).
    """
    reviewer = clause_reviewer or agents.clause_reviewer
    meta_fn = metadata_extractor or agents.metadata_extractor
    warnings: list[str] = []
    agent_failed = False
    meta = _empty_meta()
    commercial = None
    if include_commercial:
        from app.commercial import DEFAULT_BUDGET_INR, compare_demo_quotations

        try:
            commercial = compare_demo_quotations(
                budget_inr=budget_inr if budget_inr is not None else DEFAULT_BUDGET_INR,
                include_baseline=True,
            )
        except FileNotFoundError as exc:
            warnings.append(str(exc))

    if doc is None:
        try:
            doc = steps.intake(req)
        except IntakeError as exc:
            return SliceReviewResult(
                run_id="intake-failed",
                file_name=req.pdf_path,
                playbook_version=req.playbook_version,
                extraction_status="error",
                findings=[],
                routing_decision="manual_review",
                routing_reasons=[f"intake error: {exc}"],
                warnings=[str(exc)],
                status="failed",
            )

    warnings.extend(doc.preflight.warnings)

    if doc.preflight.extraction_status in ("needs_ocr", "empty", "error"):
        decision, reasons = _manual_route(doc)
        return SliceReviewResult(
            run_id=doc.run_id,
            file_name=doc.file_name,
            playbook_version=doc.playbook_version,
            extraction_status=doc.preflight.extraction_status,
            metadata=None,
            metadata_unknown_fields=[],
            findings=[],
            routing_decision=decision,
            routing_reasons=reasons,
            warnings=warnings + ["Clause review skipped: extraction not usable."],
            status=_package_status(doc, agent_failed=False),
        )

    # --- Phase 2: metadata (failures must not fabricate values) -------------
    if not skip_metadata:
        try:
            raw_meta = meta_fn(doc)
            if not isinstance(raw_meta, ContractMetadata):
                raw_meta = ContractMetadata.model_validate(raw_meta)
            meta = raw_meta
        except (ValidationError, ValueError, RuntimeError, TypeError) as exc:
            agent_failed = True
            meta = _empty_meta()
            warnings.append(
                f"Metadata extractor failed ({type(exc).__name__}): {exc}. "
                "Fields left unknown; no values were fabricated."
            )

    playbook = steps.load_playbook(req.playbook_version)
    topic_by_id = {r.id: r.topic for r in playbook.rules}

    try:
        clauses = reviewer(
            ClauseReviewInput(pages=doc.pages, rules=playbook.prompt_view())
        )
    except (ValidationError, ValueError, RuntimeError, TypeError) as exc:
        agent_failed = True
        decision, reasons = _manual_route(
            doc,
            extra_reasons=["clause agent failure — manual review required"],
        )
        # Still ground metadata so unsupported quotes are stripped.
        grounded_meta = steps.check_meta(meta, doc)
        return SliceReviewResult(
            run_id=doc.run_id,
            file_name=doc.file_name,
            playbook_version=doc.playbook_version,
            extraction_status=doc.preflight.extraction_status,
            metadata=grounded_meta,
            metadata_unknown_fields=_unknown_fields(grounded_meta),
            findings=[],
            routing_decision="manual_review",
            routing_reasons=reasons,
            warnings=warnings
            + [
                f"Clause reviewer failed ({type(exc).__name__}): {exc}. "
                "Do not interpret this as 'no risks found'."
            ],
            status="partial",
        )

    if not isinstance(clauses, ClauseFindings):
        clauses = ClauseFindings.model_validate(clauses)

    grounded = steps.ground(
        GroundingInput(
            doc=doc,
            meta=meta,
            clauses=clauses,
            playbook=playbook,
        )
    )
    scored = steps.apply_rules(
        RulesInput(grounded=grounded, playbook=playbook, pages=doc.pages)
    )
    flags = SecurityFlags(
        pre=doc.preflight.injection_hits,
        model=clauses.suspicious_instructions,
    )
    route = steps.route(
        RouteInput(scored=scored, preflight=doc.preflight, flags=flags)
    )

    findings = [
        _to_slice_finding(
            f, topic_by_id.get(f.grounded.finding.rule_id, f.grounded.finding.rule_id)
        )
        for f in scored.findings
    ]

    for f in findings:
        if f.grounding_status == "verified":
            if not f.evidence or f.grounding_detail not in ("exact", "page_corrected"):
                raise RuntimeError(
                    f"Invariant broken: {f.rule_id} marked verified without "
                    "matched evidence"
                )

    return SliceReviewResult(
        run_id=doc.run_id,
        file_name=doc.file_name,
        playbook_version=doc.playbook_version,
        extraction_status=doc.preflight.extraction_status,
        metadata=grounded.meta,
        metadata_unknown_fields=_unknown_fields(grounded.meta),
        findings=findings,
        quotation_comparison=commercial,
        routing_decision=route.route,
        routing_reasons=list(route.trace),
        warnings=warnings,
        dropped=list(grounded.dropped),
        status=_package_status(doc, agent_failed=agent_failed),
    )
