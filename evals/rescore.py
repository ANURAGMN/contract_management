"""Re-apply the *current* deterministic rules to recorded live model output.

A recording stores what Gemini returned (grounded findings + metadata). Severity,
outcome and route are computed by code, so a rule fix can be checked against real
model output without spending quota: rebuild the grounded result, re-run
``apply_rules`` and ``route``, and compare with what was recorded.

    python evals/rescore.py      # prints recorded vs re-scored route per contract
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import steps  # noqa: E402
from app.schemas import GroundedResult, ReviewPackage, ReviewRequest, RouteInput, RulesInput  # noqa: E402

CONTRACTS = ROOT / "data" / "contracts"


def rescore(pkg: ReviewPackage) -> ReviewPackage:
    """Same model output, current rules. Returns a copy; manual-review packages pass through."""
    if pkg.status != "complete" and not pkg.findings:
        return pkg
    pdf = CONTRACTS / (pkg.file_name or "")
    pages, preflight = [], None
    if pdf.exists():
        doc = steps.intake(ReviewRequest(pdf_path=str(pdf), playbook_version=pkg.playbook_version or "demo-v1"))
        pages, preflight = doc.pages, doc.preflight
    playbook = steps.load_playbook(pkg.playbook_version or "demo-v1")
    grounded = GroundedResult(meta=pkg.metadata, findings=[sf.grounded for sf in pkg.findings],
                              dropped=pkg.dropped)
    scored = steps.apply_rules(RulesInput(grounded=grounded, playbook=playbook, pages=pages))
    if preflight is None:
        from app.schemas import Preflight

        preflight = Preflight(unreliable_extraction=False)
    route = steps.route(RouteInput(scored=scored, preflight=preflight, flags=pkg.security))
    # Obligation dates are deterministic from the extracted metadata. Credit claims also need the
    # performance CSV the run was given, which is not stored, so recorded ones are kept as they are.
    obligations = list(pkg.obligations)
    if pkg.metadata is not None:
        from app.schemas import ObligationInput

        fresh, _ = steps.build_obligations(ObligationInput(meta=pkg.metadata, perf=None, today=steps.today()))
        obligations = fresh + [o for o in pkg.obligations if o.kind == "credit_claim"]
    out = pkg.model_copy(deep=True)
    out.obligations = obligations
    out.findings = scored.findings
    out.route = route
    out.status = "manual_review" if route.route == "manual_review" else "complete"
    out.stage_status = {**(pkg.stage_status or {}), "rules": "rescored_with_current_rules"}
    return out


def main() -> int:
    from app import recordings

    for label, rec in sorted(recordings.latest_by_label("review").items()):
        pkg = ReviewPackage.model_validate(rec.payload)
        new = rescore(pkg)
        changed = [
            f"{a.grounded.finding.rule_id}: {a.status}→{b.status}"
            for a, b in zip(pkg.findings, new.findings) if a.status != b.status
        ]
        print(f"{label:<16} route {pkg.route.route:<18} → {new.route.route:<18} {', '.join(changed) or 'no change'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
