"""Phase 1B smoke: PDF → grounded clause review → routing.

Offline (no API key) — uses a fixture-aligned fake reviewer:
    python scripts/smoke_clause_slice.py data/contracts/contract_a.pdf --fake

Live Gemini clause reviewer:
    python scripts/smoke_clause_slice.py data/contracts/contract_a.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import (  # noqa: E402
    ClauseFinding,
    ClauseFindings,
    ClauseReviewInput,
    ContractDoc,
    ContractMetadata,
    Evidence,
    Field_,
    ReviewRequest,
)
from app.slice import run_clause_slice  # noqa: E402

# Verbatim spans from data/contracts/src/contract_a.md (must match PDF text).
_FAKE_A = ClauseFindings(
    findings=[
        ClauseFinding(
            rule_id="PAYMENT-001",
            found=True,
            extracted_value="Net 45",
            evidence=[
                Evidence(
                    quote=(
                        "The Buyer shall pay all undisputed invoices within "
                        "forty-five (45) days of the date of a valid invoice (Net 45)."
                    ),
                    page=1,
                )
            ],
        ),
        ClauseFinding(
            rule_id="SLA-001",
            found=True,
            extracted_value="99.0% with service credits",
            evidence=[
                Evidence(
                    quote=(
                        "The Supplier shall maintain a measurable service level of "
                        "99.0% platform availability per calendar month."
                    ),
                    page=2,
                )
            ],
        ),
        ClauseFinding(
            rule_id="DATA-001",
            found=True,
            extracted_value="DPA present",
            evidence=[
                Evidence(
                    quote=(
                        'The parties have entered into a Data Processing Agreement '
                        '("DPA") in the form attached as Schedule 2'
                    ),
                    page=3,
                )
            ],
        ),
    ]
)


def _fake_reviewer(_inp: ClauseReviewInput) -> ClauseFindings:
    return _FAKE_A


def _fld(value: str | None, quote: str | None = None, page: int = 1) -> Field_:
    return Field_(
        value=value,
        evidence=Evidence(quote=quote, page=page) if quote else None,
        uncertainty=None if value else "not found in supplied text",
    )


def _fake_meta(_doc: ContractDoc) -> ContractMetadata:
    base = {name: Field_() for name in ContractMetadata.model_fields}
    base.update(
        {
            "supplier": _fld(
                "Brightline IT Services LLP",
                'Brightline IT Services LLP ("Supplier")',
                1,
            ),
            "buyer": _fld("Acme Retail India Private Limited", "Acme Retail India Private Limited", 1),
            "effective_date": _fld(
                "2026-01-01",
                'entered into as of 1 January 2026 (the "Effective Date")',
                1,
            ),
            "expiry_date": _fld(
                "2027-12-31",
                'continues until 31 December 2027 (the "Initial Term")',
                1,
            ),
            "payment_terms_days": _fld(
                "45",
                "The Buyer shall pay all undisputed invoices within forty-five (45) days "
                "of the date of a valid invoice (Net 45).",
                1,
            ),
            "sla_target_pct": _fld(
                "99.0",
                "measurable service level of 99.0% platform availability",
                2,
            ),
            "processes_personal_data": _fld(
                "yes",
                'Data Processing Agreement ("DPA")',
                3,
            ),
        }
    )
    return ContractMetadata(**base)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1B/2 clause+metadata slice smoke")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--playbook", default="demo-slice-v1")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use fixture-aligned fake agents (no API key)",
    )
    args = parser.parse_args()

    req = ReviewRequest(pdf_path=str(args.pdf), playbook_version=args.playbook)
    kwargs = {}
    if args.fake:
        kwargs["clause_reviewer"] = _fake_reviewer
        kwargs["metadata_extractor"] = _fake_meta
    result = run_clause_slice(req, **kwargs)
    payload = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2))

    verified = [f for f in result.findings if f.grounding_status == "verified"]
    known = (
        len(ContractMetadata.model_fields) - len(result.metadata_unknown_fields)
        if result.metadata
        else 0
    )
    print(
        f"\n# summary: status={result.status} route={result.routing_decision} "
        f"verified={len(verified)}/{len(result.findings)} "
        f"meta_known={known}/{len(ContractMetadata.model_fields)} fake={args.fake}",
        file=sys.stderr,
    )
    if result.status == "failed":
        return 1
    if not args.fake and result.status == "partial" and not result.findings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
