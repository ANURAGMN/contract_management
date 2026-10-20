"""Phase 1B grounded clause-review slice — offline (no live LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import (
    ClauseFinding,
    ClauseFindings,
    ClauseReviewInput,
    ContractDoc,
    Evidence,
    Page,
    Preflight,
    ReviewRequest,
)
from app.slice import run_clause_slice

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_A = ROOT / "data" / "contracts" / "contract_a.pdf"

PAYMENT_QUOTE = (
    "The Buyer shall pay all undisputed invoices within forty-five (45) days "
    "of the date of a valid invoice (Net 45)."
)
SLA_QUOTE = (
    "The Supplier shall maintain a measurable service level of 99.0% platform "
    "availability per calendar month."
)
DPA_QUOTE = (
    'The parties have entered into a Data Processing Agreement ("DPA") in the '
    "form attached as Schedule 2"
)


def _doc_from_pages(pages: list[Page], *, run_id: str = "t-slice") -> ContractDoc:
    from app.steps import build_preflight

    return ContractDoc(
        run_id=run_id,
        file_name="synthetic.pdf",
        playbook_version="demo-slice-v1",
        pages=pages,
        preflight=build_preflight(pages),
    )


def _page(n: int, body: str) -> Page:
    return Page(number=n, text=f"[PAGE {n}]\n{body}", character_count=len(body))


def _good_reviewer(_inp: ClauseReviewInput) -> ClauseFindings:
    return ClauseFindings(
        findings=[
            ClauseFinding(
                rule_id="PAYMENT-001",
                found=True,
                extracted_value="Net 45",
                evidence=[Evidence(quote=PAYMENT_QUOTE, page=1)],
            ),
            ClauseFinding(
                rule_id="SLA-001",
                found=True,
                extracted_value="99.0% with service credits",
                evidence=[Evidence(quote=SLA_QUOTE, page=2)],
            ),
            ClauseFinding(
                rule_id="DATA-001",
                found=True,
                extracted_value="DPA present",
                evidence=[Evidence(quote=DPA_QUOTE, page=3)],
            ),
        ]
    )


def _fabricating_reviewer(_inp: ClauseReviewInput) -> ClauseFindings:
    """Returns a quote that does not exist in the contract."""
    return ClauseFindings(
        findings=[
            ClauseFinding(
                rule_id="PAYMENT-001",
                found=True,
                extracted_value="Net 15",
                evidence=[
                    Evidence(
                        quote="The Buyer shall pay within Net 15 business days.",
                        page=1,
                    )
                ],
            ),
            ClauseFinding(rule_id="SLA-001", found=False),
            ClauseFinding(rule_id="DATA-001", found=False),
        ]
    )


def _missing_rules_reviewer(_inp: ClauseReviewInput) -> ClauseFindings:
    return ClauseFindings(findings=[])


def _broken_reviewer(_inp: ClauseReviewInput) -> ClauseFindings:
    raise ValidationError.from_exception_data(
        "ClauseFindings",
        [{"type": "missing", "loc": ("findings",), "input": {}, "ctx": {}}],
    )


@pytest.mark.skipif(not CONTRACT_A.exists(), reason="fixtures missing")
def test_slice_contract_a_grounded_end_to_end():
    result = run_clause_slice(
        ReviewRequest(
            pdf_path=str(CONTRACT_A),
            playbook_version="demo-slice-v1",
        ),
        clause_reviewer=_good_reviewer,
        skip_metadata=True,
    )
    assert result.status == "complete"
    assert result.extraction_status == "ok"
    assert len(result.findings) == 3

    by_id = {f.rule_id: f for f in result.findings}
    assert by_id["PAYMENT-001"].grounding_status == "verified"
    assert by_id["PAYMENT-001"].evidence[0].page == 1
    assert "Net 45" in (by_id["PAYMENT-001"].extracted_value or "")
    assert by_id["SLA-001"].grounding_status == "verified"
    assert by_id["DATA-001"].grounding_status == "verified"

    # No verified finding without a matched quote.
    for f in result.findings:
        if f.grounding_status == "verified":
            assert f.evidence
            assert f.grounding_detail in ("exact", "page_corrected")


@pytest.mark.skipif(not CONTRACT_A.exists(), reason="fixtures missing")
def test_unsupported_quote_cannot_be_verified():
    result = run_clause_slice(
        ReviewRequest(
            pdf_path=str(CONTRACT_A),
            playbook_version="demo-slice-v1",
        ),
        clause_reviewer=_fabricating_reviewer,
        skip_metadata=True,
    )
    payment = next(f for f in result.findings if f.rule_id == "PAYMENT-001")
    assert payment.grounding_status == "rejected"
    assert payment.grounding_detail == "failed"
    assert not payment.evidence
    assert any(d.reason == "not_in_text" for d in result.dropped)


@pytest.mark.skipif(not CONTRACT_A.exists(), reason="fixtures missing")
def test_missing_clause_findings_are_missing_not_verified():
    result = run_clause_slice(
        ReviewRequest(
            pdf_path=str(CONTRACT_A),
            playbook_version="demo-slice-v1",
        ),
        clause_reviewer=_missing_rules_reviewer,
        skip_metadata=True,
    )
    assert len(result.findings) == 3
    assert all(f.grounding_status == "missing" for f in result.findings)
    assert all(f.grounding_detail == "missing" for f in result.findings)
    assert result.routing_decision == "manual_review"


@pytest.mark.skipif(not CONTRACT_A.exists(), reason="fixtures missing")
def test_agent_schema_failure_returns_partial_not_clean():
    result = run_clause_slice(
        ReviewRequest(
            pdf_path=str(CONTRACT_A),
            playbook_version="demo-slice-v1",
        ),
        clause_reviewer=_broken_reviewer,
        skip_metadata=True,
    )
    assert result.status == "partial"
    assert result.findings == []
    assert result.routing_decision == "manual_review"
    assert any("Clause reviewer failed" in w for w in result.warnings)
    assert any("no risks found" in w for w in result.warnings)


def test_clean_synthetic_contract_preferred_payment():
    """Clean text + accurate fake findings → verified preferred payment."""
    body = (
        "MASTER AGREEMENT\n"
        + ("x" * 400)
        + "\n"
        + PAYMENT_QUOTE
        + "\n"
        + ("y" * 400)
    )
    doc = _doc_from_pages([_page(1, body), _page(2, "z" * 800), _page(3, "w" * 800)])
    assert doc.preflight.extraction_status == "ok"

    def reviewer(_inp: ClauseReviewInput) -> ClauseFindings:
        return ClauseFindings(
            findings=[
                ClauseFinding(
                    rule_id="PAYMENT-001",
                    found=True,
                    extracted_value="Net 45",
                    evidence=[Evidence(quote=PAYMENT_QUOTE, page=1)],
                ),
                ClauseFinding(rule_id="SLA-001", found=False, note="not a service SOW"),
                ClauseFinding(rule_id="DATA-001", found=False),
            ]
        )

    result = run_clause_slice(
        ReviewRequest(pdf_path="synthetic.pdf", playbook_version="demo-slice-v1"),
        clause_reviewer=reviewer,
        doc=doc,
        skip_metadata=True,
    )
    payment = next(f for f in result.findings if f.rule_id == "PAYMENT-001")
    assert payment.grounding_status == "verified"
    assert payment.status == "preferred"


def test_wrong_page_is_corrected_still_verified():
    body = ("pad " * 200) + SLA_QUOTE + (" pad" * 200)
    doc = _doc_from_pages(
        [_page(1, "irrelevant " * 200), _page(2, body), _page(3, "tail " * 200)]
    )

    def reviewer(_inp: ClauseReviewInput) -> ClauseFindings:
        return ClauseFindings(
            findings=[
                ClauseFinding(rule_id="PAYMENT-001", found=False),
                ClauseFinding(
                    rule_id="SLA-001",
                    found=True,
                    extracted_value="99.0%",
                    evidence=[Evidence(quote=SLA_QUOTE, page=1)],  # wrong page
                ),
                ClauseFinding(rule_id="DATA-001", found=False),
            ]
        )

    result = run_clause_slice(
        ReviewRequest(pdf_path="synthetic.pdf", playbook_version="demo-slice-v1"),
        clause_reviewer=reviewer,
        doc=doc,
        skip_metadata=True,
    )
    sla = next(f for f in result.findings if f.rule_id == "SLA-001")
    assert sla.grounding_status == "verified"
    assert sla.grounding_detail == "page_corrected"
    assert sla.evidence[0].page == 2
