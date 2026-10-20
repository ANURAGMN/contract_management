"""Phase 2 metadata extraction — offline (no live LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import (
    ClauseFinding,
    ClauseFindings,
    ClauseReviewInput,
    ContractDoc,
    ContractMetadata,
    Evidence,
    Field_,
    ReviewRequest,
)
from app.slice import run_clause_slice
from app.steps import check_meta, intake

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_A = ROOT / "data" / "contracts" / "contract_a.pdf"

SUPPLIER_QUOTE = 'Brightline IT Services LLP ("Supplier")'
BUYER_QUOTE = "Acme Retail India Private Limited"
PAYMENT_QUOTE = (
    "The Buyer shall pay all undisputed invoices within forty-five (45) days "
    "of the date of a valid invoice (Net 45)."
)
FAKE_QUOTE = "This supplier is CloudSoft Technologies Pvt Ltd only."


def _field(value: str | None, quote: str | None = None, page: int = 1, **kw) -> Field_:
    ev = Evidence(quote=quote, page=page) if quote else None
    return Field_(value=value, evidence=ev, **kw)


def _meta(**overrides) -> ContractMetadata:
    base = {name: Field_() for name in ContractMetadata.model_fields}
    base.update(overrides)
    return ContractMetadata(**base)


def _minimal_clause_reviewer(_inp: ClauseReviewInput) -> ClauseFindings:
    return ClauseFindings(
        findings=[
            ClauseFinding(rule_id="PAYMENT-001", found=False),
            ClauseFinding(rule_id="SLA-001", found=False),
            ClauseFinding(rule_id="DATA-001", found=False),
        ]
    )


def _good_meta_extractor(_doc: ContractDoc) -> ContractMetadata:
    return _meta(
        supplier=_field("Brightline IT Services LLP", SUPPLIER_QUOTE, 1),
        buyer=_field("Acme Retail India Private Limited", BUYER_QUOTE, 1),
        effective_date=_field(
            "2026-01-01",
            "entered into as of 1 January 2026 (the \"Effective Date\")",
            1,
        ),
        expiry_date=_field(
            "2027-12-31",
            "continues until 31 December 2027 (the \"Initial Term\")",
            1,
        ),
        payment_terms_days=_field("45", PAYMENT_QUOTE, 1),
        # Intentionally unknown — contract has no governing-law detail in this fake.
        governing_law=_field(
            None, uncertainty="not found in supplied text"
        ),
        contract_value=_field(
            None, uncertainty="not found in supplied text"
        ),
        processes_personal_data=_field(
            "yes",
            'Data Processing Agreement ("DPA")',
            3,
        ),
    )


def _hallucinating_meta_extractor(_doc: ContractDoc) -> ContractMetadata:
    return _meta(
        supplier=_field("CloudSoft Technologies", FAKE_QUOTE, 1),
        payment_terms_days=_field("45", PAYMENT_QUOTE, 1),
    )


def _broken_meta_extractor(_doc: ContractDoc) -> ContractMetadata:
    raise ValidationError.from_exception_data(
        "ContractMetadata",
        [{"type": "missing", "loc": ("supplier",), "input": {}, "ctx": {}}],
    )


@pytest.mark.skipif(not CONTRACT_A.exists(), reason="fixtures missing")
def test_metadata_unknown_stays_unknown():
    result = run_clause_slice(
        ReviewRequest(pdf_path=str(CONTRACT_A), playbook_version="demo-slice-v1"),
        clause_reviewer=_minimal_clause_reviewer,
        metadata_extractor=_good_meta_extractor,
    )
    assert result.metadata is not None
    assert result.metadata.governing_law.value is None
    assert result.metadata.contract_value.value is None
    assert "governing_law" in result.metadata_unknown_fields
    assert "contract_value" in result.metadata_unknown_fields
    # Known fields retain values + evidence.
    assert result.metadata.supplier.value == "Brightline IT Services LLP"
    assert result.metadata.supplier.evidence is not None
    assert result.metadata.payment_terms_days.value == "45"


@pytest.mark.skipif(not CONTRACT_A.exists(), reason="fixtures missing")
def test_metadata_unsupported_evidence_stripped():
    result = run_clause_slice(
        ReviewRequest(pdf_path=str(CONTRACT_A), playbook_version="demo-slice-v1"),
        clause_reviewer=_minimal_clause_reviewer,
        metadata_extractor=_hallucinating_meta_extractor,
    )
    assert result.metadata is not None
    # Fabricated supplier quote must not survive grounding — value cleared too.
    assert result.metadata.supplier.evidence is None
    assert result.metadata.supplier.value is None
    assert result.metadata.supplier.uncertainty is not None
    assert "quote not found" in (result.metadata.supplier.uncertainty or "")
    assert "supplier" in result.metadata_unknown_fields
    # Payment quote that exists in the PDF keeps value + evidence.
    assert result.metadata.payment_terms_days.value == "45"
    assert result.metadata.payment_terms_days.evidence is not None


@pytest.mark.skipif(not CONTRACT_A.exists(), reason="fixtures missing")
def test_metadata_agent_failure_no_fabrication():
    result = run_clause_slice(
        ReviewRequest(pdf_path=str(CONTRACT_A), playbook_version="demo-slice-v1"),
        clause_reviewer=_minimal_clause_reviewer,
        metadata_extractor=_broken_meta_extractor,
    )
    assert result.status == "partial"
    assert result.metadata is not None
    assert all(getattr(result.metadata, n).value is None for n in ContractMetadata.model_fields)
    assert any("Metadata extractor failed" in w for w in result.warnings)
    assert any("no values were fabricated" in w for w in result.warnings)
    assert set(result.metadata_unknown_fields) == set(ContractMetadata.model_fields)


@pytest.mark.skipif(not CONTRACT_A.exists(), reason="fixtures missing")
def test_check_meta_unit_directly():
    doc = intake(
        ReviewRequest(pdf_path=str(CONTRACT_A), playbook_version="demo-slice-v1")
    )
    meta = _meta(supplier=_field("X", FAKE_QUOTE, 1))
    cleaned = check_meta(meta, doc)
    assert cleaned.supplier.evidence is None
    assert cleaned.supplier.value is None
    assert "quote not found" in (cleaned.supplier.uncertainty or "")
