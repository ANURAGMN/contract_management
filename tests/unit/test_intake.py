"""Phase 1 intake tests — no live LLM required."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from app.errors import IntakeError
from app.schemas import ReviewRequest
from app.steps import (
    MAX_UPLOAD_BYTES,
    extract_pdf_bytes,
    intake,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "data" / "contracts"


def _text_pdf_bytes(pages: list[str]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf)
    for body in pages:
        y = 800
        remaining = body
        while remaining and y > 40:
            c.drawString(40, y, remaining[:95])
            remaining = remaining[95:]
            y -= 12
        c.showPage()
    c.save()
    return buf.getvalue()


def _blank_pdf_bytes(n_pages: int = 1) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf)
    for _ in range(n_pages):
        c.showPage()
    c.save()
    return buf.getvalue()


def test_intake_rejects_non_pdf(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(IntakeError) as exc:
        intake(ReviewRequest(pdf_path=str(path)))
    assert exc.value.code == "unsupported_type"


def test_intake_rejects_empty_file(tmp_path: Path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    with pytest.raises(IntakeError) as exc:
        intake(ReviewRequest(pdf_path=str(path)))
    assert exc.value.code == "empty_file"


def test_intake_rejects_missing_file(tmp_path: Path):
    with pytest.raises(IntakeError) as exc:
        intake(ReviewRequest(pdf_path=str(tmp_path / "missing.pdf")))
    assert exc.value.code == "not_found"


def test_intake_rejects_oversized(tmp_path: Path):
    path = tmp_path / "huge.pdf"
    # Cheap size check: write just over the limit without a valid PDF body.
    # Validation rejects on size before parse.
    path.write_bytes(b"%PDF-1.4\n" + b"x" * (MAX_UPLOAD_BYTES))
    with pytest.raises(IntakeError) as exc:
        intake(ReviewRequest(pdf_path=str(path)))
    assert exc.value.code == "too_large"


def test_extract_text_pdf_ok():
    body = ("ABC Technologies Master Services Agreement. " * 40).strip()
    data = _text_pdf_bytes([body, body])
    doc = extract_pdf_bytes(data, filename="sample.pdf")
    assert doc.preflight.extraction_status == "ok"
    assert not doc.preflight.unreliable_extraction
    assert len(doc.pages) == 2
    assert doc.pages[0].number == 1
    assert doc.pages[0].text.startswith("[PAGE 1]")
    assert doc.pages[0].character_count > 0
    assert doc.pages[0].character_count == len(
        doc.pages[0].text.replace("[PAGE 1]\n", "", 1).strip()
    )


def test_extract_blank_pages_need_ocr():
    data = _blank_pdf_bytes(2)
    doc = extract_pdf_bytes(data, filename="blank.pdf")
    assert doc.preflight.extraction_status in ("empty", "needs_ocr")
    assert doc.preflight.unreliable_extraction
    assert doc.preflight.warnings
    assert all(p.character_count == 0 for p in doc.pages)


def test_extract_corrupt_pdf_classified():
    doc = extract_pdf_bytes(b"%PDF-1.4\nnot-a-real-pdf%%%%", filename="corrupt.pdf")
    assert doc.preflight.extraction_status == "error"
    assert doc.preflight.unreliable_extraction
    assert doc.pages == []
    assert any("Corrupt" in w or "unreadable" in w.lower() for w in doc.preflight.warnings)


def test_extract_bytes_rejects_wrong_extension():
    with pytest.raises(IntakeError) as exc:
        extract_pdf_bytes(b"%PDF", filename="x.docx")
    assert exc.value.code == "unsupported_type"


@pytest.mark.skipif(
    not (CONTRACTS / "contract_a.pdf").exists(),
    reason="Run scripts/make_fixtures.py first",
)
def test_intake_contract_a_fixture():
    doc = intake(
        ReviewRequest(
            pdf_path=str(CONTRACTS / "contract_a.pdf"),
            playbook_version="demo-v1",
        )
    )
    assert doc.pages
    assert doc.pages[0].text.startswith("[PAGE 1]")
    assert doc.pages[0].character_count > 0
    assert doc.preflight.extraction_status == "ok"
    assert not doc.preflight.unreliable_extraction


@pytest.mark.skipif(
    not (CONTRACTS / "contract_e.pdf").exists(),
    reason="Run scripts/make_fixtures.py first",
)
def test_intake_flags_injection_on_contract_e():
    doc = intake(
        ReviewRequest(
            pdf_path=str(CONTRACTS / "contract_e.pdf"),
            playbook_version="demo-v1",
        )
    )
    assert doc.preflight.injection_hits
    assert any("AI reviewers" in h.quote for h in doc.preflight.injection_hits)
