"""CLI smoke test for page-aware PDF intake (Phase 1).

Usage (from repo root, venv active):
    python scripts/smoke_intake.py data/contracts/contract_a.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.errors import IntakeError  # noqa: E402
from app.schemas import ReviewRequest  # noqa: E402
from app.steps import intake  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test PDF intake")
    parser.add_argument("pdf", type=Path, help="Path to a supplier contract PDF")
    parser.add_argument(
        "--playbook",
        default="demo-v1",
        help="Playbook version id (default: demo-v1)",
    )
    args = parser.parse_args()

    try:
        doc = intake(
            ReviewRequest(pdf_path=str(args.pdf), playbook_version=args.playbook)
        )
    except IntakeError as exc:
        print(json.dumps({"ok": False, "error": str(exc), "code": exc.code}, indent=2))
        return 1

    summary = {
        "ok": True,
        "run_id": doc.run_id,
        "file_name": doc.file_name,
        "page_count": len(doc.pages),
        "extraction_status": doc.preflight.extraction_status,
        "unreliable_extraction": doc.preflight.unreliable_extraction,
        "low_text_pages": doc.preflight.low_text_pages,
        "warnings": doc.preflight.warnings,
        "injection_hit_count": len(doc.preflight.injection_hits),
        "pages": [
            {
                "number": p.number,
                "character_count": p.character_count,
                "preview": p.text[:120].replace("\n", " "),
            }
            for p in doc.pages
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0 if doc.preflight.extraction_status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
