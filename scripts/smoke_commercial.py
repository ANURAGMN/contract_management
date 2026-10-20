"""Phase 4 smoke: deterministic quotation comparison (no API key).

Usage:
    python scripts/smoke_commercial.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.commercial import compare_demo_quotations  # noqa: E402


def main() -> int:
    cmp = compare_demo_quotations(include_baseline=True)
    print(json.dumps(cmp.model_dump(mode="json"), indent=2))
    print(
        f"\n# summary: requisition={cmp.requisition_id} "
        f"budget_inr={cmp.budget_inr:,.0f} suppliers={len(cmp.rows)}",
        file=sys.stderr,
    )
    for r in cmp.rows:
        flag = " LOWEST" if r.is_lowest_annual_fee else ""
        sla = "SLA_OK" if r.sla_compliant else "SLA_FAIL"
        print(
            f"# {r.supplier_name}: annual={r.annual_recurring_fee} "
            f"var={r.budget_variance} {sla} term={r.term_months} "
            f"scope={r.scope_comparability}{flag}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
