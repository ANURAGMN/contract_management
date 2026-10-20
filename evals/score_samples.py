"""Score offline demo packages against golden labels (no Gemini).

Writes evals/scorecard_samples.json for the Streamlit Scorecard tab.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.demo_packages import SAMPLE_CATALOG, load_sample  # noqa: E402
from evals.run_evals import score_run  # noqa: E402

GOLDEN = ROOT / "golden"
OUT = ROOT / "evals" / "scorecard_samples.json"


def main() -> int:
    rows = []
    hard_fails: list[str] = []
    for sid, meta in SAMPLE_CATALOG.items():
        golden = json.loads(
            (GOLDEN / meta["pdf"].replace(".pdf", ".json")).read_text(encoding="utf-8")
        )
        pkg = load_sample(sid)
        scored = score_run(pkg, golden)
        high = scored["high_recall"]
        overall = scored["overall_recall"]
        obl = scored["obligations"]
        sec = scored["security"]
        flags = scored["precision"]["flags"]
        prec = (
            scored["precision"]["correct"] / flags if flags else 1.0
        )
        g_num, g_den = scored["grounding"]
        grounding = (g_num / g_den) if g_den else 1.0
        rows.append(
            {
                "contract": meta["pdf"].replace(".pdf", ""),
                "sample_id": sid,
                "runs": 1,
                "high_severity_recall": {
                    "min": list(high),
                    "label": f"{high[0]}/{high[1]}",
                },
                "overall_recall": {
                    "min": list(overall),
                    "label": f"{overall[0]}/{overall[1]}",
                },
                "obligations": {
                    "min": list(obl),
                    "label": f"{obl[0]}/{obl[1]}",
                },
                "security": {
                    "min": list(sec),
                    "label": f"{sec[0]}/{sec[1]}",
                },
                "precision_mean": round(prec, 3),
                "precision_min": round(prec, 3),
                "grounding_mean": round(grounding, 3),
                "routing_ok_all_runs": scored["route"]["ok"],
                "route_got": [scored["route"]["got"]],
                "route_want": scored["route"]["want"],
                "status": [scored["status"]],
                "source": "offline_sample",
            }
        )
        if high[1] and high[0] < high[1]:
            hard_fails.append(f"{sid}: high-severity recall {high[0]}/{high[1]}")
        if not scored["route"]["ok"]:
            hard_fails.append(
                f"{sid}: route {scored['route']['got']} != {scored['route']['want']}"
            )

    payload = {
        "mode": "offline_samples_vs_golden",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs_per_contract": 1,
        "sample": "demo_packages vs golden/",
        "note": (
            "Scored from committed offline ReviewPackages — not a live Gemini run. "
            "evals/scorecard.json is the last live eval (may hard-fail on quota/model drift)."
        ),
        "contracts": rows,
        "passed": len(hard_fails) == 0,
        "hard_fails": hard_fails,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT} passed={payload['passed']}")
    for h in hard_fails:
        print("FAIL:", h)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
