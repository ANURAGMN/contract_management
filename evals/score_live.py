"""Scorecard: recorded *live* Gemini reviews scored against the golden labels.

Takes the most recent live recording of each contract (made from the UI or by
``run_evals.py``), plus the RAG answer key, and computes hard gates, quality and
operational metrics (see ``evals/metrics.py``). No model calls.

    python evals/score_live.py            # live recordings  -> evals/scorecard_live.json
    python evals/score_live.py --fixtures # reference fixtures (self-check only)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import recordings  # noqa: E402
from app.demo_packages import list_samples, load_sample  # noqa: E402
from app.schemas import ReviewPackage  # noqa: E402
from evals.metrics import Item, evaluate, rag_counts  # noqa: E402

GOLDEN = ROOT / "golden"
OUT = ROOT / "evals" / "scorecard_live.json"


def _goldens() -> dict[str, dict]:
    return {gf.stem: json.loads(gf.read_text(encoding="utf-8")) for gf in sorted(GOLDEN.glob("contract_*.json"))}


def live_items(rescored: bool = False) -> list[Item]:
    """Latest live recording per contract. ``rescored``: same model output, current rules
    (see evals/rescore.py) — model calls, timings and quotes are unchanged."""
    from evals.rescore import rescore

    latest = recordings.latest_by_label("review")
    items = []
    for stem, golden in _goldens().items():
        rec = latest.get(golden["contract"])
        if not rec:
            continue
        pkg = ReviewPackage.model_validate(rec.payload)
        items.append(Item(
            contract=stem, pkg=rescore(pkg) if rescored else pkg, golden=golden,
            calls=rec.ai_calls, seconds=rec.seconds, recorded_at=rec.recorded_at, models=rec.models,
            recorded_route=pkg.route.route if rescored else None,
        ))
    return items


def fixture_items() -> list[Item]:
    by_file = {}
    for s in list_samples():
        pkg = load_sample(s["id"])
        by_file[pkg.file_name] = pkg
    return [Item(contract=stem, pkg=by_file[g["contract"]], golden=g)
            for stem, g in _goldens().items() if g["contract"] in by_file]


def build(kind: str = "live", *, rules: str = "current") -> dict:
    """kind: "live" | "fixtures". rules (live only): "current" re-applies today's rules to the
    recorded model output; "as_recorded" scores the package exactly as it was saved."""
    live = kind == "live"
    current = live and rules == "current"
    items = live_items(rescored=current) if live else fixture_items()
    expected = list(_goldens()) if live else sorted({it.contract for it in items})
    from evals import intake_eval, negotiation_eval

    neg = negotiation_eval.live_results() if live else negotiation_eval.scripted_results()
    result = evaluate(items, expected_contracts=expected, live=live, rag=rag_counts(), negotiation=neg,
                      intake=intake_eval.live_results() if live else None)
    result.update({
        "mode": "live_recordings_vs_golden" if live else "fixtures_vs_golden",
        "rules": ("current" if current else "as_recorded") if live else "fixture",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", action="store_true", help="score the hand-built fixtures instead")
    ap.add_argument("--as-recorded", action="store_true",
                    help="score live packages with the rules they were recorded with (default: current rules)")
    args = ap.parse_args()
    out = build("fixtures" if args.fixtures else "live", rules="as_recorded" if args.as_recorded else "current")
    path = OUT if not args.fixtures else ROOT / "evals" / "scorecard_fixtures.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"overall={out['overall']}  gates {out['gates_passed']}/{out['gates_applicable']}  "
          f"missing={out['missing']}")
    for m in out["metrics"]:
        v = m["value"]
        shown = "N/A" if v is None else (f"{v:.0%}" if m["unit"] == "%" else f"{v:.2f}{'' if m['unit'] in ('ratio', 'count') else ' s'}")
        frac = f" ({m['num']:g}/{m['den']:g})" if m["den"] else ""
        print(f"  [{m['group']:<11}] {m['name']:<36} {shown:>7}{frac:<10} {m['status']}")
    print(f"wrote {path}")
    return 0 if out["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
