"""Primary eval harness.

Runs each contract N times (default 3) through run_review(), scores the output
against the golden labels in code, and writes evals/scorecard.json and
evals/scorecard.md. Golden labels are never placed into model context.

Metrics (per the demo plan):
  - high-severity recall (hard fail if any high issue missed in any run)
  - overall recall, precision (decoys count against it)
  - citation grounding, page accuracy
  - metadata accuracy, routing accuracy
  - counts reported next to percentages; the minimum across N runs is reported.

Usage (from the repo root, inside the venv):
    python evals/run_evals.py --runs 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runner import record_review, run_review  # noqa: E402
from app.schemas import ReviewRequest  # noqa: E402

GOLDEN = ROOT / "golden"
CONTRACTS = ROOT / "data" / "contracts"
CSV = ROOT / "data" / "performance" / "supplier_performance.csv"
OUT = ROOT / "evals"


# --------------------------------------------------------------------------- #
# Text matching
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_WORD.findall((s or "").lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def coverage(span: str, text: str) -> float:
    """Share of the answer-key span's words that appear in the quote."""
    ts, tt = _tokens(span), _tokens(text)
    return len(ts & tt) / len(ts) if ts else 0.0


def overlaps(span: str, text: str, thresh: float = 0.5) -> bool:
    """Quote matches the answer-key clause: word Jaccard >= 0.5, or the quote contains
    >= 90% of the clause's words and is at most 3x its length (the model quoted the
    clause plus a neighbouring sentence — same clause, not a different one)."""
    if jaccard(span, text) >= thresh:
        return True
    ts, tt = _tokens(span), _tokens(text)
    return bool(ts) and coverage(span, text) >= 0.9 and len(tt) <= 3 * len(ts)


# --------------------------------------------------------------------------- #
# Per-run scoring
# --------------------------------------------------------------------------- #
def _finding_span(sf) -> str:
    return " ".join(e.quote for e in sf.grounded.finding.evidence)


def _best_overlap(golden_span: str, sf) -> bool:
    """A deviation matches on span if ANY single evidence quote overlaps the
    golden span (Jaccard >= 0.5). Matching per-quote avoids diluting the score
    when the model cites several sentences."""
    quotes = [e.quote for e in sf.grounded.finding.evidence] or [""]
    return any(overlaps(golden_span, q) for q in quotes)


def score_run(pkg, golden: dict) -> dict:
    issues = golden.get("issues", [])
    deviations = [i for i in issues if i["type"] == "deviation"]
    obligations = [i for i in issues if i["type"] == "obligation"]
    securities = [i for i in issues if i["type"] == "security"]
    decoys = golden.get("decoys", [])

    findings = pkg.findings

    # ---- deviations ---- #
    dev_matched = []
    for g in deviations:
        hit = any(
            sf.grounded.finding.rule_id == g["rule_id"]
            and sf.status == g["status"]
            and _best_overlap(g["span"], sf)
            for sf in findings
        )
        dev_matched.append((g, hit))

    high = [(g, h) for g, h in dev_matched if g["severity"] == "high"]
    high_recall = (sum(h for _, h in high), len(high))
    overall = deviations and (sum(h for _, h in dev_matched), len(dev_matched)) or (0, 0)

    # ---- obligations ---- #
    obl_hits = 0
    for g in obligations:
        for o in pkg.obligations:
            if o.kind != g["kind"]:
                continue
            due_ok = (g.get("due") is None) or (str(o.due) == str(g.get("due")))
            amt_ok = (g.get("amount_inr") is None and o.amount_inr is None) or (
                o.amount_inr is not None
                and g.get("amount_inr") is not None
                and abs(o.amount_inr - g["amount_inr"]) <= 1
            )
            if due_ok and amt_ok:
                obl_hits += 1
                break

    # ---- security ---- #
    sec_hits = 0
    flagged = [e.quote for e in pkg.security.pre] + [e.quote for e in pkg.security.model]
    for g in securities:
        if any(overlaps(g["span"], q) for q in flagged):
            sec_hits += 1

    # ---- precision (flags = walk_away / not_found findings) ---- #
    flags = [sf for sf in findings if sf.status in ("walk_away", "not_found")]
    correct, false_pos, decoy_hits = 0, 0, 0
    for sf in flags:
        span = _finding_span(sf)
        matched = any(
            sf.grounded.finding.rule_id == g["rule_id"] and _best_overlap(g["span"], sf)
            for g in deviations
        )
        if matched:
            correct += 1
        else:
            if any(overlaps(d["span"], span) for d in decoys):
                decoy_hits += 1
            false_pos += 1

    # ---- grounding ---- #
    # Skip absent_claimed / not_applicable — context for missing clauses, not
    # claims that a matching clause quote was located in the PDF.
    ev_total = sum(
        len(sf.grounded.finding.evidence)
        for sf in findings
        if sf.grounded.grounding not in ("absent_claimed", "not_applicable")
    )
    ev_grounded = sum(
        1
        for sf in findings
        if sf.grounded.grounding in ("exact", "page_corrected")
        for _ in sf.grounded.finding.evidence
    )
    grounding_breakdown = {}
    for sf in findings:
        grounding_breakdown[sf.grounded.grounding] = (
            grounding_breakdown.get(sf.grounded.grounding, 0) + 1
        )

    # ---- routing + metadata ---- #
    route_ok = pkg.route.route == golden["expected_route"]
    meta_total, meta_ok = 0, 0
    gmeta = golden.get("metadata", {})
    if pkg.metadata is not None:
        for field, expected in gmeta.items():
            meta_total += 1
            got = getattr(pkg.metadata, field, None)
            val = got.value if got is not None else None
            if val and _norm_meta(val) == _norm_meta(str(expected)):
                meta_ok += 1

    return {
        "high_recall": high_recall,
        "overall_recall": overall,
        "obligations": (obl_hits, len(obligations)),
        "security": (sec_hits, len(securities)),
        "precision": {"correct": correct, "flags": len(flags), "decoy_hits": decoy_hits},
        "grounding": (ev_grounded, ev_total),
        "grounding_breakdown": grounding_breakdown,
        "route": {"ok": route_ok, "got": pkg.route.route, "want": golden["expected_route"]},
        "metadata": (meta_ok, meta_total),
        "status": pkg.status,
    }


def _norm_meta(v: str) -> str:
    v = v.strip().lower()
    m = re.search(r"\d+", v)
    return m.group(0) if m and len(v) < 20 else re.sub(r"\s+", " ", v)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _min_frac(runs, key):
    """Return (min numerator across runs, denominator) for a (num, den) metric."""
    pairs = [r[key] for r in runs]
    den = pairs[0][1] if pairs else 0
    num = min((p[0] for p in pairs), default=0)
    return num, den


def aggregate(contract: str, runs: list[dict]) -> dict:
    hi = _min_frac(runs, "high_recall")
    ov = _min_frac(runs, "overall_recall")
    ob = _min_frac(runs, "obligations")
    se = _min_frac(runs, "security")
    gr = _min_frac(runs, "metadata")
    prec = [
        (r["precision"]["correct"] / r["precision"]["flags"]) if r["precision"]["flags"] else 1.0
        for r in runs
    ]
    grounding = [
        (r["grounding"][0] / r["grounding"][1]) if r["grounding"][1] else 1.0 for r in runs
    ]
    routes_ok = all(r["route"]["ok"] for r in runs)
    return {
        "contract": contract,
        "runs": len(runs),
        "high_severity_recall": {"min": hi, "label": f"{hi[0]}/{hi[1]}"},
        "overall_recall": {"min": ov, "label": f"{ov[0]}/{ov[1]}"},
        "obligations": {"min": ob, "label": f"{ob[0]}/{ob[1]}"},
        "security": {"min": se, "label": f"{se[0]}/{se[1]}"},
        "precision_mean": round(mean(prec), 3) if prec else None,
        "precision_min": round(min(prec), 3) if prec else None,
        "grounding_mean": round(mean(grounding), 3) if grounding else None,
        "metadata_accuracy": {"min": gr, "label": f"{gr[0]}/{gr[1]}"},
        "routing_ok_all_runs": routes_ok,
        "route_got": [r["route"]["got"] for r in runs],
        "route_want": runs[0]["route"]["want"],
        "status": [r["status"] for r in runs],
        "grounding_breakdown": [r["grounding_breakdown"] for r in runs],
    }


# --------------------------------------------------------------------------- #
# Hard-fail rules
# --------------------------------------------------------------------------- #
def hard_fails(results: list[dict]) -> list[str]:
    fails = []
    for r in results:
        hi_num, hi_den = r["high_severity_recall"]["min"]
        if hi_den and hi_num < hi_den:
            fails.append(f"{r['contract']}: high-severity issue missed ({r['high_severity_recall']['label']})")
        # Any contract that must escalate reaching low_risk_queue.
        if r["route_want"] != "low_risk_queue" and "low_risk_queue" in r["route_got"]:
            fails.append(f"{r['contract']}: fail-closed breach, reached low_risk_queue")
    return fails


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--contracts", nargs="*", help="e.g. contract_b (defaults to all)")
    args = ap.parse_args()

    golden_files = sorted(GOLDEN.glob("contract_*.json"))
    results = []
    for gf in golden_files:
        golden = json.loads(gf.read_text(encoding="utf-8"))
        stem = gf.stem
        if args.contracts and stem not in args.contracts:
            continue
        pdf = CONTRACTS / golden["contract"]
        if not pdf.exists():
            print(f"SKIP {stem}: {pdf} not found (run scripts/make_fixtures.py)")
            continue
        req = ReviewRequest(
            pdf_path=str(pdf),
            csv_path=str(CSV) if golden.get("supplier_id") else None,
            supplier_id=golden.get("supplier_id"),
            playbook_version="demo-v1",
        )
        runs = []
        for i in range(args.runs):
            print(f"  {stem} run {i + 1}/{args.runs} ...")
            pkg = run_review(req)
            record_review(pkg)  # keep every live run for Replay mode + live scorecard
            runs.append(score_run(pkg, golden))
        results.append(aggregate(stem, runs))

    if not results:
        print("No contracts scored. Generate fixtures first.")
        return 1

    fails = hard_fails(results)
    n_issues = sum(
        len(json.loads(p.read_text(encoding="utf-8")).get("issues", []))
        for p in GOLDEN.glob("contract_*.json")
    )
    n_decoys = sum(
        len(json.loads(p.read_text(encoding="utf-8")).get("decoys", []))
        for p in GOLDEN.glob("contract_*.json")
    )
    scorecard = {
        "runs_per_contract": args.runs,
        "sample": f"{n_issues} golden issues + {n_decoys} decoys",
        "contracts": results,
        "hard_fails": fails,
        "passed": not fails,
    }
    (OUT / "scorecard.json").write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    (OUT / "scorecard.md").write_text(_markdown(scorecard), encoding="utf-8")
    print("\n" + _markdown(scorecard))
    print(f"\nWrote {OUT / 'scorecard.json'} and scorecard.md")
    return 0 if not fails else 2


def _markdown(sc: dict) -> str:
    lines = [
        "# Scorecard",
        "",
        f"- Runs per contract: **{sc['runs_per_contract']}**  -  Sample: {sc['sample']}",
        f"- Result: {'PASS' if sc['passed'] else 'HARD FAIL'}",
        "",
        "| Contract | High recall (min) | Overall (min) | Obligations | Security | Precision (min) | Grounding (mean) | Route |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sc["contracts"]:
        route = "OK" if r["routing_ok_all_runs"] else f"MISS {r['route_got']} (want {r['route_want']})"
        lines.append(
            f"| {r['contract']} | {r['high_severity_recall']['label']} | "
            f"{r['overall_recall']['label']} | {r['obligations']['label']} | "
            f"{r['security']['label']} | {r['precision_min']} | {r['grounding_mean']} | {route} |"
        )
    if sc["hard_fails"]:
        lines += ["", "## Hard fails", *[f"- {f}" for f in sc["hard_fails"]]]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
