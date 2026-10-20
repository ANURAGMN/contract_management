"""Regression: offline samples + negotiation must match golden labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.demo_packages import SAMPLE_CATALOG, load_sample
from app.negotiation import load_scenario
from app.negotiation_agents import run_scripted_live_demo
from app.negotiation_runs import run_full
from evals.run_evals import score_run

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "golden"


def _golden_for_sample(pdf_name: str) -> dict:
    path = GOLDEN / pdf_name.replace(".pdf", ".json")
    assert path.exists(), f"missing golden {path.name}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_golden_files_cover_all_five_contracts():
    expected = {f"contract_{x}.json" for x in "abcde"}
    found = {p.name for p in GOLDEN.glob("contract_*.json")}
    assert expected <= found


@pytest.mark.parametrize("sid,meta", list(SAMPLE_CATALOG.items()))
def test_offline_sample_matches_golden(sid, meta):
    golden = _golden_for_sample(meta["pdf"])
    pkg = load_sample(sid)
    scored = score_run(pkg, golden)

    assert scored["route"]["ok"], (
        f"{sid}: route {scored['route']['got']} != {scored['route']['want']}"
    )

    hi_num, hi_den = scored["high_recall"]
    if hi_den:
        assert hi_num == hi_den, f"{sid}: high-severity recall {hi_num}/{hi_den}"

    ov_num, ov_den = scored["overall_recall"]
    if ov_den:
        assert ov_num == ov_den, f"{sid}: overall recall {ov_num}/{ov_den}"

    ob_num, ob_den = scored["obligations"]
    if ob_den:
        assert ob_num == ob_den, f"{sid}: obligations {ob_num}/{ob_den}"

    se_num, se_den = scored["security"]
    if se_den:
        assert se_num == se_den, f"{sid}: security {se_num}/{se_den}"

    flags = scored["precision"]["flags"]
    if flags:
        prec = scored["precision"]["correct"] / flags
        assert prec >= 0.75, f"{sid}: precision {prec:.2f} below 0.75"

    g_num, g_den = scored["grounding"]
    if g_den:
        assert g_num / g_den >= 0.95, f"{sid}: grounding {g_num}/{g_den}"


def test_negotiation_golden_scripted_reaches_aip():
    neg = json.loads((GOLDEN / "negotiation_cloudsoft.json").read_text(encoding="utf-8"))
    sc = load_scenario()
    assert sc.scenario_id == neg["scenario_id"]
    assert sc.policy.budget_ceiling_inr == neg["budget_ceiling_inr"]
    state = run_scripted_live_demo(sc, max_cycles=2)
    assert state.status == neg["expected_scripted_terminal"]
    if neg.get("liability_never_auto_agreed"):
        assert "liability" not in state.agreed_terms


def test_negotiation_run_full_scripted_matches_golden():
    neg = json.loads((GOLDEN / "negotiation_cloudsoft.json").read_text(encoding="utf-8"))
    run = run_full(load_scenario(), "scripted", max_cycles=2)
    assert run.state.status == neg["expected_scripted_terminal"]
    assert len(run.state.thread) >= 2
    assert all(mid in run.origins for mid in (m.message_id for m in run.state.thread))
