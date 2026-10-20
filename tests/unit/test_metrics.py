"""Evaluation metrics: hard gates, N/A handling, decoys, routing and RAG."""

from __future__ import annotations

import json
from pathlib import Path

from app.demo_packages import load_sample
from evals.metrics import Item, Metric, count_contract, evaluate, rag_counts

GOLDEN = Path(__file__).resolve().parents[2] / "golden"


def _golden(stem: str) -> dict:
    return json.loads((GOLDEN / f"{stem}.json").read_text(encoding="utf-8"))


def _by_key(result: dict) -> dict:
    return {m["key"]: m for m in result["metrics"]}


def test_metric_with_no_cases_is_na_not_zero_or_pass():
    m = Metric("x", "X", "detection", "gate", "how", 1.0)
    assert m.result() is None and m.status() == "na"


def test_lower_is_better_metric():
    m = Metric("fp", "FP", "detection", "quality", "how", 0.10, "<=", num=1, den=4)
    assert m.status() == "below_target"
    m.num = 0
    assert m.status() == "pass"


def test_fixtures_pass_all_gates():
    items = [Item(c, load_sample(s), _golden(c)) for c, s in [
        ("contract_b", "contract_b_legal"), ("contract_e", "contract_e_procurement")]]
    res = evaluate(items, expected_contracts=["contract_b", "contract_e"], live=False, rag=None)
    m = _by_key(res)
    assert res["overall"] == "pass"
    assert m["critical_recall"]["num"] == m["critical_recall"]["den"] == 2
    assert m["decoy_fp"]["value"] == 0.0  # contract_b decoy (uncapped IP indemnity) not flagged
    assert m["schema_validity"]["status"] == "na"  # fixtures make no model calls


def test_wrong_route_fails_a_hard_gate_even_with_perfect_recall():
    pkg = load_sample("contract_e_procurement").model_copy(deep=True)
    pkg.route.route = "legal"
    res = evaluate([Item("contract_e", pkg, _golden("contract_e"))], expected_contracts=["contract_e"],
                   live=False, rag=None)
    assert res["overall"] == "fail"
    assert "Route accuracy" in res["failed_gates"]
    assert _by_key(res)["overall_recall"]["status"] == "pass"


def test_missed_critical_issue_fails():
    pkg = load_sample("contract_b_legal").model_copy(deep=True)
    pkg.findings = [f for f in pkg.findings if f.grounded.finding.rule_id != "LIABILITY-002"]
    c = count_contract(pkg, _golden("contract_b"))
    assert c["critical_recall"] == (1, 2)


def test_flagging_a_decoy_counts_as_false_positive():
    pkg = load_sample("contract_b_legal").model_copy(deep=True)
    decoy = _golden("contract_b")["decoys"][0]
    target = next(f for f in pkg.findings if f.grounded.finding.rule_id == "LIABILITY-002")
    target.grounded.finding.evidence[0].quote = decoy["span"]
    c = count_contract(pkg, _golden("contract_b"))
    assert c["decoy_fp"] == (1, 1)


def test_missing_contracts_make_result_incomplete():
    items = [Item("contract_b", load_sample("contract_b_legal"), _golden("contract_b"))]
    res = evaluate(items, expected_contracts=["contract_a", "contract_b"], live=False, rag=None)
    assert res["overall"] == "incomplete" and res["missing"] == ["contract_a"]


def test_rag_metrics_use_hand_labels():
    r = rag_counts()
    assert r["in_scope"] >= 8
    assert 0 <= r["recall"] <= 1 and 0 <= r["ndcg"] <= 1
    assert r["version"][0] == r["version"][1]  # never retrieves retired/draft policy normatively
