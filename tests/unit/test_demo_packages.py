"""Offline sample packages + redline guard checks."""

from __future__ import annotations

from app.demo_packages import (
    BUILDERS,
    list_samples,
    load_sample,
    write_all_samples,
)
from app.schemas import Evidence, Redline, ReviewPackage
from app.steps import check_redline, load_playbook


def test_catalog_covers_all_five_fixtures():
    ids = {s["id"] for s in list_samples()}
    assert ids == {
        "contract_a_clean",
        "contract_b_legal",
        "contract_c_legal",
        "contract_d_legal",
        "contract_e_procurement",
    }


def test_build_and_roundtrip_samples(tmp_path, monkeypatch):
    from app import demo_packages

    monkeypatch.setattr(demo_packages, "DEMO_DIR", tmp_path)
    written = write_all_samples()
    assert len(written) == len(BUILDERS)
    for sid in BUILDERS:
        pkg = load_sample(sid)
        assert isinstance(pkg, ReviewPackage)
        assert pkg.is_sample is True
        assert pkg.quotation_comparison is not None
        assert len(pkg.quotation_comparison.rows) == 3


def test_sample_routes_match_golden():
    """Each offline sample must route to its golden expected_route."""
    import json
    from pathlib import Path

    from app.demo_packages import SAMPLE_CATALOG

    root = Path(__file__).resolve().parents[2]
    for sid, meta in SAMPLE_CATALOG.items():
        golden = json.loads(
            (root / "golden" / meta["pdf"].replace(".pdf", ".json")).read_text(
                encoding="utf-8"
            )
        )
        pkg = load_sample(sid)
        assert pkg.route.route == golden["expected_route"], sid


def test_sample_c_and_d_are_legal():
    for sid in ("contract_c_legal", "contract_d_legal"):
        pkg = BUILDERS[sid]()
        assert pkg.route.route == "legal"
        assert len(pkg.redlines) >= 1
        assert all(cr.passed for cr in pkg.redlines)


def test_sample_e_procurement_with_security_flag():
    pkg = BUILDERS["contract_e_procurement"]()
    assert pkg.route.route == "procurement_owner"
    assert pkg.security.any() is True
    assert any(o.kind == "credit_claim" for o in pkg.obligations)
    assert any(o.urgent for o in pkg.obligations)


def test_sample_a_is_low_risk():
    pkg = BUILDERS["contract_a_clean"]()
    assert pkg.route.route == "low_risk_queue"
    assert pkg.status == "complete"
    assert pkg.file_name == "contract_a.pdf"
    assert pkg.redlines == []
    assert any(o.kind == "sla_review" for o in pkg.obligations)
    assert pkg.metadata is not None
    assert pkg.metadata.supplier.value == "Brightline IT Services LLP"


def test_sample_b_is_legal_with_redlines():
    pkg = BUILDERS["contract_b_legal"]()
    assert pkg.route.route == "legal"
    assert any(f.grounded.finding.rule_id == "LIABILITY-002" for f in pkg.findings)
    assert len(pkg.redlines) >= 2
    assert all(cr.passed for cr in pkg.redlines)
    walk = [f for f in pkg.findings if f.status == "walk_away"]
    assert len(walk) >= 3


def test_check_redline_rejects_new_number():
    pb = load_playbook("demo-v1")
    rule = pb.rule("PRICE-001")
    assert rule is not None
    bad = Redline(
        rule_id="PRICE-001",
        original=Evidence(quote="Fees may increase annually.", page=1),
        proposed_text="Fees may increase by 12% each year.",
        rationale="test",
    )
    checked = check_redline(bad, rule.fallback_text)
    assert checked.passed is False
    assert any("new number" in p for p in checked.problems)
