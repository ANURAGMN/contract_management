"""Phase 4 commercial comparison — pure Python, no LLM."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.commercial import (
    DEFAULT_BUDGET_INR,
    compare_demo_quotations,
    compare_quotations,
    load_demo_quotations,
)
from app.schemas import Quotation


def _q(**kwargs) -> Quotation:
    base = dict(
        supplier_name="X",
        currency="INR",
        source_filename="x.json",
    )
    base.update(kwargs)
    return Quotation(**base)


def test_demo_fixtures_load():
    quotes = load_demo_quotations()
    assert len(quotes) == 3
    names = {q.supplier_name for q in quotes}
    assert names == {"CloudSoft", "TechNova", "DataSphere"}


def test_demo_comparison_hand_checked():
    """Hand-checked against PR-2026-1042 numbers (absolute INR)."""
    cmp = compare_demo_quotations(include_baseline=True)
    assert cmp.budget_inr == 4800000
    by = {r.supplier_name: r for r in cmp.rows}

    # CloudSoft: 50L, over budget by 2L, SLA ok, 12 months
    cs = by["CloudSoft"]
    assert cs.annual_recurring_fee == 5000000
    assert cs.budget_variance == 200000
    assert cs.budget_variance_pct == pytest.approx(4.17, abs=0.01)
    assert cs.cost_per_user == 10000.0  # 5_000_000 / 500
    assert cs.sla_compliant is True
    assert cs.term_mismatch is False
    assert cs.total_cost_over_horizon == 5150000.0  # 5M + 150k one-time
    assert cs.scope_comparability == "comparable"

    # TechNova: cheapest annual, but SLA fail
    tn = by["TechNova"]
    assert tn.annual_recurring_fee == 4400000
    assert tn.budget_variance == -400000
    assert tn.sla_compliant is False
    assert tn.is_lowest_annual_fee is True
    assert any("below required" in w for w in tn.warnings)

    # DataSphere: 24-month term mismatch vs 12-month horizon
    ds = by["DataSphere"]
    assert ds.term_months == 24
    assert ds.term_mismatch is True
    assert ds.sla_compliant is True
    # Total for 12-month horizon: 4.7M * 1 + 200k
    assert ds.total_cost_over_horizon == 4900000.0

    assert by["TechNova"].is_lowest_annual_fee is True
    assert by["CloudSoft"].is_lowest_annual_fee is False


def test_missing_annual_fee_not_comparable():
    q = _q(user_count=500, term_months=12, uptime_percent=99.9)
    cmp = compare_quotations([q], budget_inr=DEFAULT_BUDGET_INR)
    row = cmp.rows[0]
    assert row.scope_comparability == "not_comparable"
    assert row.total_cost_over_horizon is None
    assert row.cost_per_user is None
    assert "annual_recurring_fee" in row.unresolved_fields
    assert row.not_comparable_reason


def test_missing_users_skips_cost_per_user_but_keeps_annual():
    q = _q(
        annual_recurring_fee=4800000,
        term_months=12,
        uptime_percent=99.9,
        taxes_included=True,
        support_scope="full",
        payment_terms="Net 30",
    )
    cmp = compare_quotations([q])
    row = cmp.rows[0]
    assert row.annual_recurring_fee == 4800000
    assert row.cost_per_user is None
    assert "user_count" in row.unresolved_fields
    assert row.scope_comparability == "not_comparable"


def test_budget_variance_decimal_safe():
    q = _q(
        annual_recurring_fee=4800000,
        one_time_fee=0,
        user_count=500,
        term_months=12,
        uptime_percent=99.9,
        taxes_included=True,
        support_scope="full",
        payment_terms="Net 30",
    )
    cmp = compare_quotations([q], budget_inr=Decimal("4800000"))
    assert cmp.rows[0].budget_variance == 0.0


def test_term_mismatch_flagged():
    q = _q(
        annual_recurring_fee=4700000,
        one_time_fee=0,
        user_count=500,
        term_months=24,
        uptime_percent=99.9,
        taxes_included=True,
        support_scope="full",
        payment_terms="Net 30",
    )
    cmp = compare_quotations([q], comparison_horizon_months=12)
    assert cmp.rows[0].term_mismatch is True
    assert any("24 months" in w for w in cmp.rows[0].warnings)


def test_does_not_declare_winner():
    cmp = compare_demo_quotations(include_baseline=False)
    # Schema has is_lowest_annual_fee only — no "recommended" / "best" field.
    assert all(not hasattr(r, "recommended") for r in cmp.rows)
    assert sum(1 for r in cmp.rows if r.is_lowest_annual_fee) >= 1
