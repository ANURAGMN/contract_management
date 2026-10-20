"""Milestone B — obligation register & date calculator (injected today)."""

from __future__ import annotations

from datetime import date

from app.demo_packages import build_contract_a_package, build_contract_d_package
from app.obligations import (
    add_months,
    build_register,
    compute_notice_deadline,
    confirm_obligation,
    merge_registers,
    parse_date,
    stable_obligation_id,
    term_end_from_anchor,
)
from app.schemas import ContractMetadata, Field_


def _empty_meta(**overrides) -> ContractMetadata:
    base = {n: Field_() for n in ContractMetadata.model_fields}
    base.update(overrides)
    return ContractMetadata(**base)


def test_parse_date():
    assert parse_date("2026-01-01") == date(2026, 1, 1)
    assert parse_date("not-a-date") is None


def test_add_months_leap_year():
    # 29 Feb 2028 + 12 months -> 28 Feb 2029
    assert add_months(date(2028, 2, 29), 12) == date(2029, 2, 28)


def test_term_end_from_anchor():
    assert term_end_from_anchor(date(2026, 1, 1), 12) == date(2026, 12, 31)


def test_notice_deadline_calendar_days():
    deadline, expl = compute_notice_deadline(date(2027, 12, 31), 60)
    assert deadline == date(2027, 11, 1)
    assert "60" in expl


def test_notice_deadline_business_days_unresolved():
    deadline, expl = compute_notice_deadline(
        date(2027, 12, 31), 60, business_days_wording=True
    )
    assert deadline is None
    assert "business" in expl.lower()


def test_notice_deadline_missing_term_end():
    deadline, expl = compute_notice_deadline(None, 60)
    assert deadline is None
    assert "not grounded" in expl.lower()


def test_stable_id_dedupes():
    a = stable_obligation_id("c1", "renewal_notice", "sixty days")
    b = stable_obligation_id("c1", "renewal_notice", "sixty days")
    c = stable_obligation_id("c1", "renewal_notice", "ninety days")
    assert a == b
    assert a != c


def test_register_from_sample_a():
    pkg = build_contract_a_package()
    assert pkg.metadata is not None
    today = date(2026, 10, 19)
    reg = build_register(pkg.metadata, contract_id="contract_a", today=today)
    assert any(a.anchor_type == "term_end" and a.value == date(2027, 12, 31) for a in reg.anchors)
    notice = next(o for o in reg.obligations if o.obligation_type == "renewal_notice")
    assert notice.due_date == date(2027, 11, 1)
    assert notice.needs_human_confirmation is False
    assert notice.trigger_date == date(2027, 12, 31)  # distinct from notice deadline
    assert any(o.obligation_type == "payment" for o in reg.obligations)
    assert any(o.obligation_type == "sla" for o in reg.obligations)
    assert any(o.obligation_type == "privacy" for o in reg.obligations)
    upcoming = reg.upcoming(within_days=400)
    assert notice in upcoming or notice.due_date is not None


def test_missing_expiry_needs_confirmation():
    meta = _empty_meta(
        effective_date=Field_(value=None),
        expiry_date=Field_(value=None),
        notice_period_days=Field_(value="60"),
    )
    reg = build_register(meta, contract_id="x", today=date(2026, 1, 1))
    notice = next(o for o in reg.obligations if o.obligation_type == "renewal_notice")
    assert notice.due_date is None
    assert notice.needs_human_confirmation is True


def test_merge_preserves_confirmed():
    pkg = build_contract_a_package()
    today = date(2026, 10, 19)
    first = build_register(pkg.metadata, contract_id="contract_a", today=today)
    oid = first.obligations[0].obligation_id
    first = confirm_obligation(first, oid, owner_user="anurag")
    assert first.by_id(oid).status == "confirmed"

    second = build_register(
        pkg.metadata, contract_id="contract_a", today=today, existing=first
    )
    assert second.by_id(oid).status == "confirmed"
    assert second.by_id(oid).owner_user == "anurag"


def test_sample_d_builds_register():
    pkg = build_contract_d_package()
    reg = build_register(pkg.metadata, contract_id="contract_d", today=date(2026, 6, 1))
    assert reg.obligations
