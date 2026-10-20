"""Milestone B (spec §20) — obligation register & deterministic date math.

All tests inject ``today`` and never touch the wall clock.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.obligations import (
    ContractObligation,
    ObligationRegister,
    add_months,
    build_register,
    compute_notice_deadline,
    confirm_obligation,
    load_register,
    merge_registers,
    save_register,
    stable_obligation_id,
    term_end_from_anchor,
)
from app.schemas import ContractMetadata, Evidence, Field_

TODAY = date(2026, 10, 19)


def _f(value=None, quote=None, page=1):
    return Field_(value=value, evidence=Evidence(quote=quote, page=page) if quote else None)


def _meta(**fields) -> ContractMetadata:
    base = {name: Field_() for name in ContractMetadata.model_fields}
    base.update(fields)
    return ContractMetadata(**base)


# --------------------------------------------------------------------------- #
# Date calculator
# --------------------------------------------------------------------------- #
def test_add_months_clamps_leap_day():
    assert add_months(date(2028, 2, 29), 12) == date(2029, 2, 28)
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2026, 12, 15), 1) == date(2027, 1, 15)


def test_add_months_negative():
    assert add_months(date(2026, 3, 31), -1) == date(2026, 2, 28)


def test_term_end_from_anchor():
    assert term_end_from_anchor(date(2026, 1, 28), 12) == date(2027, 1, 27)
    assert term_end_from_anchor(None, 12) is None
    assert term_end_from_anchor(date(2026, 1, 1), None) is None


def test_notice_deadline_before_term_end_not_equal_to_expiry():
    term_end = date(2027, 1, 27)
    deadline, explanation = compute_notice_deadline(term_end, 90)
    assert deadline == date(2026, 10, 29)
    assert deadline != term_end  # the key distinction the spec demands
    assert "90 calendar days" in explanation


def test_notice_deadline_leap_year_window():
    # 60 days before 2028-02-29 crosses the leap day deterministically.
    deadline, _ = compute_notice_deadline(date(2028, 2, 29), 60)
    assert deadline == date(2027, 12, 31)


def test_notice_deadline_business_days_is_ambiguous():
    deadline, explanation = compute_notice_deadline(date(2027, 1, 27), 90, business_days_wording=True)
    assert deadline is None
    assert "business" in explanation.lower()


def test_notice_deadline_missing_anchor():
    assert compute_notice_deadline(None, 90)[0] is None
    assert compute_notice_deadline(date(2027, 1, 1), None)[0] is None


# --------------------------------------------------------------------------- #
# Register building
# --------------------------------------------------------------------------- #
def test_register_computes_grounded_notice_deadline():
    meta = _meta(
        effective_date=_f("2026-01-28", "28 January 2026"),
        expiry_date=_f("2027-01-27", "until 27 January 2027"),
        notice_period_days=_f("90", "ninety (90) days' written notice"),
        renewal_notice_basis=_f(
            "before the end of the then-current term",
            "before the end of the then-current term",
        ),
    )
    reg = build_register(meta, contract_id="c1", today=TODAY)
    renewal = next(o for o in reg.obligations if o.obligation_type == "renewal_notice")
    assert renewal.due_date == date(2026, 10, 29)
    assert renewal.needs_human_confirmation is False
    assert renewal.trigger_date == date(2027, 1, 27)
    # critical because the deadline is within 90 days of today
    assert renewal.criticality == "critical"


def test_register_flags_business_day_wording_for_confirmation():
    meta = _meta(
        expiry_date=_f("2027-01-27", "27 January 2027"),
        notice_period_days=_f("90", "90 business days"),
        renewal_notice_basis=_f("90 business days before the end", "90 business days before the end"),
    )
    reg = build_register(meta, contract_id="c2", today=TODAY)
    renewal = next(o for o in reg.obligations if o.obligation_type == "renewal_notice")
    assert renewal.due_date is None
    assert renewal.needs_human_confirmation is True


def test_register_missing_effective_date_leaves_notice_unresolved():
    meta = _meta(
        notice_period_days=_f("60", "sixty (60) days' notice"),
        renewal_notice_basis=_f("before the end of the then-current term", "x"),
    )
    reg = build_register(meta, contract_id="c3", today=TODAY)
    renewal = next(o for o in reg.obligations if o.obligation_type == "renewal_notice")
    assert renewal.due_date is None
    assert renewal.needs_human_confirmation is True


def test_register_derives_term_end_from_effective_plus_term():
    meta = _meta(
        effective_date=_f("2026-01-28", "28 January 2026"),
        renewal_term_months=_f("12", "twelve (12) month terms"),
        notice_period_days=_f("90", "ninety (90) days"),
        renewal_notice_basis=_f("before the end of the then-current term", "x"),
    )
    reg = build_register(meta, contract_id="c4", today=TODAY)
    term_end = next(a for a in reg.anchors if a.anchor_type == "term_end")
    assert term_end.value == date(2027, 1, 27)
    renewal = next(o for o in reg.obligations if o.obligation_type == "renewal_notice")
    assert renewal.due_date == date(2026, 10, 29)


def test_register_recurring_payment_and_sla():
    meta = _meta(
        payment_terms_days=_f("45", "within forty-five (45) days (Net 45)"),
        sla_target_pct=_f("99.0", "99.0% availability"),
    )
    reg = build_register(meta, contract_id="c5", today=TODAY)
    pay = next(o for o in reg.obligations if o.obligation_type == "payment")
    sla = next(o for o in reg.obligations if o.obligation_type == "sla")
    assert pay.recurrence == "per_invoice"
    assert pay.notice_period_days == 45
    assert sla.recurrence == "monthly"


def test_register_privacy_only_when_personal_data():
    with_pd = build_register(
        _meta(processes_personal_data=_f("yes", "DPA executed")), contract_id="c6", today=TODAY
    )
    without_pd = build_register(
        _meta(processes_personal_data=_f("no", "no personal data")), contract_id="c7", today=TODAY
    )
    assert any(o.obligation_type == "privacy" for o in with_pd.obligations)
    assert not any(o.obligation_type == "privacy" for o in without_pd.obligations)


def test_overdue_marking():
    meta = _meta(
        expiry_date=_f("2026-03-01", "1 March 2026"),
        notice_period_days=_f("30", "30 days"),
        renewal_notice_basis=_f("before the end of the then-current term", "x"),
    )
    reg = build_register(meta, contract_id="c8", today=TODAY)  # deadline 2026-01-30 < today
    renewal = next(o for o in reg.obligations if o.obligation_type == "renewal_notice")
    assert renewal.due_date == date(2026, 1, 30)
    assert renewal.status == "overdue"


def test_reminders_precede_notice_deadline():
    meta = _meta(
        expiry_date=_f("2027-01-27", "27 January 2027"),
        notice_period_days=_f("90", "90 days"),
        renewal_notice_basis=_f("before the end of the then-current term", "x"),
    )
    reg = build_register(meta, contract_id="c9", today=TODAY)
    renewal = next(o for o in reg.obligations if o.obligation_type == "renewal_notice")
    dates = sorted(r.reminder_date for r in reg.reminders)
    assert dates == [date(2026, 7, 31), date(2026, 8, 30), date(2026, 9, 29)]
    assert all(d < renewal.due_date for d in dates)


# --------------------------------------------------------------------------- #
# Stable identity, dedup and confirmation
# --------------------------------------------------------------------------- #
def test_stable_id_is_deterministic():
    a = stable_obligation_id("c", "payment", "Net 45")
    b = stable_obligation_id("c", "payment", "Net 45")
    assert a == b
    assert stable_obligation_id("c", "payment", "Net 30") != a


def test_reprocessing_does_not_duplicate():
    meta = _meta(
        payment_terms_days=_f("45", "Net 45"),
        sla_target_pct=_f("99.0", "99.0% availability"),
    )
    first = build_register(meta, contract_id="c10", today=TODAY)
    second = build_register(meta, contract_id="c10", today=TODAY, existing=first)
    ids_first = {o.obligation_id for o in first.obligations}
    ids_second = {o.obligation_id for o in second.obligations}
    assert ids_first == ids_second
    assert len(second.obligations) == len(first.obligations)


def test_confirmation_is_preserved_across_reruns():
    meta = _meta(payment_terms_days=_f("45", "Net 45"))
    reg = build_register(meta, contract_id="c11", today=TODAY)
    pay_id = next(o.obligation_id for o in reg.obligations if o.obligation_type == "payment")
    reg = confirm_obligation(reg, pay_id, owner_user="asha", due_date=date(2026, 12, 1))
    rerun = build_register(meta, contract_id="c11", today=TODAY, existing=reg)
    pay = rerun.by_id(pay_id)
    assert pay is not None
    assert pay.status == "confirmed"
    assert pay.owner_user == "asha"
    assert pay.due_date == date(2026, 12, 1)


def test_vanished_obligation_retired_not_dropped():
    meta_full = _meta(payment_terms_days=_f("45", "Net 45"), sla_target_pct=_f("99.0", "99%"))
    meta_reduced = _meta(payment_terms_days=_f("45", "Net 45"))
    first = build_register(meta_full, contract_id="c12", today=TODAY)
    sla_id = next(o.obligation_id for o in first.obligations if o.obligation_type == "sla")
    second = build_register(meta_reduced, contract_id="c12", today=TODAY, existing=first)
    sla = second.by_id(sla_id)
    assert sla is not None  # preserved for audit
    assert sla.status == "not_applicable"


def test_audit_log_records_events():
    meta = _meta(payment_terms_days=_f("45", "Net 45"))
    reg = build_register(meta, contract_id="c13", today=TODAY)
    assert any(e["event"] == "created" for e in reg.audit)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_persistence_roundtrip(tmp_path):
    meta = _meta(
        expiry_date=_f("2027-01-27", "27 January 2027"),
        notice_period_days=_f("90", "90 days"),
        renewal_notice_basis=_f("before the end of the then-current term", "x"),
        payment_terms_days=_f("45", "Net 45"),
    )
    reg = build_register(meta, contract_id="c14", today=TODAY)
    path = tmp_path / "register.json"
    save_register(reg, path)
    loaded = load_register(path)
    assert loaded is not None
    assert loaded.contract_id == "c14"
    assert {o.obligation_id for o in loaded.obligations} == {
        o.obligation_id for o in reg.obligations
    }
    assert load_register(tmp_path / "missing.json") is None


def test_upcoming_window():
    meta = _meta(
        expiry_date=_f("2027-01-27", "27 January 2027"),
        notice_period_days=_f("90", "90 days"),
        renewal_notice_basis=_f("before the end of the then-current term", "x"),
    )
    reg = build_register(meta, contract_id="c15", today=TODAY)
    assert [o.obligation_type for o in reg.upcoming(90)] == ["renewal_notice"]
    assert reg.upcoming(5) == []
