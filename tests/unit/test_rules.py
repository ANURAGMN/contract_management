from app import rules
from app.schemas import ClauseFinding, Evidence


def _f(rule_id, value, **kw):
    return ClauseFinding(rule_id=rule_id, found=True, extracted_value=value, **kw)


# --- parsers ---------------------------------------------------------------- #
def test_parse_net_days():
    assert rules.parse_net_days("Net 15") == 15
    assert rules.parse_net_days("15 days from invoice") == 15
    assert rules.parse_net_days("net-30") == 30
    assert rules.parse_net_days(None) is None


def test_parse_percent_months_days():
    assert rules.parse_percent("8% per year") == 8.0
    assert rules.parse_months("24 months") == 24
    assert rules.parse_days("90 days") == 90


# --- checkers --------------------------------------------------------------- #
def test_payment_thresholds():
    assert rules.check_payment("Net 45", _f("PAYMENT-001", "Net 45"), None) == "preferred"
    assert rules.check_payment("Net 30", _f("PAYMENT-001", "Net 30"), None) == "fallback"
    assert rules.check_payment("Net 15", _f("PAYMENT-001", "Net 15"), None) == "walk_away"


def test_price_above_5pct_is_walk_away():
    assert rules.check_price("8% annual increase", _f("PRICE-001", "8%"), None) == "walk_away"
    assert rules.check_price("5% increase", _f("PRICE-001", "5%"), None) == "fallback"
    assert rules.check_price("fixed for the term", _f("PRICE-001", "fixed for the term"), None) == "preferred"
    assert rules.check_price("unilateral at sole discretion", _f("PRICE-001", "unilateral"), None) == "walk_away"


def test_indemnity_uncapped_walk_away():
    f = _f(
        "LIABILITY-002",
        "uncapped general indemnity",
        evidence=[Evidence(quote="The Supplier's obligations under this Clause 12 shall not be subject to any limitation of liability", page=4)],
    )
    assert rules.check_indemnity(f.extracted_value, f, None) == "walk_away"


def test_law_checker():
    assert rules.check_law("India, Bengaluru courts", _f("LAW-001", "x"), None) == "preferred"
    assert rules.check_law("mutually agreed neutral venue", _f("LAW-001", "x"), None) == "fallback"
    assert rules.check_law("courts of Singapore exclusive jurisdiction", _f("LAW-001", "x"), None) == "walk_away"


def test_assign_checker():
    assert rules.check_assign("freely assign without consent", _f("ASSIGN-001", "x"), None) == "walk_away"
    assert rules.check_assign("assignment to affiliates on notice", _f("ASSIGN-001", "x"), None) == "fallback"
    assert rules.check_assign("prior written consent required", _f("ASSIGN-001", "x"), None) == "preferred"


def test_liability_prior_12_months_is_preferred():
    assert (
        rules.check_liability_cap(
            "1x fees paid in the prior 12 months",
            _f("LIABILITY-001", "1x fees paid in the prior 12 months"),
            None,
        )
        == "preferred"
    )
    assert (
        rules.check_liability_cap(
            "1x annual fees",
            _f("LIABILITY-001", "1x annual fees"),
            None,
        )
        == "fallback"
    )


def test_sla_ticket_only_is_walk_away():
    assert (
        rules.check_sla(
            "95% of support tickets answered on time",
            _f("SLA-001", "95% of support tickets answered on time"),
            None,
        )
        == "walk_away"
    )


def test_sla_uptime_thresholds():
    assert (
        rules.check_sla(
            "99.9% platform availability with service credits",
            _f("SLA-001", "99.9% platform availability with service credits"),
            None,
        )
        == "preferred"
    )
    assert (
        rules.check_sla(
            "99.0% platform availability with credits",
            _f("SLA-001", "99.0% platform availability with credits"),
            None,
        )
        == "fallback"
    )


def test_renewal_90_day_notice_is_fallback():
    from app.schemas import ContractMetadata, Field_

    meta = ContractMetadata(
        supplier=Field_(),
        buyer=Field_(),
        contract_value=Field_(),
        effective_date=Field_(),
        expiry_date=Field_(),
        auto_renewal=Field_(value="yes"),
        renewal_term_months=Field_(value="12 months"),
        notice_period_days=Field_(value="90 days"),
        renewal_notice_basis=Field_(),
        payment_terms_days=Field_(),
        liability_cap=Field_(),
        governing_law=Field_(),
        sla_target_pct=Field_(),
        service_credit_rule=Field_(),
        processes_personal_data=Field_(),
    )
    assert (
        rules.check_renewal(
            "12-month renewal, 90 days' notice",
            _f("RENEWAL-001", "12-month renewal, 90 days' notice"),
            meta,
        )
        == "fallback"
    )
    meta60 = meta.model_copy(deep=True)
    meta60.notice_period_days = Field_(value="60 days")
    assert (
        rules.check_renewal("60 days notice", _f("RENEWAL-001", "60 days"), meta60)
        == "preferred"
    )


# --- credit arithmetic (the 8pt x 5% capped at 20% example) ---------------- #
def test_credit_claim_capped():
    # target 95, actual 87 -> 8 points x 5% = 40%, capped at 20% of 500000 = 100000
    assert rules.check_credit_claim(95, 87, 5, 20, 500000) == 100000.0


def test_credit_claim_none_when_met():
    assert rules.check_credit_claim(95, 96, 5, 20, 500000) is None
