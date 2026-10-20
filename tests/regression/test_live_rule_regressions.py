"""Regression: rule outcomes on values Gemini actually returned in live runs.

Each case is a (rule, extracted value, verified quote) taken from a recorded live
review in data/recordings/review/. Before the fix these came back as walk_away or
unclear and sent clean contracts to manual review or Legal. The rule engine is
deterministic, so these lock in the fix without any model call.
"""

from __future__ import annotations

import pytest

from app import rules, steps
from app.schemas import (
    ClauseFinding,
    ContractMetadata,
    Evidence,
    Field_,
    GroundedFinding,
    GroundedResult,
    Preflight,
    RouteInput,
    RulesInput,
    SecurityFlags,
)

PB = steps.load_playbook("demo-v1")
RULE = {r.id: r for r in PB.rules}


def _meta(**kw) -> ContractMetadata:
    m = {k: Field_() for k in ContractMetadata.model_fields}
    for k, v in kw.items():
        m[k] = Field_(value=v)
    return ContractMetadata(**m)


def _finding(rule_id: str, value: str, quote: str, page: int = 2) -> ClauseFinding:
    return ClauseFinding(rule_id=rule_id, found=True, extracted_value=value,
                         evidence=[Evidence(quote=quote, page=page)])


CASES = [
    # id, rule, extracted value, quote, expected
    ("A carve-out indemnity", "LIABILITY-002",
     "subject to the limitation of liability in Clause 5, except that indemnities for IP infringement and data breach remain uncapped",
     "The Supplier's indemnification obligations are subject to the limitation of liability in Clause 5, except that "
     "indemnities for intellectual property infringement and for data breach shall remain uncapped.", "preferred"),
    ("E data-breach carve-out", "LIABILITY-002",
     "Subject to the cap in Clause 6, except that indemnities for data breach shall remain uncapped",
     "The Supplier shall indemnify the Buyer for third-party claims arising from the Services, subject to the cap in "
     "Clause 6, except that indemnities for data breach shall remain uncapped.", "preferred"),
    ("D IP indemnity within cap", "LIABILITY-002", "subject to the limitation of liability in Clause 6",
     "The Supplier shall indemnify the Buyer against third-party intellectual-property claims, subject to the "
     "limitation of liability in Clause 6.", "preferred"),
    ("B general indemnity outside cap", "LIABILITY-002", "uncapped",
     "The Supplier's obligations under this Clause 12 shall not be subject to any limitation of liability, and the "
     "cap set out in Clause 5 shall not apply to this Clause 12.", "walk_away"),
    ("Uncapped indemnity, no carve-out", "LIABILITY-002", "uncapped indemnity",
     "The Supplier's indemnity obligations are uncapped.", "walk_away"),
    ("thirty (30) days convenience", "TERMINATION-001", "thirty (30) days",
     "The Buyer may terminate for convenience on thirty (30) days' written notice.", "preferred"),
    ("Singapore law", "LAW-001", "Singapore",
     "This SOW is governed by the laws of Singapore, and the courts of Singapore shall have exclusive jurisdiction "
     "over any dispute.", "walk_away"),
    ("No personal data", "DATA-001", "No personal data is shared",
     "The Services do not involve the processing of personal data of the Buyer's customers or employees. No personal "
     "data is shared under this SOW.", "preferred"),
    ("Personal data, no DPA", "DATA-001", "stores and processes personal data",
     "The platform stores and processes personal data of the Buyer's employees, including names, contact details, "
     "compensation and performance records.", "not_found"),
    ("99.0% platform availability", "SLA-001", "99.0%",
     "The Supplier shall maintain a measurable service level of 99.0% platform availability per calendar month.",
     "fallback"),
    ("Ticket-only SLA", "SLA-001", "95% of support tickets answered on time",
     "The Supplier shall meet a measurable service level of 95% of support tickets answered on time per calendar "
     "month.", "walk_away"),
    ("Fees fixed for initial term", "PRICE-001", "Fixed for the Initial Term",
     "Fees are fixed for the Initial Term and are not subject to increase during that period.", "preferred"),
    ("8% annual escalator", "PRICE-001", "8%",
     "The Charges shall increase by eight percent (8%) on each anniversary of the Effective Date.", "walk_away"),
]


@pytest.mark.parametrize("cid,rule_id,value,quote,want", CASES, ids=[c[0] for c in CASES])
def test_live_value_outcome(cid, rule_id, value, quote, want):
    assert rules.check_status(RULE[rule_id], _finding(rule_id, value, quote), _meta()) == want


@pytest.mark.parametrize("text,days", [("thirty (30) days", 30), ("30-day", 30), ("90 calendar days", 90),
                                       ("sixty (60) days' notice", 60)])
def test_parse_days_legal_style(text, days):
    assert rules.parse_days(text) == days


@pytest.mark.parametrize("text,months", [("twenty-four (24) months", 24), ("12-month", 12), ("12 months", 12)])
def test_parse_months_legal_style(text, months):
    assert rules.parse_months(text) == months


def test_carve_out_indemnity_does_not_escalate_the_cap():
    """Contract E: the data-breach carve-out used to score walk-away and drag LIABILITY-001 with it."""
    cap = GroundedFinding(finding=_finding(
        "LIABILITY-001", "total fees paid by the Buyer in the twelve (12) months preceding the claim",
        "The aggregate liability of each party shall not exceed the total fees paid by the Buyer in the twelve (12) "
        "months preceding the claim."), grounding="exact")
    ind = GroundedFinding(finding=_finding(*CASES[1][1:4]), grounding="exact")
    scored = steps.apply_rules(RulesInput(grounded=GroundedResult(meta=_meta(), findings=[cap, ind], dropped=[]),
                                          playbook=PB, pages=[]))
    got = {f.grounded.finding.rule_id: f.status for f in scored.findings}
    assert got == {"LIABILITY-001": "preferred", "LIABILITY-002": "preferred"}
    route = steps.route(RouteInput(scored=scored, preflight=Preflight(unreliable_extraction=False),
                                   flags=SecurityFlags()))
    assert route.route == "low_risk_queue"


def test_cap_carve_out_sentence_is_walk_away():
    """Contract B clause 5, second sentence: the cap stops applying to the supplier's obligations."""
    f = _finding("LIABILITY-001", "cap carved out by Clause 12",
                 "The foregoing limitation shall not apply to the Supplier's obligations except as set out in "
                 "Clause 12 (Indemnity), which shall be governed by the terms of that Clause.")
    assert rules.check_status(RULE["LIABILITY-001"], f, _meta()) == "walk_away"


def test_unclear_cap_is_escalated_when_indemnity_is_outside_it():
    cap = GroundedFinding(finding=_finding("LIABILITY-001", "see clause 12", "Liability is dealt with in Clause 12."),
                          grounding="exact")
    ind = GroundedFinding(finding=_finding(*CASES[3][1:4]), grounding="exact")
    scored = steps.apply_rules(RulesInput(grounded=GroundedResult(meta=_meta(), findings=[cap, ind], dropped=[]),
                                          playbook=PB, pages=[]))
    assert {f.grounded.finding.rule_id: f.status for f in scored.findings}["LIABILITY-001"] == "walk_away"
