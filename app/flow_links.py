"""Connect the stages: request → quotations → negotiation → contract.

Everything here is deterministic. The procurement request supplies budget, SLA
and term; quotations are compared against those; a chosen quotation seeds the
negotiation scenario; and the reviewed contract is checked against the quote
from the same supplier.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from app.commercial import (
    DEFAULT_BUDGET_INR,
    DEFAULT_HORIZON_MONTHS,
    DEFAULT_REQUIRED_UPTIME,
    compare_quotations,
    load_demo_quotations,
)
from app.negotiation import load_scenario
from app.negotiation_schemas import NegotiationScenario
from app.schemas import CommercialComparison, HistoricalPriceBaseline, Quotation, ReviewPackage


def required_uptime(sla_text: Optional[str]) -> float:
    m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*%", sla_text or "")
    return float(m.group(1)) if m else float(DEFAULT_REQUIRED_UPTIME)


def comparison_for_request(req) -> CommercialComparison:
    """Compare the quotations on file against *this* request's numbers."""
    budget = req.budget_amount if req and req.budget_amount else float(DEFAULT_BUDGET_INR)
    if req and req.budget_period == "monthly":
        budget = budget * 12
    return compare_quotations(
        load_demo_quotations(),
        budget_inr=budget,
        required_uptime_percent=required_uptime(req.required_sla if req else None),
        comparison_horizon_months=(req.requested_contract_term_months if req and req.requested_contract_term_months
                                   else DEFAULT_HORIZON_MONTHS),
        requisition_id=req.request_id if req else "PR-2026-1042",
        baseline=HistoricalPriceBaseline(
            label="Prior year similar SaaS (illustrative)", currency="INR",
            annual_recurring_fee=4500000.0, year=2025, notes=["Synthetic baseline for demo only."],
        ),
    )


class ContractQuoteCheck(BaseModel):
    supplier: Optional[str] = None
    matched_quote: Optional[str] = None
    findings: list[tuple[str, str]] = Field(default_factory=list)  # (severity, message)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def contract_vs_quotes(pkg: Optional[ReviewPackage], quotes: list[Quotation]) -> ContractQuoteCheck:
    """Does the reviewed contract match a quotation we actually received?"""
    if not pkg or not pkg.metadata or not pkg.metadata.supplier.value:
        return ContractQuoteCheck()
    supplier = pkg.metadata.supplier.value
    out = ContractQuoteCheck(supplier=supplier)
    q = next((q for q in quotes if _norm(q.supplier_name) in _norm(supplier)
              or _norm(supplier).startswith(_norm(q.supplier_name))), None)
    if not q:
        out.findings.append((
            "high",
            f"No quotation on file from {supplier}. The contract was not part of this sourcing event — "
            "confirm it belongs to this request before approving.",
        ))
        return out
    out.matched_quote = q.supplier_name
    try:
        value = float(pkg.metadata.contract_value.value) if pkg.metadata.contract_value.value else None
    except ValueError:
        value = None
    if value and q.annual_recurring_fee and value > q.annual_recurring_fee * 1.001:
        out.findings.append(("high", f"Contract value ₹{value:,.0f} is above the quoted annual fee "
                                     f"₹{q.annual_recurring_fee:,.0f}."))
    sla = pkg.metadata.sla_target_pct.value if pkg.metadata.sla_target_pct else None
    if sla and q.uptime_percent:
        try:
            if float(sla) < q.uptime_percent:
                out.findings.append(("medium", f"Contract SLA {sla}% is lower than the {q.uptime_percent}% quoted."))
        except ValueError:
            pass
    if not out.findings:
        out.findings.append(("ok", f"Contract terms are consistent with the {q.supplier_name} quotation."))
    return out


def scenario_from_quote(quote: Quotation, req=None) -> NegotiationScenario:
    """Negotiation brief for a chosen quotation, with limits from the request."""
    base = load_scenario()
    sc = base.model_copy(deep=True)
    budget = float(req.budget_amount) if req and req.budget_amount else base.policy.budget_ceiling_inr
    opening = float(quote.annual_recurring_fee or base.opening_quote_inr)
    term = (req.requested_contract_term_months if req and req.requested_contract_term_months
            else base.policy.max_term_months)
    uptime = required_uptime(req.required_sla if req else None)

    sc.scenario_id = f"{_norm(quote.supplier_name)}_{(req.request_id if req else base.requisition_id).lower()}"
    sc.requisition_id = req.request_id if req else base.requisition_id
    sc.supplier.name = quote.supplier_name
    sc.opening_quote_inr = opening
    sc.policy.budget_ceiling_inr = budget
    sc.policy.target_price_inr = round(min(budget, opening) * 0.99, -4)
    sc.policy.price_floor_inr = round(min(budget * 0.96, sc.policy.target_price_inr * 0.97), -4)
    sc.policy.min_uptime_percent = uptime
    sc.policy.max_term_months = int(term)
    sc.description = (f"Bring {quote.supplier_name} within the ₹{budget:,.0f} budget on {sc.requisition_id} "
                      f"while holding {uptime}% uptime, a {term}-month term and data protection.")
    for t in sc.terms:
        if t.key == "annual_fee_inr":
            t.current_value = str(int(opening))
            t.target_value = str(int(sc.policy.target_price_inr))
            t.minimum_acceptable = str(int(sc.policy.price_floor_inr))
        elif t.key == "uptime_percent":
            t.current_value = str(quote.uptime_percent or uptime)
            t.target_value = t.minimum_acceptable = str(uptime)
        elif t.key == "term_months":
            t.current_value = str(quote.term_months or term)
            t.target_value = t.minimum_acceptable = str(term)
        elif t.key == "payment_terms" and quote.payment_terms:
            t.current_value = quote.payment_terms
    return sc
