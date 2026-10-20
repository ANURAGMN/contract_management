"""Phase 4 — deterministic supplier quotation comparison.

All money math uses ``Decimal``. Missing inputs yield ``None`` / ``not_comparable``
rather than silent zeros. Lowest price is labelled but never declared the winner.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Optional

from app.schemas import (
    CommercialComparison,
    HistoricalPriceBaseline,
    Quotation,
    SupplierComparisonRow,
)

ROOT = Path(__file__).resolve().parent.parent
QUOTATIONS_DIR = ROOT / "data" / "quotations"

# Demo defaults from PR-2026-1042 (absolute INR, not "lakh" display units).
DEFAULT_BUDGET_INR = Decimal("4800000")
DEFAULT_REQUIRED_UPTIME = Decimal("99.9")
DEFAULT_HORIZON_MONTHS = 12
DEFAULT_REQUISITION_ID = "PR-2026-1042"

_TWO = Decimal("0.01")
_MONEY = Decimal("0.01")


def _d(value: float | int | str | Decimal | None) -> Optional[Decimal]:
    if value is None:
        return None
    return Decimal(str(value))


def _money(value: Optional[Decimal]) -> Optional[float]:
    if value is None:
        return None
    return float(value.quantize(_MONEY, rounding=ROUND_HALF_UP))


def _pct(value: Optional[Decimal]) -> Optional[float]:
    if value is None:
        return None
    return float(value.quantize(_TWO, rounding=ROUND_HALF_UP))


def load_quotation(path: str | Path) -> Quotation:
    p = Path(path)
    return Quotation.model_validate_json(p.read_text(encoding="utf-8"))


def load_demo_quotations() -> list[Quotation]:
    """Load CloudSoft / TechNova / DataSphere fixtures if present."""
    names = ("cloudsoft.json", "technova.json", "datasphere.json")
    out: list[Quotation] = []
    for name in names:
        path = QUOTATIONS_DIR / name
        if path.exists():
            out.append(load_quotation(path))
    return out


def _unresolved(q: Quotation) -> list[str]:
    missing: list[str] = []
    for field in (
        "annual_recurring_fee",
        "user_count",
        "term_months",
        "uptime_percent",
        "taxes_included",
        "support_scope",
        "payment_terms",
    ):
        if getattr(q, field) is None:
            missing.append(field)
    return missing


def _scope_status(q: Quotation) -> tuple[str, Optional[str], list[str]]:
    """Return (status, reason, warnings)."""
    warnings: list[str] = []
    missing = _unresolved(q)
    critical = [f for f in ("annual_recurring_fee", "term_months", "user_count") if f in missing]
    if critical:
        return (
            "not_comparable",
            "Missing required commercial fields: " + ", ".join(critical),
            warnings,
        )
    if q.exclusions:
        warnings.append("Exclusions present — verify scope before ranking on price.")
    if q.taxes_included is None:
        warnings.append("Tax inclusion unknown — totals may not be comparable.")
    if q.support_scope is None:
        warnings.append("Support scope not stated.")
    if missing:
        return "unknown", "Some commercial fields unresolved: " + ", ".join(missing), warnings
    return "comparable", None, warnings


def compare_quotations(
    quotations: list[Quotation],
    *,
    budget_inr: float | Decimal = DEFAULT_BUDGET_INR,
    required_uptime_percent: float | Decimal = DEFAULT_REQUIRED_UPTIME,
    comparison_horizon_months: int = DEFAULT_HORIZON_MONTHS,
    requisition_id: str = DEFAULT_REQUISITION_ID,
    baseline: Optional[HistoricalPriceBaseline] = None,
) -> CommercialComparison:
    """Build a side-by-side comparison. Never ranks a 'best' supplier."""
    budget = _d(budget_inr) or DEFAULT_BUDGET_INR
    required_uptime = _d(required_uptime_percent) or DEFAULT_REQUIRED_UPTIME
    horizon = int(comparison_horizon_months)
    if horizon <= 0:
        raise ValueError("comparison_horizon_months must be positive")

    global_warnings: list[str] = []
    notes = [
        "Amounts are absolute currency units (INR), not lakh display labels.",
        "Lowest annual fee is highlighted but is not an automatic recommendation.",
        f"Total cost uses a {horizon}-month horizon when term and fees are known.",
    ]
    if baseline:
        notes.append(
            f"Historical baseline: {baseline.label} "
            f"({baseline.currency} {baseline.annual_recurring_fee}/year)."
        )

    rows: list[SupplierComparisonRow] = []
    for q in quotations:
        unresolved = _unresolved(q)
        scope, reason, warn = _scope_status(q)
        row_warnings = list(warn)
        annual = _d(q.annual_recurring_fee)
        one_time = _d(q.one_time_fee) if q.one_time_fee is not None else None
        users = q.user_count
        term = q.term_months
        uptime = _d(q.uptime_percent)

        normalized_annual = annual  # already annual recurring when present
        cost_per_user = None
        if annual is not None and users and users > 0:
            cost_per_user = annual / Decimal(users)

        budget_var = None
        budget_var_pct = None
        if annual is not None:
            budget_var = annual - budget
            if budget != 0:
                budget_var_pct = (budget_var / budget) * Decimal("100")

        total = None
        if annual is not None and term is not None:
            # Annual fee × (horizon/12) + one-time (if known). Do not assume 0 one-time.
            years = Decimal(horizon) / Decimal("12")
            total = annual * years
            if one_time is not None:
                total = total + one_time
            elif "one_time_fee" not in unresolved:
                pass
            else:
                row_warnings.append(
                    "One-time fee unknown — total excludes implementation charges."
                )
                # Still report recurring-only total with a warning.
        elif annual is None or term is None:
            reason = reason or "Cannot compute total cost without annual fee and term."

        term_mismatch = bool(term is not None and term != horizon)
        if term_mismatch:
            row_warnings.append(
                f"Term is {term} months vs comparison horizon {horizon} months."
            )

        sla_ok = None
        if uptime is not None:
            sla_ok = uptime >= required_uptime
            if not sla_ok:
                row_warnings.append(
                    f"Uptime {uptime}% is below required minimum {required_uptime}%."
                )

        vs_baseline = None
        if baseline is not None and annual is not None:
            base = _d(baseline.annual_recurring_fee)
            if base and base != 0:
                vs_baseline = ((annual - base) / base) * Decimal("100")

        if scope == "not_comparable":
            # Do not invent totals when not comparable on critical fields.
            total = None
            cost_per_user = cost_per_user if annual is not None and users else None

        rows.append(
            SupplierComparisonRow(
                supplier_name=q.supplier_name,
                source_filename=q.source_filename,
                currency=q.currency,
                annual_recurring_fee=_money(annual),
                one_time_fee=_money(one_time),
                normalized_annual_fee=_money(normalized_annual),
                total_cost_over_horizon=_money(total),
                cost_per_user=_money(cost_per_user),
                budget_variance=_money(budget_var),
                budget_variance_pct=_pct(budget_var_pct),
                vs_baseline_pct=_pct(vs_baseline),
                uptime_percent=float(uptime) if uptime is not None else None,
                sla_compliant=sla_ok,
                term_months=term,
                term_mismatch=term_mismatch,
                scope_comparability=scope,  # type: ignore[arg-type]
                not_comparable_reason=reason,
                unresolved_fields=unresolved,
                warnings=row_warnings,
            )
        )

    # Flag lowest annual fee among rows that have a fee (informational only).
    priced = [r for r in rows if r.annual_recurring_fee is not None]
    if priced:
        lowest = min(priced, key=lambda r: Decimal(str(r.annual_recurring_fee)))
        for r in rows:
            if r.annual_recurring_fee == lowest.annual_recurring_fee:
                r.is_lowest_annual_fee = True
        if any(r.scope_comparability != "comparable" for r in priced):
            global_warnings.append(
                "Lowest fee is highlighted, but not all offers are fully comparable."
            )

    if len({q.currency for q in quotations}) > 1:
        global_warnings.append("Mixed currencies — convert before comparing totals.")

    return CommercialComparison(
        requisition_id=requisition_id,
        budget_inr=float(budget),
        required_uptime_percent=float(required_uptime),
        comparison_horizon_months=horizon,
        baseline=baseline,
        rows=rows,
        warnings=global_warnings,
        notes=notes,
    )


def compare_demo_quotations(
    *,
    budget_inr: float | Decimal = DEFAULT_BUDGET_INR,
    include_baseline: bool = True,
) -> CommercialComparison:
    quotes = load_demo_quotations()
    if not quotes:
        raise FileNotFoundError(
            f"No quotation fixtures in {QUOTATIONS_DIR}. "
            "Expected cloudsoft.json, technova.json, datasphere.json."
        )
    baseline = None
    if include_baseline:
        baseline = HistoricalPriceBaseline(
            label="Prior year similar SaaS (illustrative)",
            currency="INR",
            annual_recurring_fee=4500000.0,
            year=2025,
            notes=["Synthetic baseline for demo only."],
        )
    return compare_quotations(
        quotes, budget_inr=budget_inr, baseline=baseline
    )


def dump_comparison_json(comparison: CommercialComparison, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(comparison.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
