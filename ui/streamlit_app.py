"""Supplier Contract Review — Streamlit demo UI.

Run from the repo root:
    streamlit run ui/streamlit_app.py

Pages are grouped by workflow stage (Intake → Review → Decide → Manage →
Quality). Every page calls the typed service functions in ``app/``; this file
only lays results out. Documents (Contract & findings page):
  1. Run live review — Gemini on an uploaded or bundled PDF; result saved to runs/
  2. Existing documents — reviewed samples and saved live runs, no model calls
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import theme as ui  # noqa: E402
import live_pages as live  # noqa: E402
from app.approvals import (  # noqa: E402
    DecisionRejectedError,
    UnauthorizedDecisionError,
    build_queue,
    merge_queues,
    queue_to_markdown,
    record_decision,
)
from app.commercial import compare_demo_quotations  # noqa: E402
from app.demo_packages import list_samples, load_sample  # noqa: E402
from app.negotiation import (  # noqa: E402
    NegotiationError,
    load_scenario,
    negotiation_brief,
    new_run,
    run_replay,
)
from app.negotiation_agents import (  # noqa: E402
    analyze_supplier_reply,
    draft_buyer_email,
    generate_strategy,
    get_backend,
    human_approve_send,
    plan_next_move,
    run_scripted_live_demo,
    simulate_supplier_reply,
)
from app.agents import judge_available as gemini_available  # noqa: E402
from app.obligations import build_register, confirm_obligation  # noqa: E402
from app.procurement_requests import (  # noqa: E402
    EDITABLE_STATUSES,
    LOCKED_STATUSES,
    InvalidTransitionError,
    compare_request_to_package,
    find_duplicate_candidates,
    link_contract_run,
    load_request,
    mark_eligible_for_sourcing,
    refresh_derived,
    request_more_information,
    reset_fixture_to_store,
    save_draft,
    sourcing_checklist,
    submit_request,
    update_request,
)
from app.rag import (  # noqa: E402
    ask_knowledge_base,
    build_default_retriever,
    comparisons_for_package_findings,
)
from app.schemas import ReviewPackage, ReviewRequest  # noqa: E402

DEFAULT_REQUEST_ID = "PR-2026-1042"
DEFAULT_SAMPLE = "contract_e_procurement"
DEMO_TODAY = date(2026, 10, 19)  # fixed "today" so renewal reminders are reproducible

SOURCE_LIVE = "Run live review"
SOURCE_EXISTING = "Existing documents"
CONTRACTS_DIR = ROOT / "data" / "contracts"

RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
DECISIONS = RUNS / "decisions.jsonl"

st.set_page_config(
    page_title="Contract Review",
    page_icon=":material/fact_check:",
    layout="wide",
)
ui.inject()


# --------------------------------------------------------------------------- #
# Shared state and helpers
# --------------------------------------------------------------------------- #
@st.cache_resource
def _rag_bundle():
    return build_default_retriever()


@st.cache_data
def _playbook_topics() -> dict[str, str]:
    data = json.loads((ROOT / "data" / "playbook" / "demo-v1.json").read_text(encoding="utf-8"))
    return {r["id"]: r.get("topic") or r["id"] for r in data["rules"]}


def clean(text: str) -> str:
    """Readable rule-trace text: arrows and outcome codes in plain words."""
    text = text.replace(" -> ", " → ").replace("->", "→")
    for code, (label, _) in ui.OUTCOMES.items():
        text = re.sub(rf"\b{code}\b", label.lower(), text)
    return text


@st.cache_data
def _playbook_rules() -> dict[str, dict]:
    data = json.loads((ROOT / "data" / "playbook" / "demo-v1.json").read_text(encoding="utf-8"))
    return {r["id"]: r for r in data["rules"]}


def topic(rule_id: str) -> str:
    raw = _playbook_topics().get(rule_id, rule_id)
    return raw.replace("_", " ").strip().capitalize()


def _sample_labels() -> dict[str, str]:
    return {s["id"]: s["label"].replace("Sample ", "", 1) for s in list_samples()}


def _ensure_state() -> None:
    if "procurement_req" not in st.session_state:
        try:
            req = load_request(DEFAULT_REQUEST_ID)
            if not req.risk_flags and not req.routing_recommendations:
                refresh_derived(req)  # raw fixture: compute flags/routing for display
            st.session_state.procurement_req = req
        except Exception:
            st.session_state.procurement_req = reset_fixture_to_store(DEFAULT_REQUEST_ID)
    if "pkg" not in st.session_state:
        # Open on the richest sample so every page has something to show.
        st.session_state.pkg = load_sample(DEFAULT_SAMPLE)
        st.session_state.pkg.request_id = DEFAULT_REQUEST_ID
        st.session_state.sample_id = DEFAULT_SAMPLE
        st.session_state.active_doc = f"sample:{DEFAULT_SAMPLE}"


def load_sample_into_session(sample_id: str) -> None:
    pkg = load_sample(sample_id)
    active_pr = st.session_state.get("procurement_req")
    pkg.request_id = active_pr.request_id if active_pr else DEFAULT_REQUEST_ID
    st.session_state.pkg = pkg
    st.session_state.sample_id = sample_id
    st.session_state.active_doc = f"sample:{sample_id}"
    try:
        link_contract_run(pkg.request_id, pkg.run_id, actor="demo.user")
        st.session_state.procurement_req = load_request(pkg.request_id)
    except Exception:
        pass


def contract_id(pkg: ReviewPackage) -> str:
    return (pkg.file_name or pkg.run_id or "demo").replace(".pdf", "")


def save_upload(uploaded) -> str:
    path = RUNS / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return str(path)


def cache_run(name: str, payload: str) -> None:
    (RUNS / f"cache_{name}.json").write_text(payload, encoding="utf-8")


def cached_run(name: str) -> str | None:
    p = RUNS / f"cache_{name}.json"
    return p.read_text(encoding="utf-8") if p.exists() else None


def log_decision(run_id: str, rule_id: str, action: str, proposed: str, final: str, reason: str) -> None:
    rec = {
        "run_id": run_id,
        "rule_id": rule_id,
        "action": action,
        "proposed": proposed,
        "final": final,
        "reason": reason,
        "ts": datetime.now().astimezone().isoformat(),
    }
    with DECISIONS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def read_decisions(run_id: str) -> list[dict]:
    if not DECISIONS.exists():
        return []
    out = []
    for line in DECISIONS.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("run_id") == run_id:
            out.append(rec)
    return out


def field(pkg: ReviewPackage, name: str) -> tuple[str | None, int | None]:
    if not pkg.metadata:
        return None, None
    f = getattr(pkg.metadata, name, None)
    if f is None:
        return None, None
    page = f.evidence.page if getattr(f, "evidence", None) else None
    return f.value, page


def no_package_state() -> None:
    ui.empty_state(
        "No contract loaded",
        "Run a live review or open an existing document on the Contract & findings page.",
    )


def style_outcome(val: str) -> str:
    for label, tone in ui.OUTCOMES.values():
        if val == label:
            bg, fg = ui.TONES[tone]
            return f"background-color:{bg};color:{fg};font-weight:600"
    return ""


STATUS_TONE = {
    "APPROVED_FOR_SOURCING": "green",
    "APPROVED_FOR_SIGNATURE": "green",
    "NEEDS_INFORMATION": "amber",
    "REJECTED": "red",
    "CANCELLED": "gray",
    "DRAFT": "gray",
}
REVIEW_STATES = {"PROCUREMENT_TRIAGE", "FINANCE_REVIEW", "SECURITY_REVIEW", "LEGAL_REVIEW", "NEEDS_INFORMATION"}
TEAM_LABELS = {"security_privacy": "Security & privacy", "business_owner": "Business owner"}


def team_label(team: str) -> str:
    return TEAM_LABELS.get(team, ui.humanize(team))


def plain(text: str) -> str:
    """Turn STATUS_CODES inside backend messages into words."""
    return re.sub(r"\b[A-Z]{3,}(?:_[A-Z]+)+\b|\b[A-Z]{5,}\b", lambda m: ui.humanize(m.group(0)).lower(), text or "")


TEAM_STYLE = {
    "procurement": ("PR", "blue"),
    "business_owner": ("BO", "purple"),
    "security_privacy": ("SP", "green"),
    "legal": ("LG", "amber"),
    "finance": ("FI", "gray"),
}


# --------------------------------------------------------------------------- #
# Page: Procurement request
# --------------------------------------------------------------------------- #
def _stage_index(status: str) -> int:
    if status == "DRAFT":
        return 0
    if status == "SUBMITTED":
        return 1
    if status in REVIEW_STATES:
        return 2
    if status == "APPROVED_FOR_SOURCING":
        return 3
    return 4


def _first_reached(req, statuses: set[str]):
    for e in req.audit_events:
        if e.after_status in statuses:
            return e.timestamp
    return None


def page_request() -> None:
    req = st.session_state.procurement_req
    editable = req.status in EDITABLE_STATUSES
    locked = req.status in LOCKED_STATUSES

    ui.page_header(
        req.title,
        f"{req.request_id} · {req.requesting_team} · Owner: {req.business_owner}",
        [ui.chip(ui.humanize(req.status), STATUS_TONE.get(req.status, "blue"))],
    )
    with st.container():
        a1, a2, _ = st.columns([1.4, 1, 3.6])
        with a1:
            if editable:
                if st.button("Submit request", type="primary", width="stretch"):
                    try:
                        st.session_state.procurement_req = submit_request(req.request_id, actor="demo.user")
                        st.rerun()
                    except InvalidTransitionError as exc:
                        st.error(str(exc))
            elif req.status in REVIEW_STATES | {"SUBMITTED"}:
                if st.button("Approve for sourcing", type="primary", width="stretch",
                             help="Demo shortcut acting as Procurement"):
                    try:
                        st.session_state.procurement_req = mark_eligible_for_sourcing(
                            req.request_id, actor="demo.procurement"
                        )
                        st.rerun()
                    except InvalidTransitionError as exc:
                        st.error(str(exc))
            else:
                if st.button("Open contract review", type="primary", width="stretch",
                             icon=":material/arrow_forward:"):
                    st.switch_page(PAGES["review"])
        with a2:
            with st.popover("More", icon=":material/more_horiz:", width="stretch"):
                can_need_info = req.status in REVIEW_STATES | {"SUBMITTED"} and req.status != "NEEDS_INFORMATION"
                if st.button("Request more information", disabled=not can_need_info, width="stretch"):
                    try:
                        st.session_state.procurement_req = request_more_information(
                            req.request_id, actor="demo.reviewer"
                        )
                        st.rerun()
                    except InvalidTransitionError as exc:
                        st.error(str(exc))
                st.download_button(
                    "Download JSON",
                    data=req.model_dump_json(indent=2),
                    file_name=f"{req.request_id}.json",
                    mime="application/json",
                    width="stretch",
                )
                if st.button("Reset demo request to draft", width="stretch"):
                    st.session_state.procurement_req = reset_fixture_to_store(DEFAULT_REQUEST_ID)
                    st.rerun()

    live.intake_assistant(req)

    if req.status in ("REJECTED", "CANCELLED"):
        ui.banner(f"Request {ui.humanize(req.status).lower()}", "Reset the demo request to start again.", "gray", "i")
    else:
        idx = _stage_index(req.status)
        labels = ["Draft", "Submitted", "Review", "Approved for sourcing", "Contract review"]
        reached = [
            {"DRAFT"},
            {"SUBMITTED"},
            REVIEW_STATES,
            {"APPROVED_FOR_SOURCING"},
            {"SOURCING_IN_PROGRESS", "CONTRACT_REVIEW", "APPROVED_FOR_SIGNATURE"},
        ]
        steps = []
        for i, label in enumerate(labels):
            ts = _first_reached(req, reached[i])
            if i < idx:
                steps.append((label, ui.fmt_ts(ts) if ts else "Done", "done"))
            elif i == idx:
                if req.status == "NEEDS_INFORMATION":
                    steps.append((label, "Information requested", "warn"))
                elif i == 2:
                    steps.append((label, ui.humanize(req.status), "current"))
                elif i == 3:
                    steps.append((label, "Fields locked", "final"))
                else:
                    steps.append((label, "Now", "current"))
            else:
                steps.append((label, "", "todo"))
        ui.stepper(steps)

    if req.missing_fields:
        ui.banner("Missing before submit", ", ".join(ui.humanize(f) for f in req.missing_fields), "amber", "!")

    sla = re.search(r"\d+(\.\d+)?%", req.required_sla or "")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Annual budget", ui.fmt_inr(req.budget_amount), border=True,
              help="Confirmed by finance" if req.budget_confirmed else "Not yet confirmed")
    m2.metric("Users", f"{req.number_of_users or '—'}", border=True)
    m3.metric("Initial term", f"{req.requested_contract_term_months or '—'} months", border=True)
    m4.metric("Required SLA", sla.group(0) if sla else (req.required_sla or "—"), border=True,
              help=req.required_sla)

    c1, c2 = st.columns(2)
    with c1, st.container(border=True, height="stretch"):
        st.markdown("##### Risk signals")
        if not req.risk_flags:
            st.caption("No risk signals for this request.")
        tone = {"high": "red", "medium": "amber", "low": "gray"}
        items = "".join(
            f'<li>{ui.chip(f.severity.capitalize(), tone.get(f.severity, "gray"))}'
            f'<span class="b"><b>{ui.escape(ui.humanize(f.code))}</b>'
            f"<small>{ui.escape(f.message)}</small></span></li>"
            for f in req.risk_flags
        )
        st.html(f'<ul class="cr-list">{items}</ul>')
    with c2, st.container(border=True, height="stretch"):
        st.markdown("##### Who reviews this")
        grouped: dict[str, list] = {}
        for r in req.routing_recommendations:
            grouped.setdefault(r.team, []).append(r)
        items = []
        for team, recs in grouped.items():
            initials, tone = TEAM_STYLE.get(team, (team[:2].upper(), "gray"))
            bg, fg = ui.TONES[tone]
            reasons = " ".join(r.reason for r in recs)
            codes = " · ".join(r.reason_code for r in recs)
            items.append(
                f'<li><span class="cr-av" style="background:{bg};color:{fg}">{initials}</span>'
                f'<span class="b" style="flex:1"><b>{ui.escape(team_label(team))}</b>'
                f"<small>{ui.escape(reasons)}</small></span>"
                f'<span class="cr-mono">{ui.escape(codes)}</span></li>'
            )
        st.html(f'<ul class="cr-list">{"".join(items)}</ul>')
        st.caption("Deterministic policy — the model cannot set status or routing.")

    with st.container(border=True):
        st.markdown("##### Sourcing checklist")
        items = [re.sub(r"\s*\(from request( flags)?\)\.?$", "", i) for i in sourcing_checklist(req)]
        half = (len(items) + 1) // 2
        k1, k2 = st.columns(2)
        k1.markdown("\n".join(f"- {i}" for i in items[:half]))
        k2.markdown("\n".join(f"- {i}" for i in items[half:]))

    dups = find_duplicate_candidates(req)
    if dups:
        with st.container(border=True):
            st.markdown("##### Possibly related requests")
            st.caption("Shown for a person to confirm — never merged automatically.")
            for d in dups:
                st.markdown(f"- `{d.request_id}` · {d.title} · score {d.match_score} · {', '.join(d.match_reasons)}")

    with st.expander("Request details", expanded=editable, icon=":material/edit:"):
        if locked:
            st.caption(f"Fields are locked while the request is {ui.humanize(req.status).lower()}.")
        elif not editable:
            st.caption("Fields can be edited in Draft or Needs information.")
        _request_form(req, editable)

    with st.container(border=True):
        st.markdown("##### Activity")
        events = list(reversed(req.audit_events))
        if not events:
            st.caption("No activity yet.")
        rows = []
        for e in events[:6]:
            move = ""
            if e.before_status or e.after_status:
                move = f"{ui.humanize(e.before_status) if e.before_status else '…'} → {ui.humanize(e.after_status) if e.after_status else '…'}"
            rows.append({"When": ui.fmt_ts(e.timestamp), "Who": e.actor, "What": plain(e.detail), "Status": move})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if len(events) > 6:
            with st.expander(f"All {len(events)} events"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "When": ui.fmt_ts(e.timestamp),
                                "Who": e.actor,
                                "Action": ui.humanize(e.action),
                                "From": ui.humanize(e.before_status) if e.before_status else "",
                                "To": ui.humanize(e.after_status) if e.after_status else "",
                                "Detail": plain(e.detail),
                            }
                            for e in events
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )


def _request_form(req, editable: bool) -> None:
    off = not editable
    cats = ["saas_software", "services", "hardware", "consulting", "other"]
    sens = ["public", "internal", "confidential", "personal_data", "regulated", "unknown"]
    periods = ["annual", "monthly", "project"]
    with st.form("pr_form", border=False):
        title = st.text_input("Title", value=req.title, disabled=off)
        c1, c2 = st.columns(2)
        with c1:
            requesting_team = st.text_input("Requesting team", value=req.requesting_team, disabled=off)
            requester_role = st.text_input("Requester role", value=req.requester_role, disabled=off)
            business_owner = st.text_input("Business owner", value=req.business_owner, disabled=off)
            category = st.selectbox("Category", cats, index=cats.index(req.category or "saas_software"),
                                    format_func=ui.humanize, disabled=off)
            purchase = st.selectbox(
                "New purchase or renewal", ["new_purchase", "renewal"],
                index=0 if (req.new_purchase_or_renewal or "new_purchase") == "new_purchase" else 1,
                format_func=ui.humanize, disabled=off,
            )
            number_of_users = st.number_input("Number of users", min_value=0,
                                              value=int(req.number_of_users or 0), disabled=off)
            desired_start = st.date_input("Desired start date", value=req.desired_start_date, disabled=off,
                                          format="DD/MM/YYYY")
        with c2:
            budget_amount = st.number_input("Budget amount (INR)", min_value=0.0, step=100000.0,
                                            value=float(req.budget_amount or 0), disabled=off)
            budget_period = st.selectbox("Budget period", periods,
                                         index=periods.index(req.budget_period or "annual"),
                                         format_func=ui.humanize, disabled=off)
            estimated_recurring = st.number_input("Estimated recurring cost (INR)", min_value=0.0, step=100000.0,
                                                  value=float(req.estimated_recurring_cost or 0), disabled=off)
            term_months = st.number_input("Requested term (months)", min_value=0,
                                          value=int(req.requested_contract_term_months or 0), disabled=off)
            required_sla = st.text_input("Required SLA", value=req.required_sla or "", disabled=off)
            data_sensitivity = st.selectbox("Data sensitivity", sens,
                                            index=sens.index(req.data_sensitivity or "unknown"),
                                            format_func=ui.humanize, disabled=off)
            t1, t2, t3 = st.columns(3)
            budget_confirmed = t1.checkbox("Budget confirmed", value=req.budget_confirmed, disabled=off)
            security_review_required = t2.checkbox("Security review", value=req.security_review_required,
                                                   disabled=off)
            privacy_review_required = t3.checkbox("Privacy review", value=req.privacy_review_required,
                                                  disabled=off)
        justification = st.text_area("Business justification", value=req.business_justification, disabled=off)
        capabilities = st.text_input("Requested capabilities (comma-separated)",
                                     value=", ".join(req.requested_capabilities), disabled=off)
        integrations = st.text_input("Integrations (comma-separated)", value=", ".join(req.integrations),
                                     disabled=off)
        saved = st.form_submit_button("Save draft", disabled=off)

    if not saved:
        return
    updates = {
        "title": title,
        "requesting_team": requesting_team,
        "requester_role": requester_role,
        "business_owner": business_owner,
        "category": category,
        "new_purchase_or_renewal": purchase,
        "number_of_users": int(number_of_users) or None,
        "desired_start_date": desired_start or None,
        "budget_amount": float(budget_amount) or None,
        "budget_period": budget_period,
        "budget_confirmed": budget_confirmed,
        "estimated_recurring_cost": float(estimated_recurring) or None,
        "requested_contract_term_months": int(term_months) or None,
        "required_sla": required_sla or None,
        "data_sensitivity": data_sensitivity,
        "security_review_required": security_review_required,
        "privacy_review_required": privacy_review_required,
        "business_justification": justification,
        "requested_capabilities": [x.strip() for x in capabilities.split(",") if x.strip()],
        "integrations": [x.strip() for x in integrations.split(",") if x.strip()],
    }
    try:
        save_draft(load_request(req.request_id), actor="demo.user")
        st.session_state.procurement_req = update_request(req.request_id, updates, actor="demo.user")
        st.toast("Draft saved", icon=":material/check:")
        st.rerun()
    except InvalidTransitionError as exc:
        st.error(str(exc))


# --------------------------------------------------------------------------- #
# Page: Contract & findings
# --------------------------------------------------------------------------- #
STAGE_LABELS = {
    "intake": "Intake",
    "metadata": "Metadata",
    "grounding": "Grounding",
    "rules": "Playbook rules",
    "route": "Route",
    "redlines": "Redlines",
    "security": "Security",
    "knowledge": "Policy knowledge",
}
OUTCOME_ORDER = {"walk_away": 0, "not_found": 1, "unclear": 2, "fallback": 3, "preferred": 4}


def _stage_chips(pkg: ReviewPackage) -> None:
    chips = []
    for key, label in STAGE_LABELS.items():
        val = str(pkg.stage_status.get(key, "")) if pkg.stage_status else ""
        if not val:
            continue
        low = val.lower()
        if key == "security" and "flag" in low:
            chips.append(ui.chip("⚑ Security flagged", "red"))
        elif key == "route":
            chips.append(ui.chip(f"✓ Routed: {ui.ROUTES.get(val, (ui.humanize(val),))[0]}", "blue"))
        elif any(w in low for w in ("fail", "error", "exhaust")):
            chips.append(ui.chip(f"✕ {label}", "red"))
        elif "skip" in low:
            chips.append(ui.chip(f"– {label}", "gray"))
        else:
            chips.append(f'<span class="cr-chip" style="color:#0D652D">✓ {ui.escape(label)}</span>')
    if chips:
        st.html(f'<div class="cr-pipe" aria-label="Pipeline">{"".join(chips)}</div>')


def page_review() -> None:
    live.document_source()
    pkg: ReviewPackage | None = st.session_state.get("pkg")
    if not pkg:
        ui.page_header("Contract & findings")
        no_package_state()
        return

    ui.section("Review results")
    supplier, _ = field(pkg, "supplier")
    buyer, _ = field(pkg, "buyer")
    chips = [ui.chip("Review complete", "green") if pkg.status == "complete"
             else ui.chip(ui.humanize(pkg.status), "amber")]
    chips += live.provenance_chips(pkg)
    head, act = st.columns([4, 1], vertical_alignment="bottom")
    with head:
        ui.page_header(
            supplier or pkg.file_name or "Contract",
            " · ".join(x for x in [
                f"With {buyer}" if buyer else "",
                pkg.file_name or "",
                f"Playbook {pkg.playbook_version or 'demo-v1'}",
                f"Run {pkg.run_id}",
            ] if x),
            chips,
        )
    with act, st.popover("Export", icon=":material/download:", width="stretch"):
        st.download_button("Review package (JSON)", data=pkg.model_dump_json(indent=2),
                           file_name=f"{pkg.run_id or 'review'}.json", mime="application/json", width="stretch")
        md = [f"# Review package `{pkg.run_id}`", f"- File: {pkg.file_name}", f"- Status: {pkg.status}",
              f"- Route: {pkg.route.route}", "", "## Findings"]
        md += [f"- **{sf.grounded.finding.rule_id}** — {sf.status} ({sf.severity})" for sf in pkg.findings]
        st.download_button("Summary (Markdown)", data="\n".join(md), file_name=f"{pkg.run_id or 'review'}.md",
                           mime="text/markdown", width="stretch")

    for e in pkg.security.pre + pkg.security.model:
        ui.banner(
            f"Hidden instruction to AI reviewers found on page {e.page}",
            f"“{e.quote}” It was treated as data and ignored: the route can't be lowered and a "
            "Security review task is open.",
            "red",
            "⚠",
        )
    active_pr = st.session_state.get("procurement_req")
    if active_pr:
        for flag in compare_request_to_package(active_pr, pkg):
            ui.banner(f"{ui.humanize(flag.code)} vs {active_pr.request_id}", flag.message,
                      "red" if flag.severity == "high" else "amber", "!")

    _stage_chips(pkg)

    with st.container(border=True):
        st.markdown("##### Key terms")
        if not pkg.metadata:
            st.caption("No metadata extracted.")
        else:
            value, vp = field(pkg, "contract_value")
            eff, ep = field(pkg, "effective_date")
            exp, _ = field(pkg, "expiry_date")
            pay, pp = field(pkg, "payment_terms_days")
            notice, np_ = field(pkg, "notice_period_days")
            auto, _ = field(pkg, "auto_renewal")
            cap, cp = field(pkg, "liability_cap")
            law, lp = field(pkg, "governing_law")
            renewal = "—"
            if auto is not None or notice:
                renewal = ("Auto-renews" if str(auto).lower() in ("true", "yes", "1") else "No auto-renewal")
                if notice:
                    renewal += f" · {notice} days notice"
            ui.key_terms([
                ("Contract value", ui.fmt_inr(value) if value else "—", vp),
                ("Term", f"{ui.fmt_date(eff)} – {ui.fmt_date(exp)}" if eff else "—", ep),
                ("Payment terms", f"Net {pay}" if pay else "—", pp),
                ("Renewal", renewal, np_),
                ("Liability cap", cap or "—", cp),
                ("Governing law", law or "—", lp),
            ])
            with st.expander("All extracted fields"):
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Field": ui.humanize(name),
                            "Value": f.value,
                            "Page": f.evidence.page if f.evidence else None,
                            "Source quote": f.evidence.quote if f.evidence else "",
                        }
                        for name, f in pkg.metadata.__dict__.items()
                    ]),
                    hide_index=True,
                    width="stretch",
                )

    findings = [
        sf for sf in pkg.findings
        if not (sf.status == "preferred" and not sf.grounded.finding.evidence)
    ]
    findings.sort(key=lambda sf: (OUTCOME_ORDER.get(sf.status, 9), sf.grounded.finding.rule_id))

    with st.container(border=True):
        counts: dict[str, int] = {}
        for sf in findings:
            counts[sf.status] = counts.get(sf.status, 0) + 1
        count_chips = "".join(
            ui.chip(f"{n} {ui.OUTCOMES.get(s, (ui.humanize(s), 'gray'))[0].lower()}",
                    ui.OUTCOMES.get(s, ("", "gray"))[1])
            for s, n in sorted(counts.items(), key=lambda kv: OUTCOME_ORDER.get(kv[0], 9))
        )
        st.html(f'<div class="cr-head" style="justify-content:space-between"><h5 style="margin:0;font-size:1.05rem;'
                f'font-weight:600">Findings</h5><div class="cr-head">{count_chips}</div></div>')
        if not findings:
            st.caption("No findings.")
            return
        df = pd.DataFrame([
            {
                "Clause": topic(sf.grounded.finding.rule_id),
                "Rule": sf.grounded.finding.rule_id,
                "Outcome": ui.OUTCOMES.get(sf.status, (ui.humanize(sf.status), ""))[0],
                "Severity": sf.severity.capitalize(),
                "Page": (f"p{sf.grounded.finding.evidence[0].page}" if sf.grounded.finding.evidence else "—"),
                "Why": clean(sf.rule_trace.split(":", 1)[-1].strip()),
            }
            for sf in findings
        ])
        event = st.dataframe(
            df.style.map(style_outcome, subset=["Outcome"]),
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key=f"findings_{pkg.run_id}",
            column_config={
                "Clause": st.column_config.TextColumn(width="medium"),
                "Rule": st.column_config.TextColumn(width=110),
                "Outcome": st.column_config.TextColumn(width="small"),
                "Severity": st.column_config.TextColumn(width="small"),
                "Page": st.column_config.TextColumn(width="small"),
                "Why": st.column_config.TextColumn(width="large"),
            },
        )
        st.caption("Select a row below — the panel under the table is the new side-by-side "
                   "contract vs approved policy comparison (RAG). Colour = outcome; severity = how much it matters.")
    rows = event.selection.rows if event and event.selection else []
    sel = findings[rows[0]] if rows else findings[0]
    st.markdown("##### Contract vs approved policy")
    _finding_detail(pkg, sel)

    if pkg.dropped:
        st.caption(f"{len(pkg.dropped)} ungrounded item(s) were dropped before display — "
                   "every quote shown here was string-matched against the PDF text.")


def _finding_detail(pkg: ReviewPackage, sf) -> None:
    from app.rag import ClauseKnowledgeComparison

    rule_id = sf.grounded.finding.rule_id
    # Prefer package-persisted groundings (Phase 3); fall back to live retrieval
    cmp_ = None
    if pkg.knowledge_groundings:
        for raw in pkg.knowledge_groundings:
            if raw.get("rule_id") == rule_id:
                cmp_ = ClauseKnowledgeComparison.model_validate(raw)
                break
    if cmp_ is None:
        retriever, _, _ = _rag_bundle()
        comparisons = {
            c.rule_id: c
            for c in comparisons_for_package_findings(pkg.findings, retriever=retriever)
        }
        cmp_ = comparisons.get(rule_id)
    with st.container(border=True):
        st.html(f'<div class="cr-head"><h5 style="margin:0;font-size:1.15rem;font-weight:600">'
                f'{ui.escape(topic(rule_id))}</h5>{ui.outcome_chip(sf.status)}'
                f'{ui.chip(sf.severity.capitalize() + " severity", "gray")}'
                f'<span class="cr-mono">{ui.escape(rule_id)}</span></div>')
        c1, c2 = st.columns(2)
        with c1:
            ev = sf.grounded.finding.evidence
            if ev:
                ui.quote_box(f"Contract · page {ev[0].page}", ev[0].quote, "gray")
                for extra in ev[1:]:
                    ui.quote_box(f"Contract · page {extra.page}", extra.quote, "gray")
            else:
                ui.quote_box("Contract", "No matching clause found in the contract.", "gray")
        with c2:
            if cmp_ and not cmp_.abstained and cmp_.knowledge_evidence:
                top = cmp_.knowledge_evidence[0]
                ui.quote_box(f"Approved policy · {top.title} v{top.version}", top.excerpt, "blue")
                st.caption(f"Evidence `{top.evidence_id}` · retrieval relevance {top.relevance_score} "
                           "(a rank, not a confidence)")
            else:
                rule = _playbook_rules().get(rule_id)
                if rule:
                    lines = [f"{ui.OUTCOMES[k][0]}: {rule[k]}" for k in ("preferred", "fallback", "walk_away")
                             if rule.get(k)]
                    ui.quote_box(f"Playbook rule · {rule_id}", "\n".join(lines), "blue")
                    st.caption("No policy document passage was retrieved for this clause, "
                               "so the playbook thresholds are shown instead.")
                else:
                    ui.quote_box("Approved policy", "No approved policy passage was strong enough to cite — "
                                 "the knowledge base abstained.", "blue")
        if cmp_:
            st.markdown(f"**Assessment** — {cmp_.assessment}")
            st.markdown(f"**Next step** — {cmp_.action}")
        st.caption(f"Rule trace: {clean(sf.rule_trace)}")
        if sf.grounded.warnings:
            st.caption("Warnings: " + "; ".join(sf.grounded.warnings))
        if cmp_ and len(cmp_.knowledge_evidence) > 1:
            with st.expander(f"{len(cmp_.knowledge_evidence) - 1} more policy passages"):
                for ev in cmp_.knowledge_evidence[1:]:
                    st.markdown(f"**{ev.title}** · v{ev.version} · {' › '.join(ev.heading_path)}")
                    st.write(ev.excerpt)
                    st.caption(f"`{ev.evidence_id}` · relevance {ev.relevance_score}")


# --------------------------------------------------------------------------- #
# Page: Redlines & route
# --------------------------------------------------------------------------- #
def page_redlines() -> None:
    ui.page_header("Redlines & route", "Proposed contract changes and the deterministic approval route.")
    pkg: ReviewPackage | None = st.session_state.get("pkg")
    if not pkg:
        no_package_state()
        return

    route_label, route_why = ui.ROUTES.get(pkg.route.route, (ui.humanize(pkg.route.route), ""))
    with st.container(border=True):
        c1, c2 = st.columns([2, 3])
        with c1:
            st.caption("Recommended route")
            st.html(f'<div style="font-size:1.6rem;font-weight:600;line-height:1.2">{ui.escape(route_label)}</div>'
                    f'<p class="cr-sub" style="font-size:14px">{ui.escape(route_why)}</p>')
            st.html(ui.chip("Set by rules, never by the model", "blue"))
        with c2:
            st.caption("Why — rule trace")
            items = "".join(
                f'<li><span class="cr-mono" style="background:#E8F0FE;color:#0A2D6E;padding:2px 6px;'
                f'border-radius:6px">{ui.escape(t.split(":", 1)[0])}</span>'
                f'<span class="b">{ui.escape(clean(t.split(":", 1)[-1].strip()))}</span></li>'
                for t in pkg.route.trace
            )
            st.html(f'<ul class="cr-list">{items}</ul>')

    st.markdown("#### Redlines")
    if not pkg.redlines:
        ui.empty_state("No redlines needed", "Nothing in this contract is at walk-away, so there is nothing to redraft.")
    for i, cr in enumerate(pkg.redlines):
        r = cr.redline
        with st.container(border=True):
            st.html(
                f'<div class="cr-head"><h5 style="margin:0;font-size:1.1rem;font-weight:600">'
                f"{ui.escape(topic(r.rule_id))}</h5><span class=\"cr-mono\">{ui.escape(r.rule_id)}</span>"
                + (ui.chip("✓ Passed drafting checks", "green") if cr.passed
                   else ui.chip("Needs manual drafting", "amber"))
                + "</div>"
            )
            c1, c2 = st.columns(2)
            with c1:
                ui.quote_box(f"Current · page {r.original.page}", r.original.quote, "red", strike=True)
            with c2:
                ui.quote_box("Proposed", r.proposed_text, "green")
            st.caption(r.rationale)
            if cr.problems:
                st.warning("; ".join(cr.problems))
            d1, d2, d3 = st.columns([2, 3, 1], vertical_alignment="bottom")
            with d1:
                action = st.segmented_control(
                    "Decision", ["Accept", "Edit", "Reject", "Escalate"], default="Accept", key=f"act{i}"
                ) or "Accept"
            with d2:
                reason = st.text_input("Reason (optional)", key=f"rsn{i}", placeholder="Why this decision")
            edited = r.proposed_text
            if action == "Edit":
                edited = st.text_area("Edited clause", value=r.proposed_text, key=f"edit{i}")
            with d3:
                if st.button("Log", type="primary", key=f"log{i}", width="stretch"):
                    log_decision(pkg.run_id, r.rule_id, action.lower(), r.proposed_text, edited, reason)
                    st.toast(f"{action} logged for {r.rule_id}", icon=":material/check:")

    logged = read_decisions(pkg.run_id)
    if logged:
        with st.container(border=True):
            st.markdown("##### Decision log")
            st.dataframe(
                pd.DataFrame([
                    {
                        "When": ui.fmt_ts(datetime.fromisoformat(d["ts"])),
                        "Rule": d["rule_id"],
                        "Decision": d["action"].capitalize(),
                        "Reason": d.get("reason", ""),
                    }
                    for d in reversed(logged[-10:])
                ]),
                hide_index=True,
                width="stretch",
            )


# --------------------------------------------------------------------------- #
# Page: Approvals
# --------------------------------------------------------------------------- #
ROLES = ["procurement", "legal", "security_privacy", "finance", "business_owner", "admin", "viewer"]


def page_approvals() -> None:
    pkg: ReviewPackage | None = st.session_state.get("pkg")
    head, who = st.columns([3, 2], vertical_alignment="bottom")
    with head:
        ui.page_header("Approvals", "Exception queue built from deterministic policy. No agent can approve a task.")
    if not pkg:
        no_package_state()
        return
    with who:
        r1, r2 = st.columns(2)
        mock_role = r1.selectbox("Acting as (demo)", ROLES, index=2, format_func=team_label,
                                 help="Mock role for the demo, not real auth. The service still rejects "
                                      "decisions a role isn't allowed to make.")
        reviewer = r2.text_input("Reviewer", value="demo.reviewer")

    cid = contract_id(pkg)
    if "approval_queue" not in st.session_state or st.session_state.get("approval_contract_id") != cid:
        fresh = build_queue(pkg, contract_id=cid)
        prev = st.session_state.get("approval_queue")
        if prev and prev.contract_id == cid:
            fresh = merge_queues(prev, fresh)
        st.session_state.approval_queue = fresh
        st.session_state.approval_contract_id = cid
    q = st.session_state.approval_queue

    m1, m2, m3 = st.columns(3)
    m1.metric("Open tasks", len(q.open_tasks()), border=True)
    m2.metric("Decided", len(q.tasks) - len(q.open_tasks()), border=True)
    m3.metric("Audit events", len(q.audit), border=True)

    if not q.tasks:
        ui.empty_state("Nothing to approve", "Only preferred or fallback terms — no exception tasks were created.")
        return

    tasks = q.sorted_queue() or q.tasks
    left, right = st.columns([5, 6])
    with left, st.container(border=True):
        st.markdown("##### Queue")
        event = st.dataframe(
            pd.DataFrame([
                {
                    "Task": clean(t.title),
                    "Severity": t.severity.capitalize(),
                    "Status": ui.humanize(t.status),
                }
                for t in tasks
            ]),
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key=f"queue_{cid}",
            column_config={"Task": st.column_config.TextColumn(width="large")},
        )
        rows = event.selection.rows if event and event.selection else []
        chosen = tasks[rows[0]] if rows else tasks[0]
        with st.popover("Download queue", icon=":material/download:"):
            st.download_button("JSON", data=q.model_dump_json(indent=2), file_name=f"{cid}_approvals.json",
                               mime="application/json", width="stretch")
            st.download_button("Markdown", data=queue_to_markdown(q), file_name=f"{cid}_approvals.md",
                               mime="text/markdown", width="stretch")

    with right, st.container(border=True):
        sev_tone = {"high": "red", "medium": "amber", "low": "gray"}
        st.html(f'<div class="cr-head"><h5 style="margin:0;font-size:1.1rem;font-weight:600">'
                f"{ui.escape(clean(chosen.title))}</h5>"
                f"{ui.chip(chosen.severity.capitalize(), sev_tone.get(chosen.severity, 'gray'))}"
                f"{ui.chip(ui.humanize(chosen.status), 'green' if chosen.status != 'open' else 'amber')}</div>")
        evidence = ", ".join(chosen.source_evidence_ids) or "—"
        if "security_flag" in chosen.reason_codes:
            quotes = [f"p{e.page}: “{e.quote}”" for e in pkg.security.pre + pkg.security.model]
            evidence = " · ".join(quotes) or evidence
        detail = {
            "Teams": ", ".join(team_label(x) for x in chosen.required_teams),
            "Reason": ", ".join(ui.humanize(x) for x in chosen.reason_codes),
            "Policy rule": ", ".join(chosen.matched_policy_rule_ids),
            "Findings": ", ".join(chosen.related_finding_ids) or "—",
            "Evidence": evidence,
        }
        rows_html = "".join(
            f'<dt style="color:#5B5F66">{k}</dt><dd style="margin:0">{ui.escape(v)}</dd>' for k, v in detail.items()
        )
        st.html(f'<dl style="display:grid;grid-template-columns:110px minmax(0,1fr);gap:8px 16px;padding:0 !important;'
                f'font-size:14px;margin:6px 0 10px">{rows_html}</dl>')
        cites = getattr(chosen, "knowledge_citations", None) or []
        if cites:
            with st.expander(f"Policy knowledge ({len(cites)} citation{'s' if len(cites) != 1 else ''})", expanded=True):
                for c in cites:
                    st.markdown(
                        f"**{c.title}** · v{c.version or '—'} · `{c.evidence_id}`"
                        + (f" · relevance {c.relevance_score}" if c.relevance_score is not None else "")
                    )
                    st.write(c.excerpt)
        elif getattr(chosen, "knowledge_abstained", False):
            st.caption("No approved policy passage retrieved for this approval reason.")
        if chosen.decision:
            ui.banner(f"{ui.humanize(chosen.decision)} by {chosen.decided_by}", chosen.decision_reason or "",
                      "green", "✓")
        else:
            with st.form(f"decide_{chosen.task_id}", border=False):
                d1, d2 = st.columns(2)
                decision = d1.selectbox(
                    "Decision",
                    ["approve", "request_changes", "need_information", "reject", "waive_with_authorization"],
                    format_func=ui.humanize,
                )
                auth_ref = d2.text_input("Authorization reference", placeholder="Required to waive")
                reason = st.text_area("Reason", height=80)
                if st.form_submit_button("Record decision", type="primary"):
                    try:
                        st.session_state.approval_queue = record_decision(
                            q, chosen.task_id, reviewer=reviewer, role=mock_role,  # type: ignore[arg-type]
                            decision=decision, reason=reason,  # type: ignore[arg-type]
                            authorization_reference=auth_ref or None,
                        )
                        st.toast(f"Recorded {ui.humanize(decision).lower()}", icon=":material/check:")
                        st.rerun()
                    except UnauthorizedDecisionError as exc:
                        st.error(f"Not allowed for {ui.humanize(mock_role)}: {exc}")
                    except (DecisionRejectedError, KeyError) as exc:
                        st.error(str(exc))


# --------------------------------------------------------------------------- #
# Page: Obligations
# --------------------------------------------------------------------------- #
OBL_TYPES = {"sla": "SLA", "renewal_notice": "Renewal notice"}


def _why_date(o) -> str:
    if o.notice_period_days and o.trigger_date:
        return (f"{o.notice_period_days} days before the term ends on {ui.fmt_date(o.trigger_date)}; "
                "missing it renews the contract.")
    return o.date_calculation_explanation or ""


def page_obligations() -> None:
    pkg: ReviewPackage | None = st.session_state.get("pkg")
    if not pkg or not pkg.metadata:
        ui.page_header("Obligations & renewals")
        no_package_state()
        return
    cid = contract_id(pkg)
    if "obl_register" not in st.session_state or st.session_state.get("obl_contract_id") != cid:
        st.session_state.obl_register = build_register(pkg.metadata, contract_id=cid, today=DEMO_TODAY)
        st.session_state.obl_contract_id = cid
    reg = st.session_state.obl_register
    term_end = next((a.value for a in reg.anchors if a.anchor_type == "term_end"), None)

    ui.page_header(
        "Obligations & renewals",
        (f"Term ends {ui.fmt_date(term_end)} · " if term_end else "")
        + f"As of {ui.fmt_date(DEMO_TODAY)} (fixed demo date) · the notice deadline is not the expiry date",
    )

    open_obl = [o for o in reg.obligations if o.status not in ("complete", "not_applicable")]
    soon = sorted(
        [o for o in open_obl if o.due_date and (o.due_date - DEMO_TODAY).days <= 14],
        key=lambda o: o.due_date,
    )
    for o in soon:
        days = (o.due_date - DEMO_TODAY).days
        when = f"{-days} days overdue" if days < 0 else ("due today" if days == 0 else f"due in {days} days")
        st.html(
            f'<div class="cr-banner" role="alert" style="background:#FCE8E6;color:#5C0B0B;align-items:center">'
            f'<span class="cr-date"><small>{o.due_date.strftime("%b").upper()}</small>'
            f"<b>{o.due_date.day}</b></span>"
            f"<div><b>{ui.escape(o.title)} — {when}</b><span>{ui.escape(_why_date(o))} "
            f"Owner: {ui.escape(ui.humanize(o.owner_role))}.</span></div></div>"
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Obligations", len(reg.obligations), border=True)
    m2.metric("Need confirmation", sum(1 for o in reg.obligations if o.needs_human_confirmation), border=True)
    m3.metric("Due in 14 days", len(reg.upcoming(14)), border=True)
    m4.metric("Due in 90 days", len(reg.upcoming(90)), border=True)

    with st.container(border=True):
        top, tog = st.columns([3, 1], vertical_alignment="center")
        top.markdown("##### Register")
        only_confirm = tog.toggle("Needs confirmation only", value=False)
        rows = []
        for o in reg.obligations:
            if only_confirm and not o.needs_human_confirmation:
                continue
            days = (o.due_date - DEMO_TODAY).days if o.due_date else None
            rows.append({
                "Obligation": o.title,
                "Type": OBL_TYPES.get(o.obligation_type, ui.humanize(o.obligation_type)),
                "Due": ui.fmt_date(o.due_date) if o.due_date else ui.humanize(o.recurrence) if o.recurrence else "Ongoing",
                "Days left": str(days) if days is not None else "—",
                "Owner": ui.humanize(o.owner_role),
                "Criticality": ui.humanize(o.criticality),
                "Status": ui.humanize(o.status),
                "Date check": "Needs a person" if o.needs_human_confirmation else "Clear",
            })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(
                df.style.map(lambda v: "color:#A50E0E;font-weight:600"
                             if v.lstrip("-").isdigit() and int(v) <= 14 else "", subset=["Days left"]),
                hide_index=True, width="stretch",
            )
        else:
            st.caption("Nothing needs confirmation.")

    c1, c2 = st.columns([2, 3])
    with c1, st.container(border=True):
        st.markdown("##### Confirm an obligation")
        by_id = {o.obligation_id: o for o in reg.obligations}
        pick = st.selectbox("Obligation", list(by_id), format_func=lambda i: by_id[i].title)
        owner = st.text_input("Owner", value="demo.reviewer")
        if st.button("Confirm date and owner", type="primary"):
            st.session_state.obl_register = confirm_obligation(reg, pick, owner_user=owner)
            st.toast("Obligation confirmed", icon=":material/check:")
            st.rerun()
        st.download_button("Download register (JSON)", data=reg.model_dump_json(indent=2),
                           file_name=f"{cid}_obligations.json", mime="application/json")
    with c2, st.container(border=True):
        chosen = by_id[pick]
        st.markdown(f"##### {chosen.title}")
        if chosen.description:
            st.write(chosen.description)
        if chosen.date_calculation_explanation:
            st.markdown(f"**How the date was computed** — {_why_date(chosen)}")
            st.caption(chosen.date_calculation_explanation)
        if chosen.completion_evidence_required:
            st.markdown(f"**Evidence to close it** — {chosen.completion_evidence_required}")
        for qt, p in zip(chosen.source_quotes, chosen.source_pages or [None] * len(chosen.source_quotes)):
            ui.quote_box(f"Contract · page {p}" if p else "Contract", qt, "gray")

    if pkg.credit_claim:
        cc = pkg.credit_claim
        ui.banner(f"Claimable service credits: {ui.fmt_inr(cc.total_credit_inr)}",
                  f"From the performance data over {cc.months_considered} month(s)"
                  + (" (capped)." if cc.capped else "."), "green", "₹")

    with st.expander("Date anchors"):
        st.dataframe(
            pd.DataFrame([
                {
                    "Anchor": ui.humanize(a.anchor_type),
                    "Date": ui.fmt_date(a.value) if a.value else "",
                    "Confidence": ui.humanize(a.confidence),
                    "Page": a.source_page,
                    "Quote": a.source_quote or "",
                }
                for a in reg.anchors
            ]),
            hide_index=True, width="stretch",
        )


# --------------------------------------------------------------------------- #
# Page: Knowledge
# --------------------------------------------------------------------------- #
EXAMPLES = [
    "What does our policy require for auto-renewal notice?",
    "What is the minimum uptime in the SLA policy?",
    "Which team must approve a missing DPA?",
    "What is our policy for lunar mineral royalties?",
]


def page_knowledge() -> None:
    retriever, status, docs = _rag_bundle()
    ui.page_header("Knowledge base",
                   "Extractive answers with citations — it abstains when evidence is weak. "
                   "Synthetic demo policies, not legal advice.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents", status.document_count, border=True)
    m2.metric("Approved policies", status.approved_policy_count, border=True)
    m3.metric("Chunks", status.chunk_count, border=True)
    m4.metric("Retrieval", ui.humanize(status.mode), border=True)

    with st.container(border=True):
        st.markdown("##### Ask a question")
        pick = st.pills("Try one", EXAMPLES, default=EXAMPLES[0], key="kb_pick")
        custom = st.text_input("Or type your own", placeholder="e.g. What payment terms are allowed?")
        question = custom.strip() or pick or EXAMPLES[0]
        if st.button("Search policies", type="primary", icon=":material/search:"):
            st.session_state.ask_result = ask_knowledge_base(question, retriever)
            st.session_state.ask_question = question

    result = st.session_state.get("ask_result")
    if result:
        with st.container(border=True):
            st.caption(f"Question: {st.session_state.get('ask_question', '')}")
            if result.abstained:
                ui.banner("No confident answer", result.answer, "amber", "?")
            else:
                st.markdown(result.answer)
            for ev in result.evidence:
                with st.expander(f"{ev.title} · v{ev.version} · relevance {ev.relevance_score}"):
                    st.caption(" › ".join(ev.heading_path) + f" · `{ev.evidence_id}` · {ui.humanize(ev.status)}")
                    st.write(ev.excerpt)
                    if ev.is_precedent:
                        st.info("Historical precedent — not policy.")
            with st.expander("Retrieval diagnostics"):
                st.json({
                    "query": result.audit.query,
                    "filters": result.audit.filters,
                    "returned_chunk_ids": result.audit.returned_chunk_ids,
                    "scores": result.audit.scores,
                    "selected_evidence_ids": result.audit.selected_evidence_ids,
                    "mode": result.audit.mode,
                })

    with st.expander(f"Corpus · {len(docs)} documents"):
        st.dataframe(
            pd.DataFrame([
                {
                    "Title": d.title,
                    "Type": ui.humanize(d.document_type),
                    "Status": ui.humanize(d.status),
                    "Version": d.version,
                    "Effective": ui.fmt_date(d.effective_from) if d.effective_from else "",
                }
                for d in docs
            ]),
            hide_index=True, width="stretch",
        )


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
PAGES = {
    "request": st.Page(page_request, title="Procurement request", icon=":material/assignment:",
                       url_path="request", default=True),
    "review": st.Page(page_review, title="Contract & findings", icon=":material/plagiarism:", url_path="review"),
    "redlines": st.Page(page_redlines, title="Redlines & route", icon=":material/edit_note:", url_path="redlines"),
    "approvals": st.Page(page_approvals, title="Approvals", icon=":material/task_alt:", url_path="approvals"),
    "commercial": st.Page(live.page_commercial, title="Commercial", icon=":material/bar_chart:", url_path="commercial"),
    "obligations": st.Page(page_obligations, title="Obligations", icon=":material/event:", url_path="obligations"),
    "negotiation": st.Page(live.page_negotiation, title="Negotiation", icon=":material/forum:", url_path="negotiation"),
    "knowledge": st.Page(page_knowledge, title="Knowledge", icon=":material/menu_book:", url_path="knowledge"),
    "scorecard": st.Page(live.page_scorecard, title="Scorecard", icon=":material/verified:", url_path="scorecard"),
}

live.PAGES = PAGES
_ensure_state()
nav = st.navigation({
    "Intake": [PAGES["request"]],
    "Review": [PAGES["review"]],
    "Decide": [PAGES["redlines"], PAGES["approvals"], PAGES["commercial"]],
    "Manage": [PAGES["obligations"], PAGES["negotiation"], PAGES["knowledge"]],
    "Quality": [PAGES["scorecard"]],
})

st.logo(str(Path(__file__).resolve().parent / "assets" / "logo.svg"), size="large")
with st.sidebar:
    live.sidebar_controls()
    ui.section("Active document")
    pkg_now = st.session_state.get("pkg")
    if pkg_now:
        sup, rt = live.summary(pkg_now)
        st.markdown(f"**{pkg_now.file_name or pkg_now.run_id}**  \n{sup}  \nRoute: {rt}")
        st.html("".join(live.provenance_chips(pkg_now)))
    st.page_link(PAGES["review"], label="Change document", icon=":material/swap_horiz:")
    st.caption("Synthetic contracts and policies for demonstration only.")

nav.run()
