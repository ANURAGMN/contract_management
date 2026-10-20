"""Streamlit demo UI.

Modes (sidebar):
  1. Sample demo — load expected offline packages (no Gemini)
  2. Replay cache — last successful/cached run for an uploaded PDF
  3. Live review — calls Gemini (slow; may hit free-tier limits)

Run (from the repo root, inside the venv):
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.commercial import compare_demo_quotations  # noqa: E402
from app.demo_packages import list_samples, load_sample  # noqa: E402
from app.negotiation import (  # noqa: E402
    load_scenario,
    negotiation_brief,
    new_run,
    run_replay,
)
from app.approvals import (  # noqa: E402
    DecisionRejectedError,
    UnauthorizedDecisionError,
    build_queue,
    merge_queues,
    queue_to_markdown,
    record_decision,
)
from app.obligations import (  # noqa: E402
    build_register,
    confirm_obligation,
)
from app.procurement_requests import (  # noqa: E402
    EDITABLE_STATUSES,
    InvalidTransitionError,
    LOCKED_STATUSES,
    compare_request_to_package,
    find_duplicate_candidates,
    link_contract_run,
    load_request,
    mark_eligible_for_sourcing,
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
from app.runner import run_review  # noqa: E402
from app.schemas import ReviewPackage, ReviewRequest  # noqa: E402

DEFAULT_REQUEST_ID = "PR-2026-1042"

RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
DECISIONS = RUNS / "decisions.jsonl"

st.set_page_config(page_title="Supplier Contract Review", layout="wide")


@st.cache_resource
def _rag_bundle():
    return build_default_retriever()


def save_upload(uploaded) -> str:
    path = RUNS / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return str(path)


def cache_run(name: str, payload: str) -> None:
    (RUNS / f"cache_{name}.json").write_text(payload, encoding="utf-8")


def cached_run(name: str) -> str | None:
    p = RUNS / f"cache_{name}.json"
    return p.read_text(encoding="utf-8") if p.exists() else None


def log_decision(
    run_id: str, rule_id: str, action: str, proposed: str, final: str, reason: str
) -> None:
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


SEV_COLOR = {"high": "🔴", "medium": "🟠", "low": "🟡"}
ROUTE_LABEL = {
    "manual_review": "Manual review (fail closed)",
    "legal": "Legal",
    "procurement_owner": "Procurement owner",
    "low_risk_queue": "Low-risk queue",
}


# --------------------------------------------------------------------------- #
# Sidebar — mode picker
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("How to run")
    mode = st.radio(
        "Mode",
        [
            "Sample demo (no Gemini)",
            "Replay cache",
            "Live Gemini review",
        ],
        index=0,
        help="Use Sample demo for a reliable walkthrough. Live mode needs API quota.",
    )
    st.caption("PR-2026-1042 · budget ₹48 lakh · playbook demo-v1 · samples A–E")


(
    procurement_tab,
    review,
    findings,
    redlines_tab,
    commercial_tab,
    approvals_tab,
    obligations_tab,
    knowledge_tab,
    negotiation_tab,
    score,
) = st.tabs(
    [
        "Procurement request",
        "Review",
        "Findings",
        "Redlines & route",
        "Commercial",
        "Approvals",
        "Obligations",
        "Knowledge",
        "Negotiation",
        "Scorecard",
    ]
)

with procurement_tab:
    st.header("Procurement request")
    st.caption(
        "Upstream intake before a supplier contract exists. "
        "Status transitions are validated in code — the model cannot set status. "
        "Synthetic demo data — not a real purchase requisition."
    )
    if st.button("Load / reset sample PR-2026-1042"):
        st.session_state.procurement_req = reset_fixture_to_store(DEFAULT_REQUEST_ID)
        st.success("Reset working copy of PR-2026-1042 to DRAFT")

    if "procurement_req" not in st.session_state:
        try:
            st.session_state.procurement_req = load_request(DEFAULT_REQUEST_ID)
        except Exception:
            st.session_state.procurement_req = reset_fixture_to_store(DEFAULT_REQUEST_ID)

    req = st.session_state.procurement_req
    editable = req.status in EDITABLE_STATUSES
    locked = req.status in LOCKED_STATUSES
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Request ID", req.request_id)
    m2.metric("Status", req.status)
    m3.metric("Version", req.version)
    m4.metric("Budget confirmed", "yes" if req.budget_confirmed else "no")
    if locked:
        st.info(
            f"Status `{req.status}` is locked — draft edits and submit are disabled. "
            "Use **Load / reset sample** to start over."
        )
    elif not editable:
        st.caption("Field edits are only enabled in DRAFT or NEEDS_INFORMATION.")

    with st.form("pr_form"):
        st.subheader("Request details")
        title = st.text_input("Title", value=req.title)
        c1, c2 = st.columns(2)
        with c1:
            requesting_team = st.text_input("Requesting team", value=req.requesting_team)
            requester_role = st.text_input("Requester role", value=req.requester_role)
            business_owner = st.text_input("Business owner", value=req.business_owner)
            category = st.selectbox(
                "Category",
                ["saas_software", "services", "hardware", "consulting", "other"],
                index=["saas_software", "services", "hardware", "consulting", "other"].index(
                    req.category or "saas_software"
                ),
            )
            purchase = st.selectbox(
                "New purchase or renewal",
                ["new_purchase", "renewal"],
                index=0 if (req.new_purchase_or_renewal or "new_purchase") == "new_purchase" else 1,
            )
            number_of_users = st.number_input(
                "Number of users", min_value=0, value=int(req.number_of_users or 0)
            )
            desired_start = st.text_input(
                "Desired start date (YYYY-MM-DD)",
                value=str(req.desired_start_date or ""),
            )
        with c2:
            budget_amount = st.number_input(
                "Budget amount", min_value=0.0, value=float(req.budget_amount or 0)
            )
            budget_period = st.selectbox(
                "Budget period",
                ["annual", "monthly", "project"],
                index=["annual", "monthly", "project"].index(req.budget_period or "annual"),
            )
            budget_confirmed = st.checkbox("Budget confirmed", value=req.budget_confirmed)
            estimated_recurring = st.number_input(
                "Estimated recurring cost",
                min_value=0.0,
                value=float(req.estimated_recurring_cost or 0),
            )
            term_months = st.number_input(
                "Requested term (months)",
                min_value=0,
                value=int(req.requested_contract_term_months or 0),
            )
            required_sla = st.text_input("Required SLA", value=req.required_sla or "")
            data_sensitivity = st.selectbox(
                "Data sensitivity",
                ["public", "internal", "confidential", "personal_data", "regulated", "unknown"],
                index=["public", "internal", "confidential", "personal_data", "regulated", "unknown"].index(
                    req.data_sensitivity or "unknown"
                ),
            )
            security_review_required = st.checkbox(
                "Security review required", value=req.security_review_required
            )
            privacy_review_required = st.checkbox(
                "Privacy review required", value=req.privacy_review_required
            )
        justification = st.text_area("Business justification", value=req.business_justification)
        capabilities = st.text_input(
            "Requested capabilities (comma-separated)",
            value=", ".join(req.requested_capabilities),
        )
        integrations = st.text_input(
            "Integrations (comma-separated)",
            value=", ".join(req.integrations),
        )
        saved = st.form_submit_button("Save draft", disabled=not editable)

    if saved:
        from datetime import date as _date

        if not editable:
            st.error(f"Cannot save draft while status is {req.status}")
        else:
            start_val = None
            if desired_start.strip():
                try:
                    start_val = _date.fromisoformat(desired_start.strip())
                except ValueError:
                    st.error("Desired start date must be YYYY-MM-DD")
                    start_val = req.desired_start_date
            updates = {
                "title": title,
                "requesting_team": requesting_team,
                "requester_role": requester_role,
                "business_owner": business_owner,
                "category": category,
                "new_purchase_or_renewal": purchase,
                "number_of_users": int(number_of_users) or None,
                "desired_start_date": start_val,
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
                "requested_capabilities": [
                    x.strip() for x in capabilities.split(",") if x.strip()
                ],
                "integrations": [x.strip() for x in integrations.split(",") if x.strip()],
            }
            try:
                # Ensure store copy exists, then apply field updates
                if req.status in EDITABLE_STATUSES:
                    save_draft(load_request(req.request_id), actor="demo.user")
                st.session_state.procurement_req = update_request(
                    req.request_id, updates, actor="demo.user"
                )
                st.success("Draft saved")
                st.rerun()
            except InvalidTransitionError as exc:
                st.error(str(exc))
            except Exception:
                # First-time: persist then update
                try:
                    st.session_state.procurement_req = reset_fixture_to_store(req.request_id)
                    st.session_state.procurement_req = update_request(
                        req.request_id, updates, actor="demo.user"
                    )
                    st.success("Draft saved")
                    st.rerun()
                except InvalidTransitionError as exc:
                    st.error(str(exc))

    req = st.session_state.procurement_req
    st.subheader("Completeness & risks")
    if req.missing_fields:
        st.warning("Missing fields: " + ", ".join(req.missing_fields))
    else:
        st.success("Required fields present for submit.")
    for flag in req.risk_flags:
        st.write(f"- `{flag.code}` ({flag.severity}): {flag.message}")

    st.subheader("Routing recommendations")
    for r in req.routing_recommendations:
        st.write(f"- **{r.team}** — {r.reason} (`{r.reason_code}`)")

    dups = find_duplicate_candidates(req)
    if dups:
        st.subheader("Possible related requests (human confirm — never auto-merge)")
        for d in dups:
            st.write(
                f"- `{d.request_id}` · {d.title} · score {d.match_score} · "
                + ", ".join(d.match_reasons)
            )

    st.subheader("Sourcing checklist (from this request)")
    for item in sourcing_checklist(req):
        st.write(f"- {item}")

    can_submit = req.status in EDITABLE_STATUSES
    can_need_info = req.status not in LOCKED_STATUSES and req.status not in (
        "DRAFT",
        "NEEDS_INFORMATION",
    )
    can_source = req.status not in (
        "APPROVED_FOR_SOURCING",
        "SOURCING_IN_PROGRESS",
        "CONTRACT_REVIEW",
        "APPROVED_FOR_SIGNATURE",
        "REJECTED",
        "CANCELLED",
        "DRAFT",
    ) or req.status in (
        "PROCUREMENT_TRIAGE",
        "FINANCE_REVIEW",
        "SECURITY_REVIEW",
        "LEGAL_REVIEW",
        "SUBMITTED",
        "NEEDS_INFORMATION",
    )
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        if st.button("Submit request", type="primary", disabled=not can_submit):
            try:
                st.session_state.procurement_req = submit_request(
                    req.request_id, actor="demo.user"
                )
                st.rerun()
            except InvalidTransitionError as exc:
                st.error(str(exc))
    with a2:
        if st.button("Request more information", disabled=not can_need_info):
            try:
                st.session_state.procurement_req = request_more_information(
                    req.request_id, actor="demo.reviewer"
                )
                st.rerun()
            except InvalidTransitionError as exc:
                st.error(str(exc))
    with a3:
        if st.button(
            "Mark eligible for sourcing (demo)",
            disabled=req.status in LOCKED_STATUSES,
        ):
            try:
                st.session_state.procurement_req = mark_eligible_for_sourcing(
                    req.request_id, actor="demo.procurement"
                )
                st.rerun()
            except InvalidTransitionError as exc:
                st.error(str(exc))
    with a4:
        st.download_button(
            "Download JSON",
            data=req.model_dump_json(indent=2),
            file_name=f"{req.request_id}.json",
            mime="application/json",
        )

    with st.expander("Audit events"):
        for e in req.audit_events[-20:]:
            st.caption(
                f"{e.timestamp.isoformat()} · {e.action} · {e.actor} · "
                f"{e.before_status or ''}→{e.after_status or ''} · {e.detail}"
            )

with review:
    st.header("Supplier Contract Review")
    active_pr = st.session_state.get("procurement_req")
    if active_pr:
        st.caption(f"Linked requisition: `{active_pr.request_id}` · status `{active_pr.status}`")

    pkg_preview = st.session_state.get("pkg")
    if active_pr and pkg_preview:
        for flag in compare_request_to_package(active_pr, pkg_preview):
            if flag.severity == "high":
                st.error(f"`{flag.code}`: {flag.message}")
            else:
                st.warning(f"`{flag.code}`: {flag.message}")

    if mode == "Sample demo (no Gemini)":
        samples = list_samples()
        labels = {s["id"]: s["label"] for s in samples}
        choice = st.selectbox(
            "Expected-output sample",
            options=list(labels.keys()),
            format_func=lambda sid: labels[sid],
        )
        st.info(
            "These packages are committed fixtures that match the golden expectations "
            "(route, key findings, redlines, obligations, commercial table). "
            "No model calls."
        )
        if st.button("Load sample", type="primary"):
            pkg = load_sample(choice)
            pkg.request_id = (
                active_pr.request_id if active_pr else DEFAULT_REQUEST_ID
            )
            st.session_state.pkg = pkg
            try:
                link_contract_run(pkg.request_id, pkg.run_id, actor="demo.user")
                if active_pr and active_pr.request_id == pkg.request_id:
                    st.session_state.procurement_req = load_request(pkg.request_id)
            except Exception:
                pass
            st.success(
                f"Loaded **{pkg.sample_label or choice}** (offline sample) · "
                f"request `{pkg.request_id}`."
            )

    elif mode == "Replay cache":
        pdf = st.file_uploader("Supplier contract (PDF) — must match a prior cache", type="pdf")
        if pdf and cached_run(pdf.name):
            st.caption(f"Cache found: `runs/cache_{pdf.name}.json`")
        elif pdf:
            st.warning(
                f"No cache for `{pdf.name}`. Load a **Sample demo** instead, "
                "or run Live once when quota allows."
            )
        if pdf and st.button("Replay cached review", type="primary"):
            raw = cached_run(pdf.name)
            if raw:
                st.session_state.pkg = ReviewPackage.model_validate_json(raw)
                st.success("Loaded from cache (no live model calls).")
            else:
                st.error("Cache missing.")

    else:  # Live
        pdf = st.file_uploader("Supplier contract (PDF)", type="pdf")
        csv = st.file_uploader("Supplier performance (CSV, optional)", type="csv")
        supplier_id = st.text_input("Supplier ID (for the CSV cross-check)", value="")
        st.warning(
            "Live review calls Gemini 3× and often takes 3–8+ minutes. "
            "Free-tier limits may fail closed to manual review. Prefer Sample demo."
        )
        if pdf and st.button("Run live review", type="primary"):
            path = save_upload(pdf)
            csv_path = save_upload(csv) if csv else None
            with st.status("Reviewing with Gemini…", expanded=True) as status:
                status.write("Metadata → clause review → redlines (throttled)…")
                pkg = run_review(
                    ReviewRequest(
                        pdf_path=path,
                        csv_path=csv_path,
                        supplier_id=supplier_id or None,
                        playbook_version="demo-v1",
                        ask_gate=False,
                    )
                )
                status.update(label="Done", state="complete")
            cache_run(pdf.name, pkg.model_dump_json())
            st.session_state.pkg = pkg

    pkg = st.session_state.get("pkg")
    if pkg:
        if getattr(pkg, "is_sample", False):
            st.caption(f"📦 Offline sample · {pkg.sample_label or pkg.run_id}")
        st.subheader("Intake")
        c1, c2, c3 = st.columns(3)
        c1.metric("Run", pkg.run_id)
        c2.metric("Status", pkg.status)
        c3.metric("Playbook", pkg.playbook_version or "demo-v1")
        st.write("**Stage status:**", pkg.stage_status)
        st.write(
            f"**Route:** {ROUTE_LABEL.get(pkg.route.route, pkg.route.route)} · "
            f"**File:** {pkg.file_name or '—'}"
        )
        if pkg.metadata:
            st.subheader("Metadata (with page cites)")
            rows = []
            for name, field in pkg.metadata.__dict__.items():
                rows.append(
                    {
                        "field": name,
                        "value": field.value,
                        "page": field.evidence.page if field.evidence else None,
                        "quote": field.evidence.quote if field.evidence else "",
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

        st.subheader("Export")
        st.download_button(
            "Download review package (JSON)",
            data=pkg.model_dump_json(indent=2),
            file_name=f"{pkg.run_id or 'review'}.json",
            mime="application/json",
        )
        md_lines = [
            f"# Review package `{pkg.run_id}`",
            f"- File: {pkg.file_name}",
            f"- Status: {pkg.status}",
            f"- Route: {pkg.route.route}",
            "",
            "## Findings",
        ]
        for sf in pkg.findings:
            md_lines.append(
                f"- **{sf.grounded.finding.rule_id}** — {sf.status} ({sf.severity})"
            )
        st.download_button(
            "Download review summary (Markdown)",
            data="\n".join(md_lines),
            file_name=f"{pkg.run_id or 'review'}.md",
            mime="text/markdown",
        )

with findings:
    pkg = st.session_state.get("pkg")
    if not pkg:
        st.info("Load a sample or run a review first.")
    else:
        if pkg.security.any():
            st.error("Security: text aimed at an AI reviewer was found and flagged.")
            for e in pkg.security.pre + pkg.security.model:
                st.write(f"> p{e.page}: {e.quote}")
        st.subheader("Findings")
        for sf in pkg.findings:
            if sf.status in ("preferred",) and not sf.grounded.finding.evidence:
                continue
            badge = SEV_COLOR.get(sf.severity, "")
            with st.expander(
                f"{badge} {sf.grounded.finding.rule_id} — {sf.status} ({sf.severity})"
            ):
                st.caption(sf.rule_trace)
                for e in sf.grounded.finding.evidence:
                    st.write(f"> p{e.page}: {e.quote}")
                if sf.grounded.warnings:
                    st.caption("warnings: " + "; ".join(sf.grounded.warnings))

        st.subheader("Contract vs knowledge (RAG)")
        st.caption(
            "Contract quote on the left · approved demo policy on the right. "
            "Lexical retrieval only — illustrative policy, not legal advice."
        )
        retriever, _, _ = _rag_bundle()
        for cmp_ in comparisons_for_package_findings(pkg.findings, retriever=retriever):
            with st.expander(
                f"{cmp_.rule_id or cmp_.intent} — "
                f"{'abstained' if cmp_.abstained else 'policy evidence retrieved'}",
                expanded=cmp_.intent in ("renewal", "sla"),
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Contract evidence**")
                    if cmp_.contract_quote:
                        st.write(
                            f"> p{cmp_.contract_page or '?'}: {cmp_.contract_quote}"
                        )
                    else:
                        st.caption("No contract quote on this finding.")
                with c2:
                    st.markdown("**Knowledge evidence**")
                    if cmp_.abstained or not cmp_.knowledge_evidence:
                        st.warning(cmp_.assessment)
                    else:
                        for ev in cmp_.knowledge_evidence[:3]:
                            st.markdown(
                                f"**{ev.title}** · v{ev.version} · `{ev.status}`"
                            )
                            st.caption(" > ".join(ev.heading_path))
                            st.write(ev.excerpt)
                            st.caption(
                                f"relevance score (not confidence): {ev.relevance_score} · "
                                f"`{ev.evidence_id}`"
                            )
                st.markdown("**Assessment:** " + cmp_.assessment)
                st.markdown("**Action:** " + cmp_.action)

        if pkg.dropped:
            st.caption(f"{len(pkg.dropped)} ungrounded item(s) dropped before display.")

with redlines_tab:
    pkg = st.session_state.get("pkg")
    if not pkg:
        st.info("Load a sample or run a review first.")
    else:
        st.subheader("Approval route")
        st.success(
            f"Recommended route: **{ROUTE_LABEL.get(pkg.route.route, pkg.route.route)}**"
        )
        st.caption("The model never decides the route; this is deterministic.")
        for t in pkg.route.trace:
            st.write(f"- {t}")

        st.subheader("Redlines")
        if not pkg.redlines:
            st.caption("No walk-away redlines in this package.")
        for i, cr in enumerate(pkg.redlines):
            r = cr.redline
            with st.container(border=True):
                st.markdown(
                    f"**{r.rule_id}** — "
                    f"{'✅ passed checks' if cr.passed else '⚠️ needs manual drafting'}"
                )
                col1, col2 = st.columns(2)
                col1.caption("Original")
                col1.write(r.original.quote)
                col2.caption("Proposed")
                col2.write(r.proposed_text)
                st.caption(r.rationale)
                if cr.problems:
                    st.warning("; ".join(cr.problems))
                action = st.radio(
                    "Decision",
                    ["Accept", "Edit", "Reject", "Escalate"],
                    horizontal=True,
                    key=f"act{i}",
                )
                edited = r.proposed_text
                if action == "Edit":
                    edited = st.text_area(
                        "Edited text", value=r.proposed_text, key=f"edit{i}"
                    )
                reason = st.text_input("Reason (optional)", key=f"rsn{i}")
                if st.button("Log decision", key=f"log{i}"):
                    log_decision(
                        pkg.run_id,
                        r.rule_id,
                        action.lower(),
                        r.proposed_text,
                        edited,
                        reason,
                    )
                    st.toast("Decision logged to runs/decisions.jsonl")

        st.subheader("Obligations")
        if pkg.obligations:
            st.dataframe(
                [
                    {
                        "kind": o.kind,
                        "due": str(o.due) if o.due else "",
                        "urgent": "⚠️" if o.urgent else "",
                        "uncertain": "?" if o.uncertain else "",
                        "owner": o.owner,
                        "amount_inr": o.amount_inr,
                    }
                    for o in pkg.obligations
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No obligations extracted.")
        if pkg.credit_claim:
            cc = pkg.credit_claim
            st.info(
                f"SLA cross-check: claimable credits INR {cc.total_credit_inr:,.0f} "
                f"over {cc.months_considered} month(s)"
                f"{' (capped)' if cc.capped else ''}."
            )

with commercial_tab:
    pkg = st.session_state.get("pkg")
    active_pr = st.session_state.get("procurement_req")
    cmp = None
    if pkg and getattr(pkg, "quotation_comparison", None) is not None:
        cmp = pkg.quotation_comparison
    else:
        cmp = compare_demo_quotations(include_baseline=True)

    st.subheader("Quotation comparison")
    st.caption(
        "Demo quotes (CloudSoft / TechNova / DataSphere) are for collaboration SaaS "
        "under **PR-2026-1042** — not for unrelated BPO/support sample contracts."
    )
    if pkg and pkg.file_name and "contract_e" in (pkg.file_name or ""):
        st.warning(
            "Loaded package is Sample E (TalkBridge BPO). These SaaS quotes are not "
            "comparable to that contract — treat commercial rows as the PR scenario only."
        )
    if active_pr and active_pr.category == "saas_software" and pkg and pkg.metadata:
        supplier = (pkg.metadata.supplier.value or "").lower()
        if "bpo" in supplier or "talkbridge" in supplier:
            st.error(
                "Category mismatch: requisition is SaaS collaboration; linked contract "
                "looks like BPO/support."
            )
    st.write(
        f"Requisition **{cmp.requisition_id}** · budget INR {cmp.budget_inr:,.0f} · "
        f"required uptime {cmp.required_uptime_percent}% · "
        f"horizon {cmp.comparison_horizon_months} months"
    )
    st.dataframe(
        [
            {
                "supplier": r.supplier_name,
                "annual": r.annual_recurring_fee,
                "one_time": r.one_time_fee,
                "per_user": r.cost_per_user,
                "budget_var": r.budget_variance,
                "uptime": r.uptime_percent,
                "sla_ok": r.sla_compliant,
                "term_mo": r.term_months,
                "term_mismatch": r.term_mismatch,
                "lowest_fee": r.is_lowest_annual_fee,
                "scope": r.scope_comparability,
            }
            for r in cmp.rows
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Lowest fee is highlighted for visibility only — not an automatic recommendation."
    )
    for r in cmp.rows:
        if r.warnings:
            st.write(f"**{r.supplier_name}:** " + "; ".join(r.warnings))
    for w in cmp.warnings:
        st.warning(w)
    for n in cmp.notes:
        st.caption(n)

with approvals_tab:
    st.header("Approvals & exception queue")
    st.caption(
        "Routing is deterministic policy (not an LLM guess). "
        "Mock role selector is for demo only — not production auth. "
        "No agent may set a task to approved."
    )
    pkg = st.session_state.get("pkg")
    if not pkg:
        st.info("Load a sample (or run a review) first.")
    else:
        mock_role = st.selectbox(
            "Mock reviewer role (demo)",
            options=[
                "procurement",
                "legal",
                "security_privacy",
                "finance",
                "business_owner",
                "admin",
                "viewer",
            ],
            index=1,
            help="Service layer still rejects unauthorized decisions.",
        )
        reviewer = st.text_input("Reviewer name", value="demo.reviewer")
        contract_id = (pkg.file_name or pkg.run_id or "demo").replace(".pdf", "")
        if (
            "approval_queue" not in st.session_state
            or st.session_state.get("approval_contract_id") != contract_id
        ):
            fresh = build_queue(pkg, contract_id=contract_id)
            prev = st.session_state.get("approval_queue")
            if prev and prev.contract_id == contract_id:
                fresh = merge_queues(prev, fresh)
            st.session_state.approval_queue = fresh
            st.session_state.approval_contract_id = contract_id

        q = st.session_state.approval_queue
        c1, c2, c3 = st.columns(3)
        c1.metric("Open tasks", len(q.open_tasks()))
        c2.metric("All tasks", len(q.tasks))
        c3.metric("Audit events", len(q.audit))
        st.caption(f"Policy `{q.policy_version}` · contract `{q.contract_id}`")

        rows = []
        for t in q.sorted_queue() or q.tasks:
            rows.append(
                {
                    "task": t.task_id,
                    "title": t.title[:80],
                    "severity": t.severity,
                    "status": t.status,
                    "teams": ", ".join(t.required_teams),
                    "findings": ", ".join(t.related_finding_ids),
                    "policy": ", ".join(t.matched_policy_rule_ids),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

        if not q.tasks:
            st.success("No exception tasks — preferred/fallback terms only.")
        else:
            tid = st.selectbox("Task", [t.task_id for t in q.tasks])
            chosen = next(t for t in q.tasks if t.task_id == tid)
            with st.expander("Task detail", expanded=True):
                st.write("**Title:**", chosen.title)
                st.write("**Status:**", chosen.status)
                st.write("**Teams:**", ", ".join(chosen.required_teams))
                st.write("**Reason codes:**", ", ".join(chosen.reason_codes))
                st.write("**Policy rules:**", ", ".join(chosen.matched_policy_rule_ids))
                st.write("**Evidence IDs:**", ", ".join(chosen.source_evidence_ids) or "—")
                if chosen.decision:
                    st.write(
                        f"**Decision:** `{chosen.decision}` by {chosen.decided_by} — "
                        f"{chosen.decision_reason}"
                    )

            decision = st.selectbox(
                "Decision",
                [
                    "approve",
                    "reject",
                    "request_changes",
                    "need_information",
                    "waive_with_authorization",
                ],
            )
            reason = st.text_area("Reason", value="")
            auth_ref = st.text_input(
                "Authorization reference (required for waive)",
                value="",
            )
            if st.button("Record decision", type="primary"):
                try:
                    st.session_state.approval_queue = record_decision(
                        q,
                        tid,
                        reviewer=reviewer,
                        role=mock_role,  # type: ignore[arg-type]
                        decision=decision,  # type: ignore[arg-type]
                        reason=reason,
                        authorization_reference=auth_ref or None,
                    )
                    st.success(f"Recorded `{decision}` on `{tid}`")
                    st.rerun()
                except UnauthorizedDecisionError as exc:
                    st.error(f"Unauthorized: {exc}")
                except DecisionRejectedError as exc:
                    st.error(str(exc))
                except KeyError as exc:
                    st.error(str(exc))

        st.download_button(
            "Download approval queue (JSON)",
            data=st.session_state.approval_queue.model_dump_json(indent=2),
            file_name=f"{contract_id}_approvals.json",
            mime="application/json",
        )
        st.download_button(
            "Download approval queue (Markdown)",
            data=queue_to_markdown(st.session_state.approval_queue),
            file_name=f"{contract_id}_approvals.md",
            mime="text/markdown",
        )

with obligations_tab:
    st.header("Obligations & Renewals")
    st.caption(
        "Notice deadline ≠ contract expiry. Dates are computed in Python; "
        "ambiguous wording stays unresolved until a human confirms."
    )
    pkg = st.session_state.get("pkg")
    if not pkg or not pkg.metadata:
        st.info("Load a sample (or run a review) with metadata first.")
    else:
        today = date(2026, 10, 19)  # fixed demo "today" for reproducible reminders
        contract_id = (pkg.file_name or pkg.run_id or "demo").replace(".pdf", "")
        if (
            "obl_register" not in st.session_state
            or st.session_state.get("obl_contract_id") != contract_id
        ):
            st.session_state.obl_register = build_register(
                pkg.metadata, contract_id=contract_id, today=today
            )
            st.session_state.obl_contract_id = contract_id

        reg = st.session_state.obl_register
        due_14 = reg.upcoming(14)
        due_90 = reg.upcoming(90)
        overdue = [
            o
            for o in reg.obligations
            if o.due_date is not None and o.due_date < today and o.status not in ("complete", "not_applicable")
        ]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Obligations", len(reg.obligations))
        c2.metric("Needs confirmation", sum(1 for o in reg.obligations if o.needs_human_confirmation))
        c3.metric("Due in 14 days", len(due_14))
        c4.metric("Due in 90 days", len(due_90))
        if overdue:
            st.error(
                "Overdue: "
                + "; ".join(f"{o.title} ({o.due_date})" for o in overdue[:5])
            )
        elif due_14:
            st.warning(
                "Due within 14 days: "
                + "; ".join(
                    f"{o.title} — **{o.due_date}** ({(o.due_date - today).days}d)"
                    for o in due_14
                )
            )

        st.subheader("Date anchors")
        st.dataframe(
            [
                {
                    "type": a.anchor_type,
                    "value": str(a.value) if a.value else "",
                    "confidence": a.confidence,
                    "page": a.source_page,
                    "quote": (a.source_quote or "")[:80],
                }
                for a in reg.anchors
            ],
            use_container_width=True,
            hide_index=True,
        )

        filter_conf = st.checkbox("Only needs human confirmation", value=False)
        rows = []
        for o in reg.obligations:
            if filter_conf and not o.needs_human_confirmation:
                continue
            days_left = None
            urgency = ""
            if o.due_date is not None:
                days_left = (o.due_date - today).days
                if days_left < 0:
                    urgency = "OVERDUE"
                elif days_left <= 14:
                    urgency = f"DUE SOON ({days_left}d)"
                elif o.urgent:
                    urgency = "urgent"
            rows.append(
                {
                    "urgency": urgency,
                    "id": o.obligation_id,
                    "type": o.obligation_type,
                    "title": o.title,
                    "due": str(o.due_date) if o.due_date else "",
                    "days_left": days_left if days_left is not None else "",
                    "trigger/term_end": str(o.trigger_date) if o.trigger_date else "",
                    "status": o.status,
                    "owner": o.owner_role or "",
                    "criticality": o.criticality,
                    "needs_confirm": o.needs_human_confirmation,
                    "calc": (o.date_calculation_explanation or "")[:100],
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.subheader("Confirm dates / owner")
        ids = [o.obligation_id for o in reg.obligations]
        pick = st.selectbox("Obligation", ids)
        owner = st.text_input("Owner user (demo)", value="demo.reviewer")
        if st.button("Confirm selected obligation"):
            st.session_state.obl_register = confirm_obligation(
                reg, pick, owner_user=owner
            )
            st.success(f"Confirmed `{pick}`")

        chosen = next(o for o in st.session_state.obl_register.obligations if o.obligation_id == pick)
        with st.expander("Detail / source evidence"):
            st.write(chosen.description or chosen.title)
            st.write("**Calculation:**", chosen.date_calculation_explanation)
            st.write("**Evidence required:**", chosen.completion_evidence_required)
            for q, p in zip(chosen.source_quotes, chosen.source_pages or [None] * len(chosen.source_quotes)):
                st.write(f"> p{p}: {q}")

        st.download_button(
            "Download obligation register (JSON)",
            data=st.session_state.obl_register.model_dump_json(indent=2),
            file_name=f"{contract_id}_obligations.json",
            mime="application/json",
        )

with knowledge_tab:
    st.header("Knowledge & Evidence")
    st.caption(
        "Synthetic demo corpus · lexical retrieval (Phase 1). "
        "Not real company policy or legal advice. "
        "Relevance scores are retrieval ranks, not calibrated confidence."
    )
    retriever, corpus_status, docs = _rag_bundle()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", corpus_status.document_count)
    c2.metric("Approved (non-hist)", corpus_status.approved_policy_count)
    c3.metric("Chunks", corpus_status.chunk_count)
    c4.metric("Mode", corpus_status.mode)
    st.caption(
        f"`{corpus_status.corpus_id}` · indexed {corpus_status.last_indexed_at.isoformat()} · "
        f"{corpus_status.label}"
    )

    with st.expander("Corpus documents"):
        st.dataframe(
            [
                {
                    "id": d.document_id,
                    "title": d.title,
                    "type": d.document_type,
                    "status": d.status,
                    "version": d.version,
                    "from": str(d.effective_from or ""),
                }
                for d in docs
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Ask the knowledge base")
    st.caption("Extractive answers with citations only — abstains when evidence is weak.")
    examples = [
        "What does our policy require for auto-renewal notice?",
        "What is the minimum uptime in the SLA policy?",
        "Which team must approve a missing DPA?",
        "What is our policy for lunar mineral royalties?",
    ]
    pick = st.selectbox("Example question", examples)
    custom = st.text_input("Or type your own question", value="")
    question = custom.strip() or pick
    if st.button("Retrieve", type="primary", key="ask_kb"):
        result = ask_knowledge_base(question, retriever)
        st.session_state.ask_result = result

    result = st.session_state.get("ask_result")
    if result:
        if result.abstained:
            st.warning(result.answer)
        else:
            st.success("Retrieved approved knowledge (illustrative).")
            st.text(result.answer)
        st.subheader("Retrieved sources")
        if not result.evidence:
            st.caption("No chunks selected.")
        for ev in result.evidence:
            with st.expander(
                f"{ev.title} · score {ev.relevance_score} · `{ev.evidence_id}`"
            ):
                st.write("**Section:**", " > ".join(ev.heading_path))
                st.write("**Status/version:**", f"{ev.status} / {ev.version}")
                st.write(ev.excerpt)
                st.caption(ev.source_path)
                if ev.is_precedent:
                    st.info("Historical precedent — not normative policy.")
        with st.expander("Retrieval diagnostics (demo)"):
            st.json(
                {
                    "query": result.audit.query,
                    "filters": result.audit.filters,
                    "returned_chunk_ids": result.audit.returned_chunk_ids,
                    "scores": result.audit.scores,
                    "selected_evidence_ids": result.audit.selected_evidence_ids,
                    "mode": result.audit.mode,
                }
            )

with negotiation_tab:
    st.header("Negotiation Simulator")
    st.caption(
        "Simulated mailbox only — no real email is sent. "
        "Each outbound message in the replay was human-approved in the fixture."
    )
    scenario = load_scenario()
    col_l, col_c, col_r = st.columns([1.1, 1.4, 1.1])

    if "neg_state" not in st.session_state:
        st.session_state.neg_state = None

    with col_l:
        st.subheader("Brief")
        if st.button("Load CloudSoft replay (offline)", type="primary"):
            st.session_state.neg_state = run_replay()
            st.success("Replay complete — agreement in principle (non-binding).")
        if st.button("Reset negotiation"):
            st.session_state.neg_state = None

        neg = st.session_state.neg_state
        brief = negotiation_brief(scenario, neg if neg is not None else new_run(scenario))
        st.metric("Status", brief["status"])
        st.metric("Round", brief["round"])
        st.write(f"**Budget ceiling:** INR {brief['budget_ceiling_inr']:,.0f}")
        st.write(f"**Target (illustrative):** INR {brief.get('target_price_inr') or 0:,.0f}")
        st.write(f"**Floor:** INR {brief['price_floor_inr']:,.0f}")
        st.write(
            f"**Current supplier fee:** INR {brief.get('current_supplier_fee_inr') or 0:,.0f}"
        )
        st.write("**Must-haves**")
        for m in brief["must_haves"]:
            st.write(f"- {m['label']}: target {m['target']}")
        st.write("**Open issues:**", ", ".join(brief["open_issues"]) or "—")
        st.write("**Agreed:**", ", ".join(brief["agreed_terms"]) or "—")
        if brief.get("strategy_summary"):
            st.info(brief["strategy_summary"])
        st.caption(brief["non_binding_notice"])

    with col_c:
        st.subheader("Email thread (simulated)")
        neg = st.session_state.neg_state
        if not neg or not neg.thread:
            st.info("Load the CloudSoft offline replay to see the thread.")
        else:
            for msg in neg.thread:
                who = (
                    "Buyer → Supplier"
                    if msg.direction == "buyer_to_supplier"
                    else "Supplier → Buyer"
                )
                with st.container(border=True):
                    st.markdown(
                        f"**{who}** · round {msg.round_number} · "
                        f"`{msg.approval_status}` · simulated={msg.simulated}"
                    )
                    st.caption(msg.subject)
                    st.write(msg.body)
                    if msg.linked_term_keys:
                        st.caption("Terms: " + ", ".join(msg.linked_term_keys))

    with col_r:
        st.subheader("Control room")
        neg = st.session_state.neg_state
        if not neg:
            st.caption("No active run.")
        else:
            st.write(f"**Run:** `{neg.run_id}`")
            st.write(f"**Scenario:** `{neg.scenario_id}`")
            if neg.pending_action:
                st.write(f"**Last action:** {neg.pending_action.action_type}")
                st.caption(neg.pending_action.rationale)
            st.write("**Concessions (supplier):**")
            for c in neg.concessions_supplier or ["—"]:
                st.write(f"- {c}")
            st.write("**Audit events:**", len(neg.events))
            with st.expander("Event log"):
                for e in neg.events:
                    st.write(
                        f"`{e.state_before}` → `{e.state_after}` · {e.actor} · {e.reason}"
                    )
            if neg.status == "AGREEMENT_IN_PRINCIPLE":
                st.success("Commercial agreement in principle (non-binding).")
                st.warning("Liability remains open for Legal — not auto-accepted.")

with score:
    st.subheader("Scorecard")
    sc_mode = st.radio(
        "Which scorecard?",
        [
            "Offline samples vs golden (no Gemini)",
            "Last live Gemini eval run",
        ],
        index=0,
        help="Sample demo packages are scored offline. Live scorecard.json may hard-fail on quota or model drift.",
    )
    if sc_mode.startswith("Offline"):
        sc_path = ROOT / "evals" / "scorecard_samples.json"
        if not sc_path.exists():
            st.info("Generating offline scorecard…")
            import subprocess

            subprocess.run(
                [sys.executable, str(ROOT / "evals" / "score_samples.py")],
                check=False,
            )
        if sc_path.exists():
            sc = json.loads(sc_path.read_text(encoding="utf-8"))
            st.caption(
                f"Mode: `{sc.get('mode')}` · generated {sc.get('generated_at', '?')} · "
                f"{sc.get('note', '')}"
            )
            st.write("PASS ✅" if sc.get("passed") else "HARD FAIL ❌")
            st.dataframe(
                [
                    {
                        "contract": r["contract"],
                        "sample": r.get("sample_id", ""),
                        "high recall": r["high_severity_recall"]["label"],
                        "overall": r["overall_recall"]["label"],
                        "obligations": r.get("obligations", {}).get("label", ""),
                        "security": r.get("security", {}).get("label", ""),
                        "precision (min)": r["precision_min"],
                        "grounding (mean)": r["grounding_mean"],
                        "route ok": r["routing_ok_all_runs"],
                        "route got": ",".join(r.get("route_got") or []),
                        "route want": r.get("route_want", ""),
                    }
                    for r in sc["contracts"]
                ],
                use_container_width=True,
                hide_index=True,
            )
            if sc.get("hard_fails"):
                st.error("\n".join(sc["hard_fails"]))
            if st.button("Refresh offline scorecard"):
                import subprocess

                subprocess.run(
                    [sys.executable, str(ROOT / "evals" / "score_samples.py")],
                    check=False,
                )
                st.rerun()
        else:
            st.warning("Could not build offline scorecard. Run `python evals/score_samples.py`.")
    else:
        sc_path = ROOT / "evals" / "scorecard.json"
        if not sc_path.exists():
            st.info(
                "No live scorecard yet. Run `python evals/run_evals.py --runs 3` "
                "(needs API quota). This is **not** the Sample demo mode."
            )
        else:
            sc = json.loads(sc_path.read_text(encoding="utf-8"))
            st.caption(
                f"Live Gemini eval · runs_per_contract={sc.get('runs_per_contract')} · "
                f"sample={sc.get('sample')} — may disagree with offline Sample demo packages."
            )
            st.write("PASS ✅" if sc.get("passed") else "HARD FAIL ❌")
            st.dataframe(
                [
                    {
                        "contract": r["contract"],
                        "high recall": r["high_severity_recall"]["label"],
                        "overall": r["overall_recall"]["label"],
                        "precision (min)": r["precision_min"],
                        "grounding (mean)": r["grounding_mean"],
                        "route ok": r["routing_ok_all_runs"],
                        "route got": ",".join(r.get("route_got") or []),
                        "route want": r.get("route_want", ""),
                    }
                    for r in sc["contracts"]
                ],
                use_container_width=True,
                hide_index=True,
            )
            if sc.get("hard_fails"):
                st.error("\n".join(sc["hard_fails"]))
