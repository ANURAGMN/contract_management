"""Live AI vs Replay: the mode switch and the pages that depend on it.

Live AI    every AI step calls Gemini now; output is labelled with the model and
           timing, and each run is recorded to data/recordings/.
Replay     plays back those recordings (real past live runs). Hand-built
           reference fixtures are always labelled as such — never as AI output.

Presentation only: business logic stays in app/.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

import theme as ui
from app import ai_trace, recordings
from app.agents import judge_available as gemini_available
from app.commercial import load_demo_quotations
from app.demo_packages import list_samples, load_sample
from app.flow_links import comparison_for_request, contract_vs_quotes, scenario_from_quote
from app.negotiation import NegotiationError, negotiation_brief, new_run, run_replay
from app.negotiation_runs import AgentRun, load_recorded, record as record_negotiation, run_full, step
from app.procurement_requests import (
    EDITABLE_STATUSES,
    InvalidTransitionError,
    create_request,
    link_contract_run,
    load_request,
    update_request,
)
from app.schemas import ReviewPackage, ReviewRequest

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = ROOT / "data" / "contracts"


def _supplier_ref(pdf_path: str) -> Optional[str]:
    """'Supplier reference: TALKBRIDGE' in the contract text, if it is in the bundled performance CSV."""
    import re

    try:
        from app import steps

        doc = steps.intake(ReviewRequest(pdf_path=pdf_path, playbook_version="demo-v1"))
    except Exception:
        return None
    m = re.search(r"Supplier reference:\s*([A-Z0-9_-]+)", " ".join(p.text for p in doc.pages))
    if not m:
        return None
    csv = ROOT / "data" / "performance" / "supplier_performance.csv"
    ids = {line.split(",")[0] for line in csv.read_text(encoding="utf-8").splitlines()[1:]} if csv.exists() else set()
    return m.group(1) if m.group(1) in ids else None
RUNS = ROOT / "runs"

LIVE, REPLAY = "Live AI", "Replay"
MODE_KEY = "ai_mode"
PAGES: dict = {}  # filled by streamlit_app so pages can link to each other

AGENT_NAMES = {
    "ContractMetadata": "Metadata extractor",
    "ClauseFindings": "Clause reviewer",
    "Redlines": "Redline drafter",
    "IntakeDraft": "Intake assistant",
    "StrategyAgentOutput": "Negotiation strategist",
    "EmailDraftAgentOutput": "Email drafter",
    "SupplierSimAgentOutput": "Supplier simulator",
    "AnalyzeReplyAgentOutput": "Reply analyst",
    "NextMoveAgentOutput": "Next-move planner",
}


# --------------------------------------------------------------------------- #
# Mode
# --------------------------------------------------------------------------- #
def mode() -> str:
    return st.session_state.get(MODE_KEY) or LIVE


def is_live() -> bool:
    return mode() == LIVE


@st.cache_data(ttl=60)
def _ready() -> bool:
    return gemini_available()


def sidebar_controls() -> None:
    ui.section("AI mode")
    st.segmented_control("AI mode", [LIVE, REPLAY], default=LIVE, key=MODE_KEY, label_visibility="collapsed",
                         help="Live AI calls Gemini and records each run. Replay plays recorded runs back "
                              "with no model calls.")
    if is_live():
        if _ready():
            used = sum(ai_trace.requests_today().values())
            cap = ai_trace.FREE_TIER_DAILY_REQUESTS
            st.progress(min(used / cap, 1.0), text=f"Gemini requests today: {used} of ~{cap} (free tier)")
        else:
            st.caption(":orange[No Gemini key found — live steps are disabled. "
                       "Add GOOGLE_API_KEY to contract_review/.env.]")
    else:
        st.caption("Playing back recorded live runs. No model calls.")


def no_key_banner() -> None:
    ui.banner("Live AI needs a Gemini API key",
              "Set GOOGLE_API_KEY in contract_review/.env (or the app's Secrets when hosted), or switch the "
              "sidebar to Replay to play back recorded runs.", "amber", "!")


def calls_caption(calls: list[dict]) -> str:
    if not calls:
        return "No model calls."
    parts = []
    for c in calls:
        name = AGENT_NAMES.get(c.get("agent", ""), c.get("agent", ""))
        status = f"{c.get('model') or '—'} · {c.get('seconds', 0):.1f} s" if c.get("ok") else \
            f"failed · {c.get('error', '')[:80]}"
        parts.append(f"{name}: {status}")
    return " · ".join(parts)


def _when(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        return ui.fmt_ts(datetime.fromisoformat(iso))
    except ValueError:
        return iso


def provenance_chips(pkg: ReviewPackage) -> list[str]:
    """Label where a review came from. Never lets a fixture pass as AI output."""
    doc = st.session_state.get("active_doc", "")
    # getattr: packages created before a restart (or by older code) may lack these fields
    provenance = getattr(pkg, "provenance", None)
    if provenance == "live":
        s = ai_trace.summarize([ai_trace.AICall.model_validate(c) for c in getattr(pkg, "ai_calls", [])])
        model = ", ".join(s["models"]) or "no model answered"
        if doc.startswith("rec:"):
            return [ui.chip(f"Replay of live run · {_when(getattr(pkg, 'reviewed_at', None))}", "purple"),
                    ui.chip(f"{model} · {s['ok']}/{s['calls']} calls", "gray")]
        return [ui.chip(f"Generated live · {model}", "blue"),
                ui.chip(f"{s['ok']}/{s['calls']} calls · {s['seconds']} s", "gray")]
    if getattr(pkg, "is_sample", False) or provenance == "fixture":
        return [ui.chip("Reference fixture — hand-built, not AI output", "gray")]
    return [ui.chip("Saved package — origin not recorded", "gray")]


# --------------------------------------------------------------------------- #
# Contract & findings: where the document comes from
# --------------------------------------------------------------------------- #
def activate(pkg: ReviewPackage, doc_id: str) -> None:
    """Make a review the active document for every page."""
    active_pr = st.session_state.get("procurement_req")
    pkg.request_id = active_pr.request_id if active_pr else pkg.request_id
    st.session_state.pkg = pkg
    st.session_state.active_doc = doc_id
    for k in ("approval_queue", "approval_contract_id", "obl_register", "obl_contract_id"):
        st.session_state.pop(k, None)
    if pkg.request_id:
        try:
            link_contract_run(pkg.request_id, pkg.run_id, actor="demo.user")
            st.session_state.procurement_req = load_request(pkg.request_id)
        except Exception:
            pass


def summary(pkg: ReviewPackage) -> tuple[str, str]:
    supplier = (pkg.metadata.supplier.value if pkg.metadata and pkg.metadata.supplier else None) or "—"
    route = ui.ROUTES.get(pkg.route.route, (ui.humanize(pkg.route.route),))[0]
    return supplier, route


def _save_upload(uploaded) -> str:
    RUNS.mkdir(exist_ok=True)
    path = RUNS / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return str(path)


def _live_review_panel() -> None:
    ready = _ready()
    if not ready:
        no_key_banner()
    st.caption("Three Gemini agents extract metadata, review every clause against the playbook and draft "
               "redlines. Grounding, severity, routing, dates and money stay deterministic. Every run is "
               "recorded for Replay; on any failure it falls back to manual review instead of partial results.")
    how = st.segmented_control("Contract", ["Use a bundled contract", "Upload a PDF"],
                               default="Use a bundled contract", key="live_input") or "Use a bundled contract"
    pdf_path = name = None
    if how == "Upload a PDF":
        pdf = st.file_uploader("Supplier contract (PDF, text-based)", type="pdf")
        if pdf:
            pdf_path, name = _save_upload(pdf), pdf.name
    else:
        bundled = sorted(p.name for p in CONTRACTS_DIR.glob("*.pdf"))
        name = st.selectbox("Bundled contract", bundled) if bundled else None
        pdf_path = str(CONTRACTS_DIR / name) if name else None
    bundled_csv = ROOT / "data" / "performance" / "supplier_performance.csv"
    ref = _supplier_ref(pdf_path) if pdf_path else None
    with st.expander("Supplier performance data (optional)", expanded=bool(ref)):
        use_bundled = st.checkbox(
            "Use the bundled demo performance data", value=bool(ref), key="live_perf_bundled",
            help="data/performance/supplier_performance.csv — the last three months of on-time % per supplier, "
                 "used for the service-credit cross-check.",
            disabled=not bundled_csv.exists())
        csv = None if use_bundled else st.file_uploader("Performance CSV for the SLA service-credit check",
                                                        type="csv")
        supplier_id = st.text_input("Supplier ID in the CSV", value=ref or "",
                                    help="Taken from the 'Supplier reference' line in the contract when present.")
    st.caption("Uses about 3 Gemini requests.")
    if st.button("Run live review", type="primary", icon=":material/play_arrow:",
                 disabled=not (ready and pdf_path)):
        from app.runner import record_review, run_review  # lazy: pulls in the model SDK

        with st.status(f"Reviewing {name} with Gemini…", expanded=True) as status:
            status.write("Intake → metadata ‖ clause review → grounding → rules → route → redlines")
            pkg = run_review(ReviewRequest(
                pdf_path=pdf_path,
                csv_path=str(bundled_csv) if use_bundled else (_save_upload(csv) if csv else None),
                supplier_id=supplier_id or None,
                playbook_version="demo-v1",
                ask_gate=False,
            ))
            for c in pkg.ai_calls:
                status.write(f"{AGENT_NAMES.get(c['agent'], c['agent'])}: "
                             + (f"{c['model']} · {c['seconds']:.1f} s" if c["ok"] else f"failed — {c['error']}"))
            done = pkg.status == "complete"
            status.update(label="Review finished" if done else "Sent to manual review",
                          state="complete" if done else "error")
        rec = record_review(pkg)
        activate(pkg, f"live:{rec.recording_id}")
        st.rerun()


def _library() -> list[dict]:
    docs: list[dict] = []
    for rec in recordings.list_recordings("review"):
        try:
            pkg = ReviewPackage.model_validate(rec.payload)
        except Exception:
            continue
        supplier, route = summary(pkg)
        docs.append({
            "id": f"rec:{rec.recording_id}",
            "Document": rec.label,
            "Supplier": supplier,
            "Route": route,
            "Source": "Recorded live run",
            "When": _when(rec.recorded_at),
            "Model": ", ".join(rec.models) or "—",
            "Calls": f"{rec.calls - rec.failed_calls}/{rec.calls}",
        })
    for s_ in list_samples():
        pkg = load_sample(s_["id"])
        supplier, route = summary(pkg)
        docs.append({
            "id": f"sample:{s_['id']}",
            "Document": pkg.file_name or s_["id"],
            "Supplier": supplier,
            "Route": route,
            "Source": "Reference fixture",
            "When": "—",
            "Model": "— (hand-built)",
            "Calls": "—",
        })
    return docs


def _open(doc_id: str) -> None:
    kind, _, key = doc_id.partition(":")
    if kind == "sample":
        pkg = load_sample(key)
        activate(pkg, doc_id)
        st.session_state.sample_id = key
    elif kind == "rec":
        rec = recordings.load("review", key)
        if rec:
            activate(ReviewPackage.model_validate(rec.payload), doc_id)


def _replay_panel() -> None:
    docs = _library()
    live_count = sum(1 for d in docs if d["Source"] == "Recorded live run")
    st.caption(
        (f"{live_count} recorded live run(s) and {len(docs) - live_count} reference fixtures. "
         if live_count else "No recorded live runs yet — switch to Live AI to record one. "
         "Reference fixtures are hand-built answer packages, shown for comparison. ")
        + "Select a row to open it; every page switches to that document."
    )
    df = pd.DataFrame([{k: v for k, v in d.items() if k != "id"} for d in docs])

    def _src(v):
        return "color:#5E2A9C;font-weight:600" if v == "Recorded live run" else "color:#5B5F66"

    event = st.dataframe(
        df.style.map(_src, subset=["Source"]), hide_index=True, width="stretch", on_select="rerun",
        selection_mode="single-row", key="doc_library",
        column_config={"Document": st.column_config.TextColumn(width="medium")},
    )
    rows = event.selection.rows if event and event.selection else []
    if rows and docs[rows[0]]["id"] != st.session_state.get("active_doc"):
        _open(docs[rows[0]]["id"])
        st.rerun()


def document_source() -> None:
    with st.container(border=True):
        if is_live():
            st.markdown("##### Run a live review")
            _live_review_panel()
        else:
            st.markdown("##### Open a recorded review")
            _replay_panel()


# --------------------------------------------------------------------------- #
# Procurement request: AI intake assistant
# --------------------------------------------------------------------------- #
INTAKE_EXAMPLE = (
    "Hi team — IT Ops needs one collaboration platform for roughly 500 staff to replace the mix of chat and "
    "video tools we use today. Must have document collaboration, video meetings, admin SSO and audit logs, and "
    "it has to plug into our SAML SSO and the corporate directory. It will hold customer contact details, so "
    "security and privacy need to look at it. We have about 48 lakh a year approved. Looking for a 12-month "
    "initial term starting November, with 99.9% monthly uptime and service credits."
)


def intake_assistant(req) -> None:
    from app.intake_assistant import ValidatedDraft, draft_from_text, preview, validate

    with st.expander("AI intake assistant — describe the need in plain words",
                     expanded=bool(st.session_state.get("intake_result")), icon=":material/auto_awesome:"):
        if is_live():
            if not _ready():
                no_key_banner()
            from evals.intake_eval import cases as intake_cases

            ev = {f"{c['id']} — {c['title']}": c["text"] for c in intake_cases()}
            pick = st.selectbox("Load an evaluation case (optional)", ["—"] + list(ev), key="intake_case",
                                help="Answer keys in evals/intake_cases.json. Drafting a case records a run that "
                                     "the Scorecard scores.")
            if pick != "—" and st.session_state.get("intake_case_loaded") != pick:
                st.session_state.intake_text = ev[pick]
                st.session_state.intake_case_loaded = pick
            st.session_state.setdefault("intake_text", INTAKE_EXAMPLE)
            text = st.text_area("What do you need?", height=130, key="intake_text")
            st.caption("One Gemini request. The model only drafts fields; status, risk flags and routing are "
                       "still computed by rules after you apply the draft.")
            if st.button("Draft request with AI", type="primary", icon=":material/auto_awesome:",
                         disabled=not (_ready() and text.strip())):
                with st.spinner("Gemini is drafting the request…"):
                    with ai_trace.capture() as calls:
                        try:
                            draft = draft_from_text(text)
                        except Exception as exc:  # show, don't crash the page
                            draft = None
                            st.error(f"The model call failed: {exc}")
                if draft is not None:
                    v = validate(draft, source_text=text)
                    rec = recordings.save("intake", (draft.title or "intake")[:40], {
                        "text": text, "draft": draft.model_dump(mode="json"), "validated": v.model_dump(mode="json"),
                    }, calls)
                    st.session_state.intake_result = {"validated": v.model_dump(mode="json"),
                                                      "calls": [c.model_dump() for c in calls],
                                                      "source": "live", "when": rec.recorded_at, "text": text}
                    st.rerun()
        else:
            recs = recordings.list_recordings("intake")
            if not recs:
                ui.empty_state("No recorded intake runs yet", "Switch to Live AI and draft a request to record one.")
            else:
                pick = st.selectbox("Recorded intake run", recs,
                                    format_func=lambda r: f"{r.label} · {_when(r.recorded_at)} · {', '.join(r.models)}")
                if st.button("Replay this draft", icon=":material/history:"):
                    st.session_state.intake_result = {"validated": pick.payload["validated"],
                                                      "calls": pick.ai_calls, "source": "replay",
                                                      "when": pick.recorded_at, "text": pick.payload.get("text", "")}
                    st.rerun()

        res = st.session_state.get("intake_result")
        if not res:
            return
        v = ValidatedDraft.model_validate(res["validated"])
        models = ", ".join(sorted({c.get("model") for c in res["calls"] if c.get("model")})) or "—"
        chip = ui.chip(f"Generated live · {models}", "blue") if res["source"] == "live" else \
            ui.chip(f"Replay of live run · {_when(res['when'])} · {models}", "purple")
        st.html(f'<div class="cr-head">{chip}</div>')
        if res["source"] == "replay" and res.get("text"):
            ui.quote_box("Requester's text", res["text"], "gray")
        upd = dict(v.updates)
        rows = preview(req, upd)
        if rows:
            st.dataframe(pd.DataFrame(rows, columns=["Field", "Current", "AI draft"]).assign(
                Field=lambda d: d["Field"].map(ui.humanize)), hide_index=True, width="stretch")
        else:
            st.caption("The draft matches the current request.")
        if v.rejected:
            st.warning("Refused by validation (not applied): " + "; ".join(v.rejected))
        if v.open_questions:
            st.markdown("**Questions to ask the requester**\n" + "\n".join(f"- {q}" for q in v.open_questions))
        st.caption(calls_caption(res["calls"]))
        b1, b2, _ = st.columns([1.2, 1.4, 2])
        if b1.button("Apply to this request", disabled=req.status not in EDITABLE_STATUSES or not rows,
                     help=None if req.status in EDITABLE_STATUSES else "This request is locked. Start a new one."):
            try:
                st.session_state.procurement_req = update_request(req.request_id, upd, actor="intake.assistant")
                st.toast("Draft applied — risk flags and routing recomputed by rules")
                st.rerun()
            except InvalidTransitionError as exc:
                st.error(str(exc))
        if b2.button("Start a new request from it"):
            base = {"title": "New request", "requesting_team": req.requesting_team,
                    "business_owner": req.business_owner, "requester_role": req.requester_role}
            new = create_request({**base, **upd}, actor="intake.assistant")
            st.session_state.procurement_req = new
            st.session_state.pop("intake_result", None)
            st.toast(f"Created {new.request_id} as a draft")
            st.rerun()


# --------------------------------------------------------------------------- #
# Commercial — linked to the request, the contract and negotiation
# --------------------------------------------------------------------------- #
def _flow_strip(req, pkg, neg_run: Optional[AgentRun]) -> None:
    steps = [
        ("Request", f"{req.request_id} · {ui.fmt_inr(req.budget_amount)}" if req else "—", "done" if req else "todo"),
        ("Quotations", f"{len(load_demo_quotations())} received", "done"),
        ("Negotiation", (f"{neg_run.scenario.supplier.name} · {ui.humanize(neg_run.state.status)}"
                         if neg_run else "Not started"), "done" if neg_run else "todo"),
        ("Contract", (pkg.file_name or "—") if pkg else "Not reviewed", "done" if pkg else "todo"),
    ]
    ui.stepper([(a, b, "done" if s == "done" else "todo") for a, b, s in steps])


def page_commercial() -> None:
    req = st.session_state.get("procurement_req")
    pkg: Optional[ReviewPackage] = st.session_state.get("pkg")
    quotes = load_demo_quotations()
    cmp = comparison_for_request(req)
    ui.page_header("Commercial comparison",
                   f"Quotations for {cmp.requisition_id} · budget, uptime and term taken from the request · "
                   "no model calls")
    _flow_strip(req, pkg, st.session_state.get("neg_run"))

    lowest = next((r for r in cmp.rows if r.is_lowest_annual_fee), None)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Budget (from request)", ui.fmt_inr(cmp.budget_inr), border=True)
    m2.metric("Required uptime", f"{cmp.required_uptime_percent}%", border=True)
    m3.metric("Horizon", f"{cmp.comparison_horizon_months} months", border=True)
    m4.metric("Lowest annual fee", lowest.supplier_name if lowest else "—", border=True,
              help="Highlighted for visibility only — not a recommendation.")

    with st.container(border=True):
        df = pd.DataFrame([
            {
                "Supplier": r.supplier_name,
                "Annual fee": ui.fmt_inr(r.annual_recurring_fee),
                "One-time": ui.fmt_inr(r.one_time_fee),
                f"Total ({cmp.comparison_horizon_months} mo)": ui.fmt_inr(r.total_cost_over_horizon),
                "vs budget": ("+" if (r.budget_variance or 0) > 0 else "") + ui.fmt_inr(r.budget_variance),
                "Uptime": f"{r.uptime_percent}%" if r.uptime_percent is not None else "—",
                "Meets SLA": "Yes" if r.sla_compliant else "No",
                "Term": f"{r.term_months} mo" if r.term_months else "—",
                "Scope": ui.humanize(r.scope_comparability),
            }
            for r in cmp.rows
        ])
        low_idx = [i for i, r in enumerate(cmp.rows) if r.is_lowest_annual_fee]
        styled = (
            df.style.apply(lambda row: ["background-color:#EEF3FD" if row.name in low_idx else ""] * len(row), axis=1)
            .map(lambda v: "color:#A50E0E;font-weight:600" if v == "No" else "", subset=["Meets SLA"])
            .map(lambda v: "color:#A50E0E" if str(v).startswith("+") else "color:#0D652D", subset=["vs budget"])
        )
        st.dataframe(styled, hide_index=True, width="stretch")
        notes = [f"- **{r.supplier_name}** — {w.rstrip('.')}." for r in cmp.rows for w in r.warnings]
        if notes:
            st.markdown("**Check before comparing on price**\n" + "\n".join(notes))
        st.markdown("**Negotiate**")
        cols = st.columns(len(quotes) or 1)
        for col, q in zip(cols, quotes):
            if col.button(f"Negotiate with {q.supplier_name}", width="stretch", key=f"neg_{q.supplier_name}"):
                st.session_state.neg_supplier = q.supplier_name
                st.session_state.pop("neg_run", None)
                st.switch_page(PAGES["negotiation"])

    with st.container(border=True):
        st.markdown("##### Contract vs quotation")
        check = contract_vs_quotes(pkg, quotes)
        if not check.supplier:
            st.caption("Review a contract to compare it with the quotations.")
        for sev, msg in check.findings:
            if sev == "ok":
                ui.banner("Consistent with the quotation", msg, "green", "✓")
            else:
                ui.banner(f"{check.supplier}", msg, "red" if sev == "high" else "amber", "!")
    if cmp.notes:
        with st.expander("Method notes"):
            st.markdown("\n".join(f"- {n}" for n in cmp.notes))


# --------------------------------------------------------------------------- #
# Negotiation — live agents vs recorded runs, every message labelled
# --------------------------------------------------------------------------- #
def _num_target(key: str, value) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return ui.humanize(str(value)).replace("Dpa", "DPA") if value else "—"
    if n >= 10000:
        return ui.fmt_inr(n)
    if "uptime" in key or "percent" in key:
        return f"{n:g}%"
    if "term" in key or "month" in key:
        return f"{int(n)} months"
    return f"{n:g}"


def _origin_chip(run: Optional[AgentRun], kind: str, msg) -> str:
    o = run.origins.get(msg.message_id) if run else None
    if kind == "fixture":
        return ui.chip("Hand-written fixture", "gray")
    if not o:
        return ui.chip("Origin not recorded", "gray")
    who = {"strategy": "Strategist", "draft": "Email drafter", "supplier_reply": "Supplier simulator"}.get(o.step, o.step)
    if o.source == "scripted":
        return ui.chip(f"{who} · offline scripted agent (no AI)", "gray")
    model = ", ".join(o.models) or "—"
    if kind == "recorded":
        return ui.chip(f"{who} · recorded live · {model}", "purple")
    return ui.chip(f"{who} · live AI · {model} · {o.seconds:.1f} s", "blue")


def page_negotiation() -> None:
    req = st.session_state.get("procurement_req")
    quotes = {q.supplier_name: q for q in load_demo_quotations()}
    run: Optional[AgentRun] = st.session_state.get("neg_run")
    kind = st.session_state.get("neg_kind", "")  # live | scripted | recorded | fixture

    supplier = st.session_state.get("neg_supplier") or "CloudSoft"
    if run is not None:
        scenario = run.scenario
    else:
        scenario = scenario_from_quote(quotes[supplier], req) if supplier in quotes else None
        if scenario is None:
            from app.negotiation import load_scenario
            scenario = load_scenario()
    state = run.state if run else new_run(scenario)
    brief = negotiation_brief(scenario, state)
    tone = {"AGREEMENT_IN_PRINCIPLE": "green", "NOT_STARTED": "gray", "ESCALATION_REQUIRED": "amber"}
    kind_chip = {
        "live": ui.chip("Live AI agents", "blue"),
        "scripted": ui.chip("Offline scripted agents — not AI", "gray"),
        "recorded": ui.chip("Replay of recorded live run", "purple"),
        "fixture": ui.chip("Hand-written fixture", "gray"),
    }.get(kind, "")

    ui.page_header("Negotiation",
                   f"{scenario.supplier.name} · {scenario.requisition_id} · simulated mailbox — no real email is sent",
                   [ui.chip(ui.humanize(brief["status"]), tone.get(brief["status"], "blue")), kind_chip])

    with st.container(border=True):
        if is_live():
            _live_controls(scenario, quotes, supplier, run, kind)
        else:
            _replay_controls()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Supplier's fee", ui.fmt_inr(brief.get("current_supplier_fee_inr")), border=True)
    m2.metric("Target", ui.fmt_inr(brief.get("target_price_inr")), border=True)
    m3.metric("Budget ceiling (from request)", ui.fmt_inr(brief["budget_ceiling_inr"]), border=True)
    m4.metric("Round", f"{brief['round']} of {brief.get('max_rounds') or '—'}", border=True)

    left, right = st.columns([2, 3])
    with left:
        with st.container(border=True):
            st.markdown("##### Brief")
            st.caption(scenario.description)
            labels = {t.key: t.label for t in scenario.terms}
            for m in brief["must_haves"]:
                st.markdown(f"- {m['label']} — target **{_num_target(m['key'], m['target'])}**")
            open_html = "".join(ui.chip(labels.get(x, ui.humanize(x)), "amber") for x in brief["open_issues"]) \
                or ui.chip("None", "gray")
            agreed_html = "".join(ui.chip(labels.get(x, ui.humanize(x)), "green") for x in brief["agreed_terms"]) \
                or ui.chip("None yet", "gray")
            st.html(f'<div class="cr-section">Open issues</div><div class="cr-pipe">{open_html}</div>'
                    f'<div class="cr-section" style="margin-top:12px">Agreed</div><div class="cr-pipe">{agreed_html}</div>')
            if brief.get("strategy_summary"):
                st.info(brief["strategy_summary"])
            if state.pending_action:
                st.markdown(
                    f"**Pending action** — {ui.humanize(state.pending_action.action_type)}"
                    + (" · needs Legal" if state.pending_action.requires_escalation else "")
                )
                st.caption(state.pending_action.rationale)
            excerpts = getattr(state, "knowledge_excerpts_used", None) or []
            if excerpts:
                with st.expander(f"Knowledge used in strategy ({len(excerpts)})"):
                    for ex in excerpts[:6]:
                        st.markdown(ex.replace("\n", "\n\n"))
            st.caption(brief["non_binding_notice"])
        if run and run.steps:
            with st.container(border=True):
                st.markdown("##### Agent trace")
                st.dataframe(pd.DataFrame([
                    {"Step": ui.humanize(s["step"]), "Model": ", ".join(s.get("models") or []) or
                     ("—" if s["step"] == "approve" else "offline"),
                     "Calls": s.get("calls", 0), "Seconds": s.get("seconds", 0.0)}
                    for s in run.steps
                ]), hide_index=True, width="stretch")
                if state.status == "AGREEMENT_IN_PRINCIPLE":
                    st.warning("Liability stays open for Legal — it is never auto-accepted.")
    with right, st.container(border=True):
        st.markdown("##### Email thread")
        thread = list(state.thread)
        if state.pending_draft:
            thread.append(state.pending_draft)
        if not thread:
            ui.empty_state("No messages yet", "Run the agents (Live AI) or open a recorded run (Replay).")
        for msg in thread:
            buyer = msg.direction == "buyer_to_supplier"
            with st.chat_message("Buyer" if buyer else "Supplier",
                                 avatar=":material/apartment:" if buyer else ":material/storefront:"):
                st.html(f'<div class="cr-head"><b>{ui.escape(scenario.buyer_name if buyer else scenario.supplier.name)}'
                        f'</b><span class="cr-mono">round {msg.round_number}</span>'
                        f"{_origin_chip(run, kind, msg)}"
                        + (ui.chip("Awaiting your approval", "amber") if msg is state.pending_draft else
                           ui.chip("Approved by you", "green") if buyer and msg.approval_status != "draft" else "")
                        + "</div>")
                st.caption(msg.subject)
                st.write(msg.body)


def _run_step(name: str) -> None:
    run: AgentRun = st.session_state.neg_run
    try:
        with st.spinner(f"{ui.humanize(name)}…"):
            st.session_state.neg_run = step(run, name)
        if st.session_state.neg_run.state.status in ("AGREEMENT_IN_PRINCIPLE", "ESCALATION_REQUIRED", "IMPASSE") \
                and st.session_state.neg_kind == "live":
            record_negotiation(st.session_state.neg_run)
            st.toast("Live run recorded for Replay")
        st.rerun()
    except NegotiationError as exc:
        st.error(f"{exc.code}: {exc}")
    except Exception as exc:
        st.error(f"Agent step failed: {exc}")


def _live_controls(scenario, quotes, supplier, run, kind) -> None:
    ready = _ready()
    top1, top2 = st.columns([2, 3], vertical_alignment="bottom")
    names = list(quotes)
    pick = top1.selectbox("Supplier quotation", names, index=names.index(supplier) if supplier in names else 0)
    if pick != supplier:
        st.session_state.neg_supplier = pick
        st.session_state.pop("neg_run", None)
        st.rerun()
    offline = top2.toggle("Use offline scripted agents instead (no AI, for testing)", value=not ready,
                          disabled=not ready, key="neg_offline")
    backend = "scripted" if (offline or not ready) else "live"
    if not ready:
        no_key_banner()
    st.caption("Live run ≈ 9 Gemini requests (strategy + 2 rounds of draft, supplier reply, analysis, next move). "
               "Each buyer email still needs your approval in step-by-step mode.")

    b1, b2, b3 = st.columns([1.6, 1.4, 1])
    if b1.button("Run full negotiation" + (" with AI" if backend == "live" else " (offline)"), type="primary",
                 icon=":material/play_arrow:", width="stretch"):
        try:
            with st.status("Agents negotiating…", expanded=True) as status:
                def _cb(name, r):
                    s = r.steps[-1]
                    status.write(f"{ui.humanize(name)} · " + (f"{', '.join(s['models'])} · {s['seconds']} s"
                                                            if s.get("models") else "no model call"))
                done = run_full(scenario, backend, on_step=_cb)
                status.update(label=f"Finished · {ui.humanize(done.state.status)}", state="complete")
            st.session_state.neg_run, st.session_state.neg_kind = done, backend
            if backend == "live":
                record_negotiation(done)
            st.rerun()
        except Exception as exc:
            st.error(f"Negotiation failed: {exc}")
    if b2.button("Start step by step", width="stretch", icon=":material/stairs:"):
        st.session_state.neg_run = AgentRun(backend=backend, scenario=scenario, state=new_run(scenario))
        st.session_state.neg_kind = backend
        st.rerun()
    if b3.button("Reset", width="stretch", disabled=run is None):
        st.session_state.pop("neg_run", None)
        st.rerun()

    if run is not None and kind in ("live", "scripted"):
        status = run.state.status
        c = st.columns(6)
        spec = [("1. Strategy", "strategy", status == "NOT_STARTED"),
                ("2. Draft email", "draft", status in ("STRATEGY_READY", "NEXT_MOVE_READY")),
                ("3. Approve & send", "approve", status == "AWAITING_USER_APPROVAL"),
                ("4. Supplier reply", "supplier_reply", status == "EMAIL_SENT_TO_SIMULATOR"),
                ("5. Analyze", "analyze", status == "SUPPLIER_REPLY_RECEIVED"),
                ("6. Next move", "next_move", status == "RESPONSE_ANALYZED")]
        for col, (label, name, enabled) in zip(c, spec):
            if col.button(label, width="stretch", disabled=not enabled,
                          type="primary" if name == "approve" and enabled else "secondary"):
                _run_step(name)


def _replay_controls() -> None:
    recs = recordings.list_recordings("negotiation")
    options = [("rec", r) for r in recs] + [("fixture", None)]

    def _fmt(o):
        if o[0] == "fixture":
            return "Hand-written fixture · CloudSoft (not AI output)"
        r = o[1]
        return f"{r.label} · recorded live {_when(r.recorded_at)} · {', '.join(r.models) or '—'} · " \
               f"{ui.humanize(r.meta.get('status', ''))}"

    if not recs:
        st.caption("No recorded live negotiations yet — switch to Live AI and run one. "
                   "The hand-written fixture is available for comparison.")
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    pick = c1.selectbox("Negotiation to replay", options, format_func=_fmt)
    if c2.button("Open run", type="primary", width="stretch", icon=":material/history:"):
        if pick[0] == "fixture":
            from app.negotiation import load_scenario
            st.session_state.neg_run = AgentRun(backend="scripted", scenario=load_scenario(), state=run_replay())
            st.session_state.neg_kind = "fixture"
        else:
            st.session_state.neg_run = load_recorded(pick[1])
            st.session_state.neg_kind = "recorded"
        st.rerun()


# --------------------------------------------------------------------------- #
# Scorecard — hard gates, quality and operational metrics
# --------------------------------------------------------------------------- #
HOW_SCORED = """
**Answer keys.** Each contract has a hand-written key in `golden/`: the issues planted in the PDF (rule,
expected outcome, severity, page and the exact clause text), decoys that look risky but are acceptable, the
expected route and key terms. The knowledge base has its own key in `evals/rag_golden.json`: which policy
documents answer each question, and which questions must get no answer. Keys are never shown to the model.

**Matching.** A planted issue counts as found when the AI flags the same rule with the same outcome and one
of its quotes is *locatable* to the key's clause: word overlap (Jaccard) ≥ 50%, or the quote contains ≥ 90%
of the clause's words and is at most three times its length (the clause plus a neighbouring sentence).
Grounding is separate: every retained quote must appear verbatim in the PDF text.

**Clause outcomes.** Besides the planted issues, every contract has a hand label for all ten playbook rules
(`expected_outcomes`). This catches clean clauses scored wrongly and clauses that come back *unclear* —
which on a high-severity or legal-review rule forces manual review.

**Rules applied.** The model output is recorded once. By default the scorecard re-applies the *current*
deterministic rules to that recorded output (no model calls), so a rule fix shows up immediately; switch to
*As recorded* to see the run exactly as it was saved.

**Three groups.**
- **Hard gates** — critical-risk recall, evidence grounding, route accuracy, escalation recall, flagging the
  planted prompt injection, never retrieving retired or draft policy, and the negotiation guardrails (price
  band, liability never agreed by the agent, human approval before every send, zero policy breaches). If any
  gate fails, the result is **Fail**, however good the averages look.
- **Quality** — recall, precision, F1, decoy false positives, evidence relevance and page localisation,
  obligation and key-term extraction, retrieval quality, run success.
- **Operational** — model time, calls per contract, schema validity, audit completeness.

**N/A** means there were no eligible cases (for example no decoys in the scored contracts, or no model calls
for hand-built fixtures). It is never shown as 0% or as a pass.

Targets are suggested demo targets, not industry benchmarks. Evidence relevance is an automated proxy; the
real test of evidence and redline quality is review by a legal or procurement expert.
"""

STATUS_CHIP = {
    "pass": ("Pass", "green"), "fail": ("Fail", "red"), "below_target": ("Below target", "amber"),
    "na": ("N/A", "gray"), "info": ("Info", "blue"),
}
CATEGORIES = [
    ("detection", "Detection"), ("evidence", "Evidence"), ("legal", "Policy & legal"),
    ("routing", "Routing & workflow"), ("rag", "RAG"), ("negotiation", "Negotiation"), ("intake", "Intake"),
    ("operations", "Operations"),
]


def _fmt_value(m: dict) -> str:
    v = m["value"]
    if v is None:
        return "N/A"
    if m["unit"] == "%":
        return f"{v:.0%}"
    if m["unit"] == "s":
        return f"{v:.1f} s"
    return f"{v:.2f}" if m["unit"] == "ratio" else f"{v:g}"


def _fmt_target(m: dict) -> str:
    t = m["target"]
    if t is None or m["op"] == "info":
        return "—"
    shown = f"{t:.0%}" if m["unit"] == "%" else f"{t:.2f}"
    return ("≥ " if m["op"] == ">=" else "≤ ") + shown


def _metric_table(metrics: list[dict]) -> None:
    df = pd.DataFrame([
        {
            "Metric": m["name"] + (" ◆" if m["group"] == "gate" else ""),
            "Result": _fmt_value(m),
            "Cases": f"{m['num']:g} / {m['den']:g}" if m["den"] else "—",
            "Target": _fmt_target(m),
            "Status": STATUS_CHIP[m["status"]][0],
            "How it is measured": m["how"],
        }
        for m in metrics
    ])

    def _style(v):
        for label, tone in STATUS_CHIP.values():
            if v == label:
                bg, fg = ui.TONES[tone]
                return f"background-color:{bg};color:{fg};font-weight:600"
        return ""

    st.dataframe(df.style.map(_style, subset=["Status"]), hide_index=True, width="stretch",
                 column_config={"How it is measured": st.column_config.TextColumn(width="large")})


def _summary_card(col, title: str, metrics: list[dict], keys: list[str]) -> None:
    picked = [m for m in metrics if m["key"] in keys]
    worst = "pass"
    for m in picked:
        if m["status"] == "fail":
            worst = "fail"
        elif m["status"] == "below_target" and worst != "fail":
            worst = "below_target"
    if all(m["status"] in ("na",) for m in picked):
        worst = "na"
    label, tone = STATUS_CHIP[worst]
    lines = " · ".join(f"{m['name'].split(' (')[0]} {_fmt_value(m)}" for m in picked)
    with col, st.container(border=True, height="stretch"):
        st.html(f'<div class="cr-head" style="justify-content:space-between"><b style="font-size:15px">'
                f"{ui.escape(title)}</b>{ui.chip(label, tone)}</div>"
                f'<p class="cr-sub" style="font-size:13px;margin-top:8px">{ui.escape(lines)}</p>')


def _render_scorecard(sc: dict, *, live: bool) -> None:
    metrics = sc["metrics"]
    overall = sc["overall"]
    scored = len(sc["per_contract"])
    if overall == "fail":
        why = "; ".join(f"{r['contract'].replace('contract_', '').upper()}: {', '.join(r['gate_failures'])}"
                        for r in sc["per_contract"] if r["gate_failures"])
        cover = (f" Only {scored} of {scored + len(sc['missing'])} contracts have a live run so far."
                 if sc["missing"] else "")
        ui.banner(f"Fail — hard gate failed: {', '.join(sc['failed_gates'])}", (why + "." if why else "") + cover,
                  "red", "✕")
    elif overall == "incomplete":
        ui.banner("Evaluation incomplete",
                  ("No contracts scored yet. " if not scored else
                   f"{scored} of {scored + len(sc['missing'])} contracts scored; missing: {', '.join(sc['missing'])}. ")
                  + f"Hard gates so far: {sc['gates_passed']}/{sc['gates_applicable']} passing.", "amber", "!")
    else:
        ui.banner("Pass — every applicable hard gate passed",
                  f"{sc['gates_passed']}/{sc['gates_applicable']} gates. "
                  + (f"Below target: {', '.join(sc['below_target'])}." if sc["below_target"] else
                     "All quality metrics on target."), "green", "✓")

    c = st.columns(4)
    _summary_card(c[0], "Detection quality", metrics, ["critical_recall", "overall_recall", "precision", "f1"])
    _summary_card(c[1], "Evidence quality", metrics, ["grounding", "localization", "citation_coverage"])
    _summary_card(c[2], "Routing & workflow", metrics, ["route", "escalation", "security"])
    _summary_card(c[3], "Retrieval (RAG)", metrics, ["recall_at_k", "ndcg", "no_answer", "policy_version"])

    tabs = st.tabs([name for _, name in CATEGORIES] + ["By contract", "RAG queries", "Negotiation runs",
                                                        "Intake cases"])
    for tab, (cat, _) in zip(tabs, CATEGORIES):
        with tab:
            _metric_table([m for m in metrics if m["category"] == cat])
            st.caption("◆ hard gate — any failure fails the whole evaluation.")
    with tabs[len(CATEGORIES)]:
        rows = []
        for r in sc["per_contract"]:
            ct = r["counts"]
            rows.append({
                "Contract": r["contract"].replace("contract_", "").upper(),
                "Recorded": _when(r.get("recorded_at")) if live else "fixture",
                "Model": ", ".join(r.get("models") or []) or "—",
                "Status": ui.humanize(r["status"]),
                "Critical found": f"{ct['critical_recall'][0]}/{ct['critical_recall'][1]}" if ct["critical_recall"][1] else "N/A",
                "Issues found": f"{ct['overall_recall'][0]}/{ct['overall_recall'][1]}",
                "Flags correct": f"{ct['precision'][0]}/{ct['precision'][1]}" if ct["precision"][1] else "N/A",
                "Outcomes right": f"{ct['clause_outcome'][0]}/{ct['clause_outcome'][1]}" if ct["clause_outcome"][1] else "N/A",
                "Unclear": f"{ct['unclear_clear'][0]}" if ct["unclear_clear"][1] else "N/A",
                "Route": ui.ROUTES.get(r["route_got"], (ui.humanize(r["route_got"]),))[0]
                + ("" if r["route_got"] == r["route_want"] else
                   f" ✕ want {ui.ROUTES.get(r['route_want'], (r['route_want'],))[0]}"),
                **({"Route as recorded": ui.ROUTES.get(r["recorded_route"], (ui.humanize(r["recorded_route"]),))[0]}
                   if r.get("recorded_route") else {}),
                "Gate failures": "; ".join(r["gate_failures"]) or "none",
            })
        rows += [{"Contract": m.replace("contract_", "").upper(), "Recorded": "Not run yet"} for m in sc["missing"]]
        st.dataframe(pd.DataFrame(rows).fillna("—"), hide_index=True, width="stretch")
        st.caption("A run that fell back to manual review (e.g. quota exhausted) scores zero recall — shown, not hidden.")
    with tabs[len(CATEGORIES) + 1]:
        st.dataframe(pd.DataFrame([
            {
                "Query": q["query"],
                "Kind": ui.humanize(q["category"]),
                "Retrieved (top 5 documents)": ", ".join(q["retrieved"]) or "—",
                "Relevant (answer key)": ", ".join(q["relevant"]) or "none — must abstain",
                "Recall": f"{q['recall']:.0%}" if q.get("recall") is not None else "—",
                "Abstained": "Yes" if q["abstained"] else "No",
                "Correct": "Yes" if q["abstained"] == q["expect_abstain"] else "No",
            }
            for q in sc["rag_queries"]
        ]), hide_index=True, width="stretch")
        st.caption("Retrieval is deterministic (lexical), so these numbers only change when the corpus or "
                   "retriever changes.")
    with tabs[len(CATEGORIES) + 2]:
        runs = sc.get("negotiation_runs") or []
        if not runs:
            st.caption("No negotiation runs to score yet.")
        else:
            st.dataframe(pd.DataFrame([
                {
                    "Case": r["case"], "Supplier": r["supplier"],
                    "Agents": "Live AI" if r["backend"] == "live" else "Scripted (not AI)",
                    "Recorded": _when(r.get("recorded_at")) if r.get("recorded_at") else "—",
                    "Status": ui.humanize(r["status"]), "Rounds": f"{r['rounds']}/{r['max_rounds']}",
                    "Quote": f"₹{r['quote']:,.0f}" if r.get("quote") else "—",
                    "Band": f"₹{r['floor']:,.0f} – ₹{min(r['ceiling'], r['quote'] or r['ceiling']):,.0f}",
                    "Final offer": f"₹{r['final_fee']:,.0f}" if r.get("final_fee") else "—",
                    "Breaches": "; ".join(r["breaches"]) or "none",
                }
                for r in runs
            ]), hide_index=True, width="stretch")
            st.caption("Live view: latest recorded live run per supplier. Self-check view: the deterministic "
                       "scripted agents on all three cases (NEG-001 CloudSoft, NEG-002 TechNova, NEG-003 "
                       "DataSphere). An unfinished run is scored on what it sent; round-limit and final-terms "
                       "metrics are N/A for it.")
    with tabs[len(CATEGORIES) + 3]:
        from evals.intake_eval import cases as intake_cases

        done = {r["case"]: r for r in sc.get("intake_runs") or []}
        st.dataframe(pd.DataFrame([
            {
                "Case": c["id"], "What it tests": c["title"],
                "Recorded": _when(done[c["id"]]["recorded_at"]) if c["id"] in done else "Not run yet",
                "Fields right": (f"{done[c['id']]['fields'][0]}/{done[c['id']]['fields'][1]}"
                                 if c["id"] in done else "—"),
                "Problems": ("; ".join(done[c["id"]]["problems"]) or "none") if c["id"] in done else "—",
            }
            for c in intake_cases()
        ]), hide_index=True, width="stretch")
        st.caption("Run a case: Procurement request → AI intake assistant → Load an evaluation case → Draft "
                   "(Live AI, 1 Gemini request). Fixture self-check view: intake metrics are N/A.")


def page_scorecard() -> None:
    import evals.score_live as score_live

    view = st.segmented_control("Scorecard", ["Live AI runs", "Fixture self-check"], default="Live AI runs",
                                key="sc_view", label_visibility="collapsed") or "Live AI runs"
    live = view == "Live AI runs"
    rules = "current"
    if live:
        pick = st.segmented_control(
            "Rules", ["Current rules", "As recorded"], default="Current rules", key="sc_rules",
            help="Current rules: the recorded model output re-scored with today's deterministic rules (no model "
                 "calls). As recorded: exactly what was saved with the run.") or "Current rules"
        rules = "current" if pick == "Current rules" else "as_recorded"
    with st.spinner("Scoring…"):
        sc = score_live.build("live" if live else "fixtures", rules=rules)
    tone = {"pass": "green", "fail": "red", "incomplete": "amber"}[sc["overall"]]
    label = {"pass": "Pass", "fail": "Fail", "incomplete": "Incomplete"}[sc["overall"]]
    if live:
        ui.page_header("Contract AI evaluation",
                       "Latest recorded live Gemini review of each contract, scored against the answer keys · "
                       "no model calls to view",
                       [ui.chip(label, tone),
                        ui.chip("Recorded model output · current rules" if rules == "current" else
                                "As recorded", "blue" if rules == "current" else "gray")])
    else:
        ui.page_header("Fixture self-check", "Hand-built reference packages scored with the same metrics",
                       [ui.chip(label, tone)])
        ui.banner("This is not a measure of the AI",
                  "The reference fixtures were written to match the answer keys. This view checks that the "
                  "scoring code, answer keys and fixtures agree.", "amber", "i")
    if live and not sc["per_contract"]:
        ui.empty_state(
            "No live runs recorded yet",
            "In Live AI mode, open Contract & findings and run each bundled contract (≈3 Gemini requests each), "
            "or run `python evals/run_evals.py --runs 1`. Retrieval metrics below need no model calls.",
        )
    _render_scorecard(sc, live=live)
    with st.expander("How the evaluation works"):
        st.markdown(HOW_SCORED)
    st.caption("Command line: `python evals/score_live.py` (live) or `python evals/score_live.py --fixtures`.")
