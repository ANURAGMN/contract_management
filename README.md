# Supplier Contract Review Agent

A narrow, auditable agent that reviews a supplier MSA/SOW against a versioned
playbook: it extracts metadata, flags clause deviations with **code-verified
quotes**, drafts fallback redlines, recommends an approval route from
**deterministic rules**, and surfaces renewal/SLA obligations — including a
cross-document SLA → service-credit check.

**Demo deck:** [docs/Zycus_Demo_Deck.pdf](docs/Zycus_Demo_Deck.pdf) — see
[docs/README.md](docs/README.md).

**Design principle:** the model is used only where language understanding is
needed (extract, review, draft). Everything that must be exact or reproducible —
severity, routing, dates, money, grounding — is Python. An ungrounded finding
never reaches the router, and the router never sees model opinions on severity.

## Architecture

```
intake (code) ─┬─ metadata_extractor (LLM) ─┐
               ├─ clause_reviewer   (LLM) ─┤  parallel
               └─ performance_tool  (code) ─┘
                        │
            grounding_validator (code)  → drops ungrounded quotes
                        │
            rule_engine + router (code) → severity, route, rule trace
                        │
        ┌── redline_drafter (LLM) ──┐  parallel
        └── obligations + credits (code) ┘
                        │
            assemble → ReviewPackage  (UI tabs, eval scorecard)
```

- **3 LLM agents** (`gemini-flash-latest`, temp 0–0.2): metadata, clause review,
  redline drafting. Schema-in / schema-out, no tools, no history.
- **Deterministic nodes**: intake + preflight, grounding, rule checkers, router,
  obligations, service credits, redline checks, assembly.
- Two runtimes share one engine: a plain async **orchestrator**
  (`app/orchestrator.py`, used by the evals and the Streamlit UI) and an **ADK**
  wrapper (`app/adk_workflow.py`, for `adk web`). See *ADK notes* below.

## Setup (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Put your AI Studio key in contract_review\.env:
#   GOOGLE_API_KEY=...
#   GOOGLE_GENAI_USE_VERTEXAI=FALSE
#   REVIEW_MODEL=gemini-flash-latest        # optional: pin a dated model
#   REVIEW_JUDGE_MODEL=gemini-pro-latest    # optional: eval judge model
```

> **API quota.** A free-tier AI Studio key is limited to ~20 `generate_content`
> requests/day. One contract costs 3 calls, so a 5-contract × 3-run scorecard
> needs ~45. Enable billing (or pin a paid model via `REVIEW_MODEL`) to run the
> full eval; otherwise you will see `429 RESOURCE_EXHAUSTED`, which the pipeline
> correctly handles by **failing closed to manual review**.

## Generate fixtures (one-off)

```powershell
python scripts/make_fixtures.py   # data/contracts/src/*.md -> *.pdf
```

Five synthetic contracts (A–E) carry 15 planted issues + 4 decoys + 1 injection
line. Answer keys live in `golden/` and are **never** loaded into prompts.

## Run

```powershell
python -m pytest tests/unit -q          # deterministic modules (no API key)
python -m pytest tests -q               # unit + regression
python evals/run_evals.py --runs 3      # 3-run scorecard (needs API key)
python scripts/build_demo_packages.py   # refresh offline sample ReviewPackages
streamlit run ui/streamlit_app.py       # demo UI (use Sample demo mode — no Gemini)
adk web                                  # ADK web; pick contract_review (see notes)
```

**Demo without Gemini:** in Streamlit, sidebar → **Sample demo (no Gemini)** →
pick Sample A–E → **Load sample**. Packages live in `data/demo_packages/`.

| Sample | Route | Theme |
|--------|--------|--------|
| A | low_risk_queue | Clean MSA |
| B | legal | Uncapped indemnity / liability |
| C | legal | 24-mo auto-renew, free assign, Singapore law |
| D | legal | Net 15, missing DPA, no SLA |
| E | procurement_owner | 8% price hike + prompt injection |

**Negotiation Simulator** tab → **Load CloudSoft replay (offline)** — state machine
+ simulated email thread (no real mail, no Gemini). Scenario/policy:
`data/negotiation/cloudsoft_pr2026.json`; replay fixture:
`data/negotiation/cloudsoft_replay.json`.

## Playbook

`data/playbook/demo-v1.json` — 10 rules (LIABILITY-001/002, PAYMENT-001,
RENEWAL-001, TERMINATION-001, DATA-001, SLA-001, PRICE-001, LAW-001, ASSIGN-001)
with preferred / fallback / walk-away thresholds, severity, Legal-review flag and
fallback drafting text. Editable without touching prompts; every run records the
version. *Illustrative demo policy, not legal standards.*

## Routing (deterministic, first match wins)

1. **Manual review** — extraction unreliable, or a high/Legal rule is `unclear` (fail closed).
2. **Legal** — a high rule at walk-away, or a Legal-review rule at walk-away / not-found.
3. **Procurement owner** — any other walk-away, or any security flag.
4. **Low-risk queue** — only preferred/fallback terms.

Expected fixture routes: A → low-risk, B → Legal, C → Legal, D → Legal,
E → procurement owner (+ security flag).

## Guardrails

- Contract text is **untrusted data**: regex preflight + prompt preamble +
  `suspicious_instructions`, all feeding `SecurityFlags` (any flag ⇒ at least R3,
  never lowers a route, always shown).
- **Grounding**: every quote is string-matched against the extracted page text;
  ungrounded quotes are dropped and logged.
- **Fail closed**: any stage failure ⇒ manual-review package (no "partial").
- **Redline drift**: `check_redlines()` rejects new numbers, removed defined
  terms, new obligations, or over-long rewrites; human review is the final control.
- Nothing is auto-approved, signed, emailed or persisted to an external system.

## ADK notes (Milestone 0)

The ADK 2.x workflow API (`Workflow`, `@node`, `ctx.run_node`, `RequestInput`,
`Agent(mode="single_turn")`) must be verified against the installed `google-adk`
version before relying on the `adk web` path — the exact surface is pinned at
Milestone 0 in `app/adk_workflow.py`. The orchestrator path
(`app/orchestrator.py`) has no ADK dependency, so the evals, tests and Streamlit
UI run regardless. Pin versions with `pip freeze > requirements.txt` after the
first green run.

## Limitations

Text-based PDFs only (scanned PDFs are detected and sent to manual review); no
OCR, web research, e-signature, email, or external writes; synthetic contracts
only; routing is a recommendation, not an approval.

## Phase 0 / 1 / 1B smokes

```powershell
python scripts\smoke_adk.py
python scripts\smoke_intake.py data\contracts\contract_a.pdf
python scripts\smoke_clause_slice.py data\contracts\contract_a.pdf --fake
python scripts\smoke_clause_slice.py data\contracts\contract_a.pdf   # live Gemini
python -m pytest tests\unit -q
```

Phase 1B/2 (`app/slice.py`) uses playbook `demo-slice-v1` (PAYMENT / SLA / DATA)
plus grounded metadata extraction. Unit tests inject fake agents and do not
require an API key. Ungrounded metadata quotes clear both value and evidence.

Phase 4 commercial comparison (`app/commercial.py`, no LLM):

```powershell
python scripts\smoke_commercial.py
```

Fixtures: `data/quotations/{cloudsoft,technova,datasphere}.json` — absolute INR
vs budget `4_800_000` (₹48 lakh). Lowest fee is labelled; SLA/term mismatches
are shown; nothing is auto-selected as the winner.
