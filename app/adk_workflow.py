"""ADK 2.x wrapper for the `adk web` / `adk eval` demo.

Verified against google-adk 2.12.0 (Milestone 0): `Agent`, `Workflow`, `@node`,
`ctx.run_node`, `Event`, `RequestInput`, `START` and `FunctionNode` all exist and
are used below. The LLM agents are ADK `Agent`s so they appear in `adk web`; the
deterministic steps are wrapped as `FunctionNode`s so the model/code boundary is
visible in one file. This reproduces the exact flow of `app/orchestrator.py`.

The reliable engine the evals and Streamlit use is `app/orchestrator.py`; this
module is the showcase runtime and shares the same prompts, schemas and steps.
"""

from __future__ import annotations

import asyncio
import json

from google.adk import Agent, Context, Event, Workflow
from google.adk.events import RequestInput
from google.adk.workflow import START, node
from google.genai import types

from app import steps
from app.agents import FLASH  # env-configured model (REVIEW_MODEL), shared with the genai path
from app.callbacks import audit_after_agent
from app.prompts import CLAUSE_PROMPT, METADATA_PROMPT, REDLINE_PROMPT
from app.schemas import (
    ClauseFindings,
    ClauseReviewInput,
    ContractDoc,
    ContractMetadata,
    GateDecision,
    GroundingInput,
    ObligationInput,
    RedlineInput,
    Redlines,
    ReviewRequest,
    RouteInput,
    RulesInput,
    ScoredFinding,
    SecurityFlags,
)

def _cfg(t: float) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(temperature=t, max_output_tokens=8192)


# --------------------------------------------------------------------------- #
# LLM agents (shown in adk web)
# --------------------------------------------------------------------------- #
metadata_extractor = Agent(
    name="metadata_extractor", model=FLASH, mode="single_turn",
    description="Extracts contract metadata with verbatim evidence.",
    instruction=METADATA_PROMPT,
    input_schema=ContractDoc, output_schema=ContractMetadata,
    generate_content_config=_cfg(0.0), after_agent_callback=audit_after_agent,
)
clause_reviewer = Agent(
    name="clause_reviewer", model=FLASH, mode="single_turn",
    description="Maps contract clauses to playbook rules with quotes.",
    instruction=CLAUSE_PROMPT,
    input_schema=ClauseReviewInput, output_schema=ClauseFindings,
    generate_content_config=_cfg(0.0), after_agent_callback=audit_after_agent,
)
redline_drafter = Agent(
    name="redline_drafter", model=FLASH, mode="single_turn",
    description="Drafts fallback redlines for deviating clauses.",
    instruction=REDLINE_PROMPT,
    input_schema=RedlineInput, output_schema=Redlines,
    generate_content_config=_cfg(0.2), after_agent_callback=audit_after_agent,
)


def _as_model(cls, value):
    return value if isinstance(value, cls) else cls.model_validate(value)


def _request_from_input(node_input) -> ReviewRequest:
    """Coerce the workflow entry input into a ReviewRequest.

    Accepts: a ReviewRequest, a JSON object (as text), or a bare PDF path such
    as 'data/contracts/contract_b.pdf'. Verified at Milestone 0: the Runner
    delivers the user message as a google.genai types.Content, so unwrap parts.
    """
    if isinstance(node_input, ReviewRequest):
        return node_input
    if isinstance(node_input, dict):
        return ReviewRequest.model_validate(node_input)

    text = node_input
    parts = getattr(node_input, "parts", None)
    if parts:
        text = "".join(getattr(p, "text", "") or "" for p in parts)
    text = (text or "").strip() if isinstance(text, str) else text

    if not text:
        raise ValueError(
            'Send a request, e.g. {"pdf_path": "data/contracts/contract_b.pdf", '
            '"playbook_version": "demo-v1"} or just a PDF path.'
        )
    if isinstance(text, str):
        if text.lstrip().startswith("{"):
            return ReviewRequest.model_validate(json.loads(text))
        # Treat a bare line as a PDF path (optionally "pdf | csv | supplier_id").
        bits = [b.strip() for b in text.split("|")]
        req = {"pdf_path": bits[0], "playbook_version": "demo-v1"}
        if len(bits) > 1 and bits[1]:
            req["csv_path"] = bits[1]
        if len(bits) > 2 and bits[2]:
            req["supplier_id"] = bits[2]
        return ReviewRequest.model_validate(req)
    return ReviewRequest.model_validate(text)


# --------------------------------------------------------------------------- #
# Human gate (RequestInput; only when route == legal and ask_gate)
# --------------------------------------------------------------------------- #
@node(rerun_on_resume=False)
async def legal_gate(ctx: Context, node_input):
    yield RequestInput(
        message="High-risk terms found. Confirm escalation to Legal:\n"
        + "\n".join(node_input.trace),
        payload=node_input,
        response_schema=GateDecision,
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@node(rerun_on_resume=True)
async def review_contract(ctx: Context, node_input):
    req = _request_from_input(node_input)

    doc = steps.intake(req)
    yield Event(state={"run_id": doc.run_id, "playbook_version": doc.playbook_version})
    if doc.preflight.unreliable_extraction:
        yield Event(output=steps.manual_review_package(doc, reason="extraction unreliable"))
        return

    playbook = steps.load_playbook(req.playbook_version)

    # The two LLM agents run in parallel (shown in adk web). The performance
    # tool is fast, deterministic CSV work, so it runs directly.
    meta_f = ctx.run_node(metadata_extractor, doc)
    clause_f = ctx.run_node(
        clause_reviewer, ClauseReviewInput(pages=doc.pages, rules=playbook.prompt_view())
    )
    meta_raw, clauses_raw = await asyncio.gather(meta_f, clause_f)
    perf = steps.performance_tool(req)
    meta = _as_model(ContractMetadata, meta_raw)
    clauses = _as_model(ClauseFindings, clauses_raw)

    grounded = steps.ground(
        GroundingInput(doc=doc, meta=meta, clauses=clauses, playbook=playbook)
    )
    scored = steps.apply_rules(
        RulesInput(grounded=grounded, playbook=playbook, pages=doc.pages)
    )
    flags = SecurityFlags(
        pre=doc.preflight.injection_hits, model=clauses.suspicious_instructions
    )
    route = steps.route(RouteInput(scored=scored, preflight=doc.preflight, flags=flags))
    yield Event(state={"route": route.route})

    deviating: list[ScoredFinding] = [
        f for f in scored.findings
        if f.status == "walk_away" and f.grounded.finding.evidence
    ]
    if deviating:
        redlines_raw = await ctx.run_node(
            redline_drafter,
            RedlineInput(findings=deviating, playbook=playbook.prompt_view()),
        )
        redlines = _as_model(Redlines, redlines_raw)
    else:
        redlines = Redlines(redlines=[])

    obligations, credit = steps.build_obligations(
        ObligationInput(meta=grounded.meta, perf=perf, today=steps.today())
    )
    checked = steps.check_redlines(redlines, deviating, playbook)

    gate = None
    if route.route == "legal" and req.ask_gate:
        gate = await ctx.run_node(legal_gate, route)

    yield Event(
        output=steps.assemble(
            doc, grounded, scored, route, checked, obligations, flags, credit, gate
        )
    )


root_agent = Workflow(name="contract_review", edges=[(START, review_contract)])
