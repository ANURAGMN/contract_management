"""Plain async orchestrator: the reliable engine behind run_review().

It mirrors the ADK ``review_contract`` node exactly (same order, same parallel
steps, same data contract) but has no dependency on ADK's workflow API, so the
evals and the Streamlit UI run regardless of the installed ADK version. The ADK
wrapper in ``app/adk_workflow.py`` reproduces this flow for the ``adk web`` demo.

Only the three agent calls touch a model. Severity, routing, dates and money
never pass through one; the human gate is handled in the UI / ADK layer.
"""

from __future__ import annotations

import asyncio

from app import agents, steps
from app.schemas import (
    ClauseReviewInput,
    GroundingInput,
    ObligationInput,
    RedlineInput,
    ReviewPackage,
    ReviewRequest,
    RouteInput,
    RulesInput,
    ScoredFinding,
    SecurityFlags,
)


async def _to_thread(fn, *args):
    return await asyncio.to_thread(fn, *args)


async def review(req: ReviewRequest) -> ReviewPackage:
    doc = steps.intake(req)
    if doc.preflight.unreliable_extraction:
        return steps.manual_review_package(doc, reason="extraction unreliable")

    playbook = steps.load_playbook(req.playbook_version)

    # 2a-2c run in parallel.
    meta, clauses, perf = await asyncio.gather(
        _to_thread(agents.metadata_extractor, doc),
        _to_thread(
            agents.clause_reviewer,
            ClauseReviewInput(pages=doc.pages, rules=playbook.prompt_view()),
        ),
        _to_thread(steps.performance_tool, req),
    )

    # Evidence gate: exactly one grounded result per playbook rule.
    grounded = steps.ground(
        GroundingInput(doc=doc, meta=meta, clauses=clauses, playbook=playbook)
    )
    scored = steps.apply_rules(
        RulesInput(grounded=grounded, playbook=playbook, pages=doc.pages)
    )

    flags = SecurityFlags(
        pre=doc.preflight.injection_hits, model=clauses.suspicious_instructions
    )
    route = steps.route(
        RouteInput(scored=scored, preflight=doc.preflight, flags=flags)
    )

    deviating: list[ScoredFinding] = [
        f for f in scored.findings if f.status == "walk_away" and f.grounded.finding.evidence
    ]

    # 6a-6b run in parallel.
    async def _draft():
        if not deviating:
            from app.schemas import Redlines

            return Redlines(redlines=[])
        return await _to_thread(
            agents.redline_drafter,
            RedlineInput(findings=deviating, playbook=playbook.prompt_view()),
        )

    redlines_raw, (obligations, credit) = await asyncio.gather(
        _draft(),
        _to_thread(
            steps.build_obligations,
            ObligationInput(meta=grounded.meta, perf=perf, today=steps.today()),
        ),
    )

    checked = steps.check_redlines(redlines_raw, deviating, playbook)

    return steps.assemble(
        doc, grounded, scored, route, checked, obligations, flags, credit
    )
