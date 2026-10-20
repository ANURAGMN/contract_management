"""run_review(): the single entry point used by the evals and the Streamlit UI.

It wraps the orchestrator with the fail-closed guarantee: any unhandled stage
failure (including a malformed agent response after retries) returns a
manual-review package rather than a partial result.
"""

from __future__ import annotations

import asyncio

from app import orchestrator, steps
from app.schemas import ReviewPackage, ReviewRequest


async def run_review_async(req: ReviewRequest) -> ReviewPackage:
    from app import ai_trace

    with ai_trace.capture() as calls:
        try:
            pkg = await orchestrator.review(req)
        except Exception as e:  # outside the deterministic nodes; safe to catch here
            pkg = steps.manual_review_package(
                req, reason=f"stage failed: {type(e).__name__}: {e}"
            )
    # Provenance: this package came from a live run; keep the call trace with it.
    pkg.provenance = "live"
    pkg.reviewed_at = ai_trace.now_iso()
    pkg.ai_calls = [c.model_dump() for c in calls]
    return pkg


def record_review(pkg: ReviewPackage):
    """Save a live review as a recording so Replay mode can play it back."""
    from app import recordings

    label = pkg.file_name or pkg.run_id
    return recordings.save(
        "review", label, pkg, pkg.ai_calls,
        meta={"status": pkg.status, "route": pkg.route.route},
    )


def run_review(req: ReviewRequest) -> ReviewPackage:
    """Synchronous wrapper for Streamlit / scripts."""
    return asyncio.run(run_review_async(req))
