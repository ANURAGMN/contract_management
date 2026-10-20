"""Run negotiation agents with provenance, and record live runs for Replay.

Every agent step is executed inside ``ai_trace.capture()``. Messages created by
that step are attributed to the calls it made, so the UI can label each email
"generated live by <model> in <n> s", "offline scripted agent" or "replayed".
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from app import ai_trace, recordings
from app.negotiation_agents import (
    analyze_supplier_reply,
    draft_buyer_email,
    generate_strategy,
    get_backend,
    human_approve_send,
    plan_next_move,
    simulate_supplier_reply,
)
from app.negotiation_schemas import NegotiationScenario, NegotiationState

Source = Literal["live", "scripted", "human"]


class MessageOrigin(BaseModel):
    source: Source
    step: str
    models: list[str] = Field(default_factory=list)
    seconds: float = 0.0
    calls: int = 0


class AgentRun(BaseModel):
    """A negotiation run plus where every message came from."""

    backend: Literal["live", "scripted"]
    scenario: NegotiationScenario
    state: NegotiationState
    origins: dict[str, MessageOrigin] = Field(default_factory=dict)
    steps: list[dict] = Field(default_factory=list)  # step name + call summary
    ai_calls: list[dict] = Field(default_factory=list)


STEPS: dict[str, Callable] = {
    "strategy": generate_strategy,
    "draft": draft_buyer_email,
    "approve": human_approve_send,
    "supplier_reply": simulate_supplier_reply,
    "analyze": analyze_supplier_reply,
    "next_move": plan_next_move,
}


def _message_ids(state: NegotiationState) -> set[str]:
    ids = {m.message_id for m in state.thread}
    if state.pending_draft:
        ids.add(state.pending_draft.message_id)
    return ids


def step(run: AgentRun, name: str) -> AgentRun:
    """Execute one agent step on the run, recording calls and message origins."""
    fn = STEPS[name]
    backend = get_backend(run.backend)
    before = _message_ids(run.state)
    with ai_trace.capture() as calls:
        if name == "approve":
            new_state = fn(run.state)
        else:
            new_state = fn(run.state, run.scenario, backend=backend)
    summary = ai_trace.summarize(calls)
    source: Source = "human" if name == "approve" else run.backend
    for mid in _message_ids(new_state) - before:
        run.origins[mid] = MessageOrigin(source=source, step=name, models=summary["models"],
                                         seconds=summary["seconds"], calls=summary["calls"])
    run.state = new_state
    run.steps.append({"step": name, **summary})
    run.ai_calls.extend(c.model_dump() for c in calls)
    return run


def run_full(scenario: NegotiationScenario, backend: Literal["live", "scripted"], *, max_cycles: int = 2,
             on_step: Optional[Callable[[str, AgentRun], None]] = None) -> AgentRun:
    """Strategy → (draft → approve → supplier → analyze → next move) × cycles."""
    from app.negotiation import new_run

    run = AgentRun(backend=backend, scenario=scenario, state=new_run(scenario))
    order = ["strategy"] + ["draft", "approve", "supplier_reply", "analyze", "next_move"] * max_cycles
    for i, name in enumerate(order):
        if name == "draft" and i > 1 and run.state.status != "NEXT_MOVE_READY":
            break
        run = step(run, name)
        if on_step:
            on_step(name, run)
    return run


def record(run: AgentRun) -> recordings.Recording:
    return recordings.save(
        "negotiation", run.scenario.supplier.name, run, run.ai_calls,
        meta={"status": run.state.status, "rounds": run.state.round_number,
              "requisition_id": run.scenario.requisition_id},
    )


def load_recorded(rec: recordings.Recording) -> AgentRun:
    return AgentRun.model_validate(rec.payload)
