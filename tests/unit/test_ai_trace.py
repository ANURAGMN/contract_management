"""Provenance: live calls are traced, stamped on packages and recorded for Replay."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app import agents, ai_trace, recordings, runner
from app.demo_packages import load_sample


class _Out(BaseModel):
    answer: str


class _In(BaseModel):
    q: str


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(recordings, "RECORDINGS", tmp_path / "recordings")
    monkeypatch.setattr(ai_trace, "USAGE_FILE", tmp_path / "usage.json")


def _fake_genai(monkeypatch):
    """Stand-in for google.genai so _run can execute without network."""
    import sys

    types_mod = SimpleNamespace(GenerateContentConfig=lambda **kw: kw)

    class _Err(Exception):
        pass

    errors_mod = SimpleNamespace(ClientError=_Err, ServerError=_Err)
    google = sys.modules.get("google") or SimpleNamespace()
    genai = SimpleNamespace(types=types_mod, errors=errors_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    monkeypatch.setitem(sys.modules, "google.genai.errors", errors_mod)
    if not hasattr(google, "genai"):
        monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=genai))


def test_run_records_model_and_timing(monkeypatch):
    _fake_genai(monkeypatch)
    monkeypatch.setattr(
        agents, "_call_once",
        lambda model, contents, cfg: SimpleNamespace(parsed=_Out(answer="ok"), text=""),
    )
    with ai_trace.capture() as calls:
        out = agents._run("gemini-test", "sys", _In(q="x"), _Out, 0.0)
    assert out.answer == "ok"
    assert len(calls) == 1
    assert calls[0].ok and calls[0].agent == "_Out"
    assert calls[0].model == agents.MODEL_CHAIN[agents._active[0]]


def test_run_records_failure(monkeypatch):
    _fake_genai(monkeypatch)

    def boom(model, contents, cfg):
        return SimpleNamespace(parsed=None, text="not json")

    monkeypatch.setattr(agents, "_call_once", boom)
    with ai_trace.capture() as calls:
        with pytest.raises(Exception):
            agents._run("gemini-test", "sys", _In(q="x"), _Out, 0.0)
    assert len(calls) == 1 and not calls[0].ok and calls[0].error


def test_capture_crosses_threads():
    def worker():
        ai_trace.record(ai_trace.AICall(agent="A", model="m", started_at="t", seconds=1.0, ok=True))

    async def main():
        await asyncio.to_thread(worker)

    with ai_trace.capture() as calls:
        asyncio.run(main())
    assert [c.agent for c in calls] == ["A"]


def test_live_review_is_stamped_and_recorded(monkeypatch):
    sample = load_sample("contract_e_procurement")

    async def fake_review(req):
        await asyncio.to_thread(
            ai_trace.record,
            ai_trace.AICall(agent="ClauseFindings", model="gemini-x", started_at="t", seconds=2.5, ok=True),
        )
        return sample.model_copy()

    monkeypatch.setattr(runner.orchestrator, "review", fake_review)
    pkg = runner.run_review(SimpleNamespace(pdf_path="contract_e.pdf"))
    assert pkg.provenance == "live" and pkg.reviewed_at
    assert pkg.ai_calls and pkg.ai_calls[0]["model"] == "gemini-x"

    rec = runner.record_review(pkg)
    assert rec.models == ["gemini-x"] and rec.calls == 1
    latest = recordings.latest_by_label("review")
    assert latest["contract_e.pdf"].recording_id == rec.recording_id


def test_usage_counter(monkeypatch):
    ai_trace.note_request("gemini-a")
    ai_trace.note_request("gemini-a")
    assert ai_trace.requests_today() == {"gemini-a": 2}
