"""Recorded live AI runs — the source for Replay mode.

A recording is the exact output of a live Gemini run plus its call trace, saved
as JSON under ``data/recordings/<kind>/``. Recordings live in ``data/`` (not
``runs/``) so they can be committed: the hosted demo then replays real runs,
never hand-made fixtures dressed up as AI output.

Kinds: ``review`` (ReviewPackage), ``negotiation`` (NegotiationState),
``intake`` (procurement intake-assistant draft).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = ROOT / "data" / "recordings"

Kind = Literal["review", "negotiation", "intake"]


class Recording(BaseModel):
    kind: Kind
    recording_id: str
    label: str  # e.g. the contract file name or supplier
    recorded_at: str  # ISO-8601 UTC
    models: list[str] = Field(default_factory=list)
    calls: int = 0
    failed_calls: int = 0
    seconds: float = 0.0
    ai_calls: list[dict] = Field(default_factory=list)
    payload: dict  # the produced object (model_dump(mode="json"))
    meta: dict = Field(default_factory=dict)

    @property
    def path(self) -> Path:
        return RECORDINGS / self.kind / f"{self.recording_id}.json"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:60] or "run"


def save(kind: Kind, label: str, payload: BaseModel | dict, ai_calls: list, meta: Optional[dict] = None) -> Recording:
    from app.ai_trace import AICall, summarize

    calls = [c if isinstance(c, AICall) else AICall.model_validate(c) for c in ai_calls]
    summary = summarize(calls)
    now = datetime.now(timezone.utc)
    rec = Recording(
        kind=kind,
        recording_id=f"{_slug(label)}__{now.strftime('%Y%m%dT%H%M%SZ')}",
        label=label,
        recorded_at=now.isoformat(timespec="seconds"),
        models=summary["models"],
        calls=summary["calls"],
        failed_calls=summary["failed"],
        seconds=summary["seconds"],
        ai_calls=[c.model_dump() for c in calls],
        payload=payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload,
        meta=meta or {},
    )
    rec.path.parent.mkdir(parents=True, exist_ok=True)
    rec.path.write_text(rec.model_dump_json(indent=2), encoding="utf-8")
    return rec


def list_recordings(kind: Kind) -> list[Recording]:
    folder = RECORDINGS / kind
    out: list[Recording] = []
    if not folder.exists():
        return out
    for p in folder.glob("*.json"):
        try:
            out.append(Recording.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return sorted(out, key=lambda r: r.recorded_at, reverse=True)


def load(kind: Kind, recording_id: str) -> Optional[Recording]:
    p = RECORDINGS / kind / f"{recording_id}.json"
    if not p.exists():
        return None
    return Recording.model_validate_json(p.read_text(encoding="utf-8"))


def latest_by_label(kind: Kind) -> dict[str, Recording]:
    """Most recent recording per label (e.g. per contract file)."""
    out: dict[str, Recording] = {}
    for r in list_recordings(kind):  # newest first
        out.setdefault(r.label, r)
    return out
