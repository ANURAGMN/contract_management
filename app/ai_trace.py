"""Proof of what the model actually did.

Every Gemini call made through ``app.agents._run`` is recorded here: which
agent, which model answered, how long it took, and whether it succeeded. The
UI uses this to label output as *generated live* (with model and timing) versus
replayed, and to show how much of the free-tier daily quota has been used.

Calls are collected per review/negotiation with :func:`capture`, which works
across ``asyncio.to_thread`` because context variables are copied into threads.
"""

from __future__ import annotations

import contextvars
import json
import threading
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
USAGE_FILE = ROOT / "runs" / "ai_usage.json"
FREE_TIER_DAILY_REQUESTS = 20  # AI Studio free tier, per model family (approx.)

_lock = threading.Lock()


class AICall(BaseModel):
    """One logical agent call (may include retries / model rotation)."""

    agent: str
    model: Optional[str] = None
    started_at: str
    seconds: float
    ok: bool
    attempts: int = 1
    error: Optional[str] = None


_current: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("ai_calls", default=None)


@contextmanager
def capture() -> Iterator[list[AICall]]:
    """Collect every model call made inside the block."""
    calls: list[AICall] = []
    token = _current.set(calls)
    try:
        yield calls
    finally:
        _current.reset(token)


def record(call: AICall) -> None:
    sink = _current.get()
    if sink is not None:
        with _lock:
            sink.append(call)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Daily request counter (each generate_content attempt counts against quota)
# --------------------------------------------------------------------------- #
def _read_usage() -> dict:
    try:
        return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def note_request(model: str) -> None:
    today = date.today().isoformat()
    with _lock:
        data = _read_usage()
        day = data.setdefault(today, {})
        day[model] = day.get(model, 0) + 1
        # keep a week of history at most
        for k in sorted(data)[:-7]:
            data.pop(k, None)
        try:
            USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            USAGE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass


def requests_today() -> dict[str, int]:
    return dict(_read_usage().get(date.today().isoformat(), {}))


def summarize(calls: list[AICall]) -> dict:
    ok = [c for c in calls if c.ok]
    models = sorted({c.model for c in ok if c.model})
    return {
        "calls": len(calls),
        "ok": len(ok),
        "failed": len(calls) - len(ok),
        "models": models,
        "seconds": round(sum(c.seconds for c in calls), 1),
    }
