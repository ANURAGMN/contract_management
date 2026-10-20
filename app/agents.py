"""The three single-turn LLM agents, implemented on google-genai.

Each agent is schema-in / schema-out, temperature near zero, with no tools and
no conversation history. Gemini returns structured JSON validated against the
Pydantic output schema. The ADK wrapper in ``app/adk_workflow.py`` reuses the
same prompts and schemas for the ``adk web`` demo.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

from app.prompts import (
    CLAUSE_PROMPT,
    JUDGE_PROMPT,
    METADATA_PROMPT,
    REDLINE_PROMPT,
)
from app.schemas import (
    ClauseFindings,
    ClauseReviewInput,
    ContractDoc,
    ContractMetadata,
    RedlineInput,
    Redlines,
)

T = TypeVar("T", bound=BaseModel)

_ENV_PATH = Path(__file__).resolve().parent.parent / "contract_review" / ".env"


def _load_env() -> None:
    if os.environ.get("GOOGLE_API_KEY"):
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_PATH)
    except Exception:
        # Minimal fallback parser if python-dotenv is unavailable.
        if _ENV_PATH.exists():
            for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())


@lru_cache(maxsize=1)
def _client():
    _load_env()
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Put it in contract_review/.env."
        )
    return genai.Client(api_key=api_key)


def _env(name: str, default: str) -> str:
    _load_env()
    return os.environ.get(name, default)


# Pin a dated model name before the demo via REVIEW_MODEL / REVIEW_JUDGE_MODEL.
FLASH = _env("REVIEW_MODEL", "gemini-flash-latest")
PRO = _env("REVIEW_JUDGE_MODEL", "gemini-pro-latest")  # eval judge only

# "Use the available model": a priority chain the client rotates through when a
# model is daily-exhausted (429 PerDay) or persistently unavailable (503). The
# preferred model (REVIEW_MODEL) is tried first; override the chain with
# REVIEW_MODELS (comma-separated).
_DEFAULT_CHAIN = [
    FLASH,
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3-flash-preview",
]
_chain_env = _env("REVIEW_MODELS", "")
MODEL_CHAIN = (
    [m.strip() for m in _chain_env.split(",") if m.strip()]
    if _chain_env
    else list(dict.fromkeys(_DEFAULT_CHAIN))  # de-dupe, keep order
)
_active = [0]  # index into MODEL_CHAIN of the current working model

# Free-tier models cap requests-per-minute (flash-lite = 10/min). Space calls so
# parallel/consecutive requests stay under the cap. Tune via REVIEW_MIN_INTERVAL.
_MIN_INTERVAL = float(_env("REVIEW_MIN_INTERVAL", "7"))
_rate_lock = threading.Lock()
_last_call = [0.0]


def _throttle() -> None:
    import time

    with _rate_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def _call_once(model: str, contents: str, cfg) -> object:
    """One model call with bounded retry for per-minute 429 and transient
    503/500. Raises on a per-day 429 (unrecoverable) or when retries run out."""
    import time

    from google.genai import errors as genai_errors

    client = _client()
    attempts = int(_env("REVIEW_MAX_RETRIES", "4"))
    backoff = float(_env("REVIEW_BACKOFF", "12"))
    for attempt in range(attempts):
        _throttle()
        from app import ai_trace

        ai_trace.note_request(model)
        try:
            return client.models.generate_content(model=model, contents=contents, config=cfg)
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            code = getattr(e, "code", None)
            if code == 429 and "PerDay" in str(e):
                raise  # daily cap: let the caller rotate to the next model
            if code in (429, 500, 503) and attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            raise


def _run(model: str, system: str, payload: BaseModel, schema: Type[T], temp: float) -> T:
    from google.genai import types
    from google.genai import errors as genai_errors

    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temp,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=schema,
    )
    contents = payload.model_dump_json()

    import time

    from app import ai_trace

    started, t0, tried = ai_trace.now_iso(), time.monotonic(), 0

    def _log(ok: bool, model: str | None, err: Exception | None = None) -> None:
        ai_trace.record(ai_trace.AICall(
            agent=schema.__name__, model=model, started_at=started,
            seconds=round(time.monotonic() - t0, 2), ok=ok, attempts=max(tried, 1),
            error=None if err is None else f"{type(err).__name__}: {str(err)[:200]}",
        ))

    # Rotate through the chain starting at the current active model; advance past
    # any model that is daily-exhausted or persistently unavailable.
    last_err: Exception | None = None
    n = len(MODEL_CHAIN)
    for step in range(n):
        idx = (_active[0] + step) % n
        candidate = MODEL_CHAIN[idx]
        tried += 1
        try:
            resp = _call_once(candidate, contents, cfg)
            _active[0] = idx  # stick with the model that worked
            parsed = getattr(resp, "parsed", None)
            result = parsed if isinstance(parsed, schema) else schema.model_validate_json(resp.text)
            _log(True, candidate)
            return result
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            last_err = e
            continue
        except Exception as e:  # malformed output etc. — record, then surface
            _log(False, candidate, e)
            raise
    err = last_err if last_err else RuntimeError("no model available")
    _log(False, None, err)
    raise err


# --------------------------------------------------------------------------- #
# Public agent calls (names match the guide)
# --------------------------------------------------------------------------- #
def metadata_extractor(doc: ContractDoc) -> ContractMetadata:
    return _run(FLASH, METADATA_PROMPT, doc, ContractMetadata, 0.0)


def clause_reviewer(inp: ClauseReviewInput) -> ClauseFindings:
    return _run(FLASH, CLAUSE_PROMPT, inp, ClauseFindings, 0.0)


def redline_drafter(inp: RedlineInput) -> Redlines:
    return _run(FLASH, REDLINE_PROMPT, inp, Redlines, 0.2)


def judge_available() -> bool:
    try:
        _client()
        return True
    except Exception:
        return False
