"""Minimal Google ADK agent smoke test (Phase 0).

Verifies the pinned ADK package can reach Gemini and return text.
Does not exercise the full contract-review workflow.

Usage (from repo root, venv active, GOOGLE_API_KEY set):
    python scripts/smoke_adk.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / "contract_review" / ".env")


async def _run() -> None:
    import google.adk as adk
    from google.adk import Agent, Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    version = getattr(adk, "__version__", None) or getattr(
        adk, "version", "unknown"
    )
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is not set (contract_review/.env).")

    model = os.environ.get("REVIEW_MODEL", "gemini-flash-latest")
    # Thinking-capable models may consume tokens before visible text;
    # keep max_output_tokens comfortably above a short reply.
    agent = Agent(
        name="smoke_echo",
        model=model,
        description="Phase 0 connectivity check",
        instruction=(
            "Reply with exactly: ADK_SMOKE_OK. "
            "Do not add any other words."
        ),
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0, max_output_tokens=256
        ),
    )

    session_service = InMemorySessionService()
    app_name = "smoke_adk"
    user_id = "phase0"
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id
    )
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part.from_text(text="ping")],
        ),
    ):
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                final_text += text

    print(f"google-adk version: {version}")
    print(f"model: {model}")
    print(f"session_id: {session.id}")
    print(f"response: {final_text!r}")
    if "ADK_SMOKE_OK" not in final_text:
        raise SystemExit(
            "Smoke failed: expected ADK_SMOKE_OK in model response. "
            f"Got: {final_text!r}"
        )
    print("ADK smoke: PASS")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
