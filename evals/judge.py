"""Redline judge (eval only). gemini-pro-latest, temperature 0.

Scores each redline 0-2 on three criteria. Never runs in the live pipeline and
never scores grounding or recall. Calibrate against hand grades on 10 redlines
and report judge-human agreement in the scorecard.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app import agents
from app.prompts import JUDGE_PROMPT
from app.schemas import Redline


class JudgeInput(BaseModel):
    rule_id: str
    original: str
    proposed: str
    fallback_text: str


class JudgeScore(BaseModel):
    meets_fallback: int = Field(ge=0, le=2)
    preserves_meaning: int = Field(ge=0, le=2)
    coherent: int = Field(ge=0, le=2)
    justification: str = ""

    @property
    def total(self) -> int:
        return self.meets_fallback + self.preserves_meaning + self.coherent


def judge_redline(r: Redline, fallback_text: str) -> JudgeScore:
    inp = JudgeInput(
        rule_id=r.rule_id,
        original=r.original.quote,
        proposed=r.proposed_text,
        fallback_text=fallback_text,
    )
    return agents._run(agents.PRO, JUDGE_PROMPT, inp, JudgeScore, 0.0)
