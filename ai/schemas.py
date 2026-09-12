"""Pydantic schemas for every structured Gemini response.

These are passed to the Gemini API as response schemas, so the model is
constrained to emit valid JSON rather than being asked politely for it. Any
response that fails validation is discarded by the client and treated as a
failure, which is what makes the fallback path reliable.

Deliberately constraint-light: enums and types only, no string lengths or
numeric bounds. Gemini's schema converter handles those inconsistently, and a
rejected schema is an API error rather than a graceful fallback. Ranges are
clamped in gemini_client.py instead.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MoveLiteral = Literal["COOPERATE", "DEFECT"]

ClassificationLiteral = Literal[
    "Always Cooperate",
    "Always Defect",
    "Tit-for-Tat-like",
    "Forgiving",
    "Retaliatory",
    "Opportunistic",
    "Unpredictable",
    "Mixed Strategy",
]


class OpponentMove(BaseModel):
    """One decision from the adaptive Gemini opponent."""

    move: MoveLiteral
    confidence: float = Field(description="How sure you are, from 0 to 1")
    reason: str = Field(description="One or two sentences, visible to the player")


class MatchAnalysis(BaseModel):
    """Educational read on how the player played."""

    strategy_summary: str = Field(
        description="What the player's behaviour most resembled, in one sentence"
    )
    did_well: list[str] = Field(description="Two to three specific strengths")
    weaknesses: list[str] = Field(description="One to three specific weaknesses")
    takeaway: str = Field(description="The strategic lesson, in two or three sentences")
    concept: str = Field(description="The game-theory concept most relevant here")


class StrategyClassification(BaseModel):
    """Approximate label for the player's observed behaviour."""

    classification: ClassificationLiteral
    confidence: float = Field(description="How well the label fits, from 0 to 1")
    explanation: str = Field(description="Which observed behaviour justifies the label")
    recommendation: str = Field(description="One concrete thing to try next match")


class TutorTurn(BaseModel):
    """One step of the Socratic tutor."""

    response: str = Field(description="Reaction to the learner's last answer")
    question: str = Field(description="The next question to put to them")
    assessment: Literal["correct", "partially_correct", "incorrect", "opening"]
    difficulty_hint: Literal["easier", "same", "harder"]


class TutorHint(BaseModel):
    """A nudge that must not contain the answer."""

    hint: str = Field(description="A nudge toward the answer, never the answer itself")