"""Gemini API wrapper.

Every public function here returns either a validated pydantic object or
None. Nothing raises. The caller treats None as "fall back to deterministic
logic", which is what keeps the game playable when the API is missing, rate
limited, slow, or returning nonsense.

The google-genai SDK is imported lazily inside get_client(), so the app
starts without paying for it and a missing dependency degrades into a
fallback rather than crashing on import.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from ai import prompts, schemas
from game.analytics import MatchStats

logger = logging.getLogger(__name__)

# Cheap and fast: one decision per round, latency is felt directly.
MOVE_MODEL = "gemini-3.1-flash-lite"
# Better reasoning: called once per match or per tutoring turn.
REASONING_MODEL = "gemini-3.6-flash"

MAX_ATTEMPTS = 2
TIMEOUT_MS = 20_000

T = TypeVar("T", bound=BaseModel)


@dataclass
class AIStatus:
    """What the UI needs to know, without seeing a raw exception."""

    available: bool
    message: str


def resolve_api_key() -> str | None:
    """Find the key in the environment, then Streamlit secrets.

    Checked in that order so a local shell export beats a committed default.
    Streamlit is imported inside the function to keep this module usable from
    plain scripts and tests.
    """
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value.strip()

    try:
        import streamlit as st

        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if name in st.secrets:
                return str(st.secrets[name]).strip()
    except Exception:  # noqa: BLE001 - secrets file absent is a normal state
        pass

    return None


_client: Any = None
_client_failed = False


def get_client() -> Any:
    """Build the SDK client once, or return None if that is not possible."""
    global _client, _client_failed

    if _client is not None or _client_failed:
        return _client

    key = resolve_api_key()
    if not key:
        _client_failed = True
        return None

    try:
        from google import genai

        _client = genai.Client(api_key=key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not create Gemini client: %s", exc)
        _client_failed = True
        return None

    return _client


def reset_client() -> None:
    """Forget a failed client so a fixed key can be picked up without a restart."""
    global _client, _client_failed
    _client = None
    _client_failed = False


def status() -> AIStatus:
    """A message safe to show a user, with no exception text in it."""
    if not resolve_api_key():
        return AIStatus(
            False,
            "No Gemini API key found. Add GEMINI_API_KEY to "
            ".streamlit/secrets.toml locally, or to Settings → Secrets on "
            "Streamlit Cloud. Deterministic opponents work without it.",
        )
    if get_client() is None:
        return AIStatus(
            False,
            "A Gemini API key was found, but the client could not be created. "
            "Check that google-genai is installed.",
        )
    return AIStatus(True, "Gemini connected.")


def _generate(
    system: str, prompt: str, schema: type[T], model: str, temperature: float = 0.7
) -> T | None:
    """One structured call, retried once, never raising.

    Uses the generateContent API with a response schema, so the model is
    constrained to the shape rather than asked for JSON in prose.
    """
    client = get_client()
    if client is None:
        return None

    last_error: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "system_instruction": system,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "temperature": temperature,
                    "http_options": {"timeout": TIMEOUT_MS},
                },
            )

            # The SDK parses into the schema when it can; fall back to the
            # raw text so a parse quirk does not cost us the whole call.
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, schema):
                return parsed

            text = (getattr(response, "text", "") or "").strip()
            if text:
                return schema.model_validate_json(text)

            raise ValueError("Empty response")

        except Exception as exc:  # noqa: BLE001 - all failures are equivalent here
            last_error = exc
            logger.warning(
                "Gemini call failed (attempt %s/%s): %s", attempt + 1, MAX_ATTEMPTS, exc
            )

    logger.warning("Gemini call gave up: %s", last_error)
    return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a confidence the model may have reported outside 0..1."""
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return 0.5


# --- Public calls ----------------------------------------------------------


def choose_move(
    personality: str,
    current_round: int,
    total_rounds: int,
    player_history: str,
    ai_history: str,
    player_score: float,
    ai_score: float,
    player_cooperation_rate: float,
    rules: str,
    label_a: str = "COOPERATE",
    label_b: str = "DEFECT",
) -> schemas.OpponentMove | None:
    """Ask Gemini for the adaptive opponent's next move.

    Returns None on any failure, which the caller treats as "use the fallback
    strategy for this round".
    """
    result = _generate(
        system=prompts.OPPONENT_SYSTEM,
        prompt=prompts.opponent_prompt(
            personality=personality,
            current_round=current_round,
            total_rounds=total_rounds,
            player_history=player_history,
            ai_history=ai_history,
            player_score=player_score,
            ai_score=ai_score,
            player_cooperation_rate=player_cooperation_rate,
            rules=rules,
            label_a=label_a,
            label_b=label_b,
        ),
        schema=schemas.OpponentMove,
        model=MOVE_MODEL,
        temperature=0.9,
    )
    if result is None:
        return None
    return result.model_copy(update={"confidence": _clamp(result.confidence)})


def analyze_match(
    stats: MatchStats, opponent_name: str, sequence: str
) -> schemas.MatchAnalysis | None:
    return _generate(
        system=prompts.ANALYST_SYSTEM,
        prompt=prompts.analysis_prompt(stats, opponent_name, sequence),
        schema=schemas.MatchAnalysis,
        model=REASONING_MODEL,
        temperature=0.6,
    )


def classify_strategy(
    stats: MatchStats, sequence: str
) -> schemas.StrategyClassification | None:
    result = _generate(
        system=prompts.CLASSIFIER_SYSTEM,
        prompt=prompts.classification_prompt(stats, sequence),
        schema=schemas.StrategyClassification,
        model=REASONING_MODEL,
        temperature=0.3,
    )
    if result is None:
        return None
    return result.model_copy(update={"confidence": _clamp(result.confidence)})


def tutor_turn(
    topic: str,
    difficulty: str,
    transcript: list[dict[str, str]],
    learner_answer: str | None = None,
) -> schemas.TutorTurn | None:
    return _generate(
        system=prompts.TUTOR_SYSTEM,
        prompt=prompts.tutor_prompt(topic, difficulty, transcript, learner_answer),
        schema=schemas.TutorTurn,
        model=REASONING_MODEL,
        temperature=0.7,
    )


def tutor_hint(topic: str, question: str) -> schemas.TutorHint | None:
    return _generate(
        system=prompts.HINT_SYSTEM,
        prompt=prompts.hint_prompt(topic, question),
        schema=schemas.TutorHint,
        model=REASONING_MODEL,
        temperature=0.6,
    )
