"""Deterministic core of the Prisoner's Dilemma.

This module owns the truth: moves, the payoff matrix, the inequality checks,
and round resolution. It has no Streamlit and no Gemini dependency, which is
what makes it testable and what keeps scoring out of the LLM's hands.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Move(str, Enum):
    """A single decision in one round."""

    COOPERATE = "COOPERATE"
    DEFECT = "DEFECT"

    @property
    def short(self) -> str:
        """Single-letter form used in tables and charts."""
        return "C" if self is Move.COOPERATE else "D"

    @classmethod
    def parse(cls, raw: object) -> Move:
        """Coerce loose input into a Move.

        Accepts a Move, or a string such as "cooperate", "C", " DEFECT ".
        Raises ValueError on anything else. The Gemini layer relies on this
        to validate model output before it can touch game state.
        """
        if isinstance(raw, Move):
            return raw
        if isinstance(raw, str):
            token = raw.strip().upper()
            if token in ("C", "COOPERATE"):
                return cls.COOPERATE
            if token in ("D", "DEFECT"):
                return cls.DEFECT
        raise ValueError(f"Not a valid move: {raw!r}")


# Where the AI opponent's move came from. Surfaced in the UI so the player
# can see when a Gemini call fell back to deterministic logic.
class MoveSource(str, Enum):
    PYTHON = "python"
    GEMINI = "gemini"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class PayoffMatrix:
    """Symmetric 2x2 payoff matrix in standard T/R/P/S notation.

    T = Temptation  (you defect, they cooperate)
    R = Reward      (both cooperate)
    P = Punishment  (both defect)
    S = Sucker      (you cooperate, they defect)
    """

    T: float = 5.0
    R: float = 3.0
    P: float = 1.0
    S: float = 0.0

    def payoffs(self, player: Move, opponent: Move) -> tuple[float, float]:
        """Return (player_payoff, opponent_payoff) for one pair of moves."""
        if player is Move.COOPERATE and opponent is Move.COOPERATE:
            return self.R, self.R
        if player is Move.COOPERATE and opponent is Move.DEFECT:
            return self.S, self.T
        if player is Move.DEFECT and opponent is Move.COOPERATE:
            return self.T, self.S
        return self.P, self.P

    # --- Prisoner's Dilemma conditions -------------------------------------
    # Both are computed here, in Python. Gemini is only ever asked to explain
    # what a failing condition means, never to evaluate it.

    @property
    def satisfies_ordering(self) -> bool:
        """T > R > P > S: defection dominates, mutual defection beats being suckered."""
        return self.T > self.R > self.P > self.S

    @property
    def satisfies_cooperation_bonus(self) -> bool:
        """2R > T + S: alternating exploitation is worse than steady cooperation."""
        return 2 * self.R > self.T + self.S

    @property
    def is_valid_prisoners_dilemma(self) -> bool:
        return self.satisfies_ordering and self.satisfies_cooperation_bonus

    def condition_report(self) -> list[tuple[str, bool, str]]:
        """(label, passed, explanation) rows for the Math Lab UI."""
        return [
            (
                f"T > R > P > S  ({self.T} > {self.R} > {self.P} > {self.S})",
                self.satisfies_ordering,
                "Defecting pays more than cooperating no matter what the "
                "opponent does, so defection is a dominant strategy.",
            ),
            (
                f"2R > T + S  ({2 * self.R} > {self.T + self.S})",
                self.satisfies_cooperation_bonus,
                "Sustained mutual cooperation beats taking turns exploiting "
                "each other, which is what makes cooperation worth reaching "
                "in a repeated game.",
            ),
        ]

    def as_grid(self) -> list[list[str]]:
        """Rows of the displayed matrix, player-first: 'your payoff, their payoff'."""
        return [
            [f"{self.R:g}, {self.R:g}", f"{self.S:g}, {self.T:g}"],
            [f"{self.T:g}, {self.S:g}", f"{self.P:g}, {self.P:g}"],
        ]


DEFAULT_MATRIX = PayoffMatrix()


@dataclass(frozen=True)
class RoundResult:
    """One completed round, including running totals after it resolved."""

    round_number: int
    player_move: Move
    ai_move: Move
    player_payoff: float
    ai_payoff: float
    player_total: float
    ai_total: float
    ai_source: MoveSource = MoveSource.PYTHON
    ai_reason: str | None = None
    ai_confidence: float | None = None

    @property
    def outcome(self) -> str:
        """Plain-language label for the round."""
        if self.player_move is self.ai_move:
            return (
                "Mutual cooperation"
                if self.player_move is Move.COOPERATE
                else "Mutual defection"
            )
        return (
            "You exploited the AI"
            if self.player_move is Move.DEFECT
            else "The AI exploited you"
        )

    def as_row(self) -> dict[str, object]:
        """Flat dict for the pandas round-history table."""
        return {
            "Round": self.round_number,
            "You": self.player_move.short,
            "AI": self.ai_move.short,
            "Your Payoff": self.player_payoff,
            "AI Payoff": self.ai_payoff,
            "Your Total": self.player_total,
            "AI Total": self.ai_total,
            "Outcome": self.outcome,
        }


def play_round(
    round_number: int,
    player_move: Move,
    ai_move: Move,
    player_total: float = 0.0,
    ai_total: float = 0.0,
    matrix: PayoffMatrix = DEFAULT_MATRIX,
    ai_source: MoveSource = MoveSource.PYTHON,
    ai_reason: str | None = None,
    ai_confidence: float | None = None,
) -> RoundResult:
    """Resolve one round and return it with cumulative scores applied.

    `player_total` and `ai_total` are the scores *before* this round. The
    returned RoundResult carries the scores *after* it, so session state only
    ever has to append the result and read the last entry.
    """
    if round_number < 1:
        raise ValueError("round_number must be 1 or greater")

    player_move = Move.parse(player_move)
    ai_move = Move.parse(ai_move)
    player_payoff, ai_payoff = matrix.payoffs(player_move, ai_move)

    return RoundResult(
        round_number=round_number,
        player_move=player_move,
        ai_move=ai_move,
        player_payoff=player_payoff,
        ai_payoff=ai_payoff,
        player_total=player_total + player_payoff,
        ai_total=ai_total + ai_payoff,
        ai_source=ai_source,
        ai_reason=ai_reason,
        ai_confidence=ai_confidence,
    )


def score_match(
    player_moves: list[Move],
    ai_moves: list[Move],
    matrix: PayoffMatrix = DEFAULT_MATRIX,
) -> tuple[float, float]:
    """Total both sides over a full sequence of moves.

    Used by the strategy simulator and by the tests as an independent check
    on the round-by-round accumulation in `play_round`.
    """
    if len(player_moves) != len(ai_moves):
        raise ValueError("Move sequences must be the same length")

    player_total = ai_total = 0.0
    for p, a in zip(player_moves, ai_moves):
        p_pay, a_pay = matrix.payoffs(Move.parse(p), Move.parse(a))
        player_total += p_pay
        ai_total += a_pay
    return player_total, ai_total