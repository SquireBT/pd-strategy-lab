"""Match statistics.

Everything Gemini is later asked to interpret is computed here first, in
Python, from the recorded moves. The model never counts anything; it receives
finished numbers and explains what they mean. This is the clearest expression
of the project's core rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from game.prisoner_dilemma import Move, RoundResult

C = Move.COOPERATE
D = Move.DEFECT


@dataclass(frozen=True)
class MatchStats:
    rounds: int
    player_score: float
    ai_score: float
    score_difference: float

    player_cooperation_rate: float
    player_defection_rate: float
    ai_cooperation_rate: float

    mutual_cooperation: int
    mutual_defection: int
    player_exploited_ai: int
    player_was_exploited: int

    longest_cooperation_streak: int
    longest_defection_streak: int

    retaliations: int
    forgiveness_moves: int
    unprovoked_defections: int
    first_defection_round: int | None

    tit_for_tat_similarity: float

    def to_dict(self) -> dict:
        """Flat dict for prompt construction. Rounded to keep prompts short."""
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, float):
                out[key] = round(value, 3)
        return out

    def summary_line(self) -> str:
        verdict = (
            "ahead"
            if self.score_difference > 0
            else "behind"
            if self.score_difference < 0
            else "level"
        )
        return (
            f"{self.rounds} rounds, finished {verdict} by "
            f"{abs(self.score_difference):g}, cooperating "
            f"{self.player_cooperation_rate:.0%} of the time."
        )


def _longest_streak(moves: list[Move], target: Move) -> int:
    best = current = 0
    for m in moves:
        current = current + 1 if m == target else 0
        best = max(best, current)
    return best


def _rate(moves: list[Move], target: Move) -> float:
    return sum(1 for m in moves if m == target) / len(moves) if moves else 0.0


def analyze_match(results: list[RoundResult]) -> MatchStats:
    """Reduce a completed (or in-progress) match to a statistics record."""
    if not results:
        raise ValueError("Cannot analyze a match with no rounds")

    mine = [r.player_move for r in results]
    theirs = [r.ai_move for r in results]
    pairs = list(zip(mine, theirs))

    player_score = results[-1].player_total
    ai_score = results[-1].ai_total

    # Behaviour that only makes sense relative to the *previous* round.
    retaliations = 0        # they defected, you answered with a defection
    forgiveness = 0         # they defected, you cooperated anyway
    unprovoked = 0          # they cooperated, you defected regardless
    tft_matches = 0         # you played whatever they played last round

    for i in range(1, len(results)):
        their_last = theirs[i - 1]
        my_now = mine[i]
        if their_last == D and my_now == D:
            retaliations += 1
        if their_last == D and my_now == C:
            forgiveness += 1
        if their_last == C and my_now == D:
            unprovoked += 1
        if my_now == their_last:
            tft_matches += 1

    first_defection = next(
        (i + 1 for i, m in enumerate(mine) if m == D),
        None,
    )

    return MatchStats(
        rounds=len(results),
        player_score=player_score,
        ai_score=ai_score,
        score_difference=player_score - ai_score,
        player_cooperation_rate=_rate(mine, C),
        player_defection_rate=_rate(mine, D),
        ai_cooperation_rate=_rate(theirs, C),
        mutual_cooperation=sum(1 for p, a in pairs if p == C and a == C),
        mutual_defection=sum(1 for p, a in pairs if p == D and a == D),
        player_exploited_ai=sum(1 for p, a in pairs if p == D and a == C),
        player_was_exploited=sum(1 for p, a in pairs if p == C and a == D),
        longest_cooperation_streak=_longest_streak(mine, C),
        longest_defection_streak=_longest_streak(mine, D),
        retaliations=retaliations,
        forgiveness_moves=forgiveness,
        unprovoked_defections=unprovoked,
        first_defection_round=first_defection,
        # Share of rounds after the first where you mirrored their last move.
        # A rough similarity score, not a claim that you *are* Tit-for-Tat.
        tit_for_tat_similarity=(
            tft_matches / (len(results) - 1) if len(results) > 1 else 0.0
        ),
    )


def move_sequence(results: list[RoundResult]) -> str:
    """Compact 'CCDCD / CCCDD' rendering for prompts."""
    mine = "".join(r.player_move.short for r in results)
    theirs = "".join(r.ai_move.short for r in results)
    return f"{mine} / {theirs}"