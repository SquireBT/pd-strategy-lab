"""Game definitions.

`Move` stays a binary choice — COOPERATE is always "the first option" and
DEFECT "the second" — and each game supplies its own labels on top. That
keeps every strategy, statistic and chart in the project working unchanged
across games, at the cost of some PD-flavoured internal naming.

BiMatrix covers asymmetric games, where the two players face different
payoffs. PayoffMatrix (in prisoner_dilemma.py) stays for the symmetric
T/R/P/S case, since it also carries the PD inequality checks. Both expose
`payoffs()` and `as_grid()`, so either can be handed to `play_round`.
"""

from __future__ import annotations

from dataclasses import dataclass

from game.prisoner_dilemma import DEFAULT_MATRIX, Move, PayoffMatrix


@dataclass(frozen=True)
class BiMatrix:
    """Four cells, each holding (row player's payoff, column player's payoff)."""

    aa: tuple[float, float]
    ab: tuple[float, float]
    ba: tuple[float, float]
    bb: tuple[float, float]

    def payoffs(self, player: Move, opponent: Move) -> tuple[float, float]:
        first = player == Move.COOPERATE
        second = opponent == Move.COOPERATE
        if first and second:
            return self.aa
        if first and not second:
            return self.ab
        if not first and second:
            return self.ba
        return self.bb

    def as_grid(self) -> list[list[str]]:
        def cell(pair: tuple[float, float]) -> str:
            return f"{pair[0]:g}, {pair[1]:g}"

        return [[cell(self.aa), cell(self.ab)], [cell(self.ba), cell(self.bb)]]


@dataclass(frozen=True)
class GameDef:
    key: str
    name: str
    tagline: str
    description: str
    label_a: str          # shown for Move.COOPERATE
    label_b: str          # shown for Move.DEFECT
    matrix: PayoffMatrix | BiMatrix
    concept: str
    strategic_note: str
    symmetric: bool = True
    editable_payoffs: bool = False

    def label(self, move: Move) -> str:
        return self.label_a if move == Move.COOPERATE else self.label_b

    def short(self, move: Move) -> str:
        return (self.label_a if move == Move.COOPERATE else self.label_b)[0].upper()

    @property
    def short_a(self) -> str:
        return self.label_a[0].upper()

    @property
    def short_b(self) -> str:
        return self.label_b[0].upper()

    def rules_text(self) -> str:
        """Plain-language payoff description, used in Gemini prompts."""
        grid = self.matrix.as_grid()
        return (
            f"Game: {self.name}. Each round both players choose "
            f"{self.label_a} or {self.label_b}.\n"
            f"- Both {self.label_a}: {grid[0][0]}\n"
            f"- Player {self.label_a}, you {self.label_b}: {grid[0][1]}\n"
            f"- Player {self.label_b}, you {self.label_a}: {grid[1][0]}\n"
            f"- Both {self.label_b}: {grid[1][1]}\n"
            "(Each cell reads: player's payoff, your payoff.)"
        )


GAMES: dict[str, GameDef] = {
    "prisoners_dilemma": GameDef(
        key="prisoners_dilemma",
        name="Prisoner's Dilemma",
        tagline="Mutual cooperation beats mutual betrayal — but betrayal always tempts.",
        description=(
            "Defecting pays more than cooperating no matter what the other "
            "player does, so rational play drives both sides to an outcome "
            "they both dislike. Repetition is what makes cooperation "
            "recoverable."
        ),
        label_a="Cooperate",
        label_b="Defect",
        matrix=DEFAULT_MATRIX,
        concept="Dominant strategies and the cooperation problem",
        strategic_note=(
            "Defection strictly dominates in any single round. Over many "
            "rounds, the shadow of future retaliation can sustain cooperation."
        ),
        editable_payoffs=True,
    ),
    "stag_hunt": GameDef(
        key="stag_hunt",
        name="Stag Hunt",
        tagline="The best outcome needs trust, and trust is the risk.",
        description=(
            "Hunting the stag together pays best, but only if both commit. "
            "Hunting hare is safe and pays regardless. Nothing tempts you to "
            "betray a partner here — the difficulty is believing they will "
            "show up."
        ),
        label_a="Stag",
        label_b="Hare",
        matrix=BiMatrix(aa=(4, 4), ab=(0, 3), ba=(3, 0), bb=(2, 2)),
        concept="Coordination and risk-dominant equilibria",
        strategic_note=(
            "Two pure equilibria: both Stag and both Hare. Stag pays more; "
            "Hare is safer. This is a problem of assurance, not temptation."
        ),
    ),
    "chicken": GameDef(
        key="chicken",
        name="Chicken",
        tagline="Someone has to yield. Neither of you wants it to be you.",
        description=(
            "Swerving is embarrassing but survivable. If both drive straight "
            "the result is catastrophic for both. The winning move depends "
            "entirely on convincing the other side you will not move."
        ),
        label_a="Swerve",
        label_b="Straight",
        matrix=BiMatrix(aa=(3, 3), ab=(2, 4), ba=(4, 2), bb=(0, 0)),
        concept="Anti-coordination and credible commitment",
        strategic_note=(
            "Equilibria are the two asymmetric outcomes: exactly one player "
            "yields. Mutual aggression is the worst cell for both, which is "
            "what makes threats here worth making and dangerous to make."
        ),
    ),
    "matching_pennies": GameDef(
        key="matching_pennies",
        name="Matching Pennies",
        tagline="You win by matching. The AI wins by not matching.",
        description=(
            "A strictly zero-sum game with no stable pure strategy. Any "
            "pattern you settle into can be read and beaten, so the only "
            "unexploitable play is to be genuinely unpredictable."
        ),
        label_a="Heads",
        label_b="Tails",
        matrix=BiMatrix(aa=(1, -1), ab=(-1, 1), ba=(-1, 1), bb=(1, -1)),
        concept="Zero-sum games and mixed strategies",
        strategic_note=(
            "No pure Nash equilibrium exists. The unique equilibrium is "
            "mixed: play each option half the time. Any detectable bias is "
            "exploitable, which makes this the sharpest test of the adaptive "
            "opponent."
        ),
        symmetric=False,
    ),
    "battle_of_sexes": GameDef(
        key="battle_of_sexes",
        name="Battle of the Sexes",
        tagline="You both want to agree. You disagree about on what.",
        description=(
            "Coordinating matters more than getting your preference, but you "
            "each prefer a different meeting point. Failing to coordinate "
            "pays nothing at all."
        ),
        label_a="Gallery",
        label_b="Concert",
        matrix=BiMatrix(aa=(2, 1), ab=(0, 0), ba=(0, 0), bb=(1, 2)),
        concept="Coordination with conflicting preferences",
        strategic_note=(
            "You prefer the Gallery, the AI prefers the Concert, and both of "
            "you prefer either over missing each other. Repetition lets you "
            "discover conventions such as alternating."
        ),
        symmetric=False,
    ),
}

DEFAULT_GAME = "prisoners_dilemma"


def get_game(key: str) -> GameDef:
    if key not in GAMES:
        raise KeyError(f"Unknown game: {key!r}")
    return GAMES[key]
