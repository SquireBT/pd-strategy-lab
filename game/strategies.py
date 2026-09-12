"""Deterministic strategies and the strategy-vs-strategy simulator.

Every strategy here is a pure function of the match history, which means it
survives Streamlit reruns without storing any internal state, and it can be
tested without mocking anything. No Gemini in this file by design: these are
rules, not reasoning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from game.prisoner_dilemma import DEFAULT_MATRIX, Move, PayoffMatrix

# A decision function sees its own past moves, the opponent's past moves, and
# a seeded RNG. Index i of both lists is the same round.
DecisionFn = Callable[[list[Move], list[Move], random.Random], Move]

C = Move.COOPERATE
D = Move.DEFECT


# --- Individual decision rules --------------------------------------------


def always_cooperate(mine: list[Move], theirs: list[Move], rng: random.Random) -> Move:
    return C


def always_defect(mine: list[Move], theirs: list[Move], rng: random.Random) -> Move:
    return D


def tit_for_tat(mine: list[Move], theirs: list[Move], rng: random.Random) -> Move:
    """Cooperate first, then mirror the opponent's last move."""
    return C if not theirs else theirs[-1]


def forgiving_tit_for_tat(
    mine: list[Move], theirs: list[Move], rng: random.Random
) -> Move:
    """Like Tit-for-Tat, but only retaliates after two defections in a row.

    A single defection is treated as noise, which lets the pair recover from
    an accidental breakdown instead of locking into mutual punishment.
    """
    if len(theirs) < 2:
        return C
    return D if theirs[-1] == D and theirs[-2] == D else C


def grim_trigger(mine: list[Move], theirs: list[Move], rng: random.Random) -> Move:
    """Cooperate until the opponent defects once, then defect forever."""
    return D if D in theirs else C


def aggressive(mine: list[Move], theirs: list[Move], rng: random.Random) -> Move:
    """Strongly favors defection, but can be coaxed by sustained cooperation.

    Defects by default. Only considers cooperating once the opponent has
    cooperated three rounds running, and even then only sometimes.
    """
    recent_all_cooperative = len(theirs) >= 3 and all(m == C for m in theirs[-3:])
    if recent_all_cooperative and rng.random() < 0.35:
        return C
    return C if rng.random() < 0.10 else D


def forgiving(mine: list[Move], theirs: list[Move], rng: random.Random) -> Move:
    """Cooperates readily and forgives, but will not be exploited indefinitely.

    Retaliates once the opponent's defection rate over the whole match passes
    half, then returns to cooperation as soon as that rate falls back.
    """
    if not theirs:
        return C
    defect_rate = sum(1 for m in theirs if m == D) / len(theirs)
    if defect_rate > 0.5 and theirs[-1] == D:
        return D
    return C


def opportunist(mine: list[Move], theirs: list[Move], rng: random.Random) -> Move:
    """Mirrors the opponent, but stabs when it reads them as a safe cooperator.

    Two consecutive cooperations from the opponent are treated as a signal
    that defection will go unpunished this round.
    """
    if len(theirs) >= 2 and theirs[-1] == C and theirs[-2] == C:
        return D if rng.random() < 0.7 else C
    return C if not theirs else theirs[-1]


def make_random(cooperation_probability: float = 0.5) -> DecisionFn:
    """Build a strategy that cooperates with a fixed probability."""
    p = max(0.0, min(1.0, cooperation_probability))

    def _random(mine: list[Move], theirs: list[Move], rng: random.Random) -> Move:
        return C if rng.random() < p else D

    return _random


# --- Registry --------------------------------------------------------------


@dataclass(frozen=True)
class Strategy:
    key: str
    name: str
    description: str          # one line, shown under the opponent dropdown
    decide: DecisionFn = field(repr=False)
    detail: str = ""          # how it behaves and how to play against it
    uses_gemini: bool = False


STRATEGIES: dict[str, Strategy] = {
    "tit_for_tat": Strategy(
        key="tit_for_tat",
        name="Tit-for-Tat",
        description="Cooperates first, then copies your previous move.",
        decide=tit_for_tat,
        detail=(
            "The winner of Axelrod's original tournament, and still the "
            "reference point for the whole field. It never defects first, "
            "punishes immediately, and forgives the moment you return to "
            "cooperating.\n\n"
            "**Playing against it:** you cannot beat it by more than a single "
            "round's payoff, because every defection you make comes back at "
            "you exactly once. Sustained cooperation is genuinely your best "
            "line — which is the whole lesson."
        ),
    ),
    "adaptive_gemini": Strategy(
        key="adaptive_gemini",
        name="Adaptive Gemini",
        description=(
            "Reads the whole match history with Gemini and reasons about what "
            "to do next."
        ),
        decide=tit_for_tat,  # fallback only, used when the API call fails
        detail=(
            "The only opponent here that is not a fixed rule. Each round its "
            "game state — history, scores, your cooperation rate, rounds "
            "remaining — is sent to Gemini, which returns a move, a confidence "
            "and a short reason you can read.\n\n"
            "**Playing against it:** it has no committed policy, so patterns "
            "that reliably beat a fixed strategy may not work twice. It also "
            "understands that the last round has no future to protect.\n\n"
            "If the API is unavailable it silently falls back to Tit-for-Tat "
            "and the round is marked as a fallback in the match log."
        ),
        uses_gemini=True,
    ),
    "forgiving": Strategy(
        key="forgiving",
        name="Forgiving",
        description=(
            "Cooperates by default and forgives lapses, but pushes back once "
            "you defect more often than not."
        ),
        decide=forgiving,
        detail=(
            "Judges you on your record across the entire match rather than "
            "your last move, so an isolated defection costs you nothing at "
            "all. Once your overall defection rate passes half it starts "
            "retaliating, and it stops again as soon as that rate falls "
            "back.\n\n"
            "**Playing against it:** the most exploitable opponent here, but "
            "only up to a point. Defecting just under half the time extracts "
            "real value; tipping past half turns it hostile and you lose the "
            "cooperation you were feeding on."
        ),
    ),
    "aggressive": Strategy(
        key="aggressive",
        name="Aggressive",
        description="Defects almost always. Hard, but not impossible, to win over.",
        decide=aggressive,
        detail=(
            "Defects roughly nine rounds in ten. Three cooperations from you "
            "in a row give it a real chance of cooperating back, but it never "
            "becomes reliable about it.\n\n"
            "**Playing against it:** mutual defection is the realistic "
            "ceiling. Cooperating mostly donates points, and the patience "
            "required to coax it out is rarely repaid. This is the opponent "
            "that makes defection look rational."
        ),
    ),
    "opportunist": Strategy(
        key="opportunist",
        name="Opportunist",
        description="Mirrors you, but defects when you start to look safe.",
        decide=opportunist,
        detail=(
            "Behaves like Tit-for-Tat until you cooperate twice in a row, "
            "which it treats as an opening and defects about seven times out "
            "of ten.\n\n"
            "**Playing against it:** punish the betrayal immediately. Because "
            "it mirrors, retaliation teaches it you are not a safe target and "
            "it returns to cooperating. Absorbing the hit and cooperating "
            "anyway invites the same move again."
        ),
    ),
    "grim_trigger": Strategy(
        key="grim_trigger",
        name="Grim Trigger",
        description="Cooperates until you defect once, then defects forever.",
        decide=grim_trigger,
        detail=(
            "Maximum deterrence and zero forgiveness. A single defection at "
            "any point in the match locks every remaining round into mutual "
            "defection, with no route back.\n\n"
            "**Playing against it:** perfect cooperation scores the highest "
            "total available in the game. One moment of greed is "
            "unrecoverable, which makes this the clearest demonstration of "
            "why the threat of permanent punishment sustains cooperation."
        ),
    ),
    "forgiving_tft": Strategy(
        key="forgiving_tft",
        name="Forgiving Tit-for-Tat",
        description="Retaliates only after two defections in a row.",
        decide=forgiving_tit_for_tat,
        detail=(
            "Treats a single defection as noise rather than aggression. That "
            "one change lets it recover from breakdowns that standard "
            "Tit-for-Tat would drag out for the rest of the match.\n\n"
            "**Playing against it:** alternating defection goes unpunished, "
            "which is the price of the forgiveness. Two defections back to "
            "back still draw retaliation."
        ),
    ),
    "always_cooperate": Strategy(
        key="always_cooperate",
        name="Always Cooperate",
        description="Cooperates every round regardless of what you do.",
        decide=always_cooperate,
        detail=(
            "A benchmark rather than a strategy. It scores well against other "
            "cooperative opponents and is completely defenceless against "
            "anything else — useful mainly for measuring how exploitative "
            "another strategy is."
        ),
    ),
    "always_defect": Strategy(
        key="always_defect",
        name="Always Defect",
        description="Defects every round. Never loses a round, never wins a match.",
        decide=always_defect,
        detail=(
            "The rational play in a one-shot game, and the dominant strategy "
            "by the payoff matrix alone. Over many rounds it never achieves "
            "mutual cooperation, so it reliably underperforms strategies that "
            "can."
        ),
    ),
    "random": Strategy(
        key="random",
        name="Random (50%)",
        description="Flips a fair coin each round.",
        decide=make_random(0.5),
        detail=(
            "There is no pattern to read, so there is nothing to exploit and "
            "nothing to reward. Any strategy that conditions on history is "
            "reduced to guessing, which makes this a useful control when "
            "comparing the others."
        ),
    ),
}

# Opponents offered in Play mode, in menu order.
OPPONENT_KEYS = [
    "adaptive_gemini",
    "tit_for_tat",
    "forgiving",
    "aggressive",
    "opportunist",
    "grim_trigger",
    "random",
]

# Strategies offered in the simulator, in menu order.
SIMULATOR_KEYS = [
    "always_cooperate",
    "always_defect",
    "tit_for_tat",
    "forgiving_tft",
    "grim_trigger",
    "random",
]


def get_strategy(key: str) -> Strategy:
    if key not in STRATEGIES:
        raise KeyError(f"Unknown strategy: {key!r}")
    return STRATEGIES[key]


def decide_move(
    key: str,
    own_history: list[Move],
    opponent_history: list[Move],
    rng: random.Random | None = None,
) -> Move:
    """Ask a registered strategy for its next move."""
    return get_strategy(key).decide(
        own_history, opponent_history, rng or random.Random()
    )


# --- Simulator -------------------------------------------------------------


@dataclass
class SimulationResult:
    """Outcome of one strategy playing another for a fixed number of rounds."""

    strategy_a: str
    strategy_b: str
    rounds: int
    moves_a: list[Move]
    moves_b: list[Move]
    cumulative_a: list[float]
    cumulative_b: list[float]

    @property
    def total_a(self) -> float:
        return self.cumulative_a[-1] if self.cumulative_a else 0.0

    @property
    def total_b(self) -> float:
        return self.cumulative_b[-1] if self.cumulative_b else 0.0

    @property
    def average_a(self) -> float:
        return self.total_a / self.rounds if self.rounds else 0.0

    @property
    def average_b(self) -> float:
        return self.total_b / self.rounds if self.rounds else 0.0

    @property
    def cooperation_rate_a(self) -> float:
        return _cooperation_rate(self.moves_a)

    @property
    def cooperation_rate_b(self) -> float:
        return _cooperation_rate(self.moves_b)

    @property
    def mutual_cooperation(self) -> int:
        return sum(1 for a, b in zip(self.moves_a, self.moves_b) if a == C and b == C)

    @property
    def mutual_defection(self) -> int:
        return sum(1 for a, b in zip(self.moves_a, self.moves_b) if a == D and b == D)


def _cooperation_rate(moves: list[Move]) -> float:
    return sum(1 for m in moves if m == C) / len(moves) if moves else 0.0


def simulate(
    key_a: str,
    key_b: str,
    rounds: int = 100,
    matrix: PayoffMatrix = DEFAULT_MATRIX,
    seed: int | None = 42,
) -> SimulationResult:
    """Run two registered strategies against each other.

    Entirely deterministic given a seed, including for the random strategy.
    Gemini is not involved and could not be: this is arithmetic.
    """
    if rounds < 1:
        raise ValueError("rounds must be 1 or greater")

    strat_a, strat_b = get_strategy(key_a), get_strategy(key_b)
    rng_a = random.Random(seed)
    rng_b = random.Random(None if seed is None else seed + 1)

    moves_a: list[Move] = []
    moves_b: list[Move] = []
    cum_a: list[float] = []
    cum_b: list[float] = []
    total_a = total_b = 0.0

    for _ in range(rounds):
        # Both decide from the same history snapshot: no peeking at the
        # opponent's move for the round currently being played.
        move_a = strat_a.decide(list(moves_a), list(moves_b), rng_a)
        move_b = strat_b.decide(list(moves_b), list(moves_a), rng_b)

        pay_a, pay_b = matrix.payoffs(move_a, move_b)
        total_a += pay_a
        total_b += pay_b

        moves_a.append(move_a)
        moves_b.append(move_b)
        cum_a.append(total_a)
        cum_b.append(total_b)

    return SimulationResult(
        strategy_a=strat_a.name,
        strategy_b=strat_b.name,
        rounds=rounds,
        moves_a=moves_a,
        moves_b=moves_b,
        cumulative_a=cum_a,
        cumulative_b=cum_b,
    )