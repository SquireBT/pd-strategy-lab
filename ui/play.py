"""Play mode: the repeated Prisoner's Dilemma.

All match state lives in a single session_state dict so a rerun cannot lose a
match, and so clicking Cooperate or Defect never re-initialises anything. The
payoff matrix is chosen before the match and frozen for its duration, because
changing payoffs mid-match would make the running totals meaningless.
"""

from __future__ import annotations

import random

import pandas as pd
import streamlit as st

from game.prisoner_dilemma import (
    DEFAULT_MATRIX,
    Move,
    MoveSource,
    PayoffMatrix,
    RoundResult,
    play_round,
)
from game.games import GameDef, get_game
from game.strategies import OPPONENT_KEYS, decide_move, get_strategy
from ui import charts, matrix_editor

STATE_KEY = "match"
ROUND_OPTIONS = [5, 10, 15, 20, 25, 30]


# --- State -----------------------------------------------------------------


def _new_state() -> dict:
    return {
        "started": False,
        "finished": False,
        "rounds_total": 10,
        "opponent_key": OPPONENT_KEYS[0],
        "game_key": None,   # filled in when a match starts
        "matrix": DEFAULT_MATRIX,
        "results": [],
        "seed": random.randrange(1_000_000),
    }


def get_state() -> dict:
    """The one place match state is created. Analyze mode reads this too."""
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = _new_state()
    return st.session_state[STATE_KEY]


def reset_match() -> None:
    st.session_state[STATE_KEY] = _new_state()
    st.session_state.pop("analysis", None)
    st.session_state.pop("classification", None)


def player_moves(state: dict) -> list[Move]:
    return [r.player_move for r in state["results"]]


def ai_moves(state: dict) -> list[Move]:
    return [r.ai_move for r in state["results"]]


def _game(state: dict) -> GameDef:
    """The game this match belongs to, falling back to the current selection."""
    from ui import home

    return get_game(state.get("game_key") or home.get_game_key())


def _custom_rules_rows(matrix, game: GameDef) -> list[str]:
    grid = matrix.as_grid()
    return [
        f"Both {game.label_a}: {grid[0][0]}",
        f"Player {game.label_a}, you {game.label_b}: {grid[0][1]}",
        f"Player {game.label_b}, you {game.label_a}: {grid[1][0]}",
        f"Both {game.label_b}: {grid[1][1]}",
    ]


def _cooperation_rate(moves: list[Move]) -> float:
    return sum(1 for m in moves if m == Move.COOPERATE) / len(moves) if moves else 0.0


# --- Turn logic ------------------------------------------------------------


def _gemini_move(state: dict, mine: list[Move], theirs: list[Move]):
    """Ask Gemini for a move. Returns None on any failure.

    Imported here rather than at module scope so a match against a
    deterministic opponent never loads the SDK.
    """
    from ai import gemini_client

    last = state["results"][-1] if state["results"] else None
    game = _game(state)
    # The live matrix may differ from the game default when payoffs are edited.
    rules = game.rules_text() if state["matrix"] is game.matrix else (
        f"Game: {game.name}. Each round both players choose "
        f"{game.label_a} or {game.label_b}.\n"
        + "\n".join(
            f"- {row}" for row in _custom_rules_rows(state["matrix"], game)
        )
        + "\n(Each cell reads: player's payoff, your payoff.)"
    )

    with st.spinner("Gemini is deciding..."):
        return gemini_client.choose_move(
            personality=get_strategy(state["opponent_key"]).description,
            current_round=len(mine) + 1,
            total_rounds=state["rounds_total"],
            player_history="".join(m.short for m in mine),
            ai_history="".join(m.short for m in theirs),
            player_score=last.player_total if last else 0.0,
            ai_score=last.ai_total if last else 0.0,
            player_cooperation_rate=_cooperation_rate(mine),
            rules=rules,
            label_a=game.label_a,
            label_b=game.label_b,
        )


def _take_turn(state: dict, choice: Move) -> None:
    """Resolve one round: player move in, AI move decided, payoffs applied."""
    results: list[RoundResult] = state["results"]
    mine, theirs = player_moves(state), ai_moves(state)

    # A fresh RNG seeded from the match seed plus the round number keeps the
    # random opponent reproducible across Streamlit reruns.
    rng = random.Random(state["seed"] * 1000 + len(results))

    strategy = get_strategy(state["opponent_key"])
    source = MoveSource.PYTHON
    reason: str | None = None
    confidence: float | None = None

    if strategy.uses_gemini:
        decision = _gemini_move(state, mine, theirs)
        if decision is None:
            # API missing, slow, or malformed: play on with the fallback rule.
            ai_move = strategy.decide(theirs, mine, rng)
            source = MoveSource.FALLBACK
        else:
            ai_move = Move.parse(decision.move)
            source = MoveSource.GEMINI
            reason, confidence = decision.reason, decision.confidence
    else:
        ai_move = decide_move(state["opponent_key"], theirs, mine, rng)

    last = results[-1] if results else None
    results.append(
        play_round(
            round_number=len(results) + 1,
            player_move=choice,
            ai_move=ai_move,
            player_total=last.player_total if last else 0.0,
            ai_total=last.ai_total if last else 0.0,
            matrix=state["matrix"],
            ai_source=source,
            ai_reason=reason,
            ai_confidence=confidence,
        )
    )

    if len(results) >= state["rounds_total"]:
        state["finished"] = True


# --- Setup -----------------------------------------------------------------


def _render_setup(state: dict) -> None:
    from ui import home

    game = get_game(home.get_game_key())
    st.subheader(game.name)
    st.caption(game.tagline)
    st.info(game.strategic_note, icon=":material/lightbulb:")

    opponent_key = st.selectbox(
        "Opponent", OPPONENT_KEYS, format_func=lambda k: get_strategy(k).name
    )
    opponent = get_strategy(opponent_key)
    st.caption(opponent.description)

    if opponent.uses_gemini:
        from ai import gemini_client

        ai_status = gemini_client.status()
        if not ai_status.available:
            st.warning(ai_status.message)

    rounds = st.segmented_control(
        "Rounds", ROUND_OPTIONS, default=10, key="setup_rounds"
    ) or 10

    st.divider()
    st.subheader("Payoff matrix")

    if game.editable_payoffs:
        grid_col, check_col = st.columns([3, 2])
        with grid_col:
            matrix = matrix_editor.render_matrix_picker("play", game.matrix)
        with check_col:
            valid = matrix_editor.render_conditions(matrix, verbose=False)
    else:
        # Only the Prisoner's Dilemma exposes editable payoffs; the others are
        # defined by their structure and stop being themselves if you alter it.
        matrix = game.matrix
        matrix_editor.render_matrix(matrix, labels=(game.label_a, game.label_b))
        valid = True

    _render_opponent_reference(opponent_key)

    st.divider()
    if st.button("Start match", type="primary"):
        state.update(
            started=True,
            finished=False,
            rounds_total=int(rounds),
            opponent_key=opponent_key,
            game_key=game.key,
            matrix=matrix,
            results=[],
            seed=random.randrange(1_000_000),
        )
        st.rerun()

    if not valid:
        st.caption(
            "You can still start a match with these payoffs — seeing how the "
            "strategies behave outside a true Prisoner's Dilemma is part of "
            "the point."
        )


# --- Match -----------------------------------------------------------------


def _render_metrics(state: dict) -> None:
    results: list[RoundResult] = state["results"]
    last = results[-1] if results else None
    p_score = last.player_total if last else 0.0
    a_score = last.ai_total if last else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Round", f"{len(results)} / {state['rounds_total']}")
    c2.metric("Your score", f"{p_score:g}", delta=f"{p_score - a_score:+g}")
    c3.metric("AI score", f"{a_score:g}")
    game = _game(state)
    c4.metric(f"Your {game.label_a}", f"{_cooperation_rate(player_moves(state)):.0%}")
    c5.metric(f"AI {game.label_a}", f"{_cooperation_rate(ai_moves(state)):.0%}")


def _render_controls(state: dict) -> None:
    st.write("")
    c1, c2 = st.columns(2)
    game = _game(state)
    if c1.button(game.label_a.upper(), width='stretch', key="btn_cooperate"):
        _take_turn(state, Move.COOPERATE)
        st.rerun()
    if c2.button(game.label_b.upper(), width='stretch', key="btn_defect"):
        _take_turn(state, Move.DEFECT)
        st.rerun()


def _render_last_round(state: dict) -> None:
    results: list[RoundResult] = state["results"]
    if not results:
        st.info("Make your first move. The AI has not seen anything from you yet.")
        return

    game = _game(state)
    r = results[-1]
    st.markdown(
        f"**Round {r.round_number}** — you played `{game.label(r.player_move)}`, "
        f"the AI played `{game.label(r.ai_move)}`. "
        f"You scored **{r.player_payoff:g}**, the AI scored **{r.ai_payoff:g}**. "
        f"_{r.outcome}._"
    )
    if r.ai_source == MoveSource.FALLBACK:
        st.warning(
            "Gemini was unavailable for that round. The game continued using "
            "the fallback strategy."
        )
    if r.ai_reason:
        with st.expander("Why the AI chose that"):
            st.write(r.ai_reason)
            if r.ai_confidence is not None:
                st.caption(f"Stated confidence: {r.ai_confidence:.0%}")


def _render_dashboard(state: dict) -> None:
    results: list[RoundResult] = state["results"]
    if not results:
        return

    mine, theirs = player_moves(state), ai_moves(state)
    game = _game(state)

    st.divider()
    tab_score, tab_moves, tab_table = st.tabs(
        ["Score over time", "Move history", "Round table"]
    )

    with tab_score:
        st.plotly_chart(
            charts.score_over_time(
                [r.player_total for r in results], [r.ai_total for r in results]
            ),
            width='stretch',
        )
        st.plotly_chart(
            charts.cooperation_rates(
                _cooperation_rate(mine), _cooperation_rate(theirs)
            ),
            width='stretch',
        )

    with tab_moves:
        st.plotly_chart(
            charts.move_history(
                mine, theirs, short_a=game.short_a, short_b=game.short_b
            ),
            width='stretch',
        )

    with tab_table:
        df = pd.DataFrame(
            [
                {
                    "Round": r.round_number,
                    "You": game.label(r.player_move),
                    "AI": game.label(r.ai_move),
                    "Your Payoff": r.player_payoff,
                    "AI Payoff": r.ai_payoff,
                    "Your Total": r.player_total,
                    "AI Total": r.ai_total,
                }
                for r in results
            ]
        )
        st.dataframe(df, width='stretch', hide_index=True)


def _render_match(state: dict) -> None:
    opponent = get_strategy(state["opponent_key"])
    game = _game(state)
    matrix = state["matrix"]

    st.subheader(game.name)
    st.caption(f"Opponent: **{opponent.name}** — {opponent.description}")
    if matrix != DEFAULT_MATRIX:
        st.caption(
            f"Custom payoffs in play: T={matrix.T:g}, R={matrix.R:g}, "
            f"P={matrix.P:g}, S={matrix.S:g}"
        )

    _render_metrics(state)

    if state["finished"]:
        results = state["results"]
        p, a = results[-1].player_total, results[-1].ai_total
        verdict = (
            "You finished ahead."
            if p > a
            else "The AI finished ahead."
            if a > p
            else "You finished level."
        )
        st.success(f"Match complete. {verdict} Final score {p:g} – {a:g}.")
        st.caption("Open **Analyze** in the sidebar for a breakdown of your play.")
    else:
        _render_controls(state)

    _render_last_round(state)
    _render_dashboard(state)

    st.divider()
    if st.button("New game"):
        reset_match()
        st.rerun()

    with st.expander("Payoff matrix in play"):
        matrix_editor.render_matrix(matrix, labels=(game.label_a, game.label_b))
        st.caption(
            "Payoffs are locked once a match starts. Start a new game to "
            "change them."
        )


def _render_opponent_reference(selected: str) -> None:
    """Full write-ups for every opponent, selected one first."""
    with st.expander("Opponent guide"):
        ordered = [selected] + [k for k in OPPONENT_KEYS if k != selected]
        for key in ordered:
            strat = get_strategy(key)
            label = f"{strat.name} — currently selected" if key == selected else strat.name
            st.markdown(f"### {label}")
            st.caption(strat.description)
            st.markdown(strat.detail)
            st.divider()
        st.caption(
            "Every opponent except Adaptive Gemini is a deterministic Python "
            "function of the match history. No model call is involved in any "
            "of them."
        )


def render() -> None:
    st.header("Play")
    state = get_state()
    if state["started"]:
        _render_match(state)
    else:
        _render_setup(state)
