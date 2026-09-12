"""Home: pick a game.

Selecting a game writes its key to session state and clears any match in
progress, since a half-finished Prisoner's Dilemma means nothing once the
payoffs belong to a different game.
"""

from __future__ import annotations

import streamlit as st

from game.games import DEFAULT_GAME, GAMES, get_game

GAME_KEY = "game_key"


def get_game_key() -> str:
    return st.session_state.setdefault(GAME_KEY, DEFAULT_GAME)


def select_game(key: str) -> None:
    from ui import play

    if st.session_state.get(GAME_KEY) != key:
        st.session_state[GAME_KEY] = key
        play.reset_match()


def render() -> None:
    st.header("AI Strategy Lab")
    st.caption("Play games. Learn game theory. Outsmart the AI.")

    st.markdown(
        "Each game below is a different strategic problem. The rules, payoffs "
        "and scoring are handled in Python; Gemini plays the adaptive "
        "opponent and explains what your choices mean."
    )

    current = get_game_key()
    st.divider()

    for row_start in range(0, len(GAMES), 2):
        cols = st.columns(2)
        for col, key in zip(cols, list(GAMES)[row_start : row_start + 2]):
            game = GAMES[key]
            with col:
                with st.container(border=True):
                    selected = key == current
                    st.subheader(game.name)
                    st.caption(game.tagline)
                    st.write(game.description)
                    st.caption(f"**Concept:** {game.concept}")
                    if st.button(
                        "Selected" if selected else f"Play {game.name}",
                        key=f"pick_{key}",
                        width='stretch',
                        disabled=selected,
                    ):
                        select_game(key)
                        st.rerun()

    st.divider()
    game = get_game(current)
    st.caption(
        f"Currently selected: **{game.name}**. Open **Play** in the sidebar "
        "to start a match, or **Learn** for the tutor, the payoff-matrix lab "
        "and the strategy simulator."
    )
