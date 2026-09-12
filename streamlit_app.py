"""AI Strategy Lab — entrypoint and sidebar navigation.

Deliberately thin. This file wires pages together and does nothing else:
game rules live in game/, AI calls live in ai/, and each page renders itself.

Page modules are imported inside the dispatch rather than at the top, so
opening the app only pays for the section actually being shown.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="AI Strategy Lab",
    page_icon="\N{DIRECT HIT}",
    layout="wide",
    initial_sidebar_state="expanded",
)

SECTIONS = ["Home", "Play", "Analyze", "Learn"]


def _render(section: str) -> None:
    if section == "Home":
        from ui import home

        home.render()
    elif section == "Play":
        from ui import play

        play.render()
    elif section == "Analyze":
        from ui import analysis

        analysis.render()
    elif section == "Learn":
        from ui import learn

        learn.render()


def main() -> None:
    with st.sidebar:
        st.title("AI Strategy Lab")
        st.caption("Play games. Learn game theory. Outsmart the AI.")
        st.divider()
        section = st.radio("Section", SECTIONS, label_visibility="collapsed")

        from ui import home
        from game.games import get_game

        st.caption(f"Game: **{get_game(home.get_game_key()).name}**")
        st.divider()
        st.caption(
            "Rules, payoffs and statistics are computed in Python. Gemini "
            "plays the adaptive opponent and handles explanation and teaching."
        )

    _render(section)


if __name__ == "__main__":
    main()
