"""AI Strategy Lab — entrypoint and sidebar navigation.

Deliberately thin. This file wires pages together and does nothing else:
game rules live in game/, AI calls live in ai/, and each page renders itself.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="AI Strategy Lab",
    page_icon="\N{DIRECT HIT}",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import play  # noqa: E402  (must follow set_page_config)

PAGES = {
    "Play": play.render,
    "Analyze": None,
    "Learn": None,
}


def _placeholder(name: str) -> None:
    st.header(name)
    st.info(f"{name} mode is not built yet.")


def main() -> None:
    with st.sidebar:
        st.title("AI Strategy Lab")
        st.caption("Play games. Learn game theory. Outsmart the AI.")
        st.divider()
        choice = st.radio("Section", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption(
            "Game rules, payoffs and statistics are computed in Python. "
            "Gemini handles strategy, explanation and teaching."
        )

    renderer = PAGES[choice]
    if renderer is None:
        _placeholder(choice)
    else:
        renderer()


if __name__ == "__main__":
    main()