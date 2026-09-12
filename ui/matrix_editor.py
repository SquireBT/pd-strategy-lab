"""Editable payoff matrix, plus the Prisoner's Dilemma condition checks.

The editor is the matrix: you type into the 2x2 grid and the T/R/P/S values
follow, rather than editing four abstract letters and watching a grid update.
Shared between Play mode and the Math Lab in Learn mode.

Every inequality is evaluated by PayoffMatrix in Python. Gemini is only ever
asked to explain what a result means, never to compute it.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from game.prisoner_dilemma import DEFAULT_MATRIX, PayoffMatrix

ROWS = ["You: Cooperate", "You: Defect"]
COLS = ["AI: Cooperate", "AI: Defect"]

# Named payoff sets. Each is a complete game; "Custom" unlocks the editor.
PRESETS: dict[str, PayoffMatrix] = {
    "Classic": PayoffMatrix(T=5, R=3, P=1, S=0),
    "Narrow": PayoffMatrix(T=4, R=3, P=1, S=0),
    "High stakes": PayoffMatrix(T=10, R=6, P=2, S=0),
    "Harsh": PayoffMatrix(T=5, R=3, P=1, S=-2),
    "Stag Hunt": PayoffMatrix(T=3, R=5, P=1, S=0),
}

PRESET_NOTES: dict[str, str] = {
    "Classic": "The standard 5 / 3 / 1 / 0 Prisoner's Dilemma.",
    "Narrow": "Temptation barely beats mutual cooperation, so defecting buys little.",
    "High stakes": "Same structure, larger numbers. Mistakes cost more.",
    "Harsh": "Being exploited is actively punishing, not merely unrewarding.",
    "Stag Hunt": (
        "Not a Prisoner's Dilemma: cooperation pays best, but only if the "
        "other side cooperates too. A coordination problem, not a temptation "
        "problem."
    ),
}


def _to_frame(matrix: PayoffMatrix) -> pd.DataFrame:
    """Your payoff in each of the four cells.

    The game is symmetric, so these four numbers are exactly T, R, P and S:
        (C, C) = R      (C, D) = S
        (D, C) = T      (D, D) = P
    """
    return pd.DataFrame(
        [[matrix.R, matrix.S], [matrix.T, matrix.P]],
        index=ROWS,
        columns=COLS,
        dtype=float,
    )


def _cell(frame: pd.DataFrame, row: int, col: int, fallback: float) -> float:
    """Read one cell, tolerating a value the user cleared."""
    try:
        value = float(frame.iloc[row, col])
    except (TypeError, ValueError):
        return fallback
    return fallback if math.isnan(value) else value


def _from_frame(frame: pd.DataFrame, fallback: PayoffMatrix) -> PayoffMatrix:
    return PayoffMatrix(
        R=_cell(frame, 0, 0, fallback.R),
        S=_cell(frame, 0, 1, fallback.S),
        T=_cell(frame, 1, 0, fallback.T),
        P=_cell(frame, 1, 1, fallback.P),
    )


def render_editor(
    prefix: str = "pd", default: PayoffMatrix = DEFAULT_MATRIX
) -> PayoffMatrix:
    """Draw an editable 2x2 grid and return the matrix it describes.

    Each cell holds *your* payoff for that pair of moves. The AI's payoff is
    the mirror image, since the game is symmetric.
    """
    data_key = f"{prefix}_grid_data"
    if data_key not in st.session_state:
        st.session_state[data_key] = _to_frame(default)

    edited = st.data_editor(
        st.session_state[data_key],
        key=f"{prefix}_grid_editor",
        width='stretch',
        column_config={
            col: st.column_config.NumberColumn(col, step=0.5, format="%g")
            for col in COLS
        },
    )
    st.session_state[data_key] = edited

    matrix = _from_frame(edited, default)
    st.caption(
        "Each cell is **your** payoff. The AI's payoff is the mirror image — "
        f"currently T={matrix.T:g}, R={matrix.R:g}, P={matrix.P:g}, "
        f"S={matrix.S:g}."
    )
    return matrix


def reset_editor(prefix: str = "pd", matrix: PayoffMatrix = DEFAULT_MATRIX) -> None:
    """Drop the stored grid so the editor rebuilds from `matrix`."""
    st.session_state.pop(f"{prefix}_grid_data", None)
    st.session_state.pop(f"{prefix}_grid_editor", None)


def render_matrix(
    matrix: PayoffMatrix,
    caption: bool = True,
    labels: tuple[str, str] = ("Cooperate", "Defect"),
) -> None:
    """Read-only view showing both payoffs per cell."""
    a, b = labels
    rows = [f"You: {a}", f"You: {b}"]
    cols = [f"AI: {a}", f"AI: {b}"]
    grid = pd.DataFrame(matrix.as_grid(), index=rows, columns=cols)
    st.dataframe(grid, width='stretch')
    if caption:
        st.caption(
            "Each cell reads **your payoff, AI payoff**. The first number is "
            "always yours."
        )


def render_conditions(matrix: PayoffMatrix, verbose: bool = True) -> bool:
    """Show whether the payoffs form a standard Prisoner's Dilemma.

    Returns the verdict so callers can decide whether to warn.
    """
    for label, passed, explanation in matrix.condition_report():
        if passed:
            st.success(f"**{label}** — satisfied")
        else:
            st.error(f"**{label}** — not satisfied")
        if verbose:
            st.caption(explanation)

    valid = matrix.is_valid_prisoners_dilemma
    if not valid:
        st.warning(
            "These payoffs are not a standard Prisoner's Dilemma. Still "
            "playable — the strategic logic just changes."
        )
    return valid


def render_matrix_picker(
    prefix: str = "pd", default: PayoffMatrix = DEFAULT_MATRIX
) -> PayoffMatrix:
    """Choose a preset payoff set, or unlock the grid to build your own.

    Presets render as a read-only 2x2 grid. Only "Custom" exposes editable
    cells, which keeps the common case to a single click.
    """
    options = list(PRESETS) + ["Custom"]
    choice = (
        st.segmented_control(
            "Payoffs", options, default="Classic", key=f"{prefix}_preset"
        )
        or "Classic"
    )

    seed_key = f"{prefix}_seed"

    if choice == "Custom":
        # Start the editor from whichever preset was showing last, so
        # switching to Custom does not reset the numbers on screen.
        matrix = render_editor(prefix, st.session_state.get(seed_key, default))
    else:
        matrix = PRESETS[choice]
        st.session_state[seed_key] = matrix
        # Drop any stored grid so re-entering Custom picks up this preset.
        reset_editor(prefix)
        render_matrix(matrix)
        st.caption(PRESET_NOTES[choice])

    return matrix
