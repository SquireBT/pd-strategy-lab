"""Learn mode.

Three tabs with deliberately different division of labour:

- Tutor: Gemini does the work. Teaching is language reasoning.
- Math Lab: Python does the work. Inequalities are arithmetic, and asking a
  model to evaluate them would be both slower and less reliable.
- Simulator: Python does the work. Running 100 rounds of two fixed strategies
  is a loop, not a judgement call.

That split is the point of the section as much as the content is.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from game.strategies import SIMULATOR_KEYS, get_strategy, simulate
from ui import charts, matrix_editor

TUTOR_KEY = "tutor"

TOPICS = [
    "Prisoner's Dilemma",
    "Nash Equilibrium",
    "Dominant Strategies",
    "Repeated Games",
    "Tit-for-Tat",
    "Mixed Strategies",
    "Zero-Sum Games",
    "Evolutionary Game Theory",
]

DIFFICULTIES = ["Beginner", "Intermediate", "Graduate"]

ASSESSMENT_LABELS = {
    "correct": ("Correct", "✅"),
    "partially_correct": ("Partly there", "🟡"),
    "incorrect": ("Not quite", "🔴"),
    "opening": ("", ""),
}


# --- Tutor -----------------------------------------------------------------


def _tutor_state() -> dict:
    return st.session_state.setdefault(
        TUTOR_KEY,
        {
            "topic": TOPICS[0],
            "difficulty": "Beginner",
            "transcript": [],   # [{"role": "tutor"|"learner", "content": str}]
            "question": None,
            "hint": None,
            "started": False,
            "scores": [],       # assessments, oldest first
        },
    )


def _reset_tutor(topic: str, difficulty: str) -> None:
    st.session_state[TUTOR_KEY] = {
        "topic": topic,
        "difficulty": difficulty,
        "transcript": [],
        "question": None,
        "hint": None,
        "started": False,
        "scores": [],
    }


def _advance_tutor(state: dict, answer: str | None) -> bool:
    """Take one tutoring turn. Returns False if the call failed."""
    from ai import gemini_client

    with st.spinner("Thinking..."):
        turn = gemini_client.tutor_turn(
            topic=state["topic"],
            difficulty=state["difficulty"],
            transcript=state["transcript"],
            learner_answer=answer,
        )

    if turn is None:
        return False

    if answer:
        state["transcript"].append({"role": "learner", "content": answer})
    state["transcript"].append(
        {"role": "tutor", "content": f"{turn.response}\n\n{turn.question}"}
    )
    state["question"] = turn.question
    state["hint"] = None
    state["started"] = True
    if turn.assessment != "opening":
        state["scores"].append(turn.assessment)
    return True


def _render_tutor() -> None:
    from ai import gemini_client

    state = _tutor_state()
    ai_status = gemini_client.status()

    col_topic, col_level = st.columns([2, 1])
    topic = col_topic.selectbox(
        "Topic", TOPICS, index=TOPICS.index(state["topic"])
    )
    difficulty = col_level.selectbox(
        "Level", DIFFICULTIES, index=DIFFICULTIES.index(state["difficulty"])
    )

    if topic != state["topic"] or difficulty != state["difficulty"]:
        _reset_tutor(topic, difficulty)
        state = _tutor_state()

    if not ai_status.available:
        st.warning(ai_status.message)
        st.caption(
            "The Math Lab and Strategy Simulator tabs are pure Python and "
            "work without an API key."
        )
        return

    st.caption(
        "The tutor asks rather than lectures. Answer in your own words — it "
        "responds to your reasoning, not your vocabulary."
    )

    if not state["started"]:
        if st.button("Start session", type="primary"):
            if not _advance_tutor(state, None):
                st.error("Could not reach Gemini. Try again in a moment.")
            st.rerun()
        return

    for entry in state["transcript"]:
        with st.chat_message("assistant" if entry["role"] == "tutor" else "user"):
            st.markdown(entry["content"])

    if state["hint"]:
        st.info(f"**Hint:** {state['hint']}")

    answer = st.chat_input("Your answer")
    if answer:
        if not _advance_tutor(state, answer):
            st.error("Could not reach Gemini. Your answer was not lost — retry.")
        st.rerun()

    col_hint, col_reset, col_score = st.columns([1, 1, 2])
    if col_hint.button("Hint", width='stretch'):
        with st.spinner("Finding a nudge..."):
            hint = gemini_client.tutor_hint(state["topic"], state["question"] or "")
        state["hint"] = hint.hint if hint else "No hint available right now."
        st.rerun()

    if col_reset.button("Restart topic", width='stretch'):
        _reset_tutor(state["topic"], state["difficulty"])
        st.rerun()

    if state["scores"]:
        correct = sum(1 for s in state["scores"] if s == "correct")
        partial = sum(1 for s in state["scores"] if s == "partially_correct")
        col_score.caption(
            f"{correct} correct, {partial} partly correct, "
            f"{len(state['scores'])} answered."
        )


# --- Math Lab --------------------------------------------------------------


def _render_math_lab() -> None:
    st.caption(
        "Change the payoffs and watch the structure of the game change with "
        "them. Both inequalities are evaluated in Python — no model is asked "
        "whether one number exceeds another."
    )

    grid_col, check_col = st.columns([3, 2])
    with grid_col:
        matrix = matrix_editor.render_matrix_picker("lab")
    with check_col:
        matrix_editor.render_conditions(matrix, verbose=True)

    st.divider()
    st.markdown(
        "**Why these two conditions?**\n\n"
        f"`T > R > P > S` makes defection a dominant strategy: with "
        f"T={matrix.T:g} beating R={matrix.R:g}, and P={matrix.P:g} beating "
        f"S={matrix.S:g}, defecting pays more whatever the other player does. "
        "That is what creates the dilemma — both players reasoning correctly "
        "arrive at an outcome both would rather avoid.\n\n"
        f"`2R > T + S` matters only in repeated play. With 2R={2 * matrix.R:g} "
        f"and T+S={matrix.T + matrix.S:g}, it decides whether steady mutual "
        "cooperation beats taking turns exploiting each other. Without it, "
        "alternating betrayal would be the efficient arrangement and "
        "cooperation would not be worth reaching."
    )


# --- Simulator -------------------------------------------------------------


def _render_simulator() -> None:
    st.caption(
        "Two fixed strategies, played head to head. Entirely deterministic "
        "Python: same inputs, same result, every time. This is the kind of "
        "work a language model should not be doing."
    )

    col_a, col_b, col_r = st.columns([2, 2, 1])
    key_a = col_a.selectbox(
        "Strategy A", SIMULATOR_KEYS, format_func=lambda k: get_strategy(k).name
    )
    key_b = col_b.selectbox(
        "Strategy B",
        SIMULATOR_KEYS,
        index=2,
        format_func=lambda k: get_strategy(k).name,
    )
    rounds = col_r.number_input("Rounds", 10, 500, 100, step=10)

    result = simulate(key_a, key_b, rounds=int(rounds), seed=42)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(result.strategy_a, f"{result.total_a:g}")
    c2.metric(result.strategy_b, f"{result.total_b:g}")
    c3.metric("Avg / round (A)", f"{result.average_a:.2f}")
    c4.metric("Avg / round (B)", f"{result.average_b:.2f}")

    st.plotly_chart(
        charts.simulation_trajectory(
            result.cumulative_a,
            result.cumulative_b,
            result.strategy_a,
            result.strategy_b,
        ),
        width='stretch',
    )

    summary = pd.DataFrame(
        [
            ("Cooperation rate", f"{result.cooperation_rate_a:.0%}",
             f"{result.cooperation_rate_b:.0%}"),
            ("Total score", f"{result.total_a:g}", f"{result.total_b:g}"),
            ("Average payoff", f"{result.average_a:.2f}", f"{result.average_b:.2f}"),
            ("Mutual cooperation", str(result.mutual_cooperation),
             str(result.mutual_cooperation)),
            ("Mutual defection", str(result.mutual_defection),
             str(result.mutual_defection)),
        ],
        columns=["Metric", result.strategy_a, result.strategy_b],
    )
    st.dataframe(summary, width='stretch', hide_index=True)

    with st.expander("What to look for"):
        st.markdown(
            "- **Tit-for-Tat vs Always Defect** loses by exactly one round's "
            "worth of payoff, and never by more. It cannot be exploited twice.\n"
            "- **Tit-for-Tat vs itself** cooperates every round — the highest "
            "joint score available.\n"
            "- **Grim Trigger vs Random** collapses almost immediately, since "
            "a random opponent defects early and the trigger never resets.\n"
            "- **Always Cooperate vs Always Defect** is the worst possible "
            "result for the cooperator, and shows why niceness alone is not a "
            "strategy."
        )


def render() -> None:
    st.header("Learn")
    tab_tutor, tab_math, tab_sim = st.tabs(
        ["Socratic tutor", "Math Lab", "Strategy simulator"]
    )
    with tab_tutor:
        _render_tutor()
    with tab_math:
        _render_math_lab()
    with tab_sim:
        _render_simulator()
