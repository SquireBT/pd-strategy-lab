"""Analyze mode.

Every number on this page is computed by game/analytics.py. The Gemini
sections interpret those numbers and are optional: if the API is unavailable
the statistics, tables and charts all still render.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from game.analytics import MatchStats, analyze_match, move_sequence
from game.strategies import get_strategy
from ui import charts, play

ANALYSIS_KEY = "analysis"
CLASSIFICATION_KEY = "classification"


def _render_stats(stats: MatchStats) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Your score", f"{stats.player_score:g}")
    c2.metric("AI score", f"{stats.ai_score:g}")
    c3.metric("Difference", f"{stats.score_difference:+g}")
    c4.metric("Rounds", stats.rounds)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("You cooperated", f"{stats.player_cooperation_rate:.0%}")
    c2.metric("AI cooperated", f"{stats.ai_cooperation_rate:.0%}")
    c3.metric("Mutual cooperation", stats.mutual_cooperation)
    c4.metric("Mutual defection", stats.mutual_defection)

    rounds = stats.rounds
    # Every value is rendered as text: mixing ints with "never" and a
    # percentage in one column makes pandas fall back to object dtype, which
    # Arrow renders inconsistently.
    breakdown = pd.DataFrame(
        [
            ("You exploited the AI", f"{stats.player_exploited_ai} of {rounds}"),
            ("You were exploited", f"{stats.player_was_exploited} of {rounds}"),
            ("Retaliations", f"{stats.retaliations}"),
            ("Forgiveness moves", f"{stats.forgiveness_moves}"),
            ("Unprovoked defections", f"{stats.unprovoked_defections}"),
            (
                "Longest cooperation streak",
                f"{stats.longest_cooperation_streak} rounds",
            ),
            ("Longest defection streak", f"{stats.longest_defection_streak} rounds"),
            (
                "First defection",
                f"round {stats.first_defection_round}"
                if stats.first_defection_round
                else "never defected",
            ),
            (
                "Mirrored AI's last move",
                f"{stats.tit_for_tat_similarity:.0%} of rounds",
            ),
        ],
        columns=["Behaviour", "Value"],
    )
    st.dataframe(breakdown, width='stretch', hide_index=True)
    st.caption(
        "Retaliation, forgiveness and unprovoked defection compare your move "
        "against the AI's *previous* move, so round 1 is excluded — they can "
        f"total at most {max(rounds - 1, 0)} here."
    )


def _render_gemini_sections(state: dict, stats: MatchStats) -> None:
    from ai import gemini_client

    ai_status = gemini_client.status()
    opponent_name = get_strategy(state["opponent_key"]).name
    sequence = move_sequence(state["results"])

    st.subheader("Strategic analysis")

    if not ai_status.available:
        st.info(ai_status.message)
        st.caption(
            "The statistics above are unaffected — they were computed in "
            "Python from your recorded moves."
        )
        return

    if st.button("Analyze this match", type="primary"):
        with st.spinner("Reading your match..."):
            st.session_state[ANALYSIS_KEY] = gemini_client.analyze_match(
                stats, opponent_name, sequence
            )
            st.session_state[CLASSIFICATION_KEY] = gemini_client.classify_strategy(
                stats, sequence
            )

    analysis = st.session_state.get(ANALYSIS_KEY)
    classification = st.session_state.get(CLASSIFICATION_KEY)

    if analysis is None and classification is None:
        st.caption("Press the button to have Gemini interpret the numbers above.")
        return

    if analysis is None:
        st.warning(
            "Gemini could not produce an analysis this time. The statistics "
            "above are still valid."
        )
    else:
        st.markdown(f"**Your strategy** — {analysis.strategy_summary}")

        col_good, col_bad = st.columns(2)
        with col_good:
            st.markdown("**What you did well**")
            for item in analysis.did_well:
                st.markdown(f"- {item}")
        with col_bad:
            st.markdown("**Potential weaknesses**")
            if analysis.weaknesses:
                for item in analysis.weaknesses:
                    st.markdown(f"- {item}")
            else:
                st.caption("Nothing significant flagged.")

        st.markdown("**Strategic takeaway**")
        st.write(analysis.takeaway)
        st.caption(f"Relevant concept: {analysis.concept}")

    if classification is not None:
        st.divider()
        st.subheader("Strategy classification")
        c1, c2 = st.columns([2, 1])
        c1.metric("Closest match", classification.classification)
        c2.metric("Confidence", f"{classification.confidence:.0%}")
        st.write(classification.explanation)
        st.markdown(f"**Try next time:** {classification.recommendation}")
        st.caption(
            "This label is an approximation from observed behaviour over a "
            "limited number of rounds, not a proof. It is based on your move "
            "pattern, not on whether you won."
        )


def render() -> None:
    st.header("Analyze")

    state = play.get_state()
    results = state["results"]

    if not results:
        st.info("No match to analyze yet. Play some rounds first.")
        return

    if not state["finished"]:
        st.caption(
            f"Match in progress — analyzing the first {len(results)} "
            f"round{'s' if len(results) != 1 else ''}."
        )

    stats = analyze_match(results)
    st.caption(
        f"Against **{get_strategy(state['opponent_key']).name}** — "
        f"{stats.summary_line()}"
    )

    _render_stats(stats)

    st.divider()
    tab_score, tab_moves = st.tabs(["Score over time", "Move history"])
    with tab_score:
        st.plotly_chart(
            charts.score_over_time(
                [r.player_total for r in results], [r.ai_total for r in results]
            ),
            width='stretch',
        )
    with tab_moves:
        st.plotly_chart(
            charts.move_history(
                play.player_moves(state), play.ai_moves(state)
            ),
            width='stretch',
        )

    st.divider()
    _render_gemini_sections(state, stats)