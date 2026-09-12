"""Plotly figure builders.

Colors are tuned for the dark purple theme in .streamlit/config.toml.

Plotly is imported lazily inside each builder rather than at module load.
It is by far the heaviest dependency in the project, and screens that draw
no charts should not pay for it.

Kept separate from the page code so Play and Analyze render identical charts,
and so the pages stay readable. These functions only shape data that Python
already computed; they never decide anything.
"""

from __future__ import annotations

from game.prisoner_dilemma import Move

PLAYER_COLOR = "#A78BFA"
AI_COLOR = "#F0883E"
COOPERATE_COLOR = "#3FB984"
DEFECT_COLOR = "#E5484D"


def _base_layout(fig, height: int = 300):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    return fig


def score_over_time(
    player_cumulative: list[float], ai_cumulative: list[float]
) -> go.Figure:
    """Cumulative score for both sides, round by round."""
    import plotly.graph_objects as go

    rounds = list(range(1, len(player_cumulative) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=player_cumulative,
            name="You",
            mode="lines+markers",
            line=dict(color=PLAYER_COLOR, width=2.5),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=ai_cumulative,
            name="AI",
            mode="lines+markers",
            line=dict(color=AI_COLOR, width=2.5),
            marker=dict(size=6),
        )
    )
    fig.update_xaxes(title="Round", showgrid=False, dtick=1 if len(rounds) <= 20 else 5)
    fig.update_yaxes(title="Cumulative score", gridcolor="rgba(167,139,250,0.18)")
    return _base_layout(fig)


def cooperation_rates(player_rate: float, ai_rate: float) -> go.Figure:
    """Side-by-side cooperation percentages."""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Bar(
            x=["You", "AI"],
            y=[player_rate * 100, ai_rate * 100],
            marker_color=[PLAYER_COLOR, AI_COLOR],
            text=[f"{player_rate:.0%}", f"{ai_rate:.0%}"],
            textposition="outside",
            width=0.5,
        )
    )
    fig.update_yaxes(
        title="Cooperation rate",
        range=[0, 115],
        ticksuffix="%",
        gridcolor="rgba(167,139,250,0.18)",
    )
    fig.update_xaxes(showgrid=False)
    return _base_layout(fig, height=260)


def move_history(
    player_moves: list[Move],
    ai_moves: list[Move],
    short_a: str = "C",
    short_b: str = "D",
) -> go.Figure:
    """A two-row strip of markers showing how the match unfolded."""
    import plotly.graph_objects as go

    rounds = list(range(1, len(player_moves) + 1))
    fig = go.Figure()

    for label, moves, row in (("You", player_moves, 1), ("AI", ai_moves, 0)):
        fig.add_trace(
            go.Scatter(
                x=rounds,
                y=[row] * len(moves),
                mode="markers+text",
                name=label,
                text=[
                    short_a if m == Move.COOPERATE else short_b for m in moves
                ],
                textposition="middle center",
                textfont=dict(color="white", size=11),
                marker=dict(
                    size=24,
                    color=[
                        COOPERATE_COLOR if m == Move.COOPERATE else DEFECT_COLOR
                        for m in moves
                    ],
                    symbol="square",
                ),
                hovertemplate=f"{label}, round %{{x}}: %{{text}}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["AI", "You"],
        range=[-0.6, 1.6],
        showgrid=False,
    )
    fig.update_xaxes(
        title="Round", showgrid=False, dtick=1 if len(rounds) <= 20 else 5
    )
    return _base_layout(fig, height=220)


def simulation_trajectory(
    cumulative_a: list[float],
    cumulative_b: list[float],
    name_a: str,
    name_b: str,
) -> go.Figure:
    """Score trajectory for a simulated strategy matchup."""
    import plotly.graph_objects as go

    rounds = list(range(1, len(cumulative_a) + 1))
    fig = go.Figure()
    for cum, name, color in (
        (cumulative_a, name_a, PLAYER_COLOR),
        (cumulative_b, name_b, AI_COLOR),
    ):
        fig.add_trace(
            go.Scatter(
                x=rounds, y=cum, name=name, mode="lines", line=dict(color=color, width=2)
            )
        )
    fig.update_xaxes(title="Round", showgrid=False)
    fig.update_yaxes(title="Cumulative score", gridcolor="rgba(167,139,250,0.18)")
    return _base_layout(fig)
