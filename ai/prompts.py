"""Prompt construction.

Kept apart from the client so prompts can be read, diffed and tuned without
touching transport code. Note what is *not* here: no prompt asks Gemini to
compute a score, count a streak, or evaluate an inequality. Every number in
these prompts was calculated in Python first.
"""

from __future__ import annotations

from game.analytics import MatchStats

OPPONENT_SYSTEM = """You are an opponent in a repeated two-player game.

Your job is to choose an option for the current round and briefly explain your \
reasoning. Play to maximise your own total score across the whole match, not \
just this round.

The rules and payoffs of the specific game are given to you each turn. Read \
them carefully: the games differ. Some reward matching the other player, some \
reward mismatching, some tempt you to betray cooperation, and at least one is \
strictly zero-sum, where any pattern you fall into can be read and punished.

Always answer with "COOPERATE" for the first listed option and "DEFECT" for \
the second, whatever those options are actually called in this game.

You are given the full history. Read it for patterns: is the player mirroring \
you, testing you, exploiting you, or playing at random? Adapt. A player who \
has cooperated consistently is worth cooperating with; a player who defects \
whenever you cooperate is not.

Do not calculate scores — they are supplied to you and are authoritative. \
Keep your reason under two sentences and write it for the player to read."""

ANALYST_SYSTEM = """You are a game theory tutor reviewing a student's \
completed Prisoner's Dilemma match.

All statistics you receive were computed from the actual recorded moves. Treat \
them as fact and interpret them. Do not recompute them, and do not claim \
patterns the numbers do not support.

Be specific and honest. If the player played badly, say so plainly and explain \
why in game-theory terms. Avoid flattery. Refer to concrete counts from the \
data rather than making vague statements. Your goal is that they understand \
something new about repeated games, not that they feel good."""

CLASSIFIER_SYSTEM = """You classify a player's observed strategy in a repeated \
Prisoner's Dilemma.

Base the label only on the behavioural statistics supplied, never on whether \
the player won. A player can lose while playing Tit-for-Tat perfectly, and win \
by getting lucky against a random opponent.

Classification is an approximation from observed behaviour over a limited \
number of rounds. Set confidence low when the match was short or the \
behaviour was inconsistent."""

TUTOR_SYSTEM = """You are a Socratic game theory tutor.

Never lecture. Explain briefly, then ask a question that makes the learner \
reason. When they answer, evaluate the reasoning rather than the wording, name \
any specific mistake, and ask a follow-up that builds on what they got right.

Adjust difficulty from their performance: if they are answering confidently, \
push harder; if they are lost, back up and rebuild the foundation. Keep each \
turn short — a few sentences and one question."""

HINT_SYSTEM = """You give hints to a game theory learner who is stuck.

A hint points at the reasoning they should be doing. It never states the \
answer, never completes the argument for them, and is at most two sentences. \
If the question asks them to identify a concept, gesture at the defining \
property of that concept without naming it."""


def opponent_prompt(
    personality: str,
    current_round: int,
    total_rounds: int,
    player_history: str,
    ai_history: str,
    player_score: float,
    ai_score: float,
    player_cooperation_rate: float,
    rules: str,
    label_a: str = "COOPERATE",
    label_b: str = "DEFECT",
) -> str:
    """Game state for one move decision.

    Only match state goes in. Nothing identifying the person, nothing from
    elsewhere in the app.
    """
    rounds_left = total_rounds - current_round + 1
    recent = player_history[-5:] if player_history else "(none yet)"

    return f"""{rules}

Answer COOPERATE to choose {label_a}, or DEFECT to choose {label_b}.

Your assigned personality: {personality}

Round {current_round} of {total_rounds} ({rounds_left} remaining, this one included).

History so far, oldest first. C = {label_a}, D = {label_b}:
- Player's moves: {player_history or '(none yet)'}
- Your moves:     {ai_history or '(none yet)'}

Player's last five moves: {recent}
Player chose {label_a} in {player_cooperation_rate:.0%} of rounds so far.

Current score — player {player_score:g}, you {ai_score:g}.

Choose your move for round {current_round}."""


def analysis_prompt(stats: MatchStats, opponent_name: str, sequence: str) -> str:
    """Computed statistics handed over for interpretation."""
    d = stats.to_dict()
    return f"""The player just finished a match against the "{opponent_name}" opponent.

Move sequence (player / opponent, oldest first):
{sequence}

Statistics computed from those moves:
- Rounds played: {d['rounds']}
- Final score: player {d['player_score']:g}, opponent {d['ai_score']:g} \
(difference {d['score_difference']:+g})
- Player cooperated {d['player_cooperation_rate']:.0%}, defected \
{d['player_defection_rate']:.0%}
- Opponent cooperated {d['ai_cooperation_rate']:.0%}
- Mutual cooperation: {d['mutual_cooperation']} rounds
- Mutual defection: {d['mutual_defection']} rounds
- Player exploited opponent: {d['player_exploited_ai']} rounds
- Player was exploited: {d['player_was_exploited']} rounds
- Longest cooperation streak: {d['longest_cooperation_streak']}
- Longest defection streak: {d['longest_defection_streak']}
- Retaliations (defected right after being defected against): {d['retaliations']}
- Forgiveness moves (cooperated right after being defected against): \
{d['forgiveness_moves']}
- Unprovoked defections (defected right after being cooperated with): \
{d['unprovoked_defections']}
- First defection: round {d['first_defection_round'] or 'never'}
- Mirrored the opponent's previous move in {d['tit_for_tat_similarity']:.0%} of rounds

Write an educational analysis of how this player played."""


def classification_prompt(stats: MatchStats, sequence: str) -> str:
    d = stats.to_dict()
    return f"""Classify this player's strategy from their observed behaviour.

Move sequence (player / opponent, oldest first):
{sequence}

- Rounds: {d['rounds']}
- Cooperation rate: {d['player_cooperation_rate']:.0%}
- Mirrored opponent's previous move: {d['tit_for_tat_similarity']:.0%} of rounds
- Retaliations: {d['retaliations']}
- Forgiveness moves: {d['forgiveness_moves']}
- Unprovoked defections: {d['unprovoked_defections']}
- Longest cooperation streak: {d['longest_cooperation_streak']}
- Longest defection streak: {d['longest_defection_streak']}
- First defection: round {d['first_defection_round'] or 'never'}

Ignore the final score entirely when choosing the label."""


def tutor_prompt(
    topic: str,
    difficulty: str,
    transcript: list[dict[str, str]],
    learner_answer: str | None,
) -> str:
    """One tutoring turn, with the conversation so far."""
    if not transcript:
        return (
            f"Topic: {topic}\nLevel: {difficulty}\n\n"
            "Open the session. Explain the core idea in two or three sentences, "
            "then ask your first question. Set assessment to 'opening'."
        )

    history = "\n".join(
        f"{'Tutor' if t['role'] == 'tutor' else 'Learner'}: {t['content']}"
        for t in transcript[-8:]
    )
    return f"""Topic: {topic}
Level: {difficulty}

Conversation so far:
{history}

The learner's latest answer: {learner_answer}

Evaluate their reasoning, then ask the next question."""


def hint_prompt(topic: str, question: str) -> str:
    return f"""Topic: {topic}

The learner is stuck on this question:
{question}

Give them one hint. Do not reveal the answer."""
