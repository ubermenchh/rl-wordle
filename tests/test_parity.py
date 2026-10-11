"""Parity of the env path with evaluate.py (Mark I, step 5).

Mark I metric: our random player and our solver get the same floor and ceiling in the new env
as in evaluate.py, within noise. Both players run through `play_game` here, with a fake
interaction whose replies come from the baseline player. The env path adds three things that
evaluate.py does not have: the reply format (the tag), the end of the episode at the first
illegal reply (choice 2a), and the mid-game states of the taskset.

Known difference, on purpose: at 3 and 4 priors the solver needs more turns in the env path
(measured 2026-10-11: 4.32 against 4.05, and 5.13 against 4.42). evaluate.py takes the states
that `sample_state` lands, so its "4 priors" are often 2 or 3 priors. The taskset redraws until
all the priors land (decision in tests/test_taskset.py), so its deep states are really deep.
Therefore the turns are compared at 0 to 2 priors only. The solve rate is compared at all.

Run from the repo root: `uv run python -m tests.test_parity`.
"""

import asyncio
import itertools
import random
from collections import defaultdict
from collections.abc import Callable
from typing import cast

import verifiers.v1 as vf
from verifiers.v1.agent import Segment

from evaluate import Player, evaluate, random_player, solver_player
from wordle_rl.taskset import WordleConfig, WordleTask, WordleTaskset, episode_reward, play_game
from wordle_rl.wordle import Game, InvalidGuess

N_TASKS = 600  # About 120 for each number of priors. The seeds are fixed, so the result is exact.


class _Trace:
    def __init__(self):
        self.info: dict = {}


class _PlayerInteraction:
    """A fake interaction whose replies come from a baseline player. It keeps its own copy of
    the game, so the player sees the state, and it records the replies for the reward."""

    def __init__(self, task: WordleTask, player: Player):
        self.game = Game(task.data.secret)
        for word in task.data.priors:
            self.game.guess(word)
        self.player = player
        self.replies: list[str] = []
        self.trace = _Trace()

    async def turn(self, message=None) -> Segment:
        word = self.player(self.game)
        try:
            self.game.guess(word)
        except InvalidGuess:
            pass  # The env sees the same reply and ends the episode.
        self.replies.append(f"<guess>{word}</guess>")
        return Segment(messages=[], root_reply=self.replies[-1])


def _play(task: WordleTask, player: Player) -> tuple[str, float, int]:
    """Outcome, reward and turns used, for one task."""
    interaction = _PlayerInteraction(task, player)
    outcome = asyncio.run(play_game(task, cast(vf.Interaction, interaction)))
    return outcome, episode_reward(task.data, interaction.replies), interaction.game.turns_used


def _env_numbers(make_player: Callable[[random.Random], Player]) -> dict[int, dict]:
    """The numbers of the env path, for each number of priors."""
    rng = random.Random(0)
    player = make_player(rng)
    tasks = itertools.islice(WordleTaskset(WordleConfig(seed=0)).load(), N_TASKS)
    rows = defaultdict(list)
    for task in tasks:
        rows[len(task.data.priors)].append(_play(task, player))
    numbers = {}
    for n_prior, games in rows.items():
        wins = [g for g in games if g[0] == "won"]
        numbers[n_prior] = {
            "solve": len(wins) / len(games),
            "turns": sum(g[2] for g in wins) / len(wins) if wins else float("nan"),
            "illegal": sum(g[0] not in ("won", "lost") for g in games) / len(games),
            "reward": sum(g[1] for g in games) / len(games),
        }
    return numbers


def _test_the_floor_matches():
    """A random legal word almost never wins, in both paths. The episodes that the env ends at
    an illegal reply (a repeat) must be rare, or choice 2a would distort the floor."""
    env = _env_numbers(random_player)
    assert sorted(env) == [0, 1, 2, 3, 4]
    for n_prior, row in env.items():
        assert evaluate(random_player, n_games=400, n_prior=n_prior)["solve_rate"] <= 0.02
        assert row["solve"] <= 0.02, (n_prior, row)
        assert row["illegal"] <= 0.03, (n_prior, row)


def _test_the_ceiling_matches():
    """The solver wins about 99% in both paths. A big gap means that the env, the tag or the
    state text hides information from a player that has it."""
    env = _env_numbers(solver_player)
    for n_prior, row in env.items():
        base = evaluate(solver_player, n_games=400, n_prior=n_prior)
        assert row["solve"] >= 0.95 and base["solve_rate"] >= 0.95, (n_prior, row, base)
        assert abs(row["solve"] - base["solve_rate"]) <= 0.05, (n_prior, row, base)
        assert row["illegal"] == 0.0, (n_prior, row)  # The solver never plays an illegal word.
        if n_prior <= 2:  # See the module docstring for 3 and 4.
            assert abs(row["turns"] - base["mean_turns"]) <= 0.3, (n_prior, row, base)


def _test_degenerate_players_score_below_the_solver():
    """Hack attempts through the whole path (PLAN.md section 7, TASKS task 5). A player that
    says nothing, repeats a word, or loops on filler must score below the solver. charbull's
    model learned the filler because doing nothing was cheaper than a wrong guess."""
    def nothing(game: Game) -> str:
        return ""

    def filler(game: Game) -> str:
        return "Final Final Final Final"

    def repeat(game: Game) -> str:
        return game.guesses[-1] if game.guesses else "CRANE"

    solver = _env_numbers(solver_player)
    solver_reward = sum(r["reward"] for r in solver.values()) / len(solver)
    for name, player in {"nothing": nothing, "filler": filler, "repeat": repeat}.items():
        env = _env_numbers(lambda rng, player=player: player)
        reward = sum(r["reward"] for r in env.values()) / len(env)
        assert reward <= 0.0, (name, reward)
        assert reward < solver_reward, (name, reward, solver_reward)


def run_tests():
    for name, test in sorted(globals().items()):
        if name.startswith("_test_"):
            test()
            print(f"  ok  {name}")
    print(f"\n{sum(n.startswith('_test_') for n in globals())} tests passed.")


if __name__ == "__main__":
    run_tests()
