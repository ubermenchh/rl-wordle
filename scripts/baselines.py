"""Random player and solver on the same held-out tasks as the model evals (Mark IV table).

The taskset is a seeded stream, so the first N tasks of split "eval" are the same tasks that
`eval_run --games N` plays on Modal. Both players use the rules of the env: the game has 6 turns
in total, and the prior guesses of the task count as turns.

Run: uv run python -m scripts.baselines --games 400
"""

import argparse
import itertools
import random

from wordle_rl.taskset import WordleConfig, WordleTaskset, episode_reward
from wordle_rl.wordle import ALLOWED, ANSWERS, Game, constraints_from, is_consistent


def pick_random(game: Game, rng: random.Random) -> str:
    # Any legal word. It is the floor: no use of the feedback at all.
    return rng.choice(sorted(ALLOWED - set(game.guesses)))


def pick_solver(game: Game, rng: random.Random) -> str:
    # Any answer that fits all feedback so far. The secret is always consistent (tests in wordle.py),
    # so the pool is never empty. This is the policy that made the SFT traces.
    c = constraints_from(game)
    pool = [w for w in ANSWERS if w not in game.guesses and is_consistent(w, c)]
    return rng.choice(pool)


def play(task, pick, rng: random.Random) -> tuple[bool, float]:
    game = Game(task.data.secret)
    for word in task.data.priors:
        game.guess(word)
    replies = []
    while not game.over:
        word = pick(game, rng)
        replies.append(f"<guess>{word}</guess>")
        game.guess(word)
    return game.won, episode_reward(task.data, replies)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tasks = list(itertools.islice(WordleTaskset(WordleConfig(split="eval")).load(), args.games))
    for name, pick in (("random player", pick_random), ("solver", pick_solver)):
        rng = random.Random(args.seed)
        results = [play(t, pick, rng) for t in tasks]
        wins = sum(w for w, _ in results)
        mean_reward = sum(r for _, r in results) / len(results)
        print(f"{name:14} wins {wins}/{len(results)} ({wins / len(results):.1%})  mean reward {mean_reward:.4f}")


if __name__ == "__main__":
    main()
