"""
Baselines and the evaluation harness.

Establishes the floor (random legal words) and the ceiling (constraint solver)
so that a model's number is interpretable. Everything is reported per n_prior:
an aggregate averages the unlearnable 0-prior stratum against the near-forced
3-4 prior strata and hides both.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Protocol

from wordle import (
    ALLOWED,
    ANSWERS,
    Game,
    InvalidGuess,
    constraints_from,
    is_consistent,
    sample_state,
)

MAX_INVALID_PER_TURN = 3

class Player(Protocol):
    def __call__(self, game: Game) -> str:
        """Propose a guess. May return something invalid - the harness measures that."""

def random_player(rng: random.Random) -> Player:
    """Floor: any legal word, ignoring all feedback."""
    pool = sorted(ALLOWED)

    def play(game: Game) -> str:
        return rng.choice(pool)
    
    return play

def solver_player(rng: random.Random) -> Player:
    """
    Ceiling: any anwer still consistent with the feedback.

    Note this knows the answer list, which the model does not -  that's why it's
    a ceiling rather than a fair comparison.
    """

    def play(game: Game) -> str:
        c = constraints_from(game)
        pool = [
            w for w in ANSWERS if w not in game.guesses and is_consistent(w, c)
        ]
        return rng.choice(pool) if pool else rng.choice(ANSWERS)

    return play

def play_game(player: Player, game: Game) -> dict:
    """Run one game to completion. Returns per-game outcome and error counts."""
    invalid = Counter()
    forfeit = False

    while not game.over:
        for _ in range(MAX_INVALID_PER_TURN):
            try:
                game.guess(player(game))
                break
            except InvalidGuess as e:
                invalid[e.kind] += 1
        else:
            forfeit = True
            break

    return {
        "won": game.won,
        "turns": game.turns_used,
        "forfeit": forfeit,
        "invalid": invalid,
    }

def evaluate(
    make_player: Callable[[random.Random], Player],
    n_games: int = 200,
    n_prior: int = 0,
    seed: int = 0,
) -> dict:
    """Play `n_games` starting from `n_prior` prior guesses. n_prior=0 = full games."""
    rng = random.Random(seed)
    player = make_player(rng)
    rows = [
        play_game(player, sample_state(n_prior, rng) if n_prior else Game(rng.choice(ANSWERS)))
        for _ in range(n_games)
    ]

    wins = [r for r in rows if r["won"]]
    invalid = Counter()
    for r in rows:
        invalid.update(r["invalid"])

    return {
        "n_prior": n_prior,
        "solve_rate": len(wins) / len(rows),
        "mean_turns": (sum(r["turns"] for r in wins) / len(wins)) if wins else float("nan"),
        "forfeit_rate": sum(r["forfeit"] for r in rows) / len(rows),
        "invalid_per_game": sum(invalid.values()) / len(rows),
        "invalid_kinds": dict(invalid),
    }

def report(name: str, make_player, n_games: int = 200, seed: int = 0) -> None:
    head = ["n_prior", "solve", "turns", "forfeit", "inv/game"]
    print(f"\n{name}")
    print(" ".join(h.rjust(8) for h in head))
    print("-" * 52)
    for n in range(5):
        r  = evaluate(make_player, n_games=n_games, n_prior=n, seed=seed)
        row = [
            str(n),
            f"{r['solve_rate']:.1%}",
            "n/a" if r["solve_rate"] == 0 else f"{r['mean_turns']:.1f}",
            f"{r['forfeit_rate']:.0%}",
            f"{r['invalid_per_game']:.1f}",
        ]
        print("  ".join(v.rjust(8) for v in row))


if __name__ == "__main__":
    report("FLOOR    (random legal word)", random_player)
    report("CEILING  (constraint solver)", solver_player)
