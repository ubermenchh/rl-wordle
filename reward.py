"""
Gated reward for one model turn. Rungs are ordered, never summed (PLAN.md §7).

Deviates from TASKS.md task 5 in one place: a legal word that ignores the
feedback scores 0.1, not 0.0. The base-model gate measured ~12% novel-valid
and ~0% consistent, so the valid/invalid boundary is where early signal lives.

The turn bonus is constant within a group in single-turn training and cancels
under RLOO; it only matters from task 11 on.
"""

import sys
import random
from wordle import (
    Game,
    InvalidGuess,
    constraints_from,
    is_consistent,
    sample_state,
    ANSWERS,
)
from render import parse_tag

def reward(text: str, game: Game) -> tuple[float, str]:
    tag = parse_tag(text)
    if tag is None:
        return (-0.1, "format")

    try:
        g = game.validate(tag)
    except InvalidGuess as e:
        return (0.0, e.kind)

    if g == game.secret:
        return (1.0 + 0.5 / (game.turns_used + 1), "solved")
    if is_consistent(g, constraints_from(game)):
        return (0.3, "consistent")

    return (0.1, "valid")

def _test_reward_ladder():
    g = Game("CRANE")
    g.guess("SLATE") # ..=.= -> A, E green; S, L, T absent
    r0 = reward("SLIME", g)
    r1 = reward("<guess>ZZZZZ</guess>", g)
    r2 = reward("<guess>SLOTH</guess>", g)
    r3 = reward("<guess>CRAZE</guess>", g)
    r4 = reward("<guess>CRANE</guess>", g)

    vals = [r[0] for r in (r0, r1, r2, r3, r4)]
    for a, b in zip(vals, vals[1:]): assert a < b, vals
    assert [r[1] for r in (r0, r1, r2, r3, r4)] == ["format", "not_a_word", "valid", "consistent", "solved"], (r0, r1, r2, r3, r4)

def _test_format_failures_are_worst():
    g = Game("CRANE")
    g.guess("SLATE")

    r = reward("", g); assert r == (-0.1, "format"), r
    r = reward("   ", g); assert r == (-0.1, "format"), r
    r = reward("CRANE", g); assert r == (-0.1, "format"), r
    r = reward("<guess>", g); assert r == (-0.1, "format"), r
    r = reward("<guess></guess>", g); assert r == (-0.1, "format"), r

def _test_invalid_kinds_are_distinct():
    g = Game("CRANE")
    g.guess("SLATE")

    r = reward("<guess>ABCD</guess>", g); assert r == (0.0, "bad_length"), r
    r = reward("<guess>ab3de</guess>", g); assert r == (0.0, "not_alpha"), r
    r = reward("<guess>SLATE</guess>", g); assert r == (0.0, "repeat"), r

def _test_repeat_never_beats_legal_word():
    rng = random.Random(0)
    for _ in range(200):
        g = sample_state(rng.randint(1, 3), rng)
        repeats = [reward(f"<guess>{w}</guess>", g)[0] for w in g.guesses]
        fresh = [reward(f"<guess>{w}</guess>", g)[0] for w in rng.sample(ANSWERS, 10) if w not in g.guesses]
        assert max(repeats) <= min(fresh), (g.history(), repeats, fresh)

def _test_earlier_solve_pays_more():
    g0 = Game("CRANE")
    g1 = Game("CRANE")
    for w in ("SLATE", "GRAVE", "BRAVE"):
        g1.guess(w)
    r0 = reward("<guess>CRANE</guess>", g0)[0]
    r1 = reward("<guess>CRANE</guess>", g1)[0]
    assert r0 == 1.5 and r1 == 1.125, (r0, r1)

def _test_reasoning_length_is_irrelevant():
    g = Game("CRAZE")
    r0 = reward("x" * 500 + "<guess>CRAZE</guess>", g)
    r1 = reward("<guess>SLOTH</guess> ... <guess>CRAZE</guess>", g)
    r2 = reward("<guess>CRAZE</guess>", g)

    assert r0 == r1 == r2, "padding and earlier tags should not change the score"

def _test_secret_is_argmax():
    rng = random.Random(0)
    for _ in range(100):
        g = sample_state(rng.randint(0, 3), rng)
        best = reward(f"<guess>{g.secret}</guess>", g)[0]
        others = [reward(f"<guess>{w}</guess>", g)[0] for w in ANSWERS[:50]]
        assert best >= max(others), (g.secret, g.history(), best, max(others))

def _test_grey_letters_are_inconsistent():
    g = Game("TRIAL")
    g.guess("WATCH")
    g.guess("DEALT")
    # after these two guesses TRIAL is the unique consistent answer in ANSWERS
    r0 = reward("<guess>CHEAT</guess>", g); assert r0[1] == "valid", r0[1]
    r2 = reward("<guess>TRIAL</guess>", g); assert r2[1] == "solved", r2[1]

def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(
        f"\n{len(tests)} tests passed. "
    )

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
