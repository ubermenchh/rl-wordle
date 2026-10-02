"""
Contract for `gate.group_signal`: does ONE group of k completions carry a learning signal?

Why this exists (PLAN.md §12 sink #2, and risk 3 of the revised plan): the gate counted a group
as informative only if 0 < #solved < k. But reward.py is a dense ladder, so a group with zero
solves can still have reward variance, and an RL step on it is not wasted. The catch is that
variance between "valid" and "invalid" only teaches spelling. So we report three flags:

    nondegen_solved      0 < #solved < k                         the original gate definition
    nondegen_reward      the group's rewards are not all equal   any gradient at all
    nondegen_consistent  some, but not all, reach the 0.3 rung   deduction signal, not spelling

Scoring uses the STRICT reward() path (tagged guess only), because that is what the RL loop
will see. gate()'s existing `solve` uses parse_lenient and measures capability; it is a
different quantity on purpose.

Run from the repo root: `uv run python -m tests.test_gate_signal`.
"""

from gate import group_signal
from reward import reward
from wordle import Game


def _game() -> Game:
    # Secret CRANE after SLATE -> ..=.= : A, E green; S, L, T absent.
    # CRANE solves; CRAZE is consistent but wrong; SLOTH is a legal word that ignores the
    # feedback; ZZZZZ is not a word; "" has no tag at all.
    g = Game("CRANE")
    g.guess("SLATE")
    return g


def _tag(w: str) -> str:
    return f"<guess>{w}</guess>"


SOLVED, CONSISTENT, VALID, NOT_A_WORD, NO_TAG = (
    _tag("CRANE"),
    _tag("CRAZE"),
    _tag("SLOTH"),
    _tag("ZZZZZ"),
    "",
)


def _test_rewards_match_reward_fn():
    # group_signal must score with reward(), not a private re-implementation that can drift.
    g = _game()
    texts = [SOLVED, CONSISTENT, VALID, NOT_A_WORD, NO_TAG]
    sig = group_signal(texts, g)
    assert sig["rewards"] == [reward(t, g)[0] for t in texts]


def _test_fully_mixed_group_trips_every_flag():
    sig = group_signal([SOLVED, CONSISTENT, VALID, NO_TAG], _game())
    assert sig["solved"] == 1
    assert sig["nondegen_solved"] and sig["nondegen_reward"] and sig["nondegen_consistent"]


def _test_all_solved_is_degenerate():
    sig = group_signal([SOLVED] * 4, _game())
    assert sig["solved"] == 4
    assert not (sig["nondegen_solved"] or sig["nondegen_reward"] or sig["nondegen_consistent"])


def _test_all_wrong_same_rung_is_degenerate():
    # Identical rewards -> zero advantage under any baseline-subtracting estimator.
    for texts in ([NO_TAG] * 4, [VALID] * 4, [CONSISTENT] * 4):
        sig = group_signal(texts, _game())
        assert not sig["nondegen_reward"], texts
        assert not sig["nondegen_solved"], texts


def _test_spelling_only_signal_is_not_a_deduction_signal():
    # The case the original gate missed AND the case that would fool a naive reward-std gate:
    # variance exists, but only between "legal word" and "not a word". No solve, no deduction.
    sig = group_signal([VALID, VALID, NOT_A_WORD, NO_TAG], _game())
    assert sig["nondegen_reward"]
    assert not sig["nondegen_solved"]
    assert not sig["nondegen_consistent"]


def _test_deduction_signal_without_any_solve():
    # Some guesses honour the feedback and some ignore it, but nothing solves yet. This is
    # the group the old gate called dead and the dense reward can still learn from.
    sig = group_signal([CONSISTENT, CONSISTENT, VALID, VALID], _game())
    assert sig["solved"] == 0
    assert not sig["nondegen_solved"]
    assert sig["nondegen_reward"] and sig["nondegen_consistent"]


def _test_solved_counts_as_consistent():
    # A solve is trivially consistent with its own feedback, so solved + consistent is a
    # uniform group on the consistency flag, while solved + valid is not.
    assert not group_signal([SOLVED, CONSISTENT], _game())["nondegen_consistent"]
    assert group_signal([SOLVED, VALID], _game())["nondegen_consistent"]


def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
