"""
Contract for `wordle.split_answers`: the train/eval partition of ANSWERS.

Why this exists (PLAN.md §7): if train reward >> eval reward you are memorizing the answer
list, and you can only see that if eval secrets were never trained on. SFT traces (task 7) are
the first place a leak can happen, so the split has to exist and be stable before then.

Run as a module from the repo root: `uv run python -m tests.test_split`.
"""

import os
import subprocess
import sys
from pathlib import Path

from render import _FEWSHOT_SPECS
from wordle import ANSWERS, split_answers

ROOT = Path(__file__).resolve().parent.parent
# Only few-shot secrets that are in ANSWERS can ever be sampled as an eval secret (PLUMB is a
# legal guess but not an answer, so it cannot leak). Reserving a non-answer is a ValueError.
FEWSHOT_SECRETS = [spec[0] for spec in _FEWSHOT_SPECS if spec[0] in ANSWERS]
assert FEWSHOT_SECRETS, "no few-shot secret is in ANSWERS: the reserved-word tests would be vacuous"


def _test_partition_is_exact():
    # Every answer lands in exactly one side: nothing lost, nothing duplicated.
    train, ev = split_answers(seed=0)
    assert set(train).isdisjoint(ev), set(train) & set(ev)
    assert sorted(train + ev) == ANSWERS


def _test_both_sides_sorted_and_unique():
    # Sorted + unique matches the invariant `_test_word_lists` enforces on ANSWERS.
    train, ev = split_answers(seed=0)
    assert train == sorted(set(train))
    assert ev == sorted(set(ev))


def _test_eval_size_follows_fraction():
    for frac in (0.1, 0.2, 0.3):
        _, ev = split_answers(seed=0, eval_frac=frac)
        assert len(ev) == round(frac * len(ANSWERS)), (frac, len(ev))


def _test_same_seed_same_split_different_seed_differs():
    assert split_answers(seed=3) == split_answers(seed=3)
    assert split_answers(seed=3)[1] != split_answers(seed=4)[1]


def _test_reserved_words_never_reach_eval():
    # Few-shot secrets appear verbatim in the prompt, so as eval secrets they would be a leak.
    for seed in range(30):
        train, ev = split_answers(seed=seed, reserved=FEWSHOT_SECRETS)
        for w in FEWSHOT_SECRETS:
            assert w not in ev, (seed, w)
            assert w in train, (seed, w)


def _test_reserved_still_hits_requested_eval_size():
    # Reserving words must not silently shrink eval: the draw comes from the non-reserved pool.
    _, ev = split_answers(seed=0, eval_frac=0.2, reserved=FEWSHOT_SECRETS)
    assert len(ev) == round(0.2 * len(ANSWERS))


def _test_bad_inputs_raise():
    # A typo'd reserved word that is silently ignored would reopen the leak it was meant to close.
    # Lambdas rather than a list of **kwargs dicts: mixed value types defeat the type checker.
    bad_calls = {
        "unknown reserved word": lambda: split_answers(seed=0, reserved=["ZZZZZ"]),
        "eval_frac=0": lambda: split_answers(seed=0, eval_frac=0.0),
        "eval_frac=1": lambda: split_answers(seed=0, eval_frac=1.0),
    }
    for name, call in bad_calls.items():
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {name}")


def _test_split_is_stable_across_processes():
    # str hashing is randomized per process. A split that draws from a set (or any
    # hash-ordered container) changes between runs, so eval words would drift from run to run.
    code = "from wordle import split_answers; print(split_answers(seed=0)[1])"
    outs = []
    for hashseed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        r = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout)
    assert outs[0] == outs[1], "split depends on PYTHONHASHSEED"


def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
