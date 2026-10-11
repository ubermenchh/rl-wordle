"""
Contract: the same seed gives the same states in EVERY process.

Why this exists: str hashing is randomized per process (PYTHONHASHSEED), so iterating a `set` of
strings gives a different order each run. sample_state drew its noise priors from a list built by
iterating the ALLOWED set, so `random.Random(0)` produced different games in different processes.
That broke the Modal-vs-local baseline parity check (2026-10-03) and would make any "fixed seed"
ablation comparison (task 10) meaningless.

Run from the repo root: `uv run python -m tests.test_reproducible`.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# noise=1.0 forces every prior down the ALLOWED-pool path, the one that used to iterate a set.
CODE = """
import random
from wordle_rl.wordle import sample_state
rng = random.Random(0)
for n in (1, 2, 3, 4):
    for _ in range(40):
        print(sample_state(n, rng, noise=1.0).history())
"""


def _run(hashseed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    r = subprocess.run(
        [sys.executable, "-c", CODE], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def _test_sample_state_is_independent_of_hash_seed():
    outs = [_run(hs) for hs in ("1", "2", "3")]
    assert outs[0] == outs[1] == outs[2], "sample_state depends on PYTHONHASHSEED"


def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
