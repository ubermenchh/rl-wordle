"""Tests for the metrics of WordleTask (Mark I, step 6b).

The first run on a real model (2026-10-11) printed only `ladder 0.00`, and `metrics` was empty.
The Mark I demo must print the win rate and the format errors, so the task reports them as
metrics. A metric does not change the reward. It reads `trace.info["outcome"]`, which
`play_game` sets, and the test uses a fake trace with only that field.

Run from the repo root: `uv run python -m tests.test_metrics`.
"""

import asyncio

import verifiers.v1 as vf

from wordle_rl.taskset import WordleData, WordleTask

OUTCOMES = ["won", "lost", "terminated", "format", "bad_length", "not_alpha", "not_a_word", "repeat"]


class _Trace:
    def __init__(self, outcome: str | None):
        self.info = {} if outcome is None else {"outcome": outcome}


def _task() -> WordleTask:
    return WordleTask(WordleData(prompt="x", secret="CRANE", priors=[]), vf.TaskConfig())


def _metric(name: str, outcome: str | None) -> float:
    return asyncio.run(getattr(_task(), name)(_Trace(outcome)))


def _test_won_is_one_only_for_a_win():
    """The win rate is the mean of this metric. A loss, a cut run and an illegal reply are 0."""
    assert {o: _metric("won", o) for o in OUTCOMES} == {o: float(o == "won") for o in OUTCOMES}


def _test_illegal_counts_every_illegal_kind():
    """The base model of 2026-10-11 ended 20 of 20 games on an illegal reply. This must show.
    A loss and a cut run are not illegal: the model played a legal game."""
    illegal = {"format", "bad_length", "not_alpha", "not_a_word", "repeat"}
    assert {o: _metric("illegal", o) for o in OUTCOMES} == {o: float(o in illegal) for o in OUTCOMES}


def _test_format_error_is_only_the_missing_tag():
    """PLAN.md section 1.1: action validity, and not the feedback, is the main failure of small
    models. We must see the missing tag apart from a wrong word."""
    assert {o: _metric("format_error", o) for o in OUTCOMES} == {o: float(o == "format") for o in OUTCOMES}


def _test_a_trace_without_an_outcome_is_not_a_win():
    """Edge: an episode that failed before play_game set the outcome (a provider error). It must
    not count as a win or crash the scoring."""
    assert _metric("won", None) == 0.0
    assert _metric("format_error", None) == 0.0


def run_tests():
    for name, test in sorted(globals().items()):
        if name.startswith("_test_"):
            test()
            print(f"  ok  {name}")
    print(f"\n{sum(n.startswith('_test_') for n in globals())} tests passed.")


if __name__ == "__main__":
    run_tests()
