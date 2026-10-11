"""
Contract for sft/teacher.py: the two functions that define what the SFT model is taught.

    explain(game, guess) -> str      the templated reasoning shown before the <guess> tag
    keep(game, guess, eval_words) -> bool   filter: is (state, guess) fit to train on?

Why property tests and not exact strings: the wording is yours to choose, but three properties are
not negotiable, because SFT teaches exactly what the traces show.

  1. The reasoning must restate what the feedback proves (green slots, must-contain letters, absent
     letters). The base model's measured failure is copying the previous guess and ignoring the
     Absent list (results/gate_2026-10-02.md), so the reasoning has to make those facts explicit.
  2. The reasoning may use only what a player can see. If explain() reads game.secret, the model is
     trained on information it will never have at inference, and learns to "know" the answer.
  3. The filter must reject guesses the model should never imitate: inconsistent, repeated,
     not-a-word, or an eval word (eval stays unseen so the post-SFT gate measures deduction).

Conventions the tests rely on: letters are written as standalone capitals ("A, C, E are absent"),
slot numbers as standalone digits ("slot 2"), and the guess appears as a whole word. So don't use a
capital "A" or "I" as an English word in the reasoning.

Run from the repo root: `uv run python -m tests.test_teacher`.
"""

import random
import re

from sft.teacher import OPENER, explain, keep
from wordle_rl.render import FEWSHOT_SECRETS
from wordle_rl.wordle import (
    ALLOWED,
    ANSWERS,
    WORD_LEN,
    Game,
    constraints_from,
    is_consistent,
    sample_state,
    split_answers,
)

MAX_WORDS = 80  # keeps targets short: the gate allows 128 new tokens, and a short target is easier to learn


def _states(n_per_prior: int = 25, seed: int = 0):
    """(game, guess) pairs: sampled mid-game states and a random consistent guess, like the solver."""
    rng = random.Random(seed)
    for n_prior in range(5):
        for _ in range(n_per_prior):
            g = sample_state(n_prior, rng)
            c = constraints_from(g)
            pool = [w for w in ANSWERS if w not in g.guesses and is_consistent(w, c)]
            yield g, rng.choice(pool)


def _letters(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z]\b", text))


def _numbers(text: str) -> set[int]:
    return {int(d) for d in re.findall(r"\b\d\b", text)}


def _test_deterministic():
    for g, guess in _states():
        assert explain(g, guess) == explain(g, guess)


def _test_is_short_and_contains_the_guess():
    for g, guess in _states():
        text = explain(g, guess)
        assert len(text.split()) <= MAX_WORDS, (len(text.split()), text)
        assert re.search(rf"\b{guess}\b", text), (guess, text)


def _test_restates_every_absent_letter():
    # The measured failure: the model ignores the Absent list. The reasoning must name each letter.
    for g, guess in _states():
        c = constraints_from(g)
        absent = {ch for ch, n in c.max_count.items() if n == 0}
        missing = absent - _letters(explain(g, guess))
        assert not missing, (g.history(), missing, explain(g, guess))


def _test_restates_green_slots_and_must_contain_letters():
    for g, guess in _states():
        c = constraints_from(g)
        text = explain(g, guess)
        for i, ch in c.fixed.items():
            assert (i + 1) in _numbers(text) and ch in _letters(text), (i, ch, text)
        placed = {ch: sum(1 for v in c.fixed.values() if v == ch) for ch in c.fixed.values()}
        needed = {ch for ch, n in c.min_count.items() if n > placed.get(ch, 0)}
        assert needed <= _letters(text), (g.history(), needed, text)


def _test_mentions_no_letter_the_state_does_not_justify():
    # Any standalone letter must be tied to the state or the guess; otherwise it is a made-up claim.
    for g, guess in _states():
        c = constraints_from(g)
        known = set(guess) | set(c.fixed.values()) | set(c.min_count) | set(c.max_count)
        for s in c.forbidden.values():
            known |= s
        stray = _letters(explain(g, guess)) - known
        assert not stray, (g.history(), guess, stray)


def _test_empty_history_makes_no_letter_claims():
    # Nothing is known at prior 0, so there is nothing to restate: no standalone letters or slots.
    g = Game("CRANE")
    text = explain(g, "SLATE")
    assert not _letters(text) and not _numbers(text), text


def _test_never_reads_the_secret():
    # Same visible history, different hidden secret -> identical reasoning. A leak would show here.
    rng = random.Random(1)
    checked = 0
    for g, guess in _states():
        if not g.guesses:
            continue
        for other in rng.sample(ANSWERS, 60):
            if other == g.secret or other in g.guesses:
                continue
            twin = Game(other)
            try:
                for w in g.guesses:
                    twin.guess(w)
            except Exception:
                continue
            if twin.feedback == g.feedback:  # same visible state, different secret
                assert explain(g, guess) == explain(twin, guess), (g.history(), g.secret, other)
                checked += 1
                break
    assert checked >= 10, f"only {checked} twin states found; the test would be vacuous"


def _test_keep_accepts_a_good_trace():
    g = Game("CRANE")
    g.guess("SLATE")
    assert keep(g, "CRAZE", set())
    assert keep(g, "CRANE", set())  # the secret is a fine target: it is consistent, not an oracle label


def _test_keep_rejects_bad_traces():
    g = Game("CRANE")
    g.guess("SLATE")  # feedback ..=.= : A, E green; S, L, T absent
    assert not keep(g, "SLOTH", set()), "inconsistent: uses absent letters S, L, T"
    assert not keep(g, "SLATE", set()), "repeat"
    assert not keep(g, "ZZZZZ", set()), "not a word"
    assert not keep(g, "ABCD", set()), "wrong length"
    assert not keep(g, "CRAZE", {"CRAZE"}), "eval word: eval must stay unseen"


def _test_keep_agrees_with_is_consistent_on_sampled_states():
    # Cross-check against the reward's own definition on many states: a kept guess must be one the
    # reward would score as at least "consistent" (0.3), otherwise SFT teaches a lower rung.
    for g, guess in _states():
        assert keep(g, guess, set()), (g.history(), guess)
        assert len(guess) == WORD_LEN


def _test_opener_is_a_legal_word_outside_the_eval_split():
    # The driver teaches OPENER at prior 0. If it were an eval word, every SFT trace would show the
    # model an eval answer, and the post-SFT gate could no longer call eval words unseen.
    _, eval_words = split_answers(reserved=[w for w in FEWSHOT_SECRETS if w in ANSWERS])
    assert OPENER in ALLOWED and len(OPENER) == WORD_LEN
    assert OPENER not in eval_words
    assert keep(Game("BLOOD"), OPENER, set(eval_words)), "the opener must pass its own filter"


def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
