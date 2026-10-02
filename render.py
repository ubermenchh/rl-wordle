"""
Prompt rendering: Game -> model-facing text, and parsing the model's reply.

1. Everything letter-spaced. On Qwen3-0.6B "C R A N E" is 5 tokens; "CRANE" is
   2 (['CR','ANE']). Measured 2026-09-21. Wordle is per-letter reasoning,
   so the model needs letters as addressable units.
2. Constraints are pre-digested into plain English. A 0.6B should spend its
   capacity choosing a word, not re-deriving bookkeeping the env already has.
"""

from __future__ import annotations

import re
from collections import Counter

from wordle import GREEN, GREY, WORD_LEN, YELLOW, Game, constraints_from

SYSTEM = f"""You are playing Wordle. Guess the secret {WORD_LEN}-letter word.

After each guess you get one symbol per letter:
    {GREEN} correct letter, correct position
    {YELLOW} letter is in the word, wrong position
    {GREY} letter is not in the word

Think briefly, then answer with exactly: <guess>WORD</guess>
Never repeat a word you have already guessed."""

def spaced(s: str) -> str:
    return " ".join(s.upper())

def render_history(game: Game) -> str:
    if not game.history():
        return "No guesses yet."
    return "\n".join(f"  {spaced(g)}  |  {spaced(fb)}" for g, fb in game.history())


def render_constraints(game: Game) -> str:
    c = constraints_from(game)

    # Counter returns 0 for missing keys, so min_count holds zero-valued entries
    # for grey letters. Filter on value, not key presence.
    required = {ch: n for ch, n in c.min_count.items() if n > 0}
    excluded = sorted(ch for ch, n in c.max_count.items() if n == 0)

    lines = []
    for i in range(WORD_LEN):
        if i in c.fixed:
            lines.append(f"  slot {i + 1}: {c.fixed[i]}")
        else:
            out = sorted(c.forbidden.get(i, set()) - set(excluded))
            tail = f"   not {' '.join(out)}" if out else ""
            lines.append(f"  slot {i + 1}: ?{tail}")

    placed = Counter(c.fixed.values())
    still_needed = sorted(ch for ch, n in required.items() if placed[ch] < n)
    if still_needed:
        lines.append(f"Must contain: {' '.join(still_needed)}")
    if excluded:
        lines.append(f"Absent:       {' '.join(excluded)}")
    if game.guesses:
        lines.append(f"Guessed:      {', '.join(game.guesses)}")
    return "\n".join(lines)

def render(game: Game) -> str:
    turns_left = game.max_turns - game.turns_used
    return (
        f"{render_history(game)}\n\n"
        f"{render_constraints(game)}\n\n"
        f"Guesses remaining: {turns_left}\n"
        f"Make your next guess."
    )

_GUESS_RE = re.compile(rf"<guess\s*>\s*([A-Za-z]{{{WORD_LEN}}})\s*</\s*guess\s*>", re.IGNORECASE)

def parse_guess(text: str) -> str | None:
    """
    Extract the guess, or None if absent/malformed.

    Takes the LAST match: models often name candidates while reasoning and then
    commit at the end, so the final tag is the actual action.
    """
    matches = _GUESS_RE.findall(text)
    return matches[-1].upper() if matches else None

_TAG_RE = re.compile(r"<guess\s*>\s*([^<]+)\s*</\s*guess\s*>", re.IGNORECASE)

def parse_tag(text: str) -> str | None:
    """Extract whatever is inside the guess tag, any length."""
    m = _TAG_RE.findall(text)
    return m[-1].upper() if m else None

_SPACED_RE = re.compile(r"(?:\b[A-Za-z]\b[^A-Za-z]+){3,}\b[A-Za-z]\b")
_LABELLED_RE = re.compile(r"guess\s*[:\-]?\s*([A-Za-z]{3,8})", re.IGNORECASE)

def parse_lenient(text: str) -> str | None:
    """
    Best-effort guess extraction, for CAPABILITY measurement only.

    Models that ignore the tag protocol still often attempt the task -- Gemma
    replies "C H E A R" because our prompt is letter-spaced. Scoring those as
    total failures measures our output format, not their Wordle ability.
    Strict protocol compliance is repoted separately as `tag`.
    """
    hit = parse_tag(text)
    if hit:
        return hit
    m = _SPACED_RE.findall(text)
    if m:
        return re.sub(r"[^A-Za-z]", "", m[-1]).upper()
    m = _LABELLED_RE.findall(text)
    if m:
        return m[-1].upper()
    words = re.findall(r"\b[A-Za-z]{5}\b", text)
    return words[-1].upper() if words else None

_FEWSHOT_SPECS = [
    (
        "PLUMB",
        ["CRANE", "SLIMY"],
        "PLUMB",
        "Slots 2 and 4 are fixed, so the shape is _ L _ M _. "
        "A, C, E, I, N, R, S, Y are absent. "
        "PLUMB has L in slot 2 and M in slot 4, and P, U, B are still allowed.",
    ),
    (
        "DOUBT",
        ["BRING", "BLOAT"],
        "DOUBT",
        "Slot 5 is fixed as T. B and O must both appear, but B cannot be in "
        "slot 1 and O cannot be in slot 3. "
        "DOUBT puts O in slot 2 and B in slot 4, ends in T, and uses no absent letters.",
    ),
]
FEWSHOT_SECRETS = [spec[0] for spec in _FEWSHOT_SPECS]

def fewshot_messages() -> list[dict]:
    """Worked examples as alternating chat turns."""
    from wordle import Game

    out = []
    for secret, priors, answer, reason in _FEWSHOT_SPECS:
        g = Game(secret)
        for p in priors:
            g.guess(p)
        out.append({"role": "user", "content": render(g)})
        out.append({"role": "assistant", "content": f"{reason}\n<guess>{answer}</guess>"})
    return out

def _test_fewshot_examples_are_valid():
    from wordle import ALLOWED, Game, constraints_from, is_consistent

    for secret, priors, answer, reason in _FEWSHOT_SPECS:
        assert answer in ALLOWED, answer
        g = Game(secret)
        for p in priors:
            assert p in ALLOWED, f"{p} is not a legal guess"
            g.guess(p)
        assert is_consistent(answer, constraints_from(g)), (secret, priors, answer)
        assert answer == secret, "examples must demonstrate a winning deduction"
        assert parse_guess(f"<guess>{answer}</guess>") == answer
        assert reason.strip()
        assert answer in reason, (answer, reason)
        assert "0" not in reason, reason
    assert len(fewshot_messages()) == 2 * len(_FEWSHOT_SPECS)

def _test_spacing_is_per_letter():
    assert spaced("crane") == "C R A N E"
    assert len(spaced("CRANE").split()) == WORD_LEN


def _test_parse_guess():
    assert parse_guess("<guess>crane</guess>") == "CRANE"
    assert parse_guess("blah <guess> Slate </guess> blah") == "SLATE"
    assert parse_guess("maybe <guess>CRANE</guess> no <guess>SLATE</guess>") == "SLATE"
    assert parse_guess("no tags here") is None
    assert parse_guess("<guess>TOOLONG</guess>") is None
    assert parse_guess("<guess>ab3de</guess>") is None


def _test_render_roundtrip():
    import random

    from wordle import sample_state

    rng = random.Random(0)
    for _ in range(50):
        g = sample_state(rng.randint(1, 3), rng)
        text = render(g)
        for gs in g.guesses:
            assert gs in text                      # history is visible
        assert "Guesses remaining" in text
        assert parse_guess(f"<guess>{g.secret}</guess>") == g.secret


def _test_summary_is_not_self_contradictory():
    import random

    from wordle import is_consistent, sample_state

    rng = random.Random(0)
    for _ in range(200):
        g = sample_state(rng.randint(1, 4), rng)
        c = constraints_from(g)
        required = {ch for ch, n in c.min_count.items() if n > 0}
        excluded = {ch for ch, n in c.max_count.items() if n == 0}
        assert not (required & excluded), (g.history(), required & excluded)
        assert is_consistent(g.secret, c)
        text = render(g)
        for ch in excluded:
            assert f"slot 1: {ch}" not in text

def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    import random
    import sys

    from wordle import sample_state

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        rng = random.Random(3)
        g = sample_state(2, rng)
        print(f"[secret: {g.secret}]\n")
        print(SYSTEM)
        print("\n" + "-" * 60 + "\n")
        print(render(g))
