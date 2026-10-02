"""
Wordle environment core: word lists, feedback scoring, game state.

Feedback symbols (single-token on every candidate tokenizer, letter spaceable):
    = correct letter, correct position  (green)
    - correct letter, wrong position    (yellow)
    . letter not in word                (grey)

Run: python wordle.py
Tests: python wordle.py test
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

WORD_LEN = 5
MAX_TURNS = 6
GREEN, YELLOW, GREY = "=", "-", "."

_ANSWERS_FILE = Path(__file__).with_name("answers.txt")
ANSWERS: list[str] = sorted({w.strip().upper() for w in _ANSWERS_FILE.read_text().split()})

_ALLOWED_FILE = Path(__file__).with_name("allowed.txt")

def load_allowed(extra: list[str] | None = None) -> set[str]:
    """"Allowed guesses: every word in allowed.txt, plus all answers."""
    if not _ALLOWED_FILE.exists():
        raise FileNotFoundError(f"{_ALLOWED_FILE} is missing; every result depends on it")

    allowed = set(ANSWERS)
    allowed.update(w.upper() for w in _ALLOWED_FILE.read_text().split())
    allowed.update(w.upper() for w in extra or [])
    return allowed


ALLOWED: set[str] = load_allowed()
_ALLOWED_SORTED: list[str] = sorted(ALLOWED)

def score(guess: str, secret: str) -> str:
    """
    Feedback for `guess` against `secret`.

    Two passes over a letter pool. Greens claim their letters first, then yellows
    draw from what remains - a single-pass version over-reports yellows on any
    word with a repeated letter (LEVEL vs EAGLE would mark both L's).
    """
    g, s = guess.upper(), secret.upper()
    if len(g) != WORD_LEN or len(s) != WORD_LEN:
        raise ValueError(f"expected {WORD_LEN} letters, got {guess!r} / {secret!r}")

    out = [GREY] * WORD_LEN
    pool = Counter(s)

    for i in range(WORD_LEN):
        if g[i] == s[i]:
            out[i] = GREEN
            pool[g[i]] -= 1

    for i in range(WORD_LEN):
        if out[i] == GREEN:
            continue
        if pool[g[i]] > 0:
            out[i] = YELLOW
            pool[g[i]] -= 1

    return "".join(out)

class InvalidGuess(Exception):
    """Rejected guess. `.kind` classifies why, so failure modes stay separable."""

    def __init__(self, kind: str, msg: str):
        super().__init__(msg)
        self.kind = kind # bad_length | not_alpha | not_a_word | repeat | over

class Game:
    def __init__(
        self,
        secret: str,
        max_turns: int = MAX_TURNS,
        allowed: set[str] | None = None,
    ):
        self.secret = secret.upper()
        self.max_turns = max_turns
        self.allowed = allowed if allowed is not None else ALLOWED
        self.guesses: list[str] = []
        self.feedback: list[str] = []

    @property
    def won(self) -> bool:
        return bool(self.guesses) and self.guesses[-1] == self.secret

    @property
    def over(self) -> bool:
        return self.won or len(self.guesses) >= self.max_turns

    @property
    def turns_used(self) -> int:
        return len(self.guesses)

    def history(self) -> list[tuple[str, str]]:
        return list(zip(self.guesses, self.feedback, strict=True))

    def validate(self, guess: str) -> str:
        g = guess.strip().upper()
        if not g.isalpha():
            raise InvalidGuess("not_alpha", f"{guess!r} is not alphabetic")
        if len(g) != WORD_LEN:
            raise InvalidGuess("bad_length", f"{guess!r} is not {WORD_LEN} letters")
        if g not in self.allowed:
            raise InvalidGuess("not_a_word", f"{g} is not in the word list")
        if g in self.guesses:
            raise InvalidGuess("repeat", f"{g} was already guessed")
        return g

    def guess(self, word: str) -> str:
        """Apply a guess. Invalid guesses raise without consuming a turn."""
        if self.over:
            raise InvalidGuess("over", "game is already finished")
        g = self.validate(word)
        fb = score(g, self.secret)
        self.guesses.append(g)
        self.feedback.append(fb)
        return fb

    def board(self) -> str:
        return "\n".join(f"  {' '.join(g)} {' '.join(f)}" for g, f in self.history())

@dataclass
class Constraints:
    """What the feedback so far proves about the secret"""

    fixed: dict[int, str] = field(default_factory=dict)             # index -> letter
    forbidden: dict[int, set[str]] = field(default_factory=dict)    # index -> letters
    min_count: Counter = field(default_factory=Counter)             # letter -> at least
    max_count: dict[str, int] = field(default_factory=dict)         # letter -> exactly

    def summary(self) -> dict:
        return {
            "green": "".join(self.fixed.get(i, "_") for i in range(WORD_LEN)),
            "required": sorted(self.min_count),
            "excluded": sorted(c for c, n in self.max_count.items() if n == 0),
        }

def constraints_from(game: Game) -> Constraints:
    """
    Derive exact letter-count constraints from feedback history.

    The subtle rule: within one guess, if letter L is marked grey *anywhere*,
    then the secret contains EXACTLY as many L's as were marked green/yellow in
    that guess. Without a grey L, you only learn a lower bound.
    """
    c = Constraints()
    for g, fb in game.history():
        seen = Counter()
        for i, (ch, sym) in enumerate(zip(g, fb, strict=True)):
            if sym == GREEN:
                c.fixed[i] = ch
                seen[ch] += 1
            elif sym == YELLOW:
                c.forbidden.setdefault(i, set()).add(ch)
                seen[ch] += 1
            else:
                c.forbidden.setdefault(i, set()).add(ch)

        for ch in set(g):
            n = seen[ch]
            c.min_count[ch] = max(c.min_count[ch], n)
            if any(g[i] == ch and fb[i] == GREY for i in range(WORD_LEN)):
                c.max_count[ch] = min(c.max_count.get(ch, n), n)

    return c

def is_consistent(word: str, c: Constraints) -> bool:
    """Could `word` still be the secret given everythin the feedback proves?"""
    w = word.upper()
    for i, ch in c.fixed.items():
        if w[i] != ch:
            return False
    for i, letters in c.forbidden.items():
        if w[i] in letters:
            return False
    for ch, n in c.min_count.items():
        if w.count(ch) < n:
            return False
    for ch, n in c.max_count.items():
        if w.count(ch) > n:
            return False
    return True

def sample_state(
        n_prior: int,
        rng: random.Random,
        secret: str | None = None,
        noise: float = 0.3,
    ) -> Game:
    """
    A game with UP TO `max_prior` prior guesses played.

    Fewer may land: once the only consistent answer left is the secret itself
    there is no legal prior to play, so we stop. Read `game.turns_used` for what
    actually happened -- do not assume it equals `max_prior`.

    `noise` is the chance a prior is drawn from all legal words instead of from
    words consistent with the feedback. Without it, every prior is a
    solver-quality move and you only ever train on states a *good* player
    reaches -- not the ones the model's own worse guesses will produce.
    """
    g = Game(secret or rng.choice(ANSWERS))
    for _ in range(n_prior):
        if rng.random() < noise:
            pool = [w for w in _ALLOWED_SORTED if w != g.secret and w not in g.guesses]
        else:
            c = constraints_from(g)
            pool = [
                w
                for w in ANSWERS
                if w != g.secret and w not in g.guesses and is_consistent(w, c)
            ]
        if not pool:
            break
        g.guess(rng.choice(pool))
    return g

def split_answers(seed: int = 0, eval_frac: float = 0.2, reserved: Collection[str] = ()) -> tuple[list[str], list[str]]:
    """Partition ANSWERS into (train, eval) secrets"""
    if not 0.0 < eval_frac < 1.0:
        raise ValueError(f"eval_frac must be in (0, 1), got {eval_frac}")
    unknown = set(reserved) - set(ANSWERS)
    if unknown:
        raise ValueError(f"reserved words not in ANSWERS: {sorted(unknown)}")

    pool = [w for w in ANSWERS if w not in set(reserved)]
    n_eval = round(eval_frac * len(ANSWERS))

    ev = random.Random(seed).sample(pool, n_eval)
    ev_set = set(ev)
    train = [w for w in ANSWERS if w not in ev_set]
    return train, sorted(ev)

def _test_word_lists():
    bad = [w for w in ANSWERS if len(w) != WORD_LEN or not w.isalpha()]
    assert not bad, bad
    assert len(ANSWERS) > 800, len(ANSWERS)
    assert ANSWERS == sorted(set(ANSWERS)), "answers must be unique and sorted"
    assert set(ANSWERS) <= ALLOWED, "every answer must be a legal guess"
    assert len(ALLOWED) > 5000, len(ALLOWED)


def _test_score_basics():
    assert score("CRANE", "CRANE") == "====="
    assert score("SLATE", "CRANE") == "..=.="   # A and E are both in position
    assert score("PLUMB", "CRANE") == "....."


def _test_score_duplicates():
    assert score("SPEED", "ERASE") == "-.--."   # both E's yellow, S yellow
    assert score("ERASE", "SPEED") == "-..--"
    assert score("ALLOY", "LOYAL") == "-----"   # anagram: all yellow, no greens
    assert score("LEVEL", "EAGLE") == "--.-."   # 2nd L grey: secret has one L
    assert score("GEESE", "THESE") == "..==="


def _test_score_invariants():
    rng = random.Random(0)
    for _ in range(2000):
        a, b = rng.choice(ANSWERS), rng.choice(ANSWERS)
        fb = score(a, b)
        assert len(fb) == WORD_LEN
        assert set(fb) <= {GREEN, YELLOW, GREY}
        assert (fb == GREEN * WORD_LEN) == (a == b), (a, b, fb)
        for i, sym in enumerate(fb):
            if sym == GREEN:
                assert a[i] == b[i]
        claimed = Counter(
            a[i] for i, sym in enumerate(fb) if sym in (GREEN, YELLOW)
        )
        for letter, n in claimed.items():
            assert n <= b.count(letter), (a, b, fb, letter)


def _test_game_flow():
    g = Game("CRANE")
    assert not g.over and g.turns_used == 0
    assert g.guess("slate") == "..=.="
    assert g.turns_used == 1 and not g.won

    for bad, kind in [
        ("abcdef", "bad_length"),
        ("ab3de", "not_alpha"),
        ("ZZZZZ", "not_a_word"),
        ("SLATE", "repeat"),
    ]:
        try:
            g.guess(bad)
        except InvalidGuess as e:
            assert e.kind == kind, (bad, e.kind, kind)
        else:
            raise AssertionError(f"{bad!r} should have raised {kind}")
    assert g.turns_used == 1, "invalid guesses must not consume a turn"

    g.guess("CRANE")
    assert g.won and g.over and g.turns_used == 2


def _test_game_loss():
    g = Game("CRANE", max_turns=2)
    g.guess("SLATE")
    g.guess("PLUMB")
    assert g.over and not g.won

def _test_constraints_exact_counts():
    g = Game("EAGLE")
    g.guess("LEVEL")                      # --.-.  -> exactly one L, at least 2 E
    c = constraints_from(g)
    assert c.max_count["L"] == 1
    assert c.min_count["E"] == 2
    assert is_consistent("EAGLE", c)
    assert not is_consistent("LLAMA", c)  # two L's
    assert not is_consistent("EERIE", c)  # no L at all


def _test_constraints_excludes_grey():
    g = Game("CRANE")
    g.guess("PLUMB")
    c = constraints_from(g)
    for ch in "PLUMB":
        assert c.max_count[ch] == 0
    assert is_consistent("CRANE", c)
    assert not is_consistent("PLANE", c)  # contains P and L


def _test_secret_always_consistent():
    rng = random.Random(1)
    for _ in range(300):
        g = sample_state(rng.randint(0, 4), rng)
        c = constraints_from(g)
        assert is_consistent(g.secret, c), (g.secret, g.history())
        assert not g.won
        assert g.secret not in g.guesses

def _test_sample_state_narrows():
    rng = random.Random(2)
    sizes = []
    for n in (0, 3):
        widths = []
        for _ in range(60):
            g = sample_state(n, rng, noise=0.0)      # test narrowing, not noise
            c = constraints_from(g)
            widths.append(sum(is_consistent(w, c) for w in ANSWERS))
        sizes.append(sum(widths) / len(widths))
    assert sizes[1] < sizes[0] / 2, sizes


def _test_noise_broadens_states():
    rng = random.Random(5)
    def mean_width(noise):
        out = []
        for _ in range(120):
            g = sample_state(3, rng, noise=noise)
            c = constraints_from(g)
            out.append(sum(is_consistent(w, c) for w in ANSWERS))
        return sum(out) / len(out)
    assert mean_width(0.5) > mean_width(0.0), "noisy priors should narrow less"

def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(
        f"\n{len(tests)} tests passed. "
        f"{len(ANSWERS)} answers, {len(ALLOWED):,} allowed guesses."
    )

def demo(seed: int = 7):
    """One game with a random-legal-word policy. This is the performance floor."""
    rng = random.Random(seed)
    g = Game(rng.choice(ANSWERS))
    pool = sorted(ALLOWED)
    while not g.over:
        try:
            g.guess(rng.choice(pool))
        except InvalidGuess:
            continue
    print(f"secret: {g.secret}\n")
    print(g.board())
    print(f"\n{'SOLVED in ' + str(g.turns_used) if g.won else 'LOST'}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        demo()
