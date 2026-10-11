"""
What the SFT model is taught: the reasoning shown before <guess>, and which examples to keep.
"""

from collections import Counter

from wordle_rl.wordle import Game, InvalidGuess, constraints_from, is_consistent

OPENER = "CRANE" # fixed first guess

def explain(game: Game, guess: str) -> str:
    """
    Templated reasoning for `guess`, built only from the feedback proves.

    Example (after SLATE, secret CRANE):
        "Fixed: slot 3 is A, slot 5 is E. Absent: L, S, T. CRAZE fits all of this."
    """
    if not game.guesses:
        return f"Nothing is known yet, so open with {guess}"

    c = constraints_from(game)
    parts = []

    if c.fixed:
        slots = ", ".join(f"slot {i+1} is {ch}" for i, ch in sorted(c.fixed.items()))
        parts.append(f"Fixed: {slots}")

    places = Counter(c.fixed.values())
    needed = sorted(ch for ch, n in c.min_count.items() if n > places[ch])
    if needed:
        parts.append(f"Must contain: {', '.join(needed)}")

    absent = sorted(ch for ch, n in c.max_count.items() if n == 0)
    if absent:
        parts.append(f"Absent: {', '.join(absent)}")

    parts.append(f"{guess} fits all of this.")
    return " ".join(parts)

def keep(game: Game, guess: str, eval_words: set[str]) -> bool:
    word = guess.strip().upper()
    try:
        game.validate(word)
    except InvalidGuess:
        return False

    return is_consistent(word, constraints_from(game)) and word not in eval_words
