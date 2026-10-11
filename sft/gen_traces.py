"""
Generate the SFT dataset: (prompt, reasoning + <guess>) pairs from a solver-style teacher.

The teacher knows only what a player sees. It picks a random word CONSISTENT with the feedback,
never the secret, so every label is derivable from its prompt. Secrets and candidate words come
from the train split only, and states whose prior guesses include an eval word are rejected, so
eval words appear nowhere in the data.

The wording of the reasoning and the filter live in sft/teacher.py (yours); this file only
assembles states, calls them, and writes JSONL.

Run: uv run python -m sft.gen_traces
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from sft.teacher import OPENER, explain, keep
from wordle_rl.render import FEWSHOT_SECRETS, SYSTEM, render
from wordle_rl.wordle import (
    ANSWERS,
    Game,
    constraints_from,
    is_consistent,
    sample_state,
    split_answers,
)


def _train_eval() -> tuple[list[str], set[str]]:
    # The canonical split, built the same way gate() builds it, so "eval" means the same words.
    train, ev = split_answers(reserved=[w for w in FEWSHOT_SECRETS if w in ANSWERS])
    return train, set(ev)


def _trace(g: Game, guess: str) -> dict:
    """One example: the chat messages SFT trains on, plus metadata for auditing (never trained on)."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": render(g)},
            {"role": "assistant", "content": f"{explain(g, guess)}\n<guess>{guess}</guess>"},
        ],
        "n_prior": g.turns_used,
        "prior_guesses": list(g.guesses),
        "secret": g.secret,
        "guess": guess,
    }


def make_traces(n_total: int = 2000, opener_copies: int = 40, seed: int = 0) -> list[dict]:
    """
    `opener_copies` identical prior-0 examples (the opener), the rest split evenly over priors 1-4.

    Prior 0 has one possible prompt, so more than a handful of copies is just weighting; it is a
    parameter so that weight is an explicit choice. Priors 1-4 are where there is something to deduce.
    """
    rest = n_total - opener_copies
    if rest <= 0 or rest % 4:
        raise ValueError(f"n_total - opener_copies must be a positive multiple of 4, got {rest}")
    per_stratum = rest // 4

    train, ev = _train_eval()
    rng = random.Random(seed)
    out: list[dict] = []

    # 1. Prior 0: the same prompt every time and the fixed opener as the answer.
    for _ in range(opener_copies):
        g = Game(rng.choice(train))
        assert keep(g, OPENER, ev), "OPENER must pass its own filter"
        out.append(_trace(g, OPENER))

    # 2. Priors 1-4: sample a mid-game state, then let the teacher pick any consistent train word.
    seen: set[tuple[str, str]] = set()
    for k in range(1, 5):
        made = 0
        while made < per_stratum:
            g = sample_state(k, rng, secret=rng.choice(train))
            # sample_state can stop early (nothing legal left), and its priors can be eval words:
            # reject both, so strata are exact and eval words never appear in a prompt.
            if g.turns_used != k or ev & set(g.guesses):
                continue
            c = constraints_from(g)
            # is_consistent first is only a speed filter; keep() stays the authority on what is fit to train on.
            pool = [w for w in train if is_consistent(w, c) and keep(g, w, ev)]
            t = _trace(g, rng.choice(pool))
            key = (t["messages"][1]["content"], t["messages"][2]["content"])
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
            made += 1

    # 3. Shuffle so a batch mixes strata (seeded, so the file is still byte-reproducible).
    rng.shuffle(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--opener-copies", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/sft_train.jsonl"))
    args = p.parse_args()

    traces = make_traces(args.n, args.opener_copies, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    by_prior = Counter(t["n_prior"] for t in traces)
    words = [len(t["messages"][2]["content"].split()) for t in traces]
    print(f"wrote {len(traces)} traces to {args.out}")
    print("by n_prior:", dict(sorted(by_prior.items())))
    print(f"completion words: mean {sum(words) / len(words):.1f}, max {max(words)}")
    for t in traces[:3]:
        print("---", t["n_prior"], "prior(s), secret", t["secret"], "->", t["guess"])
        print(t["messages"][2]["content"])


if __name__ == "__main__":
    main()
