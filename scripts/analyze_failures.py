"""How do the model's games end? Reads the trace file of an eval run (jsonl, one row per game).

Get the file: `modal volume get <volume> <eval run>/monitors/file/traces/stream/00000.jsonl.zst .`
then `zstd -d` it. The row has the game outcome in trace["info"]["outcome"] and the model's
replies in the sampled nodes.

Run: uv run python -m scripts.analyze_failures /path/to/all.jsonl
"""

import argparse
import json
from collections import Counter

from wordle_rl.render import parse_tag
from wordle_rl.wordle import ALLOWED, ANSWERS, Game, constraints_from, is_consistent


def distance(a: str, b: str) -> int:
    # Edit distance (insert, delete, change), small words: the plain table is fast enough.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def replies(trace: dict) -> list[str]:
    return [n["message"]["content"] for n in trace["nodes"] if n["message"]["role"] == "assistant" and n["sampled"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.path)]

    outcomes = Counter()
    by_priors: dict[int, Counter] = {}
    turn_of_failure = Counter()  # which reply of the game (1 = the first one) was illegal
    near_miss = Counter()  # not_a_word: distance to the closest legal word
    legal_total = legal_consistent = 0  # legal replies: do they fit all feedback so far?
    examples: dict[str, list[str]] = {}

    for row in rows:
        data = row["task"]["data"]
        trace = row["traces"][0]
        outcome = trace["info"]["outcome"]
        outcomes[outcome] += 1
        by_priors.setdefault(len(data["priors"]), Counter())[outcome] += 1

        game = Game(data["secret"])
        for word in data["priors"]:
            game.guess(word)
        for k, reply in enumerate(replies(trace), 1):
            word = parse_tag(reply)
            try:
                game.validate(word or "")
            except Exception:
                if outcome not in ("won", "lost", "terminated", "format"):
                    turn_of_failure[k] += 1
                    examples.setdefault(outcome, []).append(f"{word!r} after {game.guesses}")
                    if outcome == "not_a_word" and word:
                        near_miss[min(distance(word, w) for w in ALLOWED)] += 1
                break
            legal_total += 1
            legal_consistent += is_consistent(word, constraints_from(game))
            game.guess(word)

    n = len(rows)
    print(f"games: {n}")
    for kind, c in outcomes.most_common():
        print(f"  {kind:12} {c:4}  {c / n:.1%}")
    print("\nby number of prior guesses (games, wins, illegal-ended):")
    for k in sorted(by_priors):
        c = by_priors[k]
        total = sum(c.values())
        illegal = total - c["won"] - c["lost"] - c["terminated"]
        print(f"  {k} priors: {total:3} games, {c['won']:3} wins ({c['won'] / total:.0%}), {illegal:3} illegal-ended ({illegal / total:.0%})")
    print("\nillegal reply: which reply of the game:", dict(sorted(turn_of_failure.items())))
    print("not_a_word: edit distance to the closest legal word:", dict(sorted(near_miss.items())))
    print(f"legal replies that fit all feedback so far: {legal_consistent}/{legal_total} ({legal_consistent / max(legal_total, 1):.0%})")
    answers = set(ANSWERS)
    print(f"(the solver would use only answers: {len(answers)} words of {len(ALLOWED)} legal ones)")
    for kind, ex in examples.items():
        print(f"\nexamples of {kind}:")
        for e in ex[:6]:
            print("  ", e)


if __name__ == "__main__":
    main()
