"""Is the reasoning line of the model right, and does the word obey it?

The reply is "Fixed: slot 2 is O. Must contain: N, R. Absent: A, C. DONOR fits all of this." then
the tag. Two questions for each legal reply:
  1. Does the facts line (Fixed / Must contain / Absent) match what the feedback proves? The true
     line is the one the SFT teacher writes (sft/teacher.py::explain), so the format is the same.
  2. Does the word obey the model's own line?
Then: how do the replies that break the true constraints break them?

Run: uv run python -m scripts.check_reasoning sft=/tmp/a.jsonl rl1=/tmp/b.jsonl rl2=/tmp/c.jsonl
"""

import argparse
import json
import re
from collections import Counter

from sft.teacher import explain
from wordle_rl.render import parse_tag
from wordle_rl.wordle import Game, constraints_from, is_consistent


def facts(text: str) -> dict:
    """The three facts sections of a reasoning line, as sets. Missing section = empty set."""
    out = {"fixed": set(), "must": set(), "absent": set()}
    for part in text.split("<guess>")[0].split(". "):
        part = part.strip()
        if part.startswith("Fixed:"):
            out["fixed"] = {(int(s), c) for s, c in re.findall(r"slot (\d) is ([A-Z])", part)}
        elif part.startswith("Must contain:"):
            out["must"] = set(re.findall(r"\b([A-Z])\b", part.removeprefix("Must contain:")))
        elif part.startswith("Absent:"):
            out["absent"] = set(re.findall(r"\b([A-Z])\b", part.removeprefix("Absent:")))
    return out


def obeys_own_line(word: str, f: dict) -> bool:
    return (
        all(word[s - 1] == c for s, c in f["fixed"])
        and all(c in word for c in f["must"])
        and not any(c in word for c in f["absent"])
    )


def violations(word: str, game: Game) -> set[str]:
    c = constraints_from(game)
    found = set()
    if any(word[i] != ch for i, ch in c.fixed.items()):
        found.add("breaks a green slot")
    if any(word[i] in letters for i, letters in c.forbidden.items() if i not in c.fixed):
        found.add("puts a letter in a slot it was ruled out of")
    if any(word.count(ch) > n for ch, n in c.max_count.items()):
        found.add("uses a grey letter (or too many copies)")
    if any(word.count(ch) < n for ch, n in c.min_count.items()):
        found.add("misses a letter that must be in")
    return found


def run(name: str, path: str) -> None:
    total = right = own_ok = 0
    own_ok_when_right = n_right = own_ok_when_wrong = n_wrong = 0
    consistent_when_right = consistent_when_wrong = 0
    kinds = Counter()
    inconsistent = 0
    section_wrong = Counter()

    for line in open(path):
        row = json.loads(line)
        data = row["task"]["data"]
        game = Game(data["secret"])
        for w in data["priors"]:
            game.guess(w)
        for node in row["traces"][0]["nodes"]:
            if node["message"]["role"] != "assistant" or not node["sampled"]:
                continue
            reply = node["message"]["content"]
            word = parse_tag(reply)
            try:
                game.validate(word or "")
            except Exception:
                break  # the game ends on an illegal reply
            if not game.guesses and not data["priors"]:
                game.guess(word)
                continue  # turn 1 with no feedback: there are no facts to state
            total += 1
            stated, true = facts(reply), facts(explain(game, word))
            good = stated == true
            for k in stated:
                section_wrong[k] += stated[k] != true[k]
            own = obeys_own_line(word, stated)
            ok = is_consistent(word, constraints_from(game))
            right += good
            own_ok += own
            if good:
                n_right += 1
                own_ok_when_right += own
                consistent_when_right += ok
            else:
                n_wrong += 1
                own_ok_when_wrong += own
                consistent_when_wrong += ok
            if not ok:
                inconsistent += 1
                kinds.update(violations(word, game))
            game.guess(word)

    def pct(a: int, b: int) -> str:
        return f"{a}/{b} ({a / max(b, 1):.0%})"

    print(f"== {name}: {total} legal replies with feedback to use")
    print(f"  facts line exactly right:          {pct(right, total)}")
    print(f"  wrong sections (fixed / must / absent): {dict(section_wrong)}")
    print(f"  word obeys the model's own line:   {pct(own_ok, total)}")
    print(f"  word consistent | line right:      {pct(consistent_when_right, n_right)}")
    print(f"  word consistent | line wrong:      {pct(consistent_when_wrong, n_wrong)}")
    print(f"  inconsistent replies: {inconsistent}; how they break the constraints (can overlap):")
    for k, v in kinds.most_common():
        print(f"    {v:4} ({v / max(inconsistent, 1):.0%})  {k}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="name=path/to/all.jsonl")
    for item in parser.parse_args().runs:
        name, path = item.split("=", 1)
        run(name, path)


if __name__ == "__main__":
    main()
