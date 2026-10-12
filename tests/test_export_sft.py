"""Tests for the SFT export (Mark II, step 1).

The point of these tests: the prompt that the model is fine-tuned on must be EXACTLY the prompt
that the taskset shows during RL and eval. If the two differ, SFT teaches the model to answer a
text that it never sees again. The tests rebuild each state from the audit fields of the trace and
compare it with the exported prompt.

Run from the repo root: `uv run python -m tests.test_export_sft`.
"""

import json
import tempfile
from pathlib import Path

from sft.export_prime_rl import export
from wordle_rl.render import SYSTEM, parse_tag, render
from wordle_rl.reward import reward
from wordle_rl.taskset import WordleConfig, WordleTaskset
from wordle_rl.wordle import Game

SRC = Path(__file__).resolve().parent.parent / "data" / "sft_train.jsonl"


def _source() -> list[dict]:
    return [json.loads(line) for line in open(SRC)]


def _exported() -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "sft_wordle"
        assert export(SRC, dst) == len(_source())
        return [json.loads(line) for line in open(dst / "train.jsonl")]


def _state(row: dict) -> Game:
    game = Game(row["secret"])
    for word in row["prior_guesses"]:
        game.guess(word)
    return game


def _test_rows_have_only_the_messages_column():
    """prime-rl reads `messages`. Extra columns are only a risk, for example a secret in the data."""
    rows = _exported()
    assert all(set(r) == {"messages"} for r in rows)
    assert all([m["role"] for m in r["messages"]] == ["system", "user", "assistant"] for r in rows)


def _test_the_prompt_is_the_prompt_of_the_taskset():
    """system = the system prompt of the taskset, user = render(state). Compared with the real
    task objects of the taskset, and not only with our own functions, so a change in the taskset
    breaks this test."""
    task = next(iter(WordleTaskset(WordleConfig()).load()))
    assert task.data.system_prompt == SYSTEM
    for source, exported in zip(_source(), _exported(), strict=True):
        system, user, _ = exported["messages"]
        assert system["content"] == task.data.system_prompt
        assert user["content"] == render(_state(source))


def _test_the_answer_is_one_legal_consistent_guess_and_ends_at_the_tag():
    """The target is reasoning + one tag, and nothing after the tag. The reward of the taskset
    must call it consistent (or solved). Otherwise we teach the model a guess that RL punishes."""
    for source, exported in zip(_source(), _exported(), strict=True):
        text = exported["messages"][2]["content"]
        assert text.rstrip().endswith("</guess>") and text.count("<guess>") == 1, text
        assert parse_tag(text) == source["guess"]
        score, kind = reward(text, _state(source))
        assert kind in ("consistent", "solved"), (kind, text)


def _test_the_text_names_the_secret_only_when_the_secret_is_the_guess():
    """The reasoning line names the guess ("PULSE fits all of this."). The teacher picks any
    consistent word and never looks at the secret, so sometimes the guess is the secret by luck.
    In all other traces the secret must not be in the text: the model could not know it."""
    lucky = 0
    for source, exported in zip(_source(), _exported(), strict=True):
        before_tag = exported["messages"][2]["content"].split("<guess>")[0]
        if source["guess"] == source["secret"]:
            lucky += 1
            continue
        # Case-sensitive on purpose: the guess is written in capitals and the labels are not. The
        # secret FIXED must not match the label "Fixed:".
        assert source["secret"] not in before_tag, (source["secret"], before_tag)
    assert lucky < len(_source()) / 2, lucky  # Edge: if most guesses were the secret, the teacher is an oracle.


def run_tests():
    for name, test in sorted(globals().items()):
        if name.startswith("_test_"):
            test()
            print(f"  ok  {name}")
    print(f"\n{sum(n.startswith('_test_') for n in globals())} tests passed.")


if __name__ == "__main__":
    run_tests()
