"""
Contract for sft/gen_traces.py: the SFT dataset the model will be trained on.

Every property here guards one way the data could quietly teach the wrong thing:

  - eval words never appear as a target OR as a prior guess in a prompt, so the post-SFT gate
    measures deduction on words the model has never seen (results/gate_2026-10-02.md);
  - the teacher picks a consistent word, never "the secret" by oracle, so the label is derivable
    from the prompt alone (checked by: same prompt -> the guess is consistent, and the secret is
    not systematically the target);
  - every completion scores at least the "consistent" rung under the real reward(), so SFT can
    never teach a lower rung than the RL phase will pay for;
  - the dataset is exactly reproducible (same seed -> same bytes, in any process).

Run from the repo root: `uv run python -m tests.test_gen_traces`.
"""

import os
import subprocess
import sys
from pathlib import Path

from sft.gen_traces import make_traces
from sft.teacher import OPENER, explain
from wordle_rl.render import FEWSHOT_SECRETS, SYSTEM, parse_tag, render
from wordle_rl.reward import reward
from wordle_rl.wordle import ANSWERS, Game, split_answers

ROOT = Path(__file__).resolve().parent.parent
N, OPENER_COPIES = 200, 20
_, EVAL = split_answers(reserved=[w for w in FEWSHOT_SECRETS if w in ANSWERS])
TRACES = make_traces(n_total=N, opener_copies=OPENER_COPIES, seed=0)


def _replay(t: dict) -> Game:
    g = Game(t["secret"])
    for w in t["prior_guesses"]:
        g.guess(w)
    return g


def _test_size_and_strata():
    assert len(TRACES) == N
    by_prior = {k: sum(t["n_prior"] == k for t in TRACES) for k in range(5)}
    assert by_prior[0] == OPENER_COPIES, by_prior
    # The rest is split evenly over priors 1-4: that is where the model has something to deduce.
    assert {by_prior[k] for k in (1, 2, 3, 4)} == {(N - OPENER_COPIES) // 4}, by_prior


def _test_prior_zero_teaches_the_fixed_opener():
    for t in TRACES:
        if t["n_prior"] == 0:
            assert t["guess"] == OPENER, t["guess"]


def _test_no_eval_word_anywhere():
    # Neither as a target nor as a prior guess shown in a prompt. The secret is also a train word.
    ev = set(EVAL)
    for t in TRACES:
        assert t["guess"] not in ev, t["guess"]
        assert not ev & set(t["prior_guesses"]), (t["prior_guesses"], ev & set(t["prior_guesses"]))
        assert t["secret"] not in ev, t["secret"]


def _test_messages_are_what_the_gate_prompts_with():
    # Same system prompt and the same rendering as gate()'s plain config, with no few-shot turns:
    # SFT replaces few-shot, and train/eval prompts must look identical.
    for t in TRACES:
        g = _replay(t)
        msgs = t["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[0]["content"] == SYSTEM
        assert msgs[1]["content"] == render(g)
        assert msgs[2]["content"] == f"{explain(g, t['guess'])}\n<guess>{t['guess']}</guess>"


def _test_every_completion_earns_at_least_the_consistent_rung():
    for t in TRACES:
        g = _replay(t)
        value, kind = reward(t["messages"][2]["content"], g)
        assert kind in ("consistent", "solved"), (t["prior_guesses"], t["guess"], kind)
        assert value >= 0.3
        assert parse_tag(t["messages"][2]["content"]) == t["guess"]


def _test_the_label_is_not_an_oracle():
    # If the teacher always answered with the secret, the model would learn to "know" a word it can
    # only derive when the clues leave one candidate. Early on, many words fit, so the target must
    # often differ from the secret.
    early = [t for t in TRACES if t["n_prior"] in (1, 2)]
    differing = sum(t["guess"] != t["secret"] for t in early)
    assert differing > 0.5 * len(early), (differing, len(early))


def _test_no_duplicate_examples_beyond_the_opener():
    keys = [(t["messages"][1]["content"], t["messages"][2]["content"]) for t in TRACES if t["n_prior"] > 0]
    assert len(keys) == len(set(keys))


def _test_same_seed_same_dataset_in_any_process():
    code = (
        "import json; from sft.gen_traces import make_traces; "
        "print(json.dumps(make_traces(n_total=60, opener_copies=8, seed=3), sort_keys=True))"
    )
    outs = []
    for hs in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        r = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout)
    assert outs[0] == outs[1], "dataset depends on PYTHONHASHSEED"


def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
