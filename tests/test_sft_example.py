"""
Contract for sft.train_sft.build_example: turn one trace into (input_ids, labels).

    build_example(tokenizer, user_text, completion) -> (input_ids: list[int], labels: list[int])

This is the loss mask (AGENTS.md Rule 5, sink 1): labels are -100 everywhere the model must NOT be
trained, and equal the token id where it must. The prompt is the environment's text, so training on
it would teach the model to write the prompt. The only tokens that may carry a label are the ones
the model itself produces: the reasoning, the <guess> tag, and the end-of-turn token.

Three properties, each a way SFT silently goes wrong:
  - labels on prompt tokens (the leak): masked-in tokens must decode to the completion and nothing else;
  - a prompt that differs from inference: it must equal gate.build_prompt exactly (empty <think>
    block included), or the model is trained on inputs it will never see;
  - no end-of-turn token: the model would never learn to stop after </guess>.

Needs the Qwen3-0.6B tokenizer in the local HF cache (no network).
Run from the repo root: `uv run python -m tests.test_sft_example`.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

from gate import build_prompt
from sft.train_sft import build_example

ROOT = Path(__file__).resolve().parent.parent


def _load():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    assert tok is not None  # narrows the type for ty; from_pretrained never returns None at runtime
    return tok


TOK = _load()
IM_END = "<|im_end|>"

USER = "No guesses yet.\n\nGuesses remaining: 6\nMake your next guess."
COMPLETION = "Nothing is known yet, so open with CRANE.\n<guess>CRANE</guess>"


def _example(user=USER, completion=COMPLETION):
    ids, labels = build_example(TOK, user, completion)
    return ids, labels


def _test_ids_and_labels_align():
    ids, labels = _example()
    assert len(ids) == len(labels)
    assert all(isinstance(i, int) for i in ids + labels)


def _test_only_the_completion_and_end_token_are_unmasked():
    # Decode exactly the masked-in tokens (the Rule 5 check): they must be the model's own output.
    ids, labels = _example()
    kept = [t for t in labels if t != -100]
    assert TOK.decode(kept, skip_special_tokens=False) == COMPLETION + IM_END


def _test_masked_in_labels_equal_the_input_ids():
    # A label is just "the token that is here": any other id would train toward the wrong token.
    ids, labels = _example()
    assert all(lab == tok for tok, lab in zip(ids, labels, strict=True) if lab != -100)


def _test_prompt_matches_inference_exactly():
    ids, labels = _example()
    n_prompt = sum(1 for lab in labels if lab == -100)
    assert TOK.decode(ids[:n_prompt], skip_special_tokens=False) == build_prompt(TOK, USER)
    assert all(lab == -100 for lab in labels[:n_prompt]), "mask must be one contiguous prompt prefix"


def _test_ends_with_end_of_turn():
    ids, labels = _example()
    assert ids[-1] == TOK.convert_tokens_to_ids(IM_END) and labels[-1] == ids[-1]


def _test_on_real_traces():
    # The first 25 generated traces: every mask decodes to exactly that trace's completion.
    path = ROOT / "data" / "sft_train.jsonl"
    assert path.exists(), "run `uv run python -m sft.gen_traces` first"
    with path.open() as f:
        for _, line in zip(range(25), f, strict=False):
            m = json.loads(line)["messages"]
            ids, labels = build_example(TOK, m[1]["content"], m[2]["content"])
            kept = [t for t in labels if t != -100]
            assert TOK.decode(kept, skip_special_tokens=False) == m[2]["content"] + IM_END


def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
