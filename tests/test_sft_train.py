"""
Contract for the SFT training step in sft.train_sft: batching and one optimizer step.

    collate(examples, pad_id) -> (input_ids, labels, attention_mask)   all (B, T) long tensors
    train_step(model, optimizer, input_ids, labels, attention_mask, max_tokens, max_grad_norm)
        -> (loss: float, grad_norm: float)

The strongest test here compares train_step against a reference loop written out step by step
(zero grads, forward, loss, backward, clip, step) on a tiny model. Any wiring slip shows up as a
different weight: gradients accumulating across calls, clipping skipped or done before backward,
the loss normalised differently, or the step taken on a stale gradient.

A model that cannot memorise 8 examples has a bug in the mask, the loss or the optimizer wiring, so
the overfit test is the fastest smoke alarm for the whole SFT path (seconds, CPU, tiny model).

Run from the repo root: `uv run python -m tests.test_sft_train`.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from sft.train_sft import MAX_TOKENS, build_example, collate, sft_loss, train_step

ROOT = Path(__file__).resolve().parent.parent
V, PAD = 40, 0


class TinyLM(nn.Module):
    """Embedding -> linear. It sees only the current token, which is enough to memorise the
    deterministic next-token pairs used below. Returns an object with `.logits`, like HF models."""

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(V, 16)
        self.out = nn.Linear(16, V)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.out(self.emb(input_ids)))


def _examples():
    # Example i: prompt token (masked), then a completion of two tokens. Each id is unique to one
    # example, so every next-token target is a function of the current token and can be memorised.
    ex = []
    for i in range(8):
        p, a, b = 1 + i, 10 + i, 20 + i
        ex.append(([p, a, b], [-100, a, b]))
    return ex


def _batch(examples=None):
    return collate(examples or _examples(), PAD)


def _test_collate_pads_to_the_longest_and_masks_the_padding():
    ex = [([5, 6, 7, 8], [-100, -100, 7, 8]), ([5, 9], [-100, 9])]
    ids, labels, mask = collate(ex, PAD)
    assert ids.shape == labels.shape == mask.shape == (2, 4)
    assert ids.dtype == labels.dtype == mask.dtype == torch.long
    assert ids[1].tolist() == [5, 9, PAD, PAD]
    assert labels[1].tolist() == [-100, 9, -100, -100], "padding must never carry a label"
    assert mask[0].tolist() == [1, 1, 1, 1] and mask[1].tolist() == [1, 1, 0, 0]


def _test_collate_keeps_real_content_untouched():
    ex = _examples()
    ids, labels, _ = collate(ex, PAD)
    for row, (want_ids, want_labels) in enumerate(ex):
        assert ids[row].tolist() == want_ids and labels[row].tolist() == want_labels


def _test_train_step_matches_a_reference_loop():
    ids, labels, mask = _batch()
    max_tokens, clip, lr = 8, 0.5, 0.5

    mine, ref = TinyLM(), TinyLM()
    assert all(torch.equal(a, b) for a, b in zip(mine.parameters(), ref.parameters(), strict=True))
    opt_mine = torch.optim.SGD(mine.parameters(), lr=lr)
    opt_ref = torch.optim.SGD(ref.parameters(), lr=lr)

    for _ in range(3):  # several calls in a row: catches gradients accumulating between steps
        loss, gnorm = train_step(mine, opt_mine, ids, labels, mask, max_tokens, clip)

        opt_ref.zero_grad()
        ref_loss = sft_loss(ref(input_ids=ids, attention_mask=mask).logits, labels, max_tokens)
        ref_loss.backward()
        ref_gnorm = torch.nn.utils.clip_grad_norm_(ref.parameters(), clip)
        opt_ref.step()

        assert abs(loss - ref_loss.item()) < 1e-6
        assert abs(gnorm - ref_gnorm.item()) < 1e-6, "report the norm BEFORE clipping"
    for a, b in zip(mine.parameters(), ref.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def _test_clipping_bounds_the_update():
    # With plain SGD at lr=1 the update's norm is the (clipped) gradient norm, so it cannot exceed
    # max_grad_norm. If clipping ran before backward() or not at all, this breaks.
    ids, labels, mask = _batch()
    model = TinyLM()
    before = [p.detach().clone() for p in model.parameters()]
    _, gnorm = train_step(model, torch.optim.SGD(model.parameters(), lr=1.0), ids, labels, mask, 8, 0.01)
    moved = torch.cat([(p - b).flatten() for p, b in zip(model.parameters(), before, strict=True)]).norm()
    assert gnorm > 0.01, "the test needs an unclipped gradient bigger than the cap"
    assert moved.item() <= 0.01 + 1e-6, moved.item()


def _test_a_fully_masked_batch_changes_nothing():
    ids, labels, mask = _batch()
    labels = torch.full_like(labels, -100)
    model = TinyLM()
    before = [p.detach().clone() for p in model.parameters()]
    loss, gnorm = train_step(model, torch.optim.SGD(model.parameters(), lr=1.0), ids, labels, mask, 8, 1.0)
    assert loss == 0.0 and gnorm == 0.0
    assert all(torch.equal(p, b) for p, b in zip(model.parameters(), before, strict=True))


def _test_overfits_eight_examples():
    ids, labels, mask = _batch()
    model = TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-2)
    first, last = 0.0, 0.0
    for step in range(150):
        last, _ = train_step(model, opt, ids, labels, mask, 8, 1.0)
        if step == 0:
            first = last
    assert last < 0.1 * first, (first, last)  # memorised: the whole path is wired correctly


def _test_max_tokens_bounds_every_real_completion():
    # The loss divides by B * MAX_TOKENS. If a completion were longer, the "normalizer" would be
    # smaller than the data and the constant would stop meaning "the longest answer".
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    assert tok is not None
    longest = 0
    with (ROOT / "data" / "sft_train.jsonl").open() as f:
        for line in f:
            m = json.loads(line)["messages"]
            _, labels = build_example(tok, m[1]["content"], m[2]["content"])
            longest = max(longest, sum(lab != -100 for lab in labels))
    assert longest <= MAX_TOKENS, (longest, MAX_TOKENS)
    assert MAX_TOKENS <= 4 * longest, "MAX_TOKENS is far above any real completion; the loss scale is off"


def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
