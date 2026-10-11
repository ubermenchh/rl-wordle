"""
Contract for sft.train_sft.train: the SFT loop around collate + train_step, and the autocast option.

    train(model, examples, pad_id, *, batch_size, epochs, lr, warmup_steps, max_grad_norm,
          max_tokens=MAX_TOKENS, seed=0, log=None) -> list[dict]

One record per optimizer step: {"step", "epoch", "loss", "grad_norm", "lr", "n_examples"}.
`log`, if given, is called with each record as it is produced (this is where W&B plugs in).

Why each property matters:
  - every example is used exactly once per epoch (a loop that drops the last short batch, or
    repeats an example, quietly trains on a different dataset than the one you generated);
  - lr follows lr * min(1, (step + 1) / warmup_steps): the first AdamW steps on a freshly loaded
    model are the most violent, so they are taken small;
  - the whole run is reproducible from `seed` (shuffling included);
  - autocast=True runs the forward in bf16 while the weights stay fp32: bf16 weights would round
    away lr-sized updates (measured 2026-10-03: a 1e-5 update changes 0 of 1000 bf16 weights).

Run from the repo root: `uv run python -m tests.test_sft_loop`.
"""

import math

import torch

from sft.train_sft import train, train_step
from tests.test_sft_train import PAD, TinyLM, _batch, _examples


def _train(*, batch_size=3, epochs=2, lr=1e-2, warmup_steps=4, seed=0, log=None, model=None):
    """train() with the small tiny-model defaults; explicit keywords keep the type checker happy."""
    return train(
        model or TinyLM(), _examples(), PAD,
        batch_size=batch_size, epochs=epochs, lr=lr, warmup_steps=warmup_steps,
        max_grad_norm=1.0, max_tokens=8, seed=seed, log=log,
    )



def _test_step_count_and_every_example_once_per_epoch():
    recs = _train()
    assert len(recs) == 2 * math.ceil(8 / 3)  # 6 steps; the short last batch is kept, not dropped
    for epoch in (0, 1):
        in_epoch = [r for r in recs if r["epoch"] == epoch]
        assert sum(r["n_examples"] for r in in_epoch) == 8, in_epoch
        assert [r["n_examples"] for r in in_epoch] == [3, 3, 2]
    assert [r["step"] for r in recs] == list(range(6))


def _test_lr_warms_up_then_holds():
    recs = _train()
    for r in recs:
        want = 1e-2 * min(1.0, (r["step"] + 1) / 4)
        assert abs(r["lr"] - want) < 1e-12, (r["step"], r["lr"], want)


def _test_no_warmup_means_constant_lr():
    recs = _train(warmup_steps=0)
    assert all(abs(r["lr"] - 1e-2) < 1e-12 for r in recs)


def _test_loss_goes_down():
    recs = _train(batch_size=4, epochs=40, lr=5e-2)
    first = [r["loss"] for r in recs if r["epoch"] == 0]
    last = [r["loss"] for r in recs if r["epoch"] == 39]
    assert sum(last) / len(last) < 0.3 * sum(first) / len(first), (first, last)


def _test_same_seed_same_run_different_seed_different_order():
    a = _train(seed=1)
    b = _train(seed=1)
    c = _train(seed=2)
    assert a == b
    assert [r["loss"] for r in a] != [r["loss"] for r in c], "the shuffle must depend on the seed"


def _test_inputs_are_not_mutated():
    ex = _examples()
    snapshot = [(list(i), list(lab)) for i, lab in ex]
    train(TinyLM(), ex, PAD, batch_size=3, epochs=2, lr=1e-2, warmup_steps=4, max_grad_norm=1.0, max_tokens=8)
    assert ex == snapshot


def _test_log_callback_sees_every_record():
    seen = []
    recs = _train(log=seen.append)
    assert seen == recs


def _test_autocast_option_runs_and_gives_finite_numbers():
    ids, labels, mask = _batch()
    model = TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss, gnorm = train_step(model, opt, ids, labels, mask, 8, 1.0, autocast=True)
    assert math.isfinite(loss) and math.isfinite(gnorm)
    assert all(p.dtype == torch.float32 for p in model.parameters()), "weights must stay fp32"


def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
