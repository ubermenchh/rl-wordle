"""
Contract for sft.train_sft.sft_loss: the masked next-token cross-entropy.

    sft_loss(logits, labels, max_tokens) -> scalar tensor
    logits: (B, T, V) float, the model's scores at each position
    labels: (B, T) int, the token id at each position, or -100 where it must not be trained

Position t's logits predict token t+1, so the loss pairs logits[:, :-1] with labels[:, 1:].

The denominator is a FIXED constant, B * max_tokens (AGENTS.md Rule 5, sink 3; PLAN.md section 9,
Tier 0). Three ways to normalise, in order of how badly they can go wrong:
  - per sequence, `/ mask.sum(-1)`: every sequence is averaged first, so a token in a 20-token
    completion weighs 3x a token in a 60-token one. This is the length bias Dr. GRPO found in
    trl, OpenRLHF and verl. With a signed RL advantage it pushes lengths in a direction that
    gets misread as learning.
  - per batch, `/ mask.sum()`: every token weighs the same inside a batch, but the weight 1/N
    floats with how many tokens the batch happens to contain, so step size varies batch to batch.
  - fixed constant: every token weighs 1/(B * max_tokens), always. Nothing floats.
SFT has no advantage to bias, so the choice barely matters here. This is the same code task 8
reuses, so it is built the safe way from the start.

Run from the repo root: `uv run python -m tests.test_sft_loss`.
"""

import math

import torch
import torch.nn.functional as F

from sft.train_sft import sft_loss

B, T, V, MAX_TOKENS = 2, 5, 4, 10


def _labels(*rows):
    return torch.tensor(rows)


def _test_known_value_uniform_logits():
    # All-zero logits give a uniform distribution, so each unmasked token costs exactly ln(V).
    # 4 unmasked targets (2 per row; the shift only drops each row's first label, masked here),
    # fixed denominator B * MAX_TOKENS  ->  4 ln(4) / 20, computed by hand.
    logits = torch.zeros(B, T, V)
    labels = _labels([-100, -100, 1, 2, -100], [-100, 3, -100, -100, 0])
    expected = 4 * math.log(V) / (B * MAX_TOKENS)
    assert abs(sft_loss(logits, labels, MAX_TOKENS).item() - expected) < 1e-6


def _test_matches_reference_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(B, T, V)
    labels = _labels([-100, 2, 1, 0, 3], [-100, -100, 0, 1, 2])
    ref = F.cross_entropy(
        logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum"
    ) / (B * MAX_TOKENS)
    assert torch.allclose(sft_loss(logits, labels, MAX_TOKENS), ref, atol=1e-6)


def _test_fully_masked_batch_is_zero_not_nan():
    loss = sft_loss(torch.randn(B, T, V), torch.full((B, T), -100), MAX_TOKENS)
    assert loss.item() == 0.0 and not torch.isnan(loss)


def _test_masked_positions_get_no_gradient():
    # No gradient may flow into positions that predict a masked token, nor into the last position
    # (it has no next token). Unmasked targets do get gradient.
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = _labels([-100, -100, 1, 2, -100], [-100, 3, -100, -100, 0])
    sft_loss(logits, labels, MAX_TOKENS).backward()
    grad = logits.grad
    assert grad is not None
    assert grad[:, -1].abs().sum() == 0, "last position predicts nothing"
    assert grad[0, 0].abs().sum() == 0 and grad[0, 3].abs().sum() == 0  # next labels are masked
    assert grad[0, 1].abs().sum() > 0 and grad[0, 2].abs().sum() > 0  # next labels are unmasked


def _test_denominator_is_a_fixed_constant():
    # Same per-token cost, twice the unmasked tokens -> twice the loss. A mean over unmasked
    # tokens would return the same value both times, which is exactly the length bias.
    logits = torch.zeros(1, T, V)
    one = sft_loss(logits, _labels([-100, -100, -100, -100, 1]), MAX_TOKENS)
    two = sft_loss(logits, _labels([-100, -100, -100, 1, 1]), MAX_TOKENS)
    assert abs(two.item() - 2 * one.item()) < 1e-6, (one.item(), two.item())


def _test_bf16_logits_give_a_float32_loss():
    # subtle: log-softmax over a 150k-wide vocab in bf16 loses precision; the loss should be
    # computed in float32 whatever dtype the model emits.
    logits = torch.randn(B, T, V).to(torch.bfloat16)
    loss = sft_loss(logits, _labels([-100, 2, 1, 0, 3], [-100, -100, 0, 1, 2]), MAX_TOKENS)
    assert loss.dtype == torch.float32


def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
