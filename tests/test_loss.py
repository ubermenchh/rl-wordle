import torch

from rl.loss import rloo_loss


def run_tests():
    # Two sampled completions, four-token fixed normalization budget.
    log_probs = torch.tensor([
        [-1.0, -2.0,  0.0, 0.0],
        [-3.0,  0.0,  0.0, 0.0],
    ])
    loss_mask = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    advantages = torch.tensor([2.0, -1.0])

    # Numerator = -(((-1 - 2) * 2) + ((-3) * -1)) = 3
    # Fixed denominator = 2 completions * 4 MAX_TOKENS
    expected = torch.tensor(3.0 / 8.0)

    actual = rloo_loss(
        log_probs=log_probs,
        loss_mask=loss_mask,
        advantages=advantages,
        max_tokens=4,
    )
    assert torch.allclose(actual, expected, atol=1e-6)


if __name__ == "__main__":
    run_tests()
