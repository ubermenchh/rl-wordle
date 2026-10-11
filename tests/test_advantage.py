import torch

from rl.advantage import rloo_advantages


def run_tests():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    expected = torch.tensor([[1.0, -1 / 3, -1 / 3, -1 / 3]])
    actual = rloo_advantages(rewards)
    assert actual.shape == rewards.shape
    assert torch.allclose(actual, expected)

    equal_rewards = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    actual = rloo_advantages(equal_rewards)
    assert torch.allclose(actual, torch.zeros_like(equal_rewards))

    mixed_rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    actual = rloo_advantages(mixed_rewards)
    assert torch.allclose(actual.sum(dim=1), torch.zeros(2), atol=1e-6)

    try:
        rloo_advantages(torch.tensor([[1.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("RLOO should reject groups with k < 2")


if __name__ == "__main__":
    run_tests()
