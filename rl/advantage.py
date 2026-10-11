import torch


def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim != 2:
        raise ValueError(f"Expected rewards shape of [batch, k], got {tuple(rewards.shape)}")

    k = rewards.size(1)
    if k < 2:
        raise ValueError(
            "RLOO needs at least 2 samples per prompt to perform leave-one-out "
            f"baseline, but got k={k}"
        )

    group_totals = rewards.sum(dim=1, keepdim=True)
    leave_one_out_baseline = (group_totals - rewards) / (k - 1)
    advantages = rewards - leave_one_out_baseline

    return advantages
