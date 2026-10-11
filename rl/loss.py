import torch


def rloo_loss(
    log_probs: torch.Tensor,     # (N,T) log pi(token) of each sampled token
    loss_mask: torch.Tensor,     # (N,T) 1.0 on the model's own tokens, 0.0 else
    advantages: torch.Tensor,    # (N,)  one value for each row
    max_tokens: int,
) -> torch.Tensor:
    """Policy-gradient loss with a fixed divisor."""

    # 1. Broadcast the advantage of a row over its T tokens. (N,) -> (N,1)
    weighted = advantages.detach().unsqueeze(1) * log_probs * loss_mask # (N,T)

    # 2. Sum as the mask already removed the tokens that the model did not choose.
    # Minus, because the optimizer minimizes, and we want to raise the log-prob when A > 0.
    total = -weighted.sum()

    return total / (log_probs.size(0) * max_tokens)
