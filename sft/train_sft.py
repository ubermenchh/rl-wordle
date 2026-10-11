"""
SFT on the solver traces: turn a trace into maksed token ids, and compute the masked loss.
"""

import random

import torch
import torch.nn.functional as F

from gate import build_prompt

IM_END = "<|im_end|>" # Qwen3's end-of-turn token
MAX_TOKENS = 128

def build_example(tokenizer, user_text: str, completion: str) -> tuple[list[int], list[int]]:
    """
    Token ids for one trace, and labels that are -100 everywhere except the model's own output.

    Example (ids made up): prompt [10, 11, 12], completion + end [20, 21, 22]
        input_ids = [10, 11, 12, 20, 21, 22]
        labels    = [-100, -100, -100, 20, 21, 22]
    """
    prompt_ids = tokenizer(build_prompt(tokenizer, user_text), add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion + IM_END, add_special_tokens=False).input_ids

    return prompt_ids + completion_ids, [-100] * len(prompt_ids) + completion_ids

def sft_loss(logits: torch.Tensor, labels: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """
    Summed token cross-entropy over unmasked targets, divided by the constant B * max_tokens.
    """
    B = logits.size(0) # batch_size

    shift_logits = logits[:, :-1].float() # (B, T-1, V)
    shift_labels = labels[:, 1:]          # (B, T-1)

    total = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), # (B*(T-1), V)
        shift_labels.reshape(-1), # (B*(T-1),)
        ignore_index=-100,
        reduction="sum",
    )

    return total / (B * max_tokens)

def collate(examples: list[tuple[list[int], list[int]]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stack (input_ids, labels) pairs into rectangular tensors, padding on the right.
    """
    B = len(examples) # batch size
    T = max(len(ids) for ids, _ in examples)

    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)

    for row, (ids, lab) in enumerate(examples):
        input_ids[row, :len(ids)] = torch.tensor(ids)
        labels[row, :len(lab)] = torch.tensor(lab)
        attention_mask[row, :len(ids)] = 1

    return input_ids, labels, attention_mask

def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    max_tokens: int,
    max_grad_norm: float,
    *,
    autocast: bool = False,
) -> tuple[float, float]:
    optimizer.zero_grad()

    # forward in bf16, weights stay in fp32
    with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16, enabled=autocast):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits # (B, T, V)
        loss = sft_loss(logits, labels, max_tokens)

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    return loss.item(), grad_norm.item()

def train(
    model: torch.nn.Module,
    examples: list[tuple[list[int], list[int]]],
    pad_id: int,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    warmup_steps: int,
    max_grad_norm: float,
    max_tokens: int = MAX_TOKENS,
    seed: int = 0,
    autocast: bool = False,
    log=None,
) -> list[dict]:
    """
    Train for `epochs` passes over `examples`. Returns one record per optimizer step.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    records: list[dict] = []
    step = 0

    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)

        for start in range(0, len(order), batch_size):
            batch = [examples[i] for i in order[start:start+batch_size]]
            input_ids, labels, attention_mask = (t.to(device) for t in collate(batch, pad_id))

            step_lr = lr * min(1.0, (step + 1) / warmup_steps) if warmup_steps else lr
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            loss, grad_norm = train_step(
                model, optimizer, input_ids, labels, attention_mask,
                max_tokens, max_grad_norm, autocast=autocast,
            )
            record = {
                "step": step, "epoch": epoch, "loss": loss,
                "grad_norm": grad_norm, "lr": step_lr, "n_examples": len(batch),
            }
            records.append(record)
            if log is not None:
                log(record)
            step += 1

    return records
