from dataclasses import dataclass

import torch

from constrained import TrieProcessor, build_trie
from gate import build_prompt
from wordle_rl.render import render
from wordle_rl.reward import reward
from wordle_rl.wordle import Game


@dataclass
class Group:
    prompt_ids: list[int]
    completions: list[list[int]]
    words: list[str]

@dataclass
class Batch:
    input_ids: torch.Tensor # (N,T)
    attention_mask: torch.Tensor # (N,T)
    loss_mask: torch.Tensor # (N,T)
    rewards: torch.Tensor # (B,k)

def loss_mask_for_span(total_tokens: int, action_start: int, action_end: int) -> list[int]:
    mask = [0] * total_tokens
    for i in range(action_start, action_end):
        mask[i] = 1
    return mask

def max_action_tokens(tokenizer, words: list[str]) -> int:
    """Longest word in tokens, +1 for eos token"""
    return max(len(tokenizer.encode(w, add_special_tokens=False)) for w in words) + 1

def split_completions(gen: torch.Tensor, eos_id: int) -> list[list[int]]:
    result = []
    for row in gen:
        if eos_id in row:
            cut = (row == eos_id).nonzero()[0] + 1
        else:
            cut = len(row)
        result.append(row[:cut].tolist())
    return result

def score_group(group: Group, game: Game) -> torch.Tensor:
    return torch.tensor([reward(f"<guess>{w}</guess>", game)[0] for w in group.words], dtype=torch.float32)

def build_batch(groups: list[Group], rewards: torch.Tensor, pad_id: int) -> Batch:
    rows = []
    for g in groups:
        for c in g.completions:
            ids = g.prompt_ids + c
            mask = loss_mask_for_span(len(ids), len(g.prompt_ids), len(ids))
            rows.append((ids, mask))

    T = max(len(ids) for ids, _ in rows)
    N = len(rows)

    input_ids = torch.full((N, T), pad_id, dtype=torch.long)
    loss_mask = torch.zeros(N, T, dtype=torch.float32)
    attention_mask = torch.zeros(N, T, dtype=torch.long)

    for r, (ids, mask) in enumerate(rows):
        l_ = len(ids)
        input_ids[r, :l_] = torch.tensor(ids, dtype=torch.long)
        loss_mask[r, :l_] = torch.tensor(mask, dtype=torch.float32)
        attention_mask[r, :l_] = 1

    return Batch(input_ids, attention_mask, loss_mask, rewards)

def check_mask(batch: Batch, groups: list[Group], tokenizer):
    expected_rows = [c for g in groups for c in g.completions]

    assert batch.input_ids.shape[0] == len(expected_rows)
    assert batch.loss_mask.sum().item() == sum(len(c) for c in expected_rows)

    for r, c in enumerate(expected_rows):
        selected = batch.input_ids[r][batch.loss_mask[r] == 1].tolist()
        assert selected == c, f"row {r}: mask selects {selected}, expected {c}"

    r = 0
    print(tokenizer.decode(batch.input_ids[r][batch.loss_mask[r] == 1]))

def collect_batch(draw_group, B: int, pad_id: int, max_tries: int):
    kept: list[Group] = []
    rewards: list[torch.Tensor] = []
    tries = 0

    while len(kept) < B:
        # A smaller batch would change the loss divisor B * MAX_TOKENSa
        if tries == max_tries:
            raise RuntimeError(f"only {len(kept)} of {B} groups had sigmal after {tries} tries")

        group, r = draw_group()
        tries += 1

        # std() is 0 when all k rewards are equal. Then every advantage is 0, so the group
        # costs generation time and adds no gradient. std() is nan if k == 1, and RLOO needs k >= 2.
        if r.std() > 0:
            kept.append(group)
            rewards.append(r)

    return build_batch(kept, torch.stack(rewards), pad_id), kept, tries

def sample_group(model, tokenizer, game: Game, k: int, temp: float, pool: list[str], max_new: int) -> Group:
    text = build_prompt(tokenizer, render(game)) + "<guess>"
    ids = tokenizer(text, return_tensors="pt").to(model.device)
    trie = build_trie(tokenizer, pool)
    prompt_len = ids["input_ids"].shape[1]
    eos_token = tokenizer.eos_token_id
    out = model.generate(
        **ids,
        num_return_sequences=k,
        logits_processor=[TrieProcessor(trie, prompt_len, eos_token)],
        do_sample=True,
        temperature=temp,
        top_p=1.0,
        top_k=0,
        max_new_tokens=max_new,
        pad_token_id=tokenizer.pad_token_id or eos_token,
    )
    prompt_ids = out[0, :prompt_len].tolist()
    gen = out[:, prompt_len:]
    completions = split_completions(gen, eos_token)
    words = [tokenizer.decode(c, skip_special_tokens=True).strip().upper() for c in completions]
    return Group(prompt_ids, completions, words)
