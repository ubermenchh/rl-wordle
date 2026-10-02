"""
Trie-constrained decoding: sample only from a fixed word pool.

Removes the failure modes that domintate the funnel -- invalid words, wrong
length, repeats, -- by construction, so `consist` becomes a measurement of
deduction rather than of spelling.
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor, LogitsProcessorList

TERMINAL = "__end__"

def build_trie(tokenizer, words: list[str]) -> dict:
    """Nested dict keyed by token id. TERMINAL marks a complete word."""
    trie: dict = {}
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        node = trie
        for i in ids:
            node = node.setdefault(i, {})
        node[TERMINAL] = True
    return trie

def _allowed(trie: dict, generated: list[int], eos_id: int) -> list[int]:
    node = trie
    for tok in generated:
        nxt = node.get(tok)
        if nxt is None:
            return [eos_id]
        node = nxt
    out = [k for k in node if k != TERMINAL]
    if node.get(TERMINAL):
        out.append(eos_id)
    return out or [eos_id]

class TrieProcessor(LogitsProcessor):
    """Stateless: derives each sequence's trie position from input_ids"""

    def __init__(self, trie: dict, prompt_len: int, eos_id: int):
        self.trie, self.prompt_len, self.eos_id = trie, prompt_len, eos_id

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.prompt_len:].tolist()
            mask[b, _allowed(self.trie, gen, self.eos_id)] = 0.0
        return scores + mask

@torch.no_grad()
def sample_words(tokenizer, model, prompt: str, pool: list[str], k: int, temp: float) -> list[str]:
    """Sample k words from `pool`. Every result is a legal pool member."""
    trie = build_trie(tokenizer, pool)
    max_len = max(len(tokenizer.encode(w, add_special_tokens=False)) for w in pool)

    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = ids["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id

    out = model.generate(
        **ids,
        do_sample=True,
        temperature=temp,
        top_p=1.0,
        top_k=0,
        num_return_sequences=k,
        max_new_tokens=max_len + 1,
        pad_token_id=tokenizer.pad_token_id or eos_id,
        logits_processor=LogitsProcessorList([TrieProcessor(trie, prompt_len, eos_id)]),
    )
    gen = out[:, prompt_len:]
    return [tokenizer.decode(g, skip_special_tokens=True).strip().upper() for g in gen]


def _test_trie_covers_pool():
    """Every pool word must be reachable and terminal -- catches tokenizer drift."""
    from transformers import AutoTokenizer

    from wordle import ANSWERS

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    assert tok is not None
    pool = ANSWERS[:200]
    trie = build_trie(tok, pool)
    for w in pool:
        node = trie
        for i in tok.encode(w, add_special_tokens=False):
            assert i in node, w
            node = node[i]
        assert node.get(TERMINAL), w
        assert tok.eos_token_id in _allowed(trie, tok.encode(w, add_special_tokens=False), tok.eos_token_id)


if __name__ == "__main__":
    _test_trie_covers_pool()
    print("ok  _test_trie_covers_pool")
