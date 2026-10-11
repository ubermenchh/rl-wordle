import torch

from rl.rollout import (
    Group,
    build_batch,
    check_mask,
    collect_batch,
    loss_mask_for_span,
    score_group,
    split_completions,
)
from wordle_rl.wordle import Game


def _test_split_cuts_at_first_eos():
    """Tokens after the first eos are padding. If they stay in, we train on pad tokens."""
    EOS = 9
    gen = torch.tensor([
        [5, 6, EOS, 0, 0],         # short row, padded
        [5, 6, 7, 8, EOS],         # eos is the last token
        [5, 6, 7, 8, 4],           # Edge: no eos, keep the whole row
        [5, EOS, EOS, EOS, EOS],   # Edge: pad id equals eos id, cut at the FIRST eos
    ])
    assert split_completions(gen, EOS) == [[5, 6, EOS], [5, 6, 7, 8, EOS], [5, 6, 7, 8, 4], [5, EOS]]


def _test_score_group_follows_the_reward_ladder():
    """The sampler gives bare words. The reward needs tags. If they are missing, every word scores -0.1."""
    g = Game("CRANE")
    g.guess("SLATE")   # turns_used = 1
    grp = Group(prompt_ids=[], completions=[[]] * 4, words=["CRANE", "CRAZE", "SLOTH", "SLATE"])
    got = score_group(grp, g)
    assert got.shape == (4,)
    # solved = 1.0 + 0.5 / (1 + 1), consistent, valid but not consistent, repeat
    assert torch.allclose(got, torch.tensor([1.25, 0.3, 0.1, 0.0]))

def _test_build_batch_masks_only_the_completions():
    """Rows are in group order. The mask covers each completion and nothing else.
    If a prompt token is in the mask, we train on tokens the model did not choose."""
    EOS, PAD = 99, 0
    g0 = Group(prompt_ids=[1, 2, 3], completions=[[7, 8, EOS], [9, EOS]], words=["A", "B"])
    g1 = Group(prompt_ids=[4, 5], completions=[[7, EOS], [8, 9, EOS]], words=["C", "D"])
    rewards = torch.tensor([[1.0, 0.0], [0.5, 0.2]])
    b = build_batch([g0, g1], rewards, pad_id=PAD)

    assert b.input_ids.shape == (4, 6)
    assert b.input_ids.tolist() == [
        [1, 2, 3, 7, 8, EOS],
        [1, 2, 3, 9, EOS, PAD],   # Edge: the shorter row is padded on the right
        [4, 5, 7, EOS, PAD, PAD], # Edge: a different prompt length
        [4, 5, 8, 9, EOS, PAD],
    ]
    assert b.loss_mask.tolist() == [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
    ]
    assert b.loss_mask.sum() == 10   # 3 + 2 + 2 + 3 completion tokens
    assert b.attention_mask.tolist()[1] == [1, 1, 1, 1, 1, 0]
    assert torch.equal(b.rewards, rewards)

class _FakeTok:
    """Decode ids as space-separated numbers. Enough to see which ids are in the mask."""
    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)

def _test_check_mask_catches_a_leaking_mask():
    """If a prompt token enters the mask, check_mask must raise. A print alone is easy to miss."""
    EOS = 99
    g0 = Group(prompt_ids=[1, 2, 3], completions=[[7, 8, EOS], [9, EOS]], words=["A", "B"])
    g1 = Group(prompt_ids=[4, 5], completions=[[7, EOS], [8, 9, EOS]], words=["C", "D"])
    b = build_batch([g0, g1], torch.zeros(2, 2), pad_id=0)

    check_mask(b, [g0, g1], _FakeTok())            # the correct batch passes

    b.loss_mask[0, 2] = 1.0                         # leak: the last prompt token enters the mask
    try:
        check_mask(b, [g0, g1], _FakeTok())
    except AssertionError:
        pass
    else:
        raise AssertionError("check_mask missed a leaked prompt token")

    b2 = build_batch([g0, g1], torch.zeros(2, 2), pad_id=0)
    b2.loss_mask[3, 1] = 1.0   # row 3: the last prompt token enters the mask ...
    b2.loss_mask[3, 4] = 0.0   # ... and the eos leaves it. The count stays 10.
    try:
        check_mask(b2, [g0, g1], _FakeTok())
    except AssertionError:
        pass
    else:
        raise AssertionError("check_mask missed a shifted span")

def _test_collect_batch_drops_equal_reward_groups():
    """A group with equal rewards has zero advantage. It adds tokens and no signal."""
    EOS = 99
    flat = Group([1, 2], [[7, EOS], [8, EOS]], ["A", "B"])
    mixed = Group([1, 2], [[7, EOS], [8, EOS]], ["A", "B"])
    draws = iter([
        (flat, torch.tensor([0.1, 0.1])),
        (mixed, torch.tensor([0.3, 0.1])),
        (flat, torch.tensor([0.0, 0.0])),
        (mixed, torch.tensor([1.25, 0.1])),
    ])
    batch, groups, tries = collect_batch(lambda: next(draws), B=2, pad_id=0, max_tries=10)
    assert tries == 4
    assert len(groups) == 2
    assert torch.equal(batch.rewards, torch.tensor([[0.3, 0.1], [1.25, 0.1]]))

def _test_collect_batch_raises_when_no_group_has_signal():
    """A smaller batch would change the loss divisor B * MAX_TOKENS. Raise instead."""
    flat = Group([1], [[7, 99], [8, 99]], ["A", "B"])
    try:
        collect_batch(lambda: (flat, torch.tensor([0.1, 0.1])), B=2, pad_id=0, max_tries=5)
    except RuntimeError:
        pass
    else:
        raise AssertionError("collect_batch returned a batch with no signal")

def run_tests():
    _test_split_cuts_at_first_eos()
    _test_score_group_follows_the_reward_ladder()
    _test_build_batch_masks_only_the_completions()
    _test_check_mask_catches_a_leaking_mask()
    _test_collect_batch_drops_equal_reward_groups()
    _test_collect_batch_raises_when_no_group_has_signal()
    # Prompt tokens, generated guess tokens, then environment-added tokens.
    token_ids = [101, 102, 11, 12, 13, 201, 202]
    mask = loss_mask_for_span(
        total_tokens=len(token_ids),
        action_start=2,
        action_end=5,
    )

    assert len(mask) == len(token_ids)
    assert [token for token, keep in zip(token_ids, mask, strict=True) if keep] == [11, 12, 13]
    assert sum(mask) == 3
    assert not any(mask[:2])  # prompt is excluded
    assert not any(mask[5:])  # environment-added tokens are excluded


if __name__ == "__main__":
    run_tests()
