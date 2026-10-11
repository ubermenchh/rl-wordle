# SFT of Qwen3-1.7B on the solver traces (2026-10-11)

Mark II, step 2. `modal_prime_rl.py::sft_run`, one A100-80GB, run `sft-wordle-1011-113953`
(W&B project `rl-wordle`, run `46im53g1`). Config `configs/wordle_sft.toml`: stock `Qwen/Qwen3-1.7B`,
data `data/sft_wordle/train.jsonl` (2,000 rows, one `messages` column, made by `sft/export_prime_rl.py`),
`max_steps = 62`, batch 64, `seq_len` 1024, LR 1e-5, renderer `qwen3` with `enable_thinking = false`.

| Measure | Value |
|---|---|
| Loss, step 1 / 10 / 30 / 62 | 5.58 / 0.223 / 0.089 / 0.0445 |
| Grad norm, step 1 / 10 / 30 / 62 | 232.6 / 11.8 / 0.66 / 0.68 |
| NaN count | 0 |
| Step time | 14 s (37 s for step 1) |
| Throughput | about 4,100 tokens/s, MFU about 14.5% |
| Peak memory | 29.2 of 79.3 GiB |
| Run time | about 16 minutes (11:40:51 to 11:56:43) |
| Epochs reported by prime-rl | 7 |

Two readings, both from the log and not tested:

- **7 epochs, not 2.** I had planned 62 steps as about 2 epochs (2,000 rows / batch 64 = 31 steps). The trainer packs rows into
  sequences of `seq_len` tokens, and a batch is 64 such sequences. A row is about 298 tokens, so a step holds about 220 rows
  (4,100 tokens/s x 14 s = 57k tokens per step). 62 steps x 64 x 1,024 / (2,000 x 298) = 6.8.
- **The loss may have started to memorise at the end.** The teacher picks a random word among the consistent ones, so the word
  tokens cannot be predicted from the prompt. A loss of 0.0445 is low for that. I did not compute the loss floor. The eval on new
  states decides.

The checkpoint is `sft-wordle-1011-113953/checkpoints/step_62/trainer` (DCP format). It needs
`tools/convert_dcp_to_bf16.py` to make HF weights, which `modal_prime_rl.py::eval_run` does.

## Eval of the checkpoint, train split (NOT held-out)

`modal_prime_rl.py::eval_run`, run `eval-sft-wordle-1011-113953-1011-121117`, 100 games, stock-model settings (temperature 1.0, 256 tokens).
**This eval used `split = "train"` by mistake** (the default of `configs/wordle_eval.toml`). SFT used secrets of the same split, for about 7 epochs,
so the numbers are in-distribution and can be too good. The held-out eval is `configs/wordle_eval_heldout.toml` (the default of `eval_run` since then).

| Measure | Baseline (stock) | After SFT, train split |
|---|---|---|
| Wins | 0 | 28 |
| Lost (6 legal guesses) | 0 to 2 | 19 |
| Ended on an illegal reply | 98 to 99 | 53: 23 bad_length, 24 not_a_word, 4 repeat, 1 format, 1 not_alpha |
| Mean reward | 0.001 to 0.002 | 0.331 |
| Turns per game (assistant replies) | 1.0 | 2.7 (1: 24 games, 2: 28, 3: 24, 4: 8, 5: 12, 6: 4) |
| Legal share of all replies | about 2% | 80% (215 of 268) |
| Mean output tokens per game | 10.6 | 126.2 |

Reading: SFT moves the model from copying prior guesses to playing real games. The Mark II bar of 95% legal words is not met (80%).
The remaining failures are invented words (`ACENT`, `HONEN`) and wrong lengths: the reasoning line says "X fits all of this" for a word that
the model made up, because it has no dictionary. Rewards now differ inside a game state (0.0, 0.1, about 1.1), so RL has signal.

## Eval of the checkpoint, HELD-OUT split (the honest number)

Run `eval-eval-sft-wordle-1011-113953-1011-124348`, `configs/wordle_eval_heldout.toml` (`split = "eval"`), 100 games, 84 distinct secrets, all in the eval split.

| Measure | Train split | Held-out split |
|---|---|---|
| Wins | 28 | 5 |
| Lost | 19 | 27 |
| Ended on an illegal reply | 53 | 68 (32 not_a_word, 24 bad_length, 11 repeat, 1 not_alpha) |
| Mean reward | 0.331 | 0.0848 |
| Legal share of replies | 80% (215 of 268) | 77% (222 of 290) |
| Wins by priors (0 to 4), held-out | n/a | 1 of 21, 0 of 15, 0 of 23, 3 of 22, 1 of 19 |

Reading:
- The legal-word skill carries over (80% to 77%). The wins do not (28 to 5): most of the train-split wins were memorised answer words, after about 7 epochs.
- The vocabulary is part of it. The teacher's candidates come from the train split (`sft/gen_traces.py`: "secrets and candidate words come from the train split only"), so every SFT target is a train-split word.
- 5 wins in 100 games is about 2% to 11% (95% Wilson interval). Not measured: fewer epochs, which could memorise less.
- The Mark II bars: 95% legal words is not met (77%). Win rate above the baseline of 0 is met, by a small margin.

## Fewer epochs (same day): held-out eval of the checkpoints of run `sft-wordle-1011-142546` (30 steps, checkpoints each 10)

Same config as the 62-step run, `--max-steps 30 --ckpt-interval 10`. Held-out split, 100 games each (the three evals ran at the same time).

| Checkpoint | Epochs | Reward | Wins | Legal share of replies | Outcomes |
|---|---|---|---|---|---|
| step 10 | about 1 | 0.0030 | 0 | 46% | 38 not_a_word, 32 bad_length, 18 repeat, 3 format, 3 not_alpha, 6 lost |
| step 20 | about 2 | 0.0797 | 6 | 61% | 47 bad_length, 28 not_a_word, 7 repeat, 12 lost, 6 won |
| step 30 | about 3 | 0.0811 | 6 | 67% | 38 bad_length, 36 not_a_word, 5 repeat, 15 lost, 6 won |
| step 62 (earlier run) | about 7 | 0.068 to 0.085 | 4 to 5 | 77% | 32 not_a_word, 24 bad_length, 11 repeat, 27 lost, 5 won |

Reading: fewer epochs do not help on held-out games. The wins are flat (5 to 6) from step 20, and the legal share rises with the training (46, 61, 67, 77%).
The guess that 7 epochs memorise the answers and hurt held-out play was wrong for wins. Step 62 stays the start for RL.
The loss at step 30 is 0.086 (0.0445 at step 62). Not measured: more than 7 epochs, more data, a different LR.
