# RL on the Wordle taskset, first run: 30 steps (2026-10-11)

Mark III, first run. `modal_prime_rl.py::rl_run`, run `rl-wordle-1011-131936`, 2 x A100-80GB (1 vLLM, 1 trainer), about 41 minutes
(13:19:36 to 14:01), about $4 (my estimate: 48 minutes of 2 GPUs at the list price, including the start and the hold).
Start: the SFT checkpoint `sft-wordle-1011-113953` step 62 (converted weights). Config `configs/wordle_rl.toml`: 30 steps, 128 rollouts per step
(16 states x 8 samples), 256 tokens per turn, LR 3e-6, thinking off, trained on the train split, held-out eval of 100 games at steps 0, 5, ..., 30.

Held-out eval (split `eval`, 100 games each):

| Step | Reward | Wins | Illegal-ended | Format errors | Turns |
|---|---|---|---|---|---|
| 0 | 0.068 | 4 | 73% | 0% | 2.83 |
| 5 | 0.164 | 12 | 61% | 0% | 2.91 |
| 10 | 0.147 | 10 | 60% | 0% | 2.96 |
| 15 | 0.113 | 6 | 51% | 0% | 3.24 |
| 20 | 0.120 | 6 | 44% | 0% | 3.30 |
| 25 | 0.113 | 5 | 42% | 0% | 3.36 |
| 30 | 0.128 | 5 | 29% | 0% | 3.50 |

Training (per step): reward mean 0.19 to 0.46, no trend (0.33 at step 1, 0.33 at step 30); the share of groups with mixed outcomes (`solved_some`)
0.80 to 1.00, mean about 0.95, `solved_all` 0 at every step; entropy 0.055 to 0.076; grad norm 0.21 to 0.77; mismatch KL at most 0.011.

Reading:
- RL cuts the games that end on an illegal reply from 73% to 29%, a steady fall. It did not raise the wins: 4 to 12 to 5 of 100 is inside the noise
  (the same weights gave 4 wins here at step 0 and 5 wins in the separate eval). The reward rises because more games reach the 0.1 rung.
- The training signal is healthy (Rule 5, sink 2) and the run is stable (no entropy collapse, no grad spike).
- The training reward started at 0.33 because SFT memorised the train secrets (about 7 epochs). It is not a measure of progress; the held-out numbers are.
- Not tested: 100 steps (the held-out illegal rate was still falling at step 30), other learning rates, fewer SFT epochs.
