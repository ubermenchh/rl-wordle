# RL on Wordle, 80 steps (2026-10-11)

Run `rl-wordle-1011-131936`. Steps 1 to 30 ran first (`results/rl_wordle_30steps_2026-10-11.md`).
Steps 31 to 80 are a resume of the same run (`rl_run --resume-run`). Start: SFT checkpoint `sft-wordle-1011-113953`, step 62.
Held-out eval: split `eval`, 100 games, every 5 steps (steps 0 to 30) and every 10 steps (steps 30 to 80).

## Held-out eval

| Step | Reward | Wins of 100 | Games ended on an illegal reply | Turns |
|---|---|---|---|---|
| 0 | 0.068 | 4 | 73% | 2.83 |
| 5 | 0.164 | 12 | 61% | 2.91 |
| 10 | 0.147 | 10 | 60% | 2.96 |
| 15 | 0.113 | 6 | 51% | 3.24 |
| 20 | 0.120 | 6 | 44% | 3.30 |
| 25 | 0.113 | 5 | 42% | 3.36 |
| 30 | 0.128 | 5 | 29% | 3.50 |
| 40 | 0.146 | 7 | 30% | 3.55 |
| 50 | 0.159 | 9 | 34% | 3.58 |
| 60 | 0.130 | 7 | 41% | 3.30 |
| 70 | 0.110 | 4 | 34% | 3.65 |
| 80 | 0.195 | 14 | 46% | 3.41 |

## Training (mean over the steps)

| Steps | Train reward | Groups with a mix of solved and not solved | Entropy | Grad norm | Max mismatch KL |
|---|---|---|---|---|---|
| 1-10 | 0.303 | 0.946 | 0.067 | 0.391 | 0.0057 |
| 11-30 | 0.329 | 0.967 | 0.067 | 0.301 | 0.0113 |
| 31-50 | 0.368 | 0.997 | 0.062 | 0.313 | 0.0139 |
| 51-80 | 0.406 | 0.993 | 0.063 | 0.365 | 0.0161 |

## What it means

- Rule 5, sink 2 (learning signal): fine. Nearly every group has a mix of outcomes. No NaN, no grad-norm spike, KL stays small.
- The train reward rises in a steady line (0.30 to 0.41). This is the clearest sign that RL works.
- The held-out eval does not rise in a line. With 100 games, 14 wins has a 95% interval of about 8% to 22%, and 4 wins about 2% to 10%. Steps 70 (4 wins) and 80 (14 wins) are 10 points apart, so the noise is large.
- Illegal-ended games fell from 73% to about 30-45%. This effect is bigger than the noise. The win count is not clearly better than step 0 to 10 (12 and 10 wins).
- Do not call step 80 "the best model" from one 100-game eval. Compare steps 50 and 80 with more games before we choose what to push.
- Not measured: more than 100 games, the random player, the solver.
