# Final table: held-out Wordle, 400 games (2026-10-11)

All rows use the same task stream: split `eval` (secrets that SFT and RL never use as a secret), seed 0, first 400 tasks.
A task is a game that starts with 0 to 4 prior guesses. The game has 6 turns in total. Sampling: temperature 1.0, 256 tokens at most, thinking off.

| Player | Wins of 400 | Win rate (95% interval) | Mean reward | Games ended on an illegal reply |
|---|---|---|---|---|
| Stock Qwen3-1.7B | 0 of 100 (train split, Mac and A100) | 0% (0% to 4%) | 0.001 to 0.002 | 98% to 99% |
| After SFT (`sft-wordle-1011-113953`, step 62) | 30 | 7.5% (5% to 10.5%) | 0.107 | 69% |
| After SFT + 80 RL steps (`rl-wordle-1011-131936`, step 80) | 52 | 13% (10% to 17%) | 0.1955 | 38.5% |
| Random legal player | 0 | 0% | 0.100 | 0% |
| Solver (random word that fits all feedback) | 396 | 99.0% | 1.120 | 0% |

Commands: `eval_run --ckpt-run <run> --step <n> --games 400` (Modal, A100-80GB), and `uv run python -m scripts.baselines --games 400`.
The stock model row has only 100 games and is on the train split. It is the baseline of `results/baseline_qwen3-1.7b_2026-10-11.md`.

What it means:

- Run 1 gave 7.5% to 13% wins over SFT (about 2.6 standard errors) and illegal-ended games fell from 69% to 38.5%. Run 2 below gave a smaller gain, so read run 1 as the high end.
- Reward is not the win rate. The random legal player gets 0.100 and the SFT model gets 0.107, but the SFT model wins 7.5% of the games and the random player wins 0%. Use wins when you compare players.
- The solver wins 99%, so the game is easy for a program that tracks the feedback. The model is far from that: it wins 13%, and 38.5% of its games end on an illegal reply.
- The best published result is 14.5% full-game wins on a 7B (`PLAN.md §2`). Our number is for a different measure (mid-game states, 6 turns including the prior guesses), so do not compare them as equal.
- Not measured: step 50, step 75, other temperatures, thinking on, and the 400 games of the stock model.
- The intermediate 100-game evals during RL were noisy. Steps 70 and 80 gave 4 and 14 wins. Only the 400-game numbers belong in the table.

## Replication: run 2 (2026-10-17)

Same config and the same SFT start (step 62) as run 1. Only the seed of the training task stream differs (`env.taskset.seed = 1`).
Run `rl-wordle-1017-110828` (new Modal account, 80 steps in one run, evals every 20 steps). Cost about $10.4 (billing summary, with the failed start).
The same 400 held-out tasks as the rows above.

| Player | Wins of 400 | Win rate (95% interval) | Mean reward | Games ended on an illegal reply |
|---|---|---|---|---|
| After SFT | 30 | 7.5% (5.3% to 10.5%) | 0.107 | 69% |
| RL run 1, step 80 | 52 | 13.0% (10.1% to 16.7%) | 0.1955 | 38.5% |
| RL run 2, step 80 | 39 | 9.75% (7.2% to 13.1%) | 0.1596 | 42% |
| Both runs together | 91 of 800 | 11.4% | | |

What it means:

- RL lowers the share of games that end on an illegal reply, and it did so in both runs: 69% to 38.5% and 42%. This result repeats.
- The gain in wins is positive in both runs but not stable: +5.5 points (run 1) and +2.3 points (run 2). Run 1 against SFT is about 2.6 standard errors, run 2 against SFT is about 1.1. Run 1 against run 2 is about 1.5, so the two runs do not differ clearly.
- Both runs together (91 of 800, 11.4%) against SFT (30 of 400): about 2.1 standard errors. SFT was measured once. My read is that RL helps by a few points, and the size is between 2 and 6 points.
- The Hub model is run 1. It is the better of two runs, so 13% is a high pick. A fair summary is "10% to 13% after 80 steps of RL".
- Both runs sat at 4 to 7 wins (100 games) from step 15 or 20 to step 60 and had 13 or 14 wins at step 80. Part of this is noise: in run 2 the first 100 tasks gave 13 wins and the other 300 gave 26 (8.7%).
- Run 2 was less stable early (steps 11 to 30: mismatch KL up to 0.077, grad norm 0.71 on average, against 0.011 and 0.30 in run 1). It did not collapse.
- Not measured: a third seed, a matched SFT-only control, step 40 of run 2.
