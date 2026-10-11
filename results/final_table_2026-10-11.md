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

- RL gave a real gain over SFT: 7.5% to 13% wins. The difference is about 2.6 standard errors. Illegal-ended games fell from 69% to 38.5%.
- Reward is not the win rate. The random legal player gets 0.100 and the SFT model gets 0.107, but the SFT model wins 7.5% of the games and the random player wins 0%. Use wins when you compare players.
- The solver wins 99%, so the game is easy for a program that tracks the feedback. The model is far from that: it wins 13%, and 38.5% of its games end on an illegal reply.
- The best published result is 14.5% full-game wins on a 7B (`PLAN.md §2`). Our number is for a different measure (mid-game states, 6 turns including the prior guesses), so do not compare them as equal.
- Not measured: step 50, step 75, other temperatures, thinking on, and the 400 games of the stock model.
- The intermediate 100-game evals during RL were noisy. Steps 70 and 80 gave 4 and 14 wins. Only the 400-game numbers belong in the table.
