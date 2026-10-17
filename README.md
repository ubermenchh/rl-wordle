# rl-wordle

Qwen3-1.7B that plays Wordle, trained with SFT and then RL.
The environment is our own [verifiers](https://github.com/PrimeIntellect-ai/verifiers) taskset (`environments/wordle_rl`).
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) trains it on Modal A100 GPUs.

Model: https://huggingface.co/ubermenchh/qwen3-1.7b-wordle-rl

## Result

Held-out secrets, 400 games. A game starts with 0 to 4 prior guesses and has 6 turns in total.

| Player | Wins | Mean reward | Games ended on an illegal reply |
|---|---|---|---|
| Stock Qwen3-1.7B (100 games, train split) | 0% | 0.001 | 98% |
| After SFT (62 steps) | 7.5% | 0.107 | 69% |
| After SFT + RL (80 steps), run 1 (the Hub model) | 13% | 0.1955 | 38.5% |
| After SFT + RL (80 steps), run 2 (another seed) | 9.75% | 0.1596 | 42% |
| Random legal player | 0% | 0.100 | 0% |
| Solver | 99% | 1.120 | 0% |

Details: `results/final_table_2026-10-11.md`. Reward is not the win rate, so compare players on wins.
The Hub model is run 1, the better of two runs. A fair summary: RL gives 10% to 13% wins and about 40% illegal replies, from 7.5% and 69% after SFT.

Watch the model play one game (downloads 3.4 GB the first time):

    uv run python -m scripts.play --seed 3

## Layout

| Path | What |
|---|---|
| `environments/wordle_rl/` | Game engine, prompt, reward ladder, taskset and env |
| `sft/` | Solver traces and the export for the prime-rl SFT trainer |
| `configs/` | prime-rl configs: SFT, RL and evals |
| `modal_prime_rl.py` | Modal functions: `sft_run`, `rl_run`, `eval_run` |
| `scripts/baselines.py` | Random player and solver on the held-out tasks |
| `scripts/play.py` | One game with the model from the Hub |
| `tests/`, `check.sh` | Tests |
| `results/` | Measured results of each run |

## Reproduce

    ./check.sh                                    # tests
    uv run python -m sft.export_prime_rl          # SFT data
    uv run modal run --detach modal_prime_rl.py::sft_run
    uv run modal run --detach modal_prime_rl.py::rl_run --ckpt-run <sft-run> --step 62 --max-steps 80
    uv run modal run --detach modal_prime_rl.py::eval_run --ckpt-run <rl-run> --step 80 --games 400
    uv run python -m scripts.baselines --games 400

The runs need a Modal account with a secret `wandb-secret`. The cost was about $7 for 80 RL steps on 2 x A100-80GB.
