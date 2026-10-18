# rl-wordle

Train Qwen3-1.7B to play Wordle with reinforcement learning.

The Wordle environment is written here, as a [verifiers](https://github.com/PrimeIntellect-ai/verifiers) taskset.
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) trains the model on A100 GPUs on [Modal](https://modal.com).
This is a side project. The goal was a working pipeline and an honest measurement, not a strong Wordle player.

- Model: https://huggingface.co/ubermenchh/qwen3-1.7b-wordle-rl
- Steps: the stock model, then supervised fine-tuning (SFT) on solver traces, then RL (GRPO-style, from prime-rl).
- Result: wins on held-out words go from 0% (stock) to 7.5% (SFT) to 10% to 13% (SFT + RL). A classical solver wins 99%.

## Result

400 games with secrets that SFT and RL do not use as a secret. A game starts with 0 to 4 prior guesses and has 6 turns in total.
Sampling: temperature 1.0, 256 tokens at most, thinking off.

| Player | Wins | Mean reward | Games ended on an illegal reply |
|---|---|---|---|
| Stock Qwen3-1.7B (100 games, train split) | 0% | 0.001 | 98% |
| After SFT (62 steps) | 7.5% | 0.107 | 69% |
| After SFT + RL (80 steps), run 1 (the Hub model) | 13% | 0.1955 | 38.5% |
| After SFT + RL (80 steps), run 2 (another seed) | 9.75% | 0.1596 | 42% |
| Random legal player | 0% | 0.100 | 0% |
| Solver (random word that fits all feedback) | 99% | 1.120 | 0% |

An illegal reply (not a word, a repeat, wrong length, no tag) ends the game. Reward is not the win rate: a random player scores 0.100 and wins 0%. Compare players on wins.

The Hub model is run 1, the better of two runs, so 13% is a high pick. A fair summary: RL gives 10% to 13% wins, from 7.5% after SFT.
The best published Wordle result that I know is 14.5% on a 7B model. It uses a different measure, so the two numbers are not equal.

### What the numbers hide

I read the traces of the 400-game evals (`results/failures_2026-10-17.md`, `results/reasoning_check_2026-10-18.md`).

- **RL moved games from "illegal" to "lost".** Illegal-ended games decreased from 69% to about 40%. The share of games with legal replies and 6 turns used ("lost") increased from 24% to 48%. Wins increased by 2 to 5 points.
- **The play is not better.** In the games that stayed legal, the win rate did not increase (24%, 21% and 17% for SFT, run 1 and run 2).
- **The model mostly ignores the feedback.** Only about 1 legal reply in 5 fits all the feedback so far. This is the same before and after RL.
- **Most illegal replies are words with one wrong letter.** About 87% of the "not a word" replies are one edit from a word in the list (for example `REDER`, `PLACA`).
- **The model writes the known facts, and the word then breaks them.** The word breaks the facts in about 3 of 4 replies.

### What did not work

| Try | Result |
|---|---|
| Fewer SFT epochs (steps 10, 20, 30 instead of 62) | Worse or equal on held-out words (`results/sft_wordle_2026-10-11.md`) |
| A reward bonus for earlier replies that fit the feedback (`shaping = 0.2`, 40 steps) | No change in consistency (17.7% against 17.9%). The model made its "Absent" list shorter (`results/shaping_run3_2026-10-18.md`) |

## How it works

**The task.** Each task is a game with 0 to 4 prior guesses. The model makes the next guess.
The model sees a system prompt and the board. The format is the same at each turn, with spaces between the letters.

The symbol `=` is a correct letter in the correct position. The symbol `-` is a letter in the word, in the wrong position. The symbol `.` is a letter that is not in the word.

```
  S L A T E  |  . . = . =

  slot 1: ?
  slot 2: ?
  slot 3: A
  slot 4: ?
  slot 5: E
Absent:       L S T
Guessed:      SLATE

Guesses remaining: 5
Make your next guess.
```

It replies with a short line of facts and a tag: `<guess>WORD</guess>`. The secrets come from 900 answer words. 20% of them are held out for the eval. The legal guesses are 8,511 words.

**The reward.** One rung for the last reply of the game, scored in the board before it. The rungs are in a sequence. They are not added:

| Last reply | Reward |
|---|---|
| No `<guess>` tag | -0.1 |
| Not a legal word (wrong length, not a word, a repeat) | 0.0 |
| Legal word, but it breaks the feedback | 0.1 |
| Legal word that fits all feedback | 0.3 |
| The secret | 1.0 plus 0.5 / (turns + 1) |

An illegal reply ends the episode. The code is in `environments/wordle_rl/wordle_rl/taskset.py` and `reward.py`.

**The pipeline.**

1. **Solver traces.** A solver plays 2,000 games that have some prior guesses. Each answer is a line of facts from a template ("Fixed: slot 2 is O. Absent: A, C. DONOR fits all of this.") and a tag (`sft/gen_traces.py`, `sft/teacher.py`).
2. **SFT.** 62 steps on 1 A100 (about 16 minutes), learning rate 1e-5. prime-rl puts more than one row in each sequence, so this is about 7 epochs.
3. **RL.** prime-rl on 2 A100 GPUs (one for vLLM, one for the trainer). Each step has 128 rollouts (8 samples for each board), with 256 tokens at most for each turn. The learning rate is 3e-6, and the run has 80 steps. The values are in `configs/wordle_rl.toml`.
4. **Eval.** 400 held-out games, with the same prompts and sampling (`configs/wordle_eval_heldout_400.toml`).

Cost on Modal (A100-80GB, about $0.093 a minute for the 2-GPU job): 80 RL steps cost about $10 to $12. An eval of 400 games costs about $0.6.

## Quickstart

Look at the model as it plays one game. The first run downloads the 3.4 GB model. It runs on a Mac (Apple GPU), CUDA or CPU:

```
uv run python -m scripts.play --seed 3
uv run python -m scripts.play --secret CRANE
```

Example (seed 3, one run). The model samples at temperature 1.0, so the games are different each time:

```
model: Nothing is known yet, so open with CRANE fits all of this.
<guess>CRANE</guess>
   CRANE  .-.-.
model: Must contain: N, R. Absent: A, C, E. LOGIN fits all of this.
<guess>LOGIN</guess>
   LOGIN  .=..-
model: Fixed: slot 2 is O. Absent: A, C, E, G, I, L. BOUND fits all of this.
<guess>BOUND</guess>
   BOUND  .=.--
model: Fixed: slot 2 is O. Absent: A, B, C, E, G, I, L, U. DONOR fits all of this.
<guess>DONOR</guess>
   DONOR  =====
```

Run the tests (the game engine, the reward, the taskset and the env). Then run the random player and the solver on the held-out tasks:

```
./check.sh
uv run python -m scripts.baselines --games 400
```

The code uses `uv`. The environment and the tests run on Python 3.13. prime-rl uses Python 3.12 and an NVIDIA GPU. It runs in the Modal image.

## Reproduce the training

You must have a Modal account with GPU access, and a Weights and Biases API token in a Modal secret named `wandb-secret` (variable `WANDB_API_KEY`).

```
uv run python -m sft.export_prime_rl                       # SFT data from data/sft_train.jsonl
uv run modal run --detach modal_prime_rl.py::sft_run
uv run modal run --detach modal_prime_rl.py::eval_run --ckpt-run <sft-run> --step 62 --games 400
uv run modal run --detach modal_prime_rl.py::rl_run --ckpt-run <sft-run> --step 62 --max-steps 80 --eval-interval 20 --ckpt-interval 40
uv run modal run --detach modal_prime_rl.py::eval_run --ckpt-run <rl-run> --step 80 --games 400
```

- `rl_run` has a timeout cap in its decorator (`timeout=`). Set it above the run time and below your credit. 80 steps run for about 110 minutes, so use 8400 seconds.
- Use two plain hyphens (`--detach`). If a chat or an editor changes them to a longer dash, Modal reads the flag as a module path.
- The versions are fixed: prime-rl `73140dec` (2026-10-10) and verifiers `2cb0a6d3`, the revision that prime-rl uses. Tutorials from before verifiers v1 have a different API.
- To read the traces of a run, use `scripts/analyze_failures.py` and `scripts/check_reasoning.py` (the use is in their docstrings).

## Layout

| Path | What |
|---|---|
| `environments/wordle_rl/` | Game engine, prompt, reward ladder, taskset and env (the verifiers package) |
| `sft/` | Solver teacher, trace generator, and the export for the prime-rl SFT trainer |
| `configs/` | prime-rl configs: SFT, RL and the evals |
| `modal_prime_rl.py` | Modal image and functions: `sft_run`, `rl_run`, `eval_run` |
| `scripts/` | `play.py`, `baselines.py`, `analyze_failures.py`, `check_reasoning.py` |
| `tests/`, `check.sh` | Tests and the one-command check |
| `results/` | Measured results of each run, with the numbers in this README |
| `rl/`, `gate.py`, `constrained.py`, `evaluate.py` | An earlier RLOO attempt, written from scratch, and its pass@k gate. Paused. The project moved to prime-rl on 2026-10-11 |
| `PLAN.md`, `TASKS.md` | The first plan, from before the move |

## Limits

- There are two RL runs and one SFT run. The increase in wins is between 2 and 6 points, and it changes between seeds.
- The numbers are for games with 0 to 4 prior guesses and 6 turns in total. They are not for full games from the first guess.
- The eval has 400 games, so a win rate has an interval of about +-3 points.
- Not tested: thinking on, other temperatures, larger models, and other rewards for the use of the feedback.
- There is no license file.

## Thanks

The model that we start from is [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B). The training tools are [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) and [verifiers](https://github.com/PrimeIntellect-ai/verifiers). The GPUs are from [Modal](https://modal.com).
