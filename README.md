# rl-wordle

Teaching Qwen3-1.7B to play Wordle: supervised fine-tuning on solver traces first, then reinforcement learning.

The Wordle environment is written from scratch as a [verifiers](https://github.com/PrimeIntellect-ai/verifiers) taskset, training uses [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), and everything ran on A100s on [Modal](https://modal.com).

It works, modestly. The model goes from 0% to about 10-13% wins on held-out words, while a simple solver wins 99%. Most of what RL taught it was to stop making illegal guesses, not to play better. Details below.

Model: [ubermenchh/qwen3-1.7b-wordle-rl](https://huggingface.co/ubermenchh/qwen3-1.7b-wordle-rl)

## Results

400 games on words that were never a secret during training. Each game starts with 0 to 4 guesses already on the board, and the model gets 6 guesses in total. Sampling at temperature 1.0, thinking off.

| Player | Wins | Games ended by an illegal guess |
|---|---|---|
| Qwen3-1.7B, no training (100 games, train split) | 0% | 98% |
| + SFT | 7.5% | 69% |
| + SFT + RL, run 1 (the model on the Hub) | 13% | 38.5% |
| + SFT + RL, run 2 (different seed) | 9.75% | 42% |
| Random legal word | 0% | 0% |
| Solver (any word that fits the feedback so far) | 99% | 0% |

An illegal guess ends the game on the spot: not a word, wrong length, a repeat, or no `<guess>` tag. Run 1 is the better of the two runs, so 13% is on the optimistic side and "10-13%" is the fair number. The best published Wordle result I know of is 14.5% on a 7B model, but it is for full games from the first guess, so it is not comparable.

### What I found

I went through the traces of the 400-game evals (see `results/`).

- RL mostly fixed legality. Games that ended on an illegal guess dropped from 69% to about 40%, and those games now run out of turns instead.
- Among games that stay legal, the win rate did not go up (24%, 21%, 17% for SFT, run 1, run 2). The extra wins come from more games surviving, not from better play.
- Only about 1 guess in 5 is consistent with the feedback seen so far, before and after RL. After SFT the model writes the right constraints in its reasoning line 87% of the time, but only about 1 word in 4 obeys them.
- Nearly all illegal guesses are near misses: about 87% of the non-words are one letter away from a real word (`REDER`, `PLACA`).

Two things I tried that did not help: SFT for fewer steps (worse or equal on held-out words), and a small reward bonus for earlier guesses that fit the feedback (consistency stayed at about 18% and the model just wrote shorter reasoning lines).

## How it works

Each task is a Wordle game with 0 to 4 guesses already played, and the model makes the next one. The secret words come from 900 answers, with 20% held out for evaluation. Any of 8,511 words is a legal guess.

The model sees the board and the constraints worked out from it, with `=` for the right letter in the right spot, `-` for a letter that is in the word but elsewhere, and `.` for a letter that is not in the word:

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

It answers with a short reasoning line and `<guess>WORD</guess>`. A game is scored by its last guess:

| Last guess | Reward |
|---|---|
| No `<guess>` tag | -0.1 |
| Not a legal word | 0.0 |
| Legal, but contradicts earlier feedback | 0.1 |
| Legal and consistent with all feedback | 0.3 |
| The secret | 1.0 + 0.5 / (turns + 1) |

The training pipeline:

1. A solver plays 2,000 boards and writes a templated reasoning line plus a guess for each (`sft/`).
2. SFT on those: 62 steps on one A100, about 16 minutes. prime-rl packs several rows into each sequence, so that is closer to 7 epochs.
3. RL with prime-rl on two A100s, one for vLLM and one for training. Each step uses 128 rollouts (8 per board), learning rate 3e-6, 80 steps. Settings are in `configs/wordle_rl.toml`.

The 80-step RL run cost about $10-12 on Modal, and a 400-game eval about $0.60.

## Try it

```
uv run python -m scripts.play --seed 3
uv run python -m scripts.play --secret CRANE
```

The first run downloads the 3.4 GB model. It runs on Apple GPU, CUDA or CPU. The model samples at temperature 1.0, so games differ. One game:

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

Tests, and the random player and solver on the held-out games:

```
./check.sh
uv run python -m scripts.baselines --games 400
```

## Reproducing the training

You need a Modal account with GPU access and a Modal secret called `wandb-secret` that holds `WANDB_API_KEY`.

```
uv run python -m sft.export_prime_rl
uv run modal run --detach modal_prime_rl.py::sft_run
uv run modal run --detach modal_prime_rl.py::rl_run --ckpt-run <sft-run> --step 62 --max-steps 80 --eval-interval 20 --ckpt-interval 40
uv run modal run --detach modal_prime_rl.py::eval_run --ckpt-run <rl-run> --step 80 --games 400
```

- The SFT checkpoint needs to be converted to HF weights before RL. `eval_run` on the SFT checkpoint does that.
- `rl_run` has a `timeout=` in its decorator. 80 steps take about 110 minutes, so set it to 8400 seconds and keep an eye on your credit.
- prime-rl is pinned to `73140dec` and verifiers to `2cb0a6d3` (the revision prime-rl uses). verifiers v1 has a different API from older tutorials.
- `scripts/analyze_failures.py` and `scripts/check_reasoning.py` read the trace files of an eval run.

## Layout

| Path | What |
|---|---|
| `environments/wordle_rl/` | Game engine, prompt, reward, taskset and env |
| `sft/` | Solver teacher, trace generation, export for the prime-rl SFT trainer |
| `configs/` | prime-rl configs for SFT, RL and evals |
| `modal_prime_rl.py` | Modal image and the `sft_run`, `rl_run`, `eval_run` functions |
| `scripts/` | Play, baselines and trace analysis |
| `tests/`, `check.sh` | Tests |
| `results/` | Measurements behind the numbers above |
| `rl/`, `gate.py`, `constrained.py`, `evaluate.py` | An earlier from-scratch RLOO attempt with its pass@k gate, paused when the project moved to prime-rl |
| `PLAN.md`, `TASKS.md` | The original plan |

## Limits

- Two RL runs and one SFT run. The gain in wins is somewhere between 2 and 6 points and differs by seed.
- The games start with 0 to 4 guesses already played. This is not full Wordle from the first guess.
- 400 games gives a win rate of about +-3 points.
- Not tried: thinking mode, other temperatures, bigger models, other rewards for using the feedback.

## License

MIT, see `LICENSE`. The model weights start from Qwen3-1.7B, which has its own license.
