# Baseline: stock Qwen3-1.7B in the wordle_rl environment (2026-10-11)

Model `Qwen/Qwen3-1.7B`, thinking off, served by `mlx_lm.server` 0.32.0 on an Apple M5.
Environment `wordle-rl` (taskset split `train`, seed 0), null harness, subprocess runtime.
Sampling: temperature 1.0, 256 new tokens at most. 100 episodes, one rollout each.

Command:

    mlx_lm.server --model Qwen/Qwen3-1.7B --port 8080 --chat-template-args '{"enable_thinking": false}'
    uv run vf-eval wordle-rl -m Qwen/Qwen3-1.7B --client.base-url http://localhost:8080/v1 \
      --client.api-key-var LOCAL_KEY --sampling.temperature 1.0 --sampling.max-tokens 256 \
      -n 100 -r 1 -c 4 --no-push

| Measure | Value |
|---|---|
| Errored episodes | 0 of 100 |
| Wins (`won`) | 0 of 100 |
| Missing tag (`format_error`) | 0 of 100 |
| Games that ended on an illegal reply (`illegal`) | 98 of 100 |
| Mean reward (`ladder`) | 0.002 (98 episodes at 0.0, 2 at 0.1) |
| Legal first reply | 2 of 100 (95% Wilson interval 0.6% to 7.0%) |

Outcomes: 54 `not_a_word`, 31 `repeat`, 9 `bad_length`, 4 `not_alpha`, 2 `lost`.

| Priors | Episodes | Outcomes |
|---|---|---|
| 0 | 16 | 13 not_a_word, 3 bad_length |
| 1 | 26 | 12 repeat, 12 not_a_word, 2 not_alpha |
| 2 | 18 | 11 not_a_word, 5 repeat, 2 bad_length |
| 3 | 20 | 10 repeat, 8 not_a_word, 1 not_alpha, 1 lost |
| 4 | 20 | 10 not_a_word, 4 repeat, 4 bad_length, 1 not_alpha, 1 lost |

What it means:

- The model always writes the tag. It does not write a legal new word. It copies a prior guess, or it writes a string that is not a word. With 0 priors, the first 5 replies of a 20-game run were all `WORDL`, the name of the game in the system prompt.
- Almost all rewards are equal (0.0), so a group has no signal. RL from this model learns nothing (`AGENTS.md`, Rule 5, sink 2). SFT must come first (Mark II).
- Not measured: a run with thinking on, and other temperatures.
- The `PrimeIntellect/Qwen3-1.7B` clone always thinks (its `chat_template.jinja` has no `enable_thinking`), so it is not used here (`.jarvis/DECISIONS.md`, 2026-10-11).

## Second measurement: Modal A100, prime-rl `eval` (same day)

Same model, same game, same sampling (temperature 1.0, 256 tokens, thinking off through `sampling.extra_body.chat_template_kwargs`).
Run through prime-rl's env servers on 2 x A100-80GB, vLLM and router (`modal_prime_rl.py::spike --stage eval`, run `spike-wordle-baseline-1011-110208`,
config `configs/wordle_eval.toml`, which does not set the harness or the runtime). 100 episodes, 0 errors.

| Measure | Mac (`vf-eval`) | A100 (`prime-rl eval`) |
|---|---|---|
| Wins | 0 | 0 |
| Games ended on an illegal reply | 98 | 99 |
| Mean reward (`ladder`) | 0.002 | 0.001 |
| Outcomes: not_a_word / repeat / bad_length / not_alpha / lost | 54 / 31 / 9 / 4 / 2 | 48 / 39 / 10 / 2 / 1 |
| Format errors | 0 | 0 |
| Mean output tokens per episode | not measured | 10.6 |

The two paths agree within the noise of 100 games. The 0-prior games are identical in both (13 not_a_word, 3 bad_length), so both runs saw the same tasks.
`enable_thinking: false` reached vLLM (replies of about 10 tokens, no truncation).
This also shows that our taskset, env, reward and metrics run unchanged under prime-rl on Python 3.12.
