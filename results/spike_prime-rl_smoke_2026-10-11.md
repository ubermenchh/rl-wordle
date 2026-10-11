# Spike: prime-rl smoke test on Modal (2026-10-11)

Mark I, step 7b. The `reverse-text` example of prime-rl, 20 RL steps, run by `modal_prime_rl.py`
(`spike --stage smoke`). It is not our taskset. It proves the stack: install, GPUs, vLLM, trainer,
orchestrator, weight sync and W&B.

Setup: Modal, 2 x A100-SXM4-80GB (driver CUDA API 13.0), image `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04`
with Python 3.12, prime-rl `73140dec` (2026-10-10), verifiers `0.3.2.dev206` (`2cb0a6d33a`), torch `2.13.0+cu130`.
Config `examples/basic/reverse-text/rl.toml`: model `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`, batch 128, group size 16,
max 128 completion tokens, LR 3e-6. Run `spike-reverse-text-1011-101508`, W&B project `rl-wordle`.

Held-out eval (256 episodes of reverse-text):

| Step | Policy | Reward | Truncation | Error |
|---|---|---|---|---|
| 0 | v0 | 0.1203 | 85.5% | 0.0% |
| 5 | v5 | 0.8205 | 0.8% | 0.0% |
| 10 | v10 | 0.8306 | 0.4% | 0.0% |
| 15 | v15 | 0.8392 | 0.0% | 0.0% |
| 20 | v20 | 0.8436 | 0.8% | 0.0% |

Training: 128 of 128 rollouts trainable at every step. Step times 5 to 15 s when no eval ran. The orchestrator loop took 12 min 46 s.
Trainer: peak memory 10.7 to 11.3 GiB of 80 GiB. Entropy 0.60 at step 1, a maximum of 1.41 at step 12, 0.85 at step 20.
Mismatch KL 0.0005 at step 1, a maximum of 0.0074 at step 11, 0.0014 at step 20. Grad norm 0.24 to 1.74, no spike.

What it took (five tries; the first four died in 20 seconds or less, each costing about a minute of 2 GPUs):

1. `uv sync --locked` fails at this commit. The lock does not match the workspace (the verifiers submodule is a dev build). Use `--frozen`.
2. `ModuleNotFoundError: prime_kernels`. The trainer needs the `kernels` extra.
3. `No such file: 'inference'`. `rl` starts programs by name, so the venv `bin` must be on `PATH`.
4. `FileExistsError` on the run folder. Use a new run name for each try.
5. `No such file: 'vllm-router'`. The router is in the `disagg` extra.

Extras used: `gpu dashboard flash-attn disagg kernels`. Left out on purpose (Hopper and Blackwell kernels): `flash-attn-3 flash-attn-cute flash-mla quack`.
`flash-attn-cute` also fails to build on Modal: GitHub refuses the anonymous clone of the submodules of flash-attn-4.

Seen and not understood:

- The vllm-router logged two failed requests to the engine (`Failed to send typed request`, on `/v1/chat/completions` and on `/inference/v1/generate`). Both times the next requests returned 200.
- vLLM warned `RotaryEmbedding: Failed to load weights` at the first weight load. The run did not suffer.
- Truncation of 85.5% at step 0 is a property of the SFT model with a 128-token limit.
