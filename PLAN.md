# RL for Wordle on a Small Open Model — Research + Plan

Target: train a ~0.5B–1.7B open model with reinforcement learning to play Wordle,
implementing the RL algorithm from scratch. Compute is remote (dev machine is an
Apple Silicon Mac, no CUDA).

---

## 0. Decisions up front

| Question | Answer | Why |
|---|---|---|
| **Use Tinker?** | **No, not as the trainer.** | Smallest model on Tinker is `Qwen/Qwen3.5-4B` (verified from `models.json`). It is LoRA-only, gives you no hidden states (so no critic, ever), and hides the backward pass — i.e. it abstracts away exactly what "from scratch" means. Use it for two narrow things (§3.2). |
| **Use Modal?** | **Yes, as primary compute.** | Per-second billing, `$30/mo` free credit, infra-as-Python, Volumes for checkpoints, official GRPO examples. RL dev is bursty (run 20 min → read reward curve → edit reward → rerun); serverless is the right shape. |
| **Which model?** | `Qwen3-0.6B` primary; `gemma-3-1b-it` and `SmolLM2-360M-Instruct` are legitimate alternatives | See §3.3 for measured data. Decide by the Phase 0 pass@k gate, not by vibes. |
| **Which algorithm first?** | **RLOO**, not GRPO | RLOO ≡ Dr. GRPO up to a `G/(G-1)` learning-rate factor (proved in Dr. GRPO Appendix A). So the *simple* algorithm is the corrected state of the art. ~80 lines, no critic, no ref model, no clipping. GRPO's extra terms (`std`, `1/\|o_i\|`) are precisely the parts the literature has since identified as bugs. |
| **Multi-turn or single-turn?** | **Both, in that order — single-turn first** | Every project that got a number relied on training over *randomized mid-game states*. Single-turn removes token masking + credit assignment, so you get signal in days not weeks. Then do real multi-turn rollouts as Phase 5. |
| **vLLM for rollouts?** | **Not in v1** | Generate with HF `.generate()` in the same process/dtype and reuse the sampling logprobs → the train/inference mismatch is zero *by construction*. Add vLLM only when generation is measurably the bottleneck, and add Truncated Importance Sampling at the same time. |

---

## 1. Prior art — read this before writing code

The honest summary: **nobody has published a strong Wordle result from a small model
trained purely with RL.** Knowing the real bar prevents you from thinking your code is
broken when it is actually the task.

| Work | Model | Result | Code |
|---|---|---|---|
| [RLDiary "The $100 Agent"](https://github.com/RLDiary/Wordle-GRPO) ⭐ | Qwen2.5-7B + SFT(500 o3 traces) + GRPO | **<5% → 14.5% solve rate.** 35 steps ≈ 6,120 games, ~5h on 3×A100 ≈ **$25** | ✅ Public |
| [Predibase / DeepLearning.AI course](https://www.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/) | Qwen2.5-7B-Instruct | "7/10 games" — course marketing, no methodology | Dataset only: [`predibase/wordle-grpo`](https://huggingface.co/datasets/predibase/wordle-grpo) (76 rows) |
| [HF TRL × OpenEnv notebook](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb) | **Qwen3-1.7B** | No win rate reported — it's a tutorial | ✅ Public, Colab-runnable |
| [shreyansh26/wordle-solver](https://github.com/shreyansh26/wordle-solver) ⭐ | Qwen3-4B + SFT(Kimi-K2) + multi-turn GRPO | Best *engineering* reference; `constraint_validity_reward` is the idea to steal | ✅ Public |
| [charbull/wordle-lora-rl](https://github.com/charbull/wordle-lora-rl) ⭐ | Gemma-3-4B, MLX/LoRA on a Mac | Best *failure-mode* writeup. Read its "Lessons Learned" twice | ✅ Public |
| [TextArena](https://github.com/TextArena/TextArena) `envs/Wordle/env.py` | — | The upstream env everyone else wraps. Copy its two-pass feedback logic | ✅ Public |
| [shaikabdulfahad/wordle-qwen2-mini](https://huggingface.co/shaikabdulfahad/wordle-qwen2-mini) | **Qwen2-0.5B** + GRPO | 20 games on a T4. Existence proof that the plumbing runs at 0.5B, not that it learns | Weights only |

Corrections to common misconceptions: **OpenPipe ART has no Wordle example** (it has
2048 / Codenames / Tic-Tac-Toe — its RULER auto-reward is the wrong tool here anyway,
since Wordle's reward is perfectly verifiable). **`willccbb/verifiers` moved** to
[`PrimeIntellect-ai/verifiers`](https://github.com/PrimeIntellect-ai/verifiers); its two
Wordle envs are ~20-line TextArena wrappers contributing no Wordle-specific logic.

### The five findings that should shape your design

1. **The binding constraint is action-space validity, not understanding G/Y/X.** RLDiary
   on a *7B*: guesses were "**VERY** often neither 5 letters long, nor valid English
   words." charbull on ~1B: "they often failed to adhere to the required format,
   indicating they lacked the capacity to learn the structure from the RL signal alone.
   **For smaller models, SFT is likely not just helpful, but necessary.**"
2. **Curriculum over mid-game states is the single biggest lever.** charbull: training on
   turn-1 prompts only → barely learns; one prior guess → learning starts; **random
   history of 0–4 turns → the unlock.** Predibase's whole dataset is flattened mid-game
   states. Independently discovered by three groups.
3. **Gate the reward, don't sum it.** RLDiary used progressive gating (format 0.05 →
   valid-word 0.2 → solved 1.0). Additive rewards get hacked. charbull's first model
   learned to emit `Final Final Final...` because inaction was cheaper than a wrong guess.
4. **Pre-digest the constraints in the prompt.** charbull: symbolic feedback (`✓✓xxx`)
   underperformed a plain-English state summary (`Green Letters:`, `Words Already
   Guessed:`). Also: adding "Do not repeat any words" to the prompt beat the repetition
   *penalty*.
5. **Low temperature at eval** (0.1 ≫ 0.9) for a logical puzzle.

---

## 2. Reality check — what is actually achievable

State-of-the-art on a 7B with SFT + GRPO is **14.5%** full-game solve rate. Set
milestones accordingly, and **make the task easier until signal exists**:

- Restrict the answer list to ~500–1000 common words (Wordle's real list is ~2,300;
  allowed-guess list is ~13,000).
- Allow 6 turns, but score partial progress.
- Train on randomized mid-game states so many prompts are 1–2 guesses from solved.

Milestones, in order:
- **M1** — 0.6B emits a valid 5-letter dictionary word in the right tags >95% of the time.
- **M2** — given a mid-game state with 4 greens, it solves >80%. (This is "can it read feedback at all.")
- **M3** — measurable lift in full-game solve rate over the SFT baseline on held-out words.
- **M4** — a clean ablation table: RLOO vs GRPO vs Dr. GRPO vs +DAPO tricks.

M4 is the real deliverable. "I implemented five RL algorithms from scratch and measured
which corrections matter" is worth more than a win-rate number, and it is achievable.

---

## 3. Platform

### 3.1 Modal (primary) — verified pricing

| GPU | VRAM | $/hr | Use |
|---|---|---|---|
| T4 | 16 GB | $0.59 | too slow |
| **A10** | 24 GB | **$1.10** | **dev / debug** (~27h inside the free $30) |
| L40S | 48 GB | $1.95 | skip — capacity you don't need |
| A100 80GB | 80 GB | $2.50 | skip — worse bandwidth than H100 at 63% of price |
| **H100** | 80 GB | **$3.95** | **real runs** |

Plus, billed additively: CPU `$0.047/core/hr`, memory `$0.008/GiB/hr` — a real trainer at
8 cores + 64 GiB adds **~$0.89/hr**. Volumes `$0.09/GiB/mo`, first 1 TiB free.
Free tier: **$30/mo, 10 GPU concurrency, 1-day log retention.**
**Apply for [academic credits](https://modal.com/academics) — up to $10k.**

Counter-intuitive but correct: prefer **H100 over L40S** for real runs. GRPO wall-clock is
dominated by autoregressive rollout generation, which is memory-bandwidth-bound. H100 SXM
is ~3.35 TB/s vs L40S 864 GB/s — ~3.9× the bandwidth for 2.0× the price, so it is usually
*cheaper per completed step*. Don't assume the cheap GPU is the cheap run.

**The non-negotiable Modal facts:**
- Default function timeout is **300s**. Max is **24h**. Set `timeout=` or your run dies in 5 minutes.
- **Every GPU function is preemptible and there is no opt-out** (`nonpreemptible=True` is CPU/memory-only). Checkpoint from step 1, not as later hardening.
- Use `.spawn(...).get()`, **not `.remote()`** — `.remote()` Function Calls expire at 24h.
- `modal run --detach` or the job dies when you close your laptop.
- Put `add_local_python_source` **last** in the image chain or every code edit rebuilds vLLM.
- Log retention is 1 day on Starter → **wire up W&B on day one.**
- Do **not** spawn a Sandbox per rollout for the reward (Modal's `grpo_trl` example does that only because it executes untrusted generated code). Wordle scoring is ~10 lines of pure Python — do it in-process.

Read, in order: [`grpo_trl`](https://modal.com/docs/examples/grpo_trl) ·
[`long-training`](https://modal.com/docs/examples/long-training) (the resumable pattern) ·
[`learn_math.py`](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/reinforcement-learning/learn_math.py)
(closest existing template — multi-turn GRPO via `verifiers`).

**Step 1 of the whole project:** run `grpo_trl.py::train_vllm_colocate_mode` *unmodified*
on an A10 to validate account, image build, GPU access, and Volume writes — before
writing any Wordle code.

### 3.2 Tinker — two narrow uses, not the trainer

Verified from [`models.json`](https://tinker-docs.thinkingmachines.ai/tinker/models.json):
smallest is **`Qwen/Qwen3.5-4B`** ($0.737/M train tokens). No sub-4B exists. Llama 3.x and
Qwen3-30B-A3B were retired. It is **LoRA-only**, and you only ever receive logprobs — never
hidden states — so **a critic/value head is impossible.** A curriculum that goes REINFORCE
→ baseline → A2C → PPO-with-GAE cannot be completed on Tinker.

Where it *is* worth your money:

1. **Read [`recipes/multiplayer_rl/guess_number/`](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/multiplayer_rl/guess_number)** — a
   multi-turn guess-with-feedback game, structurally isomorphic to Wordle, that converges
   to ≥80% in **20 steps**. Free to read. Also read
   [`rl/data_processing.py`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/data_processing.py):
   their `compute_advantages` is literally `rewards - rewards.mean()` (no std, citing
   Dr. GRPO), and `trajectory_to_data` shows the prefix-merging + observation-masking you
   must get right.
2. **Teacher for SFT data / on-policy distillation.** RL transmits `O(1)` bits per episode
   (one scalar advantage); on-policy distillation transmits `O(N)` (a signal per token).
   TML measured distillation recovering an RL-trained policy in **7–10× fewer gradient
   steps ≈ 50–100× less compute**. So: RL a `Qwen3.6-35B-A3B` into a good Wordle player on
   Tinker (~$1.18/M train tokens), then distill into your 0.6B on Modal. That is a strong
   Phase 6 and a genuinely interesting result.

Estimated Tinker cost if you do use it: a converged `Qwen3.5-4B` Wordle run is **tens of
dollars** (~$40 at bs=32, group=8, 200 steps with cache discount). Cached prefill is 80%
off, which matters a lot for multi-turn.

Also read regardless of platform: [LoRA Without
Regret](https://thinkingmachines.ai/blog/lora/) — **LoRA LR ≈ 10× the full-FT LR** (~15×
for runs under 100 steps) is the mistake everyone makes; optimal LR is ~independent of
rank; apply LoRA to MLP not just attention; rank 32 is plenty for RL.

### 3.3 Model selection — measured, not assumed

Data from `scripts/tokenizer_probe.py` (runs on the Mac, no GPU, ~2 min).

**Finding 1: tokenization is not a differentiator.** Every candidate fuses bare words —
`CRANE` → `['CR','ANE']` on all six. Gemma is worse: `SPEED` is a **single token**. But
letter-spacing is clean everywhere (`C R A N E` → exactly 5 tokens; 26/26 single-token
letters, both cases). So §6's mitigation is universally available and this drops out of the
model decision.

**Finding 2: subtract the embedding table before judging "size."** All these models tie
embeddings, so:

| Model | Vocab | Embed params | ≈ Transformer params | Embed share |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 128k | 262.7M | ~975M | 21% |
| gemma-3-1b-it | 262k | 302.0M | ~700M | 30% |
| **Qwen3-0.6B** | 152k | 155.6M | **~440M** | 26% |
| Qwen2.5-0.5B-Instruct | 152k | 136.1M | ~354M | 28% |
| SmolLM2-360M-Instruct | **49k** | 47.2M | ~315M | **13%** |
| gemma-3-270m-it | 262k | 167.8M | **~100M** | **62%** |

`gemma-3-270m-it` spends 62% of its parameters on the embedding table — it is effectively a
~100M-param reasoning model. **SmolLM2-360M has ~3× its transformer capacity** while being
33% bigger nominally. Given that a 7B struggles to emit valid 5-letter words, 270M is not a
serious candidate; keep it as a pipeline smoke test only.

**Finding 3: vocab width is an RL cost.** The `batch × seq × vocab` logits tensor is
typically your largest activation, materialized for every rollout. Gemma's 262k vocab is
1.7× Qwen's and **5.3× SmolLM2's**. On a 24GB A10 at `k=8` that is the difference between
fitting and not. **For RL, small vocab is an asset** — it buys you larger group size `k`,
which is exactly what advantage-estimate quality depends on.

**Finding 4: Gemma 3 has no system role.** It silently merges it into the first user turn:

```
gemma-3-1b-it: '<bos><start_of_turn>user\nSYSTEM_TEXT\n\nUSER_TEXT<end_of_turn>\n<start_of_turn>model\n'
Qwen3-0.6B:    '<|im_start|>system\nSYSTEM_TEXT<|im_end|>\n<|im_start|>user\nUSER_TEXT...'
```

It doesn't error, it merges — exactly the class of template quirk that corrupts delta-based
token masking in Phase 5.

**Verdict.** Primary `Qwen3-0.6B`, only because all prior art and Modal's examples use Qwen,
so a failure is diagnosable as yours vs the task's. `gemma-3-1b-it` is a fine choice (and
Gemma-3-**4B** has published Wordle prior art) at the cost of 1.7× logits memory and the
system-role merge. `SmolLM2-360M-Instruct` is the interesting dark horse: a 49k vocab makes
the whole loop cheap enough to run large `k` on an A10. **Resolve this with the Phase 0
gate across all four (~$2 on an A10), not from this table.**

### 3.4 When to leave Modal

Modal is ~30–60% more expensive per GPU-hour than RunPod Community ($2.69/hr H100) or
Prime Intellect (**$2.43 on-demand / $0.94 spot**). Modal wins for bursty iteration
(break-even: if the GPU idles >60% of the time). For a single converged 72-hour run, go
elsewhere. The checkpoint-resume code you write for Modal's mandatory preemptibility ports
directly to spot instances, so this costs you nothing to keep as an option.
**Colab is disqualified** — no real persistent filesystem, opaque GPU assignment.

---

## 4. Repo layout

```
rl-wordle/
  env/
    wordle.py           # feedback fn (two-pass, duplicate-letter correct), game state
    words.py            # answer list (~500 curated) + allowed-guess list (~13k)
    render.py           # state -> prompt string. THE highest-leverage file (§6)
    test_wordle.py      # unit tests. SPEED/ERASE etc. A buggy env is an exploitable env.
  rl/
    rollout.py          # generate() + token span tracking + loss_mask construction
    advantage.py        # reinforce | rloo | grpo | drgrpo  <- the ablation lives here
    loss.py             # policy-gradient losses; normalizer is a config flag
    train.py            # the ~200-line loop. no framework.
    metrics.py          # everything in §8
  sft/
    gen_traces.py       # teacher -> SFT data
    train_sft.py
  eval/
    harness.py          # held-out words, fixed seeds, pass@k, temperature sweep
  modal_app.py          # Modal entrypoints: sft, train, eval, serve
  PLAN.md
```

Single-file-per-concern, no RL framework. Model this on
[**nanoAhaMoment**](https://github.com/McGill-NLP/nano-aha-moment) — "Single File, Single
GPU, From Scratch... **No RL Library**", and its task (CountDown) is a small symbolic
puzzle structurally close to Wordle. It has Karpathy-style YouTube lectures. **This is
your primary code template.**

---

## 5. Phased plan

### Phase 0 — Environment + exploration gate (no RL yet)

Do not skip the gate. This is where most projects silently die.

1. Write `env/wordle.py`. **Two-pass feedback**: greens first, consume matched letters,
   then yellows. Unit-test duplicate letters (`guess=SPEED, target=ERASE`).
2. Classify invalid moves separately, as TextArena does: bad format / wrong length /
   repeated guess / not in dictionary. You need these as distinct metrics.
3. **Prompt-format sweep.** Dr. GRPO §2.1: Qwen2.5-Math-7B scored **0.0** with the R1
   template and **38.2** with none — "applying templates in fact destroys the capability
   before RL reconstructs it." Test 3–4 formats, pick by pass@16.
4. **THE GATE.** Sample `k=16` games per starting state at T=1.0. Measure the fraction of
   prompts with `0 < #solved < 16`. **If that's near zero, stop.** No algorithm creates a
   signal that isn't there. Remedies: smaller answer list, more turns, few-shot prompt,
   shaped reward, SFT warm-start.

**Exit criteria:** >20% of groups have non-degenerate reward. Env tests green.

### Phase 1 — SFT warm-start

Two independent groups say this is mandatory below ~7B; you are at 0.6B. RLDiary spent
**$35** on 500 o3 traces for this.

- Generate 500–2000 traces from a strong teacher over **randomized mid-game states
  (0–4 prior guesses)** — this curriculum *is* the finding from §1.2.
- Filter to traces that are both correct and constraint-consistent.
- Train on Modal A10. Re-run the Phase 0 gate afterwards; it should improve a lot.

**Exit criteria:** M1 (>95% valid-word format compliance).

### Phase 2 — RLOO from scratch, single-turn

~80 lines. No ratio, no clip, no KL, no value net, no reference model.

```
Â_i = (k·R_i − Σ_j R_j) / (k−1)                                 # RLOO, vectorized
L   = − Σ_i Σ_t mask_{i,t} · Â_i · log π_θ(y_{i,t} | ·) / (B · MAX_TOKENS)
```

- `μ = 1`: generate → one gradient step → regenerate. Fully on-policy, so the importance
  ratio is identically 1 and clipping is dead code. RAGEN §4.3 measured Online-1 beating
  Online-5/10 — simpler *and* better.
- **Dynamic sampling from day one:** drop groups with `#solved ∈ {0, k}`, oversample to
  refill. This was DAPO's single largest ablation gain (+8 points).
- `k = 4–8`. RAGEN: with a fixed batch, **more prompts × fewer samples each** beats fewer
  prompts × many (4 beat 32 on generalization) — but you need `k > 1`.
- Grad-clip global norm 0.5. Adam `eps=1e-5` (not PyTorch's 1e-8). LR ~1e-6.
- Same engine for generation and training; reuse sampling logprobs → `δ_t = 0`.

**Exit criteria:** M2 (>80% on 4-green states), and reward moves.

### Phase 3 — Instrument before tuning anything

See §8. Non-negotiable. Also: **decode and print the masked-in tokens for one sample every
run**, and `assert loss_mask.sum() == expected_generated_token_count`.

### Phase 4 — Algorithm ablation (the real deliverable)

Each is a small, testable diff against Phase 2. Hold the reward and env fixed.

| Variant | Diff | Hypothesis |
|---|---|---|
| REINFORCE + running-mean baseline | swap `advantage.py` | worse than RLOO (higher variance) |
| GRPO as published | `+ /std`, `+ 1/\|o_i\|`, `+ β·KL_k3` | length bias + difficulty bias visible |
| Dr. GRPO | drop `std`, drop `1/\|o_i\|` | ≈ RLOO up to LR scale — **verify this empirically** |
| + Clip-Higher | `ε_low=0.2, ε_high=0.28` | fixes entropy collapse if you see it |
| + token-level loss normalizer | batch-token mean vs per-seq mean | stability, small perf gain |
| GSPO-token | sequence-level ratio, per-token advantage | best fit for multi-turn Phase 5 |

DAPO's own ablation, for calibration (AIME24): naive GRPO 30 → +overlong filter 36 →
+clip-higher 38 → +soft-overlong 41 → +token-level loss 42 → +dynamic sampling **50**.

### Phase 5 — Real multi-turn rollouts

Now the interesting part. Start with **trajectory-level advantage** (one scalar per game
broadcast to every action token) — Wordle is ≤6 turns and reward is near-terminal, so
per-turn credit is a refinement, not a prerequisite.

**Critical: mask environment tokens.** Train on env-produced tokens and you are doing
supervised learning on the environment's output — the model learns to hallucinate feedback.

The robust approach for a from-scratch build: **do not use a multi-message chat template
at all.** Build one flat string and record `(start, end)` token spans for your own
generations as you go. `loss_mask = 1` on those spans only. You control the string, so
there is no template to fight. (If you insist on chat format, read
[verl's multi-turn tokenization docs](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
first — delta-based tokenization with `add_generation_prompt=True` on the prefix, plus a
fixed `BASE_CHAT_HISTORY` to survive reasoning-stripping templates like Qwen3's, plus a
tokenization sanity check. It's a real minefield.)

Then, only if a metric demands it: per-turn advantages via **GSPO-token** (sequence-level
importance weight for stability, per-token advantage for credit — costs ~3 lines), or
DeepSeekMath §4.1.3's process-supervision form `Â_{i,t} = Σ_{index(j) ≥ t} r̃_i^{index(j)}`.

### Phase 6 — Stretch

- **On-policy distillation** from a Tinker-trained `Qwen3.6-35B-A3B` teacher (§3.2).
- **Constrained decoding** inside `<guess>` tags — mask logits to the allowed-word list.
  Since §1.1 says validity dominates the signal budget, this converts "emit a legal word"
  into "pick a good word." Arguably cheating; definitely a clean ablation. Keep `<think>`
  free-form.
- Hard-code the opener (CRANE/SOARE) and let the model take over from turn 2 — charbull
  found opening strategy basically doesn't emerge at this scale.

---

## 6. Two design choices worth more than any algorithm tweak

**Tokenization.** Wordle is per-letter reasoning; BPE fuses letters. Render feedback (and
probably the guess) **letter-spaced** — `C . R - A = N . E .` — not as a compact
emoji/symbol string. This is nearly free and plausibly the highest-leverage decision in
the project. It has nothing to do with RL.

**Pre-digest the state.** Per charbull, beat symbolic feedback with an explicit summary:

```
Green (correct position):   _ R _ _ E
Yellow (in word, wrong pos): A
Grey (not in word):          C, N, S, T
Already guessed:             CRANE, SWEAT
Do not repeat any word above. Reply: <guess>WORD</guess>
```

You are not obliged to make a 0.6B model re-derive constraints it can't hold. Let the
env do the bookkeeping and spend the model's capacity on word choice.

---

## 7. Reward design

**Gate, don't sum** (RLDiary). Additive rewards get hacked.

```
if not format_ok:                  R = -0.1    # must be the worst outcome. see below.
elif not valid_5letter_dict_word:  R =  0.0
elif solved:                       R =  1.0 + 0.5 * (1/turns_used)
else:                              R =  0.3 * constraint_consistency
```

`constraint_consistency` is the idea to steal from
[shreyansh26](https://github.com/shreyansh26/wordle-solver): build `fixed_positions`
(greens), `excluded_positions`, `required_letters`, `excluded_letters` from history, and
score whether the guess *could* be the answer given all prior feedback. It checks logical
consistency with the clues, not correctness — **a dense, verifiable, intermediate signal**,
which is exactly what RAGEN §4.4 says is missing when reasoning fails to emerge. Known
limitation to document: duplicate-letter *counts* are only approximate.

**Make format failure unequivocally the worst outcome.** charbull's model learned to emit
empty strings because the penalty for lazy inaction was milder than for a thoughtful wrong
guess.

**Set β = 0 — no KL, no reference model.** Both DAPO and Dr. GRPO argue this: with a
rule-based verifier there is no reward model to over-optimize. Saves memory, saves a model,
saves a bug surface. You may still want to *log* KL as a diagnostic — use Schulman's **k3**,
`(r − 1) − log r`, computed in log space (`(logr.exp() - 1) - logr`). It's unbiased, lower
variance than k2, and provably ≥ 0. k1 (`-log r`) has std/true ≈ **20** at small KL.

Wordle-specific hacks to defend against: non-words; repeating a guess (ensure a repeat can
never *increase* reward); memorizing the answer list (hold words out — if train reward ≫
eval reward you're memorizing); exploiting bugs in your own feedback function; degenerate
opener + ignoring feedback.

---

## 8. Instrumentation (Phase 3, non-negotiable)

| Metric | Why |
|---|---|
| **frac. groups with nonzero advantage** | **#1 health metric.** Near zero ⇒ no signal; no algorithm helps |
| **reward std within group** | earliest collapse warning — RAGEN saw std drop at step 40, reward collapse at step 90 |
| gradient norm | spike ⇒ collapse is already irreversible |
| token entropy over generated positions | entropy collapse. `R = −a·e^H + b`: your ceiling is set by starting entropy |
| **entropy of the guess distribution on fixed eval prompts** | *template collapse* — diverse-looking but input-agnostic output, invisible to token entropy |
| does the guess change when feedback changes? | is it actually conditioning on the clues, or pattern-matching? |
| held-out eval win rate | DAPO §4.3: train reward decorrelates from val accuracy |
| invalid-guess rate, repeat-guess rate (separately) | the dominant failure mode |
| mean / max response length | length-bias symptom — check your normalizer *before* blaming the reward |
| `max_t \|δ_t\|` | only if two engines. Free to compute, and the only reliable mismatch alarm |

---

## 9. Gotchas, ranked by (probability × silent damage)

**Tier 0 — fix before the first run**

1. **Loss mask leaking environment tokens.** Assert it, decode it, look at it.
2. **The `masked_mean` denominator.** Dr. GRPO audited trl, OpenRLHF, verl, SimpleRL-Zero,
   Open-Reasoner-Zero — **all length-biased.** Use a fixed global constant:
   `loss = (per_tok_loss * mask).sum() / (B * MAX_TOKENS)`, not `/ mask.sum(-1)`. Getting
   this wrong pushes correct answers shorter and wrong answers longer, and you will
   misread the resulting length growth as emergent reasoning.
3. **Group-std normalization.** Upweights the *least* informative prompts (a group with
   rewards `[1,1,1,1,1,0]` gets inflated ~2.7× vs `[1,1,1,0,0,0]`). Also `std=0` → NaN.
   Just don't divide by std.
4. **Zero-advantage groups eating your batch.** Brutal at both ends in Wordle: early,
   nothing is solved; late, everything is. Filter `0 < #solved < k` and oversample.

**Tier 1 — will destabilize the run**

5. **vLLM ↔ PyTorch logprob mismatch.** Real: `mean |δ_t|` is tiny but `max |δ_t| ≈ 1.0`,
   and the argmax can flip. Both naive fixes fail — recomputation skews the advantage-
   weighted loss (the clipped surrogate reacts asymmetrically to `sign(Â)`, so symmetric
   numerical noise becomes a biased gradient), and bypass lets the optimizer exploit
   trainer-side numerical artifacts. **KL estimators do not detect this.** Avoid the whole
   class by using one engine in v1. If you must use vLLM: try fp16 instead of bf16
   ([arXiv 2510.26788](https://arxiv.org/abs/2510.26788) — possibly the cheapest fix in the
   literature), add TIS with `τ_tok ≈ 2`, and log `δ_t`.
6. **Entropy collapse.** Mitigate with Clip-Higher (`ε_low=0.2, ε_high=0.28` — do *not*
   also raise `ε_low`, that collapses the sampling space).
7. **Echo trap / multi-turn collapse.** RAGEN: "vanilla adaptations of PPO and GRPO achieve
   early gains in agent settings but often collapse." Their Qwen-2.5-**0.5B** setup is
   exactly your scale. Mitigate with variance-based trajectory filtering (keep top 25–75%
   by reward std).

**Tier 2**

8. Prompt-template sensitivity (Phase 0.3).
9. Reward hacking (§7).
10. `<think>` blocks shrinking monotonically — RAGEN measured 66 → 17 tokens over 200
    steps, and `NoThink` variants sometimes *won*. Outcome-only reward will not induce
    reasoning; that's what the constraint-consistency term is for.

---

## 10. Reading path

**Before writing code**
1. [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — read the "13 core details" section. Most important doc here for *how to work*, even though you won't use PPO.
2. [Back to Basics: Revisiting REINFORCE for RLHF](https://arxiv.org/abs/2402.14740) — **read before the GRPO paper.** Gives REINFORCE-with-baseline and RLOO, plus the three arguments for a simple implementation (clipping fires <5% of the time; removing it slightly helps; LLM RLHF is a bandit, not a token-level MDP).
3. Sutton & Barto **Ch. 13 §13.1–13.4** only. Everything here is Ch. 13 with a transformer attached. Skip Ch. 6–12.
4. [Schulman, Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) — 5 min, permanently useful.

**Code to read (not clone)**
5. [**nanoAhaMoment**](https://github.com/McGill-NLP/nano-aha-moment) ⭐ your template. Has lectures.
6. [TinyZero](https://github.com/Jiayi-Pan/TinyZero) · [sail-sg/understand-r1-zero](https://github.com/sail-sg/understand-r1-zero) (the *correct* unbiased loss) · [open-r1](https://github.com/huggingface/open-r1) (check "how does everyone do X")
7. [verl multi-turn tokenization docs](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html) — required before Phase 5.

**Papers, in this order**
8. [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) — §4.1 only, plus §5.2.1 ("Towards a Unified Paradigm", genuinely clarifying) and §4.1.3 (process supervision = per-turn credit).
9. [Dr. GRPO — Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) ⭐ **highest information-per-page paper on this list.** §3.1 (two biases), §3.2 (the `masked_mean` listing), Appendix A (the RLOO equivalence proof), §2.1 (template sensitivity, pass@8 as exploration check).
10. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — read *after* Dr. GRPO so you're appropriately skeptical of the "aha moment" and length-growth narratives.
11. [DAPO](https://arxiv.org/abs/2503.14476) — §3.1–3.4, Table 1, §4.3.
12. [GSPO](https://arxiv.org/abs/2507.18071) — §3 (sharpest conceptual argument in this literature: a token-level "importance weight" from a single sample isn't correcting any distribution, it's noise), §4.3 GSPO-token.
13. [RAGEN](https://arxiv.org/abs/2504.20073) ⭐ most directly relevant to a multi-turn game at 0.5B. Echo trap, StarPO-S, rollout shaping.
14. [Entropy Mechanism of RL](https://arxiv.org/abs/2505.22617) · [ArCHer](https://arxiv.org/abs/2402.19446) (concept only) — when things break.

---

## 11. Budget

| Item | Cost |
|---|---|
| Phase 0–1 dev on A10 | ~$0 (free $30/mo) |
| SFT teacher traces (API) | ~$20–35 |
| Phase 2–4 training, H100 ~60h incl. CPU/mem overhead | ~$290 |
| Optional Tinker teacher + distillation | ~$40–80 |
| **Total** | **~$150–400** |

Calibration: RLDiary got 3× improvement on a 7B for **$25** of GPU (3×A100, 5h). If you're
spending far more than that per experiment, the experiment is too big.
Apply for [Modal academic credits](https://modal.com/academics) ($10k) first.

---

## 12. The three things most likely to sink this

1. **Loss mask leaking environment tokens.**
2. **No learning signal** — a 0.6B never solves a Wordle. Measure pass@k *first* (Phase 0.4).
3. **A normalizer copied from a framework that had it wrong.**

## The one fact to hold onto

**Dr. GRPO ≡ RLOO up to a `G/(G−1)` learning-rate factor.** The "advanced" algorithm you'd
reach for is the *simple* one, and GRPO's extra terms are precisely the parts the
literature later identified as bugs. Write RLOO. You get the corrected state of the art in
~80 readable lines, and every subsequent paper becomes a small, testable diff against it.
