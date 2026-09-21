# Build Tasks — RL for Wordle on a Small Open Model

Actionable breakdown of `PLAN.md`. Each task is a checkpoint: when it's done, something
runs and produces output you can read. Deep rationale, exact math, and citations live in
`PLAN.md` — this file is the sequence.

**Ordering principle:** tasks 1–4 run entirely on your Mac for free. Task 4 is a hard
go/no-go gate. Do not provision cloud GPUs until it passes.

| # | Task | Where | Rough effort |
|---|---|---|---|
| 1 | Foundation: env core + tests | local | half a day |
| 2 | Prompt rendering + state sampler | local | 2–3 hrs |
| 3 | Baselines + eval harness | local | half a day |
| 4 | **THE GATE: pass@k viability check** | local (MPS) | 2–3 hrs |
| 5 | Reward function | local | 2–3 hrs |
| 6 | Modal harness | cloud | half a day |
| 7 | SFT warm-start | cloud | 1 day |
| 8 | RLOO from scratch | cloud | 1–2 days |
| 9 | Instrumentation | cloud | half a day |
| 10 | **Algorithm ablation (the deliverable)** | cloud | 3–5 days |
| 11 | Multi-turn rollouts | cloud | 3–5 days |

---

## Task 1 — Foundation: environment core

**Goal.** A tested Wordle engine in one file. Everything downstream trusts this, so it
gets tests now rather than later.

**Build**
- Answer list: ~500 curated *common* 5-letter words. Keep it small and common —
  `PLAN.md §2`: if a 0.6B model can never solve a game, there is no reward signal to learn
  from.
- Allowed-guess list: macOS ships `/usr/share/dict/words` → 8,506 five-letter words, no
  network needed. Union with the answer list so every answer is a legal guess.
- `score(guess, secret) -> str` returning 5 symbols. Use `=` green, `-` yellow, `.` grey
  (single-token on every candidate tokenizer, and letter-spaceable).
- `Game` class: `guesses`, `feedback`, `won`, `over`, `turns_used`, `history()`.
- `InvalidGuess` exception carrying a `.kind`: `bad_length` / `not_alpha` / `not_a_word` /
  `repeat`. **Keep these distinct** — action validity, not feedback comprehension, is the
  dominant failure mode for small models (`PLAN.md §1.1`), and you can't fix what you don't
  measure separately.
- Invalid guesses must **not** consume a turn.
- A `__main__` demo that plays one game and prints the board.

**Scoring must be two-pass.** Greens claim their letters from a `Counter` pool first, then
yellows draw from what remains. A single-pass version over-reports yellows and will pass
casual testing.

**Verify — these four cases are the ones that catch a broken scorer**
```
SPEED vs ERASE  ->  -.--.    both E's yellow, S yellow
ALLOY vs LOYAL  ->  -----    anagram: all yellow, zero greens
LEVEL vs EAGLE  ->  --.-.    2nd L grey (secret has only one L)
GEESE vs THESE  ->  ..===
```
Plus property tests over random pairs: `all-green ⟺ guess == secret`; a green at index `i`
implies `guess[i] == secret[i]`; feedback never claims more of a letter than the secret
contains.

**Exit criteria.** Tests pass. Demo plays a game and loses (that's your floor — headroom
exists).

**Known limitation to note.** web2 contains archaic words (`SKAFF`, `BRUGH`, `RUPIE`), so
"valid word" is *weaker* than real Wordle. Permissive, not wrong — but your valid-word
metric will read better than it should. Intersect with a frequency list if it bothers you.

---

## Task 2 — Prompt rendering + mid-game state sampler

**Goal.** Turn a `Game` into a prompt string, and generate the training distribution.
These belong together: the sampler produces what the renderer displays.

**Build**
- `render(game) -> str`. **Letter-space everything**: `C R A N E` tokenizes to exactly 5
  tokens on every candidate model, whereas `CRANE` fuses to 2. This is measured, not
  assumed (`PLAN.md §3.3`), and it's close to free.
- **Pre-digest the constraints** into plain English rather than making the model re-derive
  them:
  ```
  Green (correct position):    _ R _ _ E
  Yellow (in word, wrong pos): A
  Grey (not in word):          C, N, S, T
  Already guessed:             CRANE, SWEAT
  Do not repeat any word above.
  ```
  charbull found this beat raw symbol strings, and that an explicit "do not repeat"
  instruction beat a repetition *penalty*.
- `parse_guess(text) -> str | None` extracting from `<guess>WORD</guess>`. Must tolerate
  whitespace and case. This is your action parser — parser bugs are reward bugs.
- `sample_state(n_prior: int) -> Game`: a game with `n_prior` plausible prior guesses
  already applied. Draw `n_prior` uniformly from 0–4.

**Why the sampler matters most.** Three independent groups found the 0–4 prior-guess
curriculum to be the single biggest lever. Training only on turn-1 prompts barely learns.

**Exit criteria.** Print 5 sampled states at each `n_prior` and read them. They should be
unambiguous to *you* at a glance. Round-trip test: `parse_guess(render_of_a_guess) == guess`.

---

## Task 3 — Baselines and eval harness

**Goal.** Establish the floor and the ceiling before any ML, so that later numbers mean
something.

**Build**
- `constraints_from(game)`: derive `fixed_positions` (greens), `excluded_positions`,
  `required_letters`, `excluded_letters` from history.
- `is_consistent(guess, game) -> bool`: could this guess still be the answer given all
  prior feedback? Note the known gap: duplicate-letter *counts* are only approximate.
- `RandomPlayer` — any legal word. **The floor.**
- `SolverPlayer` — filter the answer list to consistent candidates, pick by max expected
  information (or just pick the first). **The ceiling**; a good solver gets ~99% in ≤6.
- `evaluate(player, n_games, seed) -> dict` reporting: solve rate, mean turns when solved,
  invalid-guess rate broken out **by kind**, repeat rate.

**Exit criteria.** A printed table with both baselines. You now know the interval any model
result has to be interpreted inside. `is_consistent` gets reused by the reward in task 5,
so test it directly.

---

## Task 4 — THE GATE: pass@k viability check

**This is the most important task in the project.** It answers whether the whole thing is
possible, costs nothing, and runs on your Mac.

**Build**
- Load a candidate on MPS via `transformers`, sample `k=16` completions per state at
  temperature 1.0.
- For each state compute `#solved`. Report the **fraction of states with
  `0 < #solved < k`** — states that are neither always-solved nor never-solved.
- Sweep: 4 candidate models × 3–4 prompt formats × `n_prior ∈ {0..4}`.

**Candidates** (measured parameter breakdown in `PLAN.md §3.3`):
`Qwen3-0.6B` (default — all prior art uses Qwen, so failures are diagnosable) ·
`gemma-3-1b-it` (fine; costs 1.7× logits memory, and no system role) ·
`SmolLM2-360M-Instruct` (49k vocab → cheapest loop, largest affordable `k`) ·
skip `gemma-3-270m-it` (62% of its params are the embedding table — ~100M effective).

**The decision rule**
- `>20%` non-degenerate → proceed to task 5.
- `~0%` → **stop.** No algorithm creates a signal that isn't there. Make the task easier:
  shrink the answer list, raise `n_prior` so states start closer to solved, add few-shot
  examples, or go straight to task 7 (SFT) and re-gate.

**Also sweep the prompt format here.** Dr. GRPO measured Qwen2.5-Math-7B scoring **0.0**
with one template and **38.2** with none — applying a template destroyed the capability
before RL could rebuild it. Pick by pass@16.

**Exit criteria.** A table of non-degenerate fractions, and a chosen (model, format) pair
with evidence behind it.

---

## Task 5 — Reward function

**Goal.** A verifiable reward that resists hacking.

**Build — gate, don't sum.** Additive rewards get hacked.
```
if not format_ok:                  R = -0.1     # must be the worst outcome
elif not valid_5letter_dict_word:  R =  0.0
elif solved:                       R =  1.0 + 0.5 * (1 / turns_used)
else:                              R =  0.3 * constraint_consistency
```
- `constraint_consistency` reuses `is_consistent` from task 3. It's a **dense, verifiable
  intermediate signal** — the thing RAGEN identifies as missing when reasoning fails to
  emerge. Win/lose alone is far too sparse at ~1–15% success.
- **Format failure must be unequivocally worst.** charbull's model learned to emit empty
  strings because lazy inaction was cheaper than a thoughtful wrong guess.
- **Set β = 0. No KL term, no reference model.** With a rule-based verifier there's no
  reward model to over-optimize (DAPO and Dr. GRPO both argue this). Saves a model in
  memory and a whole class of bugs.

**Verify — write tests that actively try to cheat**
- Empty string, whitespace, gibberish, a 4-letter word, a repeated guess → all score
  ≤ a genuine wrong guess.
- A repeat can never *increase* reward.
- Solving in 3 beats solving in 5.

**Exit criteria.** Reward tests pass, including the hack attempts.

---

## Task 6 — Modal harness

**Goal.** Reproducible remote execution with checkpoints that survive preemption.

**Build**
- **First: run Modal's `grpo_trl.py::train_vllm_colocate_mode` completely unmodified on an
  A10.** Validates account, image build, GPU access, and Volume writes before any of your
  code is in the picture.
- `modal_app.py`: image (pin `trl`/`vllm`/`transformers` exactly — they break each other),
  a `modal.Volume` for checkpoints, entrypoints for `eval` / `sft` / `train`.
- Port task 3's `evaluate()` to run on an A10 and confirm it matches your local numbers.

**Non-negotiable Modal facts**
- Default timeout is **300s**; max is 24h. Set `timeout=` or your run dies in 5 minutes.
- **Every GPU function is preemptible with no opt-out.** Checkpoint from step 1.
- Use `.spawn(...).get()`, **not `.remote()`** — `.remote()` calls expire at 24h.
- `modal run --detach`, or it dies when you close your laptop.
- Put `add_local_python_source` **last** in the image chain or every edit rebuilds vLLM.
- Logs are retained 1 day on the free tier → **wire up W&B now.**
- Score the reward in-process. Don't spawn a Sandbox per rollout.

**Exit criteria.** Remote eval reproduces local eval. A checkpoint survives a killed-and-
restarted run.

---

## Task 7 — SFT warm-start

**Goal.** Get the model over the format/validity wall so RL can work on strategy.

Two independent groups report this is **mandatory** below ~7B, and you're at ~0.6B. Even a
7B "VERY often" emitted non-words before SFT.

**Build**
- Generate 500–2000 traces from a strong teacher **over randomized mid-game states** (task
  2's sampler — same curriculum). RLDiary spent ~$35 on 500 traces.
- Filter to traces that are both correct *and* constraint-consistent.
- Train on an A10. Small LoRA rank is fine.
- **Re-run task 4's gate afterwards.** It should improve substantially.

**Exit criteria.** >95% valid-word format compliance, and a higher non-degenerate fraction
than the base model.

---

## Task 8 — RLOO from scratch

**Goal.** Your own policy-gradient loop, single-turn, no RL framework.

**Why RLOO and not GRPO.** Dr. GRPO proved RLOO ≡ Dr. GRPO up to a `G/(G-1)` learning-rate
factor. So the *simple* algorithm is the corrected state of the art, and GRPO's extra terms
(`/std`, `1/|o_i|`) are exactly the parts later identified as bugs. ~80 lines. Every
subsequent paper becomes a small, testable diff against this.

**Build**
```
Â_i = (k·R_i − Σ_j R_j) / (k − 1)                                    # RLOO
L   = − Σ_i Σ_t mask_{i,t} · Â_i · log π_θ(y_{i,t}) / (B · MAX_TOKENS)
```
- No ratio, no clip, no KL, no value net, no reference model.
- `μ = 1`: generate → one gradient step → regenerate. Fully on-policy, so the importance
  ratio is identically 1 and clipping would be dead code. Measured better *and* simpler.
- **Dynamic sampling from day one:** drop groups where `#solved ∈ {0, k}`, oversample to
  refill. This was DAPO's single largest ablation gain (+8 points).
- `k = 4–8`. More prompts × fewer samples each beats fewer prompts × many, but you need
  `k > 1`.
- Grad-clip global norm 0.5. Adam `eps=1e-5` (not PyTorch's 1e-8). LR ~1e-6.
- **Generate with HF `.generate()` in-process and reuse the sampling logprobs.** This makes
  the train/inference logprob mismatch zero by construction. Add vLLM only when generation
  is measurably the bottleneck, and add Truncated Importance Sampling at the same time.

**The denominator matters.** Use a fixed global constant: `(per_tok_loss * mask).sum() /
(B * MAX_TOKENS)`, **not** `/ mask.sum(-1)`. Dr. GRPO audited trl, OpenRLHF, verl,
SimpleRL-Zero and Open-Reasoner-Zero and found all of them length-biased here. Getting it
wrong pushes correct answers shorter and wrong answers longer — and you'll misread the
resulting length growth as emergent reasoning.

**Exit criteria.** Reward moves on a tiny run. Solve rate on 4-green states >80%.

---

## Task 9 — Instrumentation

**Goal.** See failures before they're fatal. Do this *before* tuning anything.

| Metric | Why |
|---|---|
| **frac. groups with nonzero advantage** | **#1 health metric.** Near zero ⇒ no signal |
| **reward std within group** | earliest collapse warning — std drops ~50 steps before reward does |
| gradient norm | a spike means collapse is already irreversible |
| token entropy over generated positions | entropy collapse; your ceiling is set by starting entropy |
| **entropy of the guess distribution on fixed eval prompts** | *template collapse* — invisible to token entropy |
| does the guess change when the feedback changes? | is it conditioning on clues or pattern-matching? |
| held-out eval solve rate | train reward decorrelates from val accuracy |
| invalid-guess rate + repeat rate, **separately by kind** | the dominant failure mode |
| mean / max response length | length-bias symptom |

**Two assertions to run every single time:** `loss_mask.sum() == expected_generated_tokens`,
and decode-and-print the masked-in tokens for one sample. You should see *only* your
guesses. This is the #1 way to silently waste a week.

---

## Task 10 — Algorithm ablation (the real deliverable)

**Goal.** Measure which corrections in the literature actually matter on your task. Hold
the reward, env, and seeds fixed; change one thing at a time.

| Variant | Diff from task 8 | Hypothesis |
|---|---|---|
| REINFORCE + running-mean baseline | swap the advantage fn | worse than RLOO (higher variance) |
| GRPO as published | `+ /std`, `+ 1/\|o_i\|`, `+ β·KL` | length + difficulty bias become visible |
| Dr. GRPO | drop `std`, drop `1/\|o_i\|` | ≈ RLOO up to LR scale — **verify empirically** |
| + Clip-Higher | `ε_low=0.2, ε_high=0.28` | fixes entropy collapse if you see it |
| + token-level normalizer | batch-token mean vs per-seq mean | stability |
| GSPO-token | sequence-level ratio, per-token advantage | best fit for task 11 |

Calibration from DAPO's own ablation (AIME24): naive GRPO 30 → +overlong filter 36 →
+clip-higher 38 → +soft-overlong 41 → +token-level loss 42 → +dynamic sampling **50**.

**This table is worth more than a win-rate number.** State of the art on a *7B* is 14.5%
full-game solve rate; you will not beat that at 0.6B. "I implemented five estimators from
scratch and measured which corrections matter" is the achievable, defensible result.

---

## Task 11 — Multi-turn rollouts

**Goal.** Real 6-turn games instead of single flattened states.

**Build**
- Start with **trajectory-level advantage**: one scalar per game, broadcast to every action
  token. Wordle is ≤6 turns with near-terminal reward, so per-turn credit is a refinement,
  not a prerequisite.
- **Mask environment tokens.** Train on env-produced tokens and you're doing supervised
  learning on the environment's output — the model learns to hallucinate feedback.
- **Do not use a chat template.** Build one flat string and record `(start, end)` token
  spans for your own generations as you append them. You control the string, so there's no
  template to fight. (Gemma silently folds the system role into the first user turn —
  exactly the quirk that corrupts delta-based masking.)
- Expect instability: RAGEN found vanilla PPO/GRPO "achieve early gains in agent settings
  but often collapse," on a Qwen2.5-**0.5B** setup — your exact scale. Mitigate with
  variance-based trajectory filtering (keep the top 25–75% by reward std).

**Then, only if a metric demands it:** per-turn advantages via GSPO-token (sequence-level
importance weight for stability, per-token advantage for credit — ~3 lines).

**Exit criteria.** Measurable lift in full-game solve rate over the SFT baseline on
held-out words.

---

## The three things most likely to sink this

1. **A loss mask leaking environment tokens.** Assert it, decode it, look at it.
2. **No learning signal** — the model never solves a game. That's what task 4 is for.
3. **A loss normalizer copied from a framework that had it wrong.** They mostly did.

## Budget

~$150–400 total. Tasks 1–5 are free. Calibration: RLDiary got a 3× improvement on a 7B for
**$25** of GPU. If a single experiment costs much more than that, it's too big.
Apply for [Modal academic credits](https://modal.com/academics) ($10k) before starting.
