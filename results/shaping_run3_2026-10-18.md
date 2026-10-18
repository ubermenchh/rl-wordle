# Run 3: a bonus for earlier replies that fit the feedback (2026-10-18)

Hypothesis (from `results/reasoning_check_2026-10-18.md`): the reward scored only the last reply, so consistent earlier replies earned nothing. Run 3 adds
`shaping * (earlier replies that fit the feedback) / (number of earlier replies)` with `shaping = 0.2` to the training reward (`episode_reward`, `WordleConfig.shaping`).
The eval source has no shaping, so eval rewards compare with earlier runs.

Setup: run `rl-wordle-1018-080914`, new Modal account. Same SFT start (step 62), same config and the same task stream as run 2 (`seed = 1`). 40 steps, evals at step 0 and 40.
Baseline for the comparison: run 2 at step 40 (`rl-wordle-1017-110828`, same seed, no shaping). Both evaluated on the same 400 held-out tasks.
Cost: about $6 for the run (my estimate from the monthly billing summary), about $0.6 for each 400-game eval.

Pass rule, set before the run: among legal replies, the share that fit all feedback rises by 10 points or more over the baseline at step 40, and the illegal-ended share does not rise.

| 400 held-out games, step 40 | Run 2 (baseline) | Run 3 (shaping 0.2) |
|---|---|---|
| Wins | 35 (8.8%) | 25 (6.2%) |
| Lost (legal replies, 6 turns used) | 205 (51.2%) | 207 (51.7%) |
| Illegal-ended games | 160 (40.0%) | 168 (42.0%) |
| Legal replies that fit all feedback | 212 of 1,185 (17.9%) | 210 of 1,186 (17.7%) |
| Word obeys the model's own line | 19% | 30% |
| Mean reply length | 112 characters | 102 characters |
| Held-out reward, 100 games (step 40) | 0.107 | 0.121 |

Result: the rule is not met. The consistency share is the same (17.9% against 17.7%, a rise of 0 points, and 10 were needed). The illegal-ended share is 2 points higher, which is within the noise. Wins are lower (25 against 35, about 1.3 standard errors), also within the noise.

What changed instead:

- The model learned to shorten its "Absent" list. In 861 of 1,117 replies the stated absent letters are a strict subset of the true ones (49 of 1,116 in the baseline). Example: `Absent: B, E.` where the true line has 8 letters. The "facts line exactly right" share fell from 74% to 18% for this reason.
- A shorter list leaves fewer chances for the word to break its own line, so "word obeys the model's own line" rose from 19% to 30%. The word is no more consistent with the real feedback.
- Late in the run, entropy (0.099 against 0.079), grad norm (0.67 against 0.53) and the masked share (0.005 against 0.002) were higher than in run 2. No collapse.
- Training reward is not comparable between the runs, because run 3 includes the bonus.

What it means:

- A bonus of 0.2 on earlier replies did not raise consistency in 40 steps. Run 2 also stayed near 18% at step 80, so more steps of the old reward did not help either.
- Not tested, and I cannot say which is the cause: a larger bonus, more steps, a dense reward per reply (not averaged), or a limit of the model: choosing a word that satisfies many constraints at once may be hard for 1.7B in a single pass.
- One seed, one 400-game eval for each run. A difference smaller than about 6 points of consistency would not show.
