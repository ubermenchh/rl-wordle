# Rule 5 check of the RL run `rl-wordle-1011-131936` (2026-10-17)

## Sink 1: loss mask

Data: the first trace chunk of the run (`monitors/file/traces/stream/00000.jsonl.zst`), training rows from step 1 on.
5,989 traces, 740,519 tokens with the mask on. Each node has `token_ids`, `mask` and `sampled`.

| Check | Result |
|---|---|
| Mask on in a node that is not from the assistant (system, user) | 0 nodes |
| Mask on in an assistant node that was not sampled | 0 nodes |
| Sampled assistant nodes with some tokens off | 15,679, all the same: the header `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` is off, and all later tokens are on |

Four decoded samples of masked-in tokens, for example:
`Fixed: slot 1 is E, slot 4 is R, slot 5 is Y. Absent: A, B, H, I, L, M, O, P, S, U, V. ENTRY fits all of this.\n<guess>ENTRY</guess><|im_end|>`

Result: only the guesses of the model are in the loss. The feedback and the prompt are not.
The `<think>` header is part of the prompt (thinking off), and it is the same in the evals.
Not checked: the second trace chunk, and the batch that the trainer builds from these nodes.

## Sink 2: learning signal

Measured in `results/rl_wordle_80steps_2026-10-11.md`: 95% to 99% of the groups have a mix of solved and not solved.

## Sink 3: loss normalizer

Resolved configs (`configs/attempt_2/resolved`): `loss.type = ipo`, `eps = 0.3`, `loss_aggregation = token`, `length_weighted_baseline = False`, `length_penalty = None`.
The docs of prime-rl `73140dec` (`docs/algorithms.md`) say that with `token` every `rl` token in the batch has the same weight.
So the denominator is shared by the whole batch, and it is not the length of each answer (the bias that Dr. GRPO found).
It is not the fixed constant of `AGENTS.md`, Rule 5: the denominator changes from batch to batch.
I read the docs and the `IPOLoss` code (it returns a sum), not the line in the trainer that does the division.
