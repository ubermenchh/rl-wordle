"""
The pass@k visibility gate.

Measures whether a model produces a LEARNABLE signal, not whether it's good.
The number that matters is `nondegen`: the fraction of states where some but not
all of k samples solve it. Groups that are all-right or all-wrong have zero
advantage under any policy-gradient estimator, so they contribute no gradient/

If nondegen is ~0 at every stratum, stop. No algorithm creates signal that isn't
there.
"""

from __future__ import annotations

import argparse
import random
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constrained import sample_words
from wordle_rl.render import (
    FEWSHOT_SECRETS,
    SYSTEM,
    fewshot_messages,
    parse_lenient,
    parse_tag,
    render,
)
from wordle_rl.reward import reward
from wordle_rl.wordle import (
    ALLOWED,
    ANSWERS,
    WORD_LEN,
    Game,
    constraints_from,
    is_consistent,
    sample_state,
    split_answers,
)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def load(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
    model.to(DEVICE) # ty: ignore[invalid-argument-type]
    model.eval()
    return tokenizer, model

def build_prompt(tokenizer, user_text: str, fewshot: bool = False) -> str:
    """Apply the chat template, coping with families that reject a system role."""
    msgs = [{"role": "system", "content": SYSTEM}]
    if fewshot:
        msgs += fewshot_messages()
    msgs.append({"role": "user", "content": user_text})
    kwargs = {"tokenize": False, "add_generation_prompt": True}

    try:
        return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except Exception:
        pass
    try:
        return tokenizer.apply_chat_template(msgs, **kwargs)
    except Exception:
        # No system role (Gemma): fold SYSTEM into the first user turn and keep
        # the few-shot turns, rather than dropping them.
        rest = [dict(m) for m in msgs[1:]]
        rest[0]["content"] = f"{SYSTEM}\n\n{rest[0]['content']}"
        return tokenizer.apply_chat_template(rest, **kwargs)


@torch.no_grad()
def sample_k(tokenizer, model, prompt: str, k: int, temp: float, max_new: int) -> list[str]:
    ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **ids,
        do_sample=True,
        temperature=temp,
        top_p=1.0,
        top_k=0,
        num_return_sequences=k,
        max_new_tokens=max_new,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen = out[:, ids["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)

def group_signal(texts: list[str], game: Game) -> dict:
    scored = [reward(t, game) for t in texts]
    rewards = [r for r, _ in scored]
    kinds = [kind for _, kind in scored]

    solved = kinds.count("solved")
    n_consistent = sum(kind in ("consistent", "solved") for kind in kinds)

    return {
        "rewards": rewards,
        "solved": solved,
        "nondegen_solved": 0 < solved < len(texts),
        "nondegen_reward": len(set(rewards)) > 1,
        "nondegen_consistent": 0 < n_consistent < len(texts),
    }

def gate(
    model_id: str,
    max_prior: int,
    n_states: int,
    k: int,
    temp: float,
    max_new: int,
    seed: int,
    tokenizer=None,
    model=None,
    fewshot: bool = False,
    constrained: bool = False,
) -> dict:
    if tokenizer is None:
        tokenizer, model = load(model_id)
    rng = random.Random(seed)
    _, eval_words = split_answers(reserved=[w for w in FEWSHOT_SECRETS if w in ANSWERS])

    examples = {
        "missing_tag": [],
        "wrong_length": [],
        "not_allowed": [],
    }
    tagged = len5 = valid = consistent = repeat = solved = total = used = 0
    nondegen = 0
    nd_reward = nd_consist = 0

    for _ in range(n_states):
        g = sample_state(max_prior, rng, secret=rng.choice(eval_words))
        c = constraints_from(g)
        used += g.turns_used

        if constrained:
            pool = [w for w in sorted(ALLOWED) if w not in g.guesses]
            words = sample_words(tokenizer, model,
                                 build_prompt(tokenizer, render(g), fewshot) + "<guess>",
                                 pool, k, temp)
            texts = [f"<guess>{w}</guess>" for w in words]
        else:
            texts = sample_k(tokenizer, model,
                             build_prompt(tokenizer, render(g), fewshot), k, temp, max_new)

        sig = group_signal(texts, g)
        nd_reward += sig["nondegen_reward"]
        nd_consist += sig["nondegen_consistent"]

        hits = 0
        for t in texts:
            total += 1
            tag = parse_tag(t)
            if tag is not None:
                tagged += 1

            raw = parse_lenient(t)

            if tag is None:
                kind = "missing_tag"
            elif raw is None or len(raw) != WORD_LEN:
                kind = "wrong_length"
            elif raw not in ALLOWED:
                kind = "not_allowed"
            else:
                kind = None

            if kind is not None and len(examples[kind]) < 3:
                examples[kind].append({
                    "history": g.history(),
                    "output": t,
                    "parsed": raw,
                })

            if raw is None or len(raw) != WORD_LEN:
                continue
            len5 += 1
            if raw in g.guesses:
                repeat += 1
            if raw not in ALLOWED:
                continue
            valid += 1
            if is_consistent(raw, c):
                consistent += 1
            if raw == g.secret:
                hits += 1
        solved += hits
        if 0 < hits < k:
            nondegen += 1

    return {
        "max_prior": max_prior,
        "tag": tagged / total,
        "len5": len5 / total,
        "valid": valid / total,
        "repeat": repeat / total,
        "consistent": consistent / total,
        "solve": solved / total,
        "nondegen": nondegen / n_states,
        "nd_reward": nd_reward / n_states,
        "nd_consist": nd_consist / n_states,
        "n_used": used / n_states,
        "examples": examples,
    }

def summarize(model_id: str, args) -> dict:
    """
    One row per model. Funnel averaged over constrained strata (1-4); solve
    and nondegen only over priors 3-4, where a single guess can realistically win.
    """
    tokenizer, model = load(model_id)
    rows = [
        gate(model_id, n, args.n_states, args.k, args.temp,
             args.max_new, args.seed, tokenizer, model, fewshot=args.fewshot, constrained=args.constrained)
        for n in range(5)
    ]
    con, deep = rows[1:], rows[3:]

    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return {
        "model": model_id.split("/")[-1],
        "tag": mean(r["tag"] for r in con),
        "len5": mean(r["len5"] for r in con),
        "valid": mean(r["valid"] for r in con),
        "repeat": mean(r["repeat"] for r in con),
        "consist": mean(r["consistent"] for r in con),
        "solve34": mean(r["solve"] for r in deep),
        "ndgn34": mean(r["nondegen"] for r in deep),
        "ndrw": mean(r["nd_reward"] for r in con),
        "ndcn": mean(r["nd_consist"] for r in con),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n-states", type=int, default=30)
    p.add_argument("-k", type=int, default=16)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", default=None, help="comma-separated model ids; prints one summary row each")
    p.add_argument("--fewshot", action="store_true")
    p.add_argument("--constrained", action="store_true")
    args = p.parse_args()

    if args.sweep:
        cols = ("tag", "len5", "valid", "repeat", "consist", "solve34", "ndgn34", "ndrw", "ndcn")
        print(f"on {DEVICE} k={args.k} temp={args.temp} states={args.n_states}")
        print(" ".join(["model".ljust(24)] + [c.rjust(8) for c in cols]))
        print("-" * (24 + 10 * len(cols)))
        for mid in args.sweep.split(","):
            try:
                s = summarize(mid.strip(), args)
            except Exception as e:
                print(f"{mid.strip()[:24].ljust(24)} SKIPPED: {type(e).__name__}: {str(e)[:60]}")
                continue

            print(" ".join([s["model"].ljust(24)]
                + [f"{s[c]:.0%}".rjust(8) for c in cols]), flush=True)

        return

    tokenizer, model = load(args.model)
    probe = build_prompt(tokenizer, render(sample_state(2, random.Random(0))), args.fewshot)
    print(f"{args.model} on {DEVICE} k={args.k} temp={args.temp} "
          f"states={args.n_states} fewshot={args.fewshot} "
          f"constrained={args.constrained} prompt_tokens={len(tokenizer(probe).input_ids)}")

    cols = ("tag", "len5", "valid", "repeat", "consistent", "solve", "nondegen", "nd_reward", "nd_consist")
    print(" ".join(h.rjust(9) for h in ("prior",) + cols))
    print("-" * (10 * (len(cols) + 1)))
    for n in range(5):
        r = gate(args.model, n, args.n_states, args.k, args.temp,
                 args.max_new, args.seed, tokenizer, model, fewshot=args.fewshot, constrained=args.constrained)
        row = [str(n)] + [f"{r[c]:.0%}" for c in cols]
        print("  ".join(v.rjust(9) for v in row), flush=True)

if __name__=="__main__":
    main()
