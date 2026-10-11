"""
Modal harness for rl-wordle (PLAN.md §3.1, TASKS.md task 6).

Right now this holds only the smoke test: does our account, image build, A10 GPU and Volume
work before any real code depends on them? Entrypoints for eval / sft / train get added here as
those tasks arrive.

Run:  uv run modal run modal_app.py            (smoke)
Cost: about $0.10 (A10 is $0.000306/s, verified on modal.com/pricing 2026-10-02).

Facts this file relies on (PLAN.md §3.1, re-checked against modal docs 2026-10-02):
  - default function timeout is 300s, so every GPU function sets `timeout=` explicitly;
  - every GPU function is preemptible, so checkpoints go to a Volume from the first step;
  - use `.spawn(...).get()`, not `.remote()`, for anything long (remote calls expire at 24h).
"""

from pathlib import Path

import modal

app = modal.App("rl-wordle")

# Checkpoints and the HF cache live here, so they survive preemption and container restarts.
CKPT = "/ckpt"  # str, not Path: Modal's `volumes=` dict is typed to take str or PurePosixPath keys
vol = modal.Volume.from_name("rl-wordle-ckpt", create_if_missing=True)

MODEL = "Qwen/Qwen3-0.6B"

# Modal keeps logs for 1 day on the free tier (PLAN.md §3.1), so W&B is the only durable record of a
# run. The secret already exists in this workspace; required_keys fails fast if it lacks the key.
WANDB_PROJECT = "rl-wordle"
wandb_secret = modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])


def init_wandb(name: str, config: dict, run_id: str | None = None):
    """
    Start (or resume) a W&B run. Call this inside a Modal function, after the secret is attached.

    Every GPU function can be preempted and restarted (PLAN.md §3.1). Without a stable `run_id`,
    each restart opens a NEW W&B run and one training curve shatters into pieces. Pass the same
    id on every attempt (e.g. stored next to the checkpoint on the Volume) and `resume="allow"`
    stitches the attempts into one run.
    """
    import wandb

    return wandb.init(
        project=WANDB_PROJECT, name=name, config=config, id=run_id, resume="allow"
    )

image = (
    # 3.13 to match pyproject's requires-python, so remote and local run the same language version.
    modal.Image.debian_slim(python_version="3.13")
    # Pinned to the versions in uv.lock: a remote number is only comparable to a local one if
    # the libraries match.
    .uv_pip_install(
        "torch==2.13.0", "transformers==5.14.1", "accelerate==1.14.0", "wandb==0.30.0"
    )
    # Model weights land on the Volume, so the second run skips the download.
    .env({"HF_HOME": f"{CKPT}/hf"})
    # LAST on purpose: these layers change on every code edit, and anything after them would
    # rebuild each time (PLAN.md §3.1: add_local_python_source last).
    .add_local_python_source("wordle_rl", "evaluate", "gate", "constrained", "sft")
    # Inside the package (/root/wordle_rl), because wordle.py finds both lists via
    # Path(__file__).with_name. Both are required: wordle.py raises at import if either is missing.
    # Note: I did not check if add_local_python_source ships .txt files, so we add them by hand.
    # The parity_check entrypoint shows if the lists are the right size (8,511 allowed words).
    .add_local_file(
        "environments/wordle_rl/wordle_rl/answers.txt", "/root/wordle_rl/answers.txt"
    )
    .add_local_file(
        "environments/wordle_rl/wordle_rl/allowed.txt", "/root/wordle_rl/allowed.txt"
    )
    # The SFT dataset is reproducible from a seed (python -m sft.gen_traces), 1.8 MB, so it ships
    # with the image rather than living on the Volume.
    .add_local_file("data/sft_train.jsonl", "/root/data/sft_train.jsonl")
)


@app.function(
    image=image, gpu="A10", volumes={CKPT: vol}, secrets=[wandb_secret], timeout=900
)
def smoke() -> dict:
    """Prove GPU access, a Volume write, a model load + generation, and that our code ships."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info: dict = {"cuda": torch.cuda.is_available()}
    info["gpu"] = torch.cuda.get_device_name(0)

    # 1. Volume write. commit() is what makes it visible to other containers.
    marker = Path(CKPT) / "smoke" / "hello.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("written by smoke()")

    # 2. Load the model on the GPU and generate a few tokens.
    tok = AutoTokenizer.from_pretrained(MODEL)
    assert tok is not None  # narrows the type for ty; from_pretrained never returns None at runtime
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto")
    model.to("cuda")  # ty: ignore[invalid-argument-type]
    ids = tok("The word of the day is", return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=8, do_sample=False)  # ty: ignore[invalid-argument-type]
    info["generated"] = tok.decode(out[0], skip_special_tokens=True)

    # 3. Our code made it into the image. ALLOWED size is the portability check: locally it is
    #    8,511 (macOS /usr/share/dict/words); a smaller number here means the remote image has no
    #    system dictionary and wordle.py silently fell back to the answer list alone.
    import wordle_rl.wordle as wordle

    info["answers"] = len(wordle.ANSWERS)
    info["allowed"] = len(wordle.ALLOWED)

    # 4. W&B: one run with a couple of metrics, to prove the secret, network and dashboard work.
    run = init_wandb("smoke", {"model": MODEL, "gpu": info["gpu"]})
    run.log({"smoke/allowed": info["allowed"], "smoke/answers": info["answers"]}, step=0)
    info["wandb_url"] = run.url
    run.finish()

    vol.commit()
    return info


@app.function(image=image, volumes={CKPT: vol}, timeout=120)
def read_back() -> str:
    """Runs in a different container, so a successful read proves the write persisted."""
    vol.reload()
    return (Path(CKPT) / "smoke" / "hello.txt").read_text()


def _baseline_table() -> dict:
    """Floor and ceiling at every n_prior, fixed seed. Deterministic, so local == remote or a bug."""
    from evaluate import evaluate, random_player, solver_player

    players = {"floor": random_player, "ceiling": solver_player}
    return {
        f"{name}/prior{n}": evaluate(make, n_games=200, n_prior=n, seed=0)
        for name, make in players.items()
        for n in range(5)
    }


@app.function(image=image, timeout=300)
def parity() -> dict:
    """CPU only: run the env self-tests and the baselines inside the Modal image."""
    import wordle_rl.wordle as wordle

    wordle.run_tests()  # raises on any failure, which fails the whole call
    return _baseline_table()


@app.local_entrypoint()
def parity_check():
    """Does a remote evaluation reproduce the local one? (TASKS.md task 6 exit criterion.)"""
    import json

    # sort_keys + json handles the NaN mean_turns the floor player produces (nan != nan in ==).
    local = json.dumps(_baseline_table(), sort_keys=True)
    remote = json.dumps(parity.remote(), sort_keys=True)
    print("PARITY OK: remote == local" if local == remote else "PARITY MISMATCH")
    if local != remote:
        print("local :", local[:400])
        print("remote:", remote[:400])


@app.local_entrypoint()
def main():
    # spawn().get(), not remote(): the habit to build now, since training runs must not use remote().
    info = smoke.spawn().get()
    for k, v in info.items():
        print(f"{k:>10}: {v}")
    print(f"{'read_back':>10}: {read_back.remote()}")


# --- SFT -----------------------------------------------------------------------------------------
# Full fine-tune of Qwen3-0.6B on the solver traces (data/sft_train.jsonl). These are STARTING values,
# typical for a small full fine-tune, not tuned: read the loss curve on W&B before changing them.
# Batch 4, not 16: logits are B x T x V floats and Qwen's vocab is ~152k, so batch 16 at ~300
# tokens needed ~3 GB per logits copy and ran out of memory on the 22 GB A10 (measured 2026-10-03).
SFT_BATCH_SIZE = 4
SFT_EPOCHS = 2
SFT_LR = 1e-5
SFT_WARMUP_STEPS = 20
SFT_MAX_GRAD_NORM = 1.0


@app.function(
    image=image,
    gpu="A10",
    volumes={CKPT: vol},
    secrets=[wandb_secret],
    timeout=3600,
    # A preempted or failed attempt restarts from scratch. Fine for a run of minutes; the RL runs
    # (hours) get real mid-run resume, because losing those costs money.
    retries=modal.Retries(max_retries=2, initial_delay=0.0),
    single_use_containers=True,
)
def sft(
    run_name: str = "sft-solver-v1",
    limit: int = 0,
    init_model: str = MODEL,
    hard_examples_name: str = "",
    hard_repeats: int = 1,
) -> dict:
    """Train on the traces and save the bf16 weights to the Volume for the post-SFT gate."""
    import json
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sft.train_sft import build_example, train

    vol.reload()
    tok = AutoTokenizer.from_pretrained(init_model)
    assert tok is not None and tok.pad_token_id is not None
    # fp32 weights + bf16 autocast forward: bf16 weights would round away lr-sized updates.
    model = AutoModelForCausalLM.from_pretrained(init_model, dtype=torch.float32)
    model.to("cuda")  # ty: ignore[invalid-argument-type]
    model.train()

    examples = []

    def append_row(row: dict) -> None:
        messages = row["messages"]
        target = f"<guess>{row['guess']}</guess>"
        examples.append(build_example(tok, messages[1]["content"], target))

    with open("/root/data/sft_train.jsonl") as f:
        for line in f:
            append_row(json.loads(line))

    if hard_examples_name:
        hard_path = Path(CKPT) / "sft" / hard_examples_name
        with hard_path.open() as f:
            for line in f:
                row = json.loads(line)
                for _ in range(hard_repeats):
                    append_row(row)

    if limit:
        examples = examples[:limit]

    config = {
        "init_model": init_model, "n_examples": len(examples), "batch_size": SFT_BATCH_SIZE,
        "epochs": SFT_EPOCHS, "lr": SFT_LR, "warmup_steps": SFT_WARMUP_STEPS,
        "max_grad_norm": SFT_MAX_GRAD_NORM, "hard_examples_name": hard_examples_name,
        "hard_repeats": hard_repeats,
    }
    run = init_wandb(run_name, config)

    def log(r: dict) -> None:
        run.log({f"sft/{k}": v for k, v in r.items() if k != "step"}, step=r["step"])

    t0 = time.time()
    records = train(
        model, examples, tok.pad_token_id,
        batch_size=SFT_BATCH_SIZE, epochs=SFT_EPOCHS, lr=SFT_LR,
        warmup_steps=SFT_WARMUP_STEPS, max_grad_norm=SFT_MAX_GRAD_NORM,
        autocast=True, log=log,
    )
    seconds = time.time() - t0

    out = Path(CKPT) / "sft" / run_name
    model.to(torch.bfloat16)  # ty: ignore[invalid-argument-type]
    model.save_pretrained(out)  # bf16 is enough for inference; half the size
    tok.save_pretrained(out)
    vol.commit()

    summary = {
        "steps": len(records),
        "first_loss": records[0]["loss"],
        "last_loss": records[-1]["loss"],
        "hard_examples_name": hard_examples_name,
        "hard_repeats": hard_repeats,
        "max_grad_norm_seen": max(r["grad_norm"] for r in records),
        "seconds": round(seconds, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        "saved_to": str(out),
        "wandb_url": run.url,
    }
    run.finish()
    return summary


@app.local_entrypoint()
def sft_run(
    run_name: str = "sft-solver-v1",
    init_model: str = MODEL,
    limit: int = 0,
    hard_examples_name: str = "",
    hard_repeats: int = 1,
):
    """`limit=64` for a quick smoke run; no limit for the real one (use --detach)."""
    for k, v in sft.spawn(
        run_name, limit, init_model, hard_examples_name, hard_repeats
    ).get().items():
        print(f"{k:>20}: {v}")

@app.function(image=image, gpu="A10", volumes={CKPT: vol}, timeout=1800)
def mine_sft_errors(model_dir: str, out_name: str, per_prior: int = 100, k: int = 4) -> dict:
    import json
    import random

    import gate as G
    from wordle_rl.render import FEWSHOT_SECRETS
    from wordle_rl.wordle import (
        ALLOWED,
        ANSWERS,
        WORD_LEN,
        Game,
        constraints_from,
        is_consistent,
        split_answers,
    )

    _, eval_words = split_answers(
        reserved=[w for w in FEWSHOT_SECRETS if w in ANSWERS]
    )
    eval_words = set(eval_words)
    candidate_words = sorted(
        word for word in ALLOWED
        if word not in eval_words and len(word) == WORD_LEN and word.isalpha()
    )
    hard_prompt_count = 0

    vol.reload()
    tok, model = G.load(model_dir)

    groups = {n: [] for n in range(5)}
    with open("/root/data/sft_train.jsonl") as f:
        for line in f:
            row = json.loads(line)
            groups[row["n_prior"]].append(row)

    rng = random.Random(0)
    selected = []
    for rows in groups.values():
        selected.extend(rng.sample(rows, min(per_prior, len(rows))))

    hard_rows = []
    failed_outputs = 0
    for row in selected:
        user_text = row["messages"][1]["content"]
        prompt = G.build_prompt(tok, user_text)
        outputs = G.sample_k(tok, model, prompt, k, 1.0, 128)

        invalid = False
        for output in outputs:
            raw = G.parse_lenient(output)
            if (
                G.parse_tag(output) is None
                or raw is None
                or len(raw) != G.WORD_LEN
                or raw not in G.ALLOWED
            ):
                invalid = True
                failed_outputs += 1

        if invalid:
            hard_prompt_count += 1

            if row["n_prior"] == 0:
                hard_rows.append(row)
                continue

            game = Game(row["secret"])
            for previous_guess in row["prior_guesses"]:
                game.guess(previous_guess)

            constraints = constraints_from(game)
            alternatives = [
                word for word in candidate_words
                if word != row["guess"]
                and word not in game.guesses
                and is_consistent(word, constraints)
            ]
            for target in rng.sample(alternatives, min(2, len(alternatives))):
                hard_rows.append({**row, "guess": target})

    out = Path(CKPT) / "sft" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in hard_rows:
            f.write(json.dumps(row) + "\n")

    vol.commit()
    return {
        "prompts_checked": len(selected),
        "hard_prompt_count": hard_prompt_count,
        "hard_targets": len(hard_rows),
        "invalid_outputs": failed_outputs,
        "saved_to": str(out),
    }

# --- Gate on a trained model -----------------------------------------------------------------------
@app.function(
    image=image, gpu="A10", volumes={CKPT: vol}, secrets=[wandb_secret], timeout=7200
)
def gate_sweep(model_dir: str, header: str, n_states: int = 100, k: int = 16, constrained: bool = False) -> str:
    """The same sweep as the local base-model gate (strata 0-4, T=1.0, plain prompt), on the GPU."""
    import gate as G

    vol.reload()  # pick up weights written by an earlier container
    tok, model = G.load(model_dir)
    cols = ("tag", "len5", "valid", "repeat", "consistent", "solve", "nondegen", "nd_reward", "nd_consist")
    lines = [header, "", "| prior | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for n in range(5):
        r = G.gate(model_dir, n, n_states, k, 1.0, 128, 0, tok, model, constrained=constrained)
        lines.append(f"| {n} | " + " | ".join(f"{r[c]:.0%}" for c in cols) + " |")

        for kind, samples in r["examples"].items():
            if samples:
                lines.append(f"\nExamples: prior={n}, type={kind}")
                lines.extend(f"- {sample!r}" for sample in samples)

    table = "\n".join(lines) + "\n"

    mode_suffix = "-constrained" if constrained else ""
    size_suffix = "" if n_states == 100 else f"-diagnostic-{n_states}"
    artifact = f"gate{mode_suffix}{size_suffix}.md"
    (Path(model_dir) / artifact).write_text(table)
    vol.commit()
    return table


@app.local_entrypoint()
def gate_run(run_name: str = "sft-v0-nopunct", n_states: int = 100, k: int = 16, constrained: bool = False):
    """Gate a saved SFT model and write results/gate_<run_name>.md (tied to the current commit)."""
    import subprocess

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True
    ).stdout.strip()
    header = (
        f"# Gate: {run_name} (A10, k={k}, {n_states} states per stratum, T=1.0, plain prompt)\n\n"
        f"Code: commit `{commit}`" + (" (with uncommitted changes)" if dirty else "")
    )
    table = gate_sweep.spawn(f"{CKPT}/sft/{run_name}", header, n_states, k, constrained).get()

    mode_suffix = "-constrained" if constrained else ""
    size_suffix = "" if n_states == 100 else f"-diagnostic-{n_states}"
    out = Path("results") / f"gate_{run_name}{mode_suffix}{size_suffix}.md"
    out.write_text(table + "\n")
    print(table)
    print(f"\nwrote {out}")
