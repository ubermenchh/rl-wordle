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
    .add_local_python_source("wordle", "render", "reward", "evaluate")
    # Beside the modules (/root), because wordle.py finds both lists via Path(__file__).with_name.
    # Both are required: wordle.py raises at import if either is missing.
    .add_local_file("answers.txt", "/root/answers.txt")
    .add_local_file("allowed.txt", "/root/allowed.txt")
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
    import wordle

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
    import wordle

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
