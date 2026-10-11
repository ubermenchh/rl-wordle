"""
Spike for Mark I, step 7 (.jarvis/ROADMAP.md): does prime-rl run on Modal A100 GPUs, and does it
play our taskset? Stop rule: if this does not run in one sitting, we switch to TRL
(.jarvis/DECISIONS.md, 2026-10-11).

Three steps, from cheap to expensive:

    uv run modal run modal_prime_rl.py::check
        CPU only, cents. Builds the image, then checks the install, the word lists, the taskset
        and the pairing of the taskset with its env. No GPU.

    uv run modal run --detach modal_prime_rl.py::spike --stage smoke
        2 x A100-80GB. The prime-rl smoke test (reverse-text, 20 steps). Proves the install, the
        GPUs, vLLM, the trainer and W&B, before our code is in the picture.

    uv run modal run --detach modal_prime_rl.py::spike --stage eval
        2 x A100-80GB. vLLM serves Qwen/Qwen3-1.7B on GPU 0, and prime-rl's `eval` plays 100
        games of our taskset. The numbers compare with results/baseline_qwen3-1.7b_2026-10-11.md.

Where to watch (the spike prints the tunnel URL in the logs):
    - the prime-rl dashboard, through a Modal tunnel (live rollouts, traces, logs);
    - W&B, project rl-wordle;
    - the Modal dashboard, modal.com/apps/ubermenchh (logs, GPU use, cost).

Facts this file relies on (checked 2026-10-11):
    - prime-rl needs Python ~=3.12 and an NVIDIA GPU; its lock uses torch with CUDA 13.0 and a
      prebuilt flash-attn wheel (prime-rl pyproject.toml);
    - Modal GPU hosts run driver 580.95.05 with CUDA API 13.0 (modal.com/docs/guide/cuda);
    - prime-rl main (73140dec, 2026-10-10) uses verifiers commit 2cb0a6d33a, the one that
      pyproject.toml pins for our local work.
Not checked: that the install works in this image, and that the dashboard answers through the
tunnel. This spike finds out.
"""

import contextlib
import glob
import os
import subprocess
import threading
import time
import urllib.request

import modal

app = modal.App("rl-wordle-prime-rl")

# Pinned. When this changes, change the `verifiers` rev in pyproject.toml to the commit of the
# deps/verifiers submodule of the new prime-rl commit.
PRIME_RL_REV = "73140dec358031ed3233002202eec5f568220790"
CUDA_IMAGE = "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04"  # The one of prime-rl's Dockerfile.cuda.
APP_DIR = "/app"  # The prime-rl checkout and its .venv.
BIN = f"{APP_DIR}/.venv/bin"  # Entry points: rl, sft, inference, eval, dashboard.
OUT = "/outputs"  # A Volume. Run folders and the HF cache survive the container.
MODEL = "Qwen/Qwen3-1.7B"
DASHBOARD_PORT = 7788  # prime-rl dashboard (docs/training.md).
RELAY_PORT = 7789  # The tunnel port. See `_start_dashboard`.

vol = modal.Volume.from_name("rl-wordle-ckpt", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])

image = (
    # add_python gives the CUDA image a Python 3.12, as prime-rl requires.
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])  # Removes the verbose logging of the base image.
    .apt_install("git", "curl", "build-essential", "ninja-build", "socat")
    .pip_install("uv==0.11.9")  # The uv version of prime-rl's Dockerfile.cuda.
    .env({"UV_LINK_MODE": "copy", "UV_PROJECT_ENVIRONMENT": f"{APP_DIR}/.venv"})
    .run_commands(
        # The submodules use ssh URLs (git@github.com:). The container has no ssh key, so use
        # https, as prime-rl's scripts/install.sh does.
        'git config --global url."https://github.com/".insteadOf "git@github.com:"',
        f"git clone https://github.com/PrimeIntellect-ai/prime-rl.git {APP_DIR}",
        f"cd {APP_DIR} && git checkout {PRIME_RL_REV}",
        f"cd {APP_DIR} && git submodule update --init -- "
        "deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config",
        # gpu = torch, vllm, ...; flash-attn = a prebuilt wheel (no compile); dashboard = the web UI.
        # --all-packages adds the example envs, as in prime-rl's Dockerfile.cuda.
        # --frozen, not --locked: at prime-rl 73140dec `uv lock --check` fails, also with all 5
        # submodules (measured 2026-10-11, locally and on Modal). The deps/verifiers submodule is a
        # dev build (0.3.2.dev206), and uv.lock was written for another one. --frozen installs the
        # locked versions without checking that the lock is current.
        # The extras that we need on an A100 (Ampere). Found by failure (2026-10-11):
        #   kernels = prime_kernels, which the trainer imports at start;
        #   disagg  = vllm-router, which `inference` starts by name.
        # Left out on purpose: flash-attn-3, flash-attn-cute, flash-mla and quack are kernels for
        # Hopper and Blackwell GPUs. flash-attn-cute pulls flash-attn-4 from git with submodules,
        # and GitHub refused the anonymous clone from the Modal builder (`could not read Username`).
        # If a later run needs one of them, add it alone and look at the error.
        f"cd {APP_DIR} && uv sync --all-packages --extra gpu --extra dashboard --extra flash-attn "
        "--extra disagg --extra kernels --frozen --no-dev",
    )
    # Our package, with its word lists. --no-deps: the venv already has verifiers (the editable
    # submodule), and PyPI verifiers 0.3.1 must not replace it.
    .add_local_dir("environments/wordle_rl", "/pkg/wordle_rl", copy=True)
    .run_commands(
        f"cd {APP_DIR} && uv pip install --python {BIN}/python --no-deps /pkg/wordle_rl"
    )
    # LAST on purpose: these change on every edit and must not rebuild the layers above.
    .add_local_file("configs/wordle_eval.toml", f"{APP_DIR}/wordle_eval.toml")
)


def _env() -> dict:
    """The environment for each command: the venv bin folder is on PATH.

    `rl` starts `inference`, `env-server`, `orchestrator` and `torchrun` by NAME. Under `uv run`
    the venv is on PATH. We call /app/.venv/bin/rl by its full path, so we must do it ourselves.
    The second smoke run died on `No such file or directory: 'inference'` (2026-10-11).
    """
    env = os.environ.copy()
    env["PATH"] = f"{BIN}:{env['PATH']}"
    return env


def _run_name(base: str) -> str:
    """A run name that is new for each try. prime-rl refuses to reuse a run folder that has
    files from an earlier run (FileExistsError, 2026-10-11), and failed tries leave files too.
    The same name is the W&B name, so each try has its own W&B run."""
    return f"{base}-{time.strftime('%m%d-%H%M%S')}"


def _run(cmd: list[str], env: dict | None = None) -> None:
    """Run a command in the prime-rl checkout. Its output goes to the Modal logs."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=APP_DIR, env=env or _env(), check=True)


@app.function(image=image, timeout=1800)
def check() -> None:
    """CPU only. Checks everything that does not need a GPU, before we pay for one."""
    # 1. The three libraries and the versions that we think we have.
    _run(
        [
            f"{BIN}/python",
            "-c",
            "import importlib.metadata as m, verifiers, prime_rl, wordle_rl;"
            "print('verifiers', m.version('verifiers'), verifiers.__file__);"
            "print('prime-rl', m.version('prime-rl'));"
            "print('wordle-rl', m.version('wordle-rl'), wordle_rl.__file__)",
        ]
    )
    # 2. The word lists. On 2026-10-02 the remote list had 900 words and the local list 8,511,
    # with no error (DECISIONS.md). The lists are package data of the wheel now.
    _run(
        [
            f"{BIN}/python",
            "-c",
            "from wordle_rl.wordle import ALLOWED, ANSWERS;"
            "print('answers', len(ANSWERS), 'allowed', len(ALLOWED));"
            "assert len(ANSWERS) == 900 and len(ALLOWED) == 8511, 'word lists differ from local'",
        ]
    )
    # 3. verifiers finds our taskset and pairs it with our env, and the env has the defaults.
    _run(
        [
            f"{BIN}/python",
            "-c",
            "import itertools;"
            "from verifiers.v1.utils.loaders import environment_class, taskset_class;"
            "ts = taskset_class('wordle-rl'); env = environment_class('wordle-rl');"
            "print('taskset', ts.__name__, '| env', env.__name__);"
            "from wordle_rl import WordleConfig, WordleEnvConfig;"
            "p = WordleEnvConfig().player;"
            "print('player', p.harness.id, p.runtime.type);"
            "assert p.harness.id == 'null' and p.runtime.type == 'subprocess';"
            "tasks = list(itertools.islice(ts(WordleConfig()).load(), 3));"
            "print([(t.data.secret, t.data.priors) for t in tasks])",
        ]
    )
    # 4. Every module that the entry points import at start. A missing extra shows up here, on
    # the CPU, and not after the container with 2 GPUs has started. A missing module fails the
    # check. Another error (for example no GPU) is only printed, because a CPU container may not
    # be able to import a module that works on a GPU.
    _run(
        [
            f"{BIN}/python",
            "-c",
            "import importlib;"
            "mods = ['prime_rl.entrypoints.rl', 'prime_rl.trainer.model', 'prime_rl.trainer.rl.train',"
            " 'prime_rl.entrypoints.inference', 'prime_rl.entrypoints.eval',"
            " 'prime_rl.entrypoints.orchestrator', 'prime_rl.entrypoints.dashboard', 'prime_kernels'];"
            "missing = [];\n"
            "for m in mods:\n"
            "    try:\n"
            "        importlib.import_module(m); print('ok     ', m)\n"
            "    except ModuleNotFoundError as e:\n"
            "        print('MISSING', m, '->', e); missing.append(m)\n"
            "    except Exception as e:\n"
            "        print('error  ', m, '->', type(e).__name__, str(e)[:120])\n"
            "assert not missing, missing",
        ]
    )
    # 5. The programs that rl and eval start by name are on PATH.
    _run(
        [
            "python",
            "-c",
            "import shutil;"
            "names = ['inference', 'env-server', 'orchestrator', 'torchrun', 'eval', 'dashboard', 'vllm-router'];"
            "found = {n: shutil.which(n) for n in names}; print(found);"
            "assert all(found.values()), [n for n, p in found.items() if not p]",
        ]
    )
    # 6. The eval config is valid. --dry-run resolves and writes the config, then exits (the help
    # of `eval`, 2026-10-11). A wrong key, for example in sampling.extra_body, fails here, on
    # the CPU, and not after vLLM has started on two GPUs.
    _run([f"{BIN}/eval", "@", f"{APP_DIR}/wordle_eval.toml", "--dry-run", "--run.name", "dry"])
    # Print what the env of our source resolved to: the harness and the runtime come from our
    # defaults, because wordle_eval.toml does not set them.
    resolved = f"{APP_DIR}/outputs/dry/configs/attempt_1/resolved/envs/eval/wordle-rl.json"
    _run(["sh", "-c", f"head -c 3000 {resolved}"])


@contextlib.contextmanager
def _stream_logs(run_name: str):
    """Copy the new lines of the run's log files to the console, with a prefix.

    `rl` writes vLLM, trainer and orchestrator logs to files in the run folder, so without this
    the Modal logs (and `modal run`) show nothing between "Launcher complete" and the end.
    """
    stop = threading.Event()
    positions: dict[str, int] = {}

    def loop() -> None:
        while not stop.is_set():
            for path in sorted(glob.glob(f"{APP_DIR}/outputs/{run_name}/logs/attempt_*/*.log")):
                with open(path, errors="replace") as f:
                    f.seek(positions.get(path, 0))
                    chunk = f.read()
                    positions[path] = f.tell()
                prefix = os.path.basename(path)[: -len(".log")]
                for line in chunk.splitlines():
                    print(f"[{prefix}] {line}", flush=True)
            stop.wait(2)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        time.sleep(3)  # One last pass, so the end of the logs is not lost.
        stop.set()
        thread.join(timeout=10)


def _wait_for(url: str, process: subprocess.Popen, timeout_s: int) -> None:
    """Wait until the vLLM server answers. Stop at once if its process has died."""
    start = time.time()
    while time.time() - start < timeout_s:
        if process.poll() is not None:
            raise RuntimeError(f"the server stopped with code {process.returncode}")
        try:
            urllib.request.urlopen(url, timeout=5)
            print(f"server ready after {time.time() - start:.0f}s", flush=True)
            return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"{url} did not answer in {timeout_s}s")


def _start_dashboard(env: dict) -> list[subprocess.Popen]:
    """Start the prime-rl dashboard and a relay on 0.0.0.0. Best effort: a failure here must
    not stop the run, because W&B and the Modal logs still work.

    The relay is there because I do not know if the dashboard listens on 127.0.0.1 only. A
    Modal tunnel may not reach such a port. socat listens on all interfaces and forwards."""
    processes = []
    try:
        processes.append(subprocess.Popen([f"{BIN}/dashboard", OUT], cwd=APP_DIR, env=env))
        processes.append(
            subprocess.Popen(
                [
                    "socat",
                    f"TCP-LISTEN:{RELAY_PORT},fork,reuseaddr",
                    f"TCP:127.0.0.1:{DASHBOARD_PORT}",
                ]
            )
        )
    except OSError as e:
        print(f"dashboard not started: {e}", flush=True)
    return processes


@app.function(
    image=image,
    gpu="A100-80GB:2",
    volumes={OUT: vol},
    secrets=[wandb_secret],
    # The cost cap: 1 hour of 2 GPUs is about $5 (Modal list price, 2026-10-10). The function
    # stops at the limit, whatever the code does.
    timeout=3600,
)
def spike(stage: str = "smoke", hold_min: int = 5) -> None:
    """Run one stage on 2 A100 GPUs, with the dashboard tunnel and W&B on.

    stage "smoke": the prime-rl smoke test (reverse-text).
    stage "eval":  the baseline eval of our taskset.
    hold_min: minutes to keep the container open after the run, to look at the dashboard.
              Each minute costs about $0.08.
    """
    env = _env()
    env["HF_HOME"] = f"{OUT}/hf"  # Model weights land on the Volume: the next run skips them.
    env["WANDB_PROJECT"] = "rl-wordle"

    # 1. What did we get? The GPUs and the driver.
    _run(["nvidia-smi", "-L"], env)
    _run(
        [
            f"{BIN}/python",
            "-c",
            "import torch;"
            "print('torch', torch.__version__, 'cuda', torch.version.cuda,"
            " 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0))",
        ],
        env,
    )

    # 2. prime-rl writes to outputs/<run name>. Point that folder at the Volume.
    if not os.path.exists(f"{APP_DIR}/outputs"):
        os.symlink(OUT, f"{APP_DIR}/outputs")

    side_processes = _start_dashboard(env)
    try:
        with modal.forward(RELAY_PORT) as tunnel:
            print(f"\n>>> DASHBOARD: {tunnel.url}\n>>> W&B: project rl-wordle\n", flush=True)
            try:
                if stage == "smoke":
                    _smoke(env)
                elif stage == "eval":
                    _eval(env)
                else:
                    raise ValueError(f"unknown stage {stage!r}: use smoke or eval")
            except BaseException:
                # A failed run must not hold 2 GPUs open (about $0.08 a minute): the first smoke
                # run failed in 20 seconds and would have held the container for 5 minutes.
                vol.commit()
                raise
            vol.commit()
            print(f"\nDone. The dashboard stays open for {hold_min} min: {tunnel.url}")
            time.sleep(hold_min * 60)
    finally:
        for process in side_processes:
            process.terminate()


def _smoke(env: dict) -> None:
    """The smoke test of prime-rl's README: 20 RL steps on reverse-text, 1 GPU for vLLM and 1
    for the trainer. The model is a 0.6B, so this is cheap. If it runs, the install is good."""
    name = _run_name("spike-reverse-text")
    print(f">>> RUN NAME (W&B and the dashboard): {name}", flush=True)
    with _stream_logs(name):
        _run(
            [
                f"{BIN}/rl",
                "@",
                "examples/basic/reverse-text/rl.toml",
                "--monitors.wandb.project",
                "rl-wordle",
                "--monitors.wandb.name",
                name,
                "--run.name",
                name,
            ],
            env,
        )


def _eval(env: dict) -> None:
    """vLLM on GPU 0, and the eval of prime-rl against it, on our taskset."""
    server = subprocess.Popen(
        [f"{BIN}/inference", "--vllm.model", MODEL],
        cwd=APP_DIR,
        env={**env, "CUDA_VISIBLE_DEVICES": "0"},
    )
    try:
        _wait_for("http://localhost:8000/v1/models", server, timeout_s=900)
        name = _run_name("spike-wordle-baseline")
        print(f">>> RUN NAME (W&B and the dashboard): {name}", flush=True)
        with _stream_logs(name):
            _run(
                [
                    f"{BIN}/eval",
                    "@",
                    f"{APP_DIR}/wordle_eval.toml",
                    "--run.name",
                    name,
                    "--monitors.wandb.project",
                    "rl-wordle",
                    "--monitors.wandb.name",
                    name,
                ],
                env,
            )
    finally:
        server.terminate()
