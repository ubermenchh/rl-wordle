"""Export the SFT traces for prime-rl's `sft` trainer.

prime-rl reads a `messages` column (a whole chat) and trains on the assistant turns only
(docs/training.md, "Dataset Format", checked 2026-10-11). `data.name` can be a local folder, so
the file goes to the Modal Volume and nothing is uploaded to the Hub.

We keep only the `messages` column. The audit fields of the traces (secret, prior guesses) stay in
data/sft_train.jsonl, which tests/test_export_sft.py uses to check each prompt.

Run: uv run python -m sft.export_prime_rl
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def export(src: Path, dst_dir: Path) -> int:
    """Write dst_dir/train.jsonl with one {"messages": [...]} row for each trace. Returns the count.

    The file name matters: a local folder with train.jsonl loads as the `train` split, and the
    trainer asks for that split (prime-rl sft/data.py).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(src) as fin, open(dst_dir / "train.jsonl", "w") as fout:
        for line in fin:
            fout.write(json.dumps({"messages": json.loads(line)["messages"]}) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=ROOT / "data" / "sft_train.jsonl")
    parser.add_argument("--dst", type=Path, default=ROOT / "data" / "sft_wordle")
    args = parser.parse_args()
    print(f"wrote {export(args.src, args.dst)} rows to {args.dst / 'train.jsonl'}")


if __name__ == "__main__":
    main()
