from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_example(offset: int) -> dict[str, list[int]]:
    # Token ids are synthetic. Labels use -100 for prompt tokens and real ids for assistant tokens.
    user = [10 + offset, 11 + offset, 12 + offset, 13 + offset]
    assistant = [30 + offset, 31 + offset, 32 + offset, 0]
    input_ids = [1] + user + [2] + assistant
    labels = [-100] * (1 + len(user) + 1) + assistant
    return {"input_ids": input_ids, "labels": labels}


def write_split(path: Path, count: int, start_offset: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for index in range(count):
            row = make_example((start_offset + index) % 20)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-examples", type=int, default=64)
    parser.add_argument("--valid-examples", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_split(out_dir / "train.jsonl", args.train_examples, 0)
    write_split(out_dir / "valid.jsonl", args.valid_examples, args.train_examples)
    print(
        f"wrote {args.train_examples} train examples and "
        f"{args.valid_examples} valid examples to {out_dir}"
    )


if __name__ == "__main__":
    main()
