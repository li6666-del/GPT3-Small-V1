from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=20000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1337)
    base = np.arange(args.tokens, dtype=np.uint32) % args.vocab_size
    noise = rng.integers(0, 4, size=args.tokens, dtype=np.uint32)
    tokens = ((base + noise) % args.vocab_size).astype(np.uint16)
    split = int(args.tokens * 0.9)
    tokens[:split].tofile(out_dir / "train.bin")
    tokens[split:].tofile(out_dir / "valid.bin")
    print(f"wrote {split} train tokens and {args.tokens - split} valid tokens to {out_dir}")


if __name__ == "__main__":
    main()
