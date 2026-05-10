from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gpt_small.tokenizer import Tokenizer


def choose_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize a UTF-8 text file into a memmap .bin file.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=Path("artifacts/tokenizer/vocab.bin"))
    parser.add_argument("--merges-path", type=Path, default=Path("artifacts/tokenizer/merges.bin"))
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=args.special_tokens,
    )
    dtype = choose_dtype(len(tokenizer.vocab))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    token_count = 0
    with args.input_path.open("r", encoding="utf-8") as source, args.output_path.open("wb") as sink:
        buffer = []
        for token_id in tokenizer.encode_iterable(source):
            buffer.append(token_id)
            if len(buffer) >= 1_000_000:
                np.asarray(buffer, dtype=dtype).tofile(sink)
                token_count += len(buffer)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(sink)
            token_count += len(buffer)

    print(f"wrote {token_count} tokens to {args.output_path} as {dtype}")


if __name__ == "__main__":
    main()
