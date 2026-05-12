from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gpt_small.tokenizer import Tokenizer


def choose_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def flush_buffer(buffer: list[int], sink, dtype: np.dtype) -> int:
    if not buffer:
        return 0
    np.asarray(buffer, dtype=dtype).tofile(sink)
    count = len(buffer)
    buffer.clear()
    return count


def tokenize_with_fast_tokenizer(
    input_path: Path,
    output_path: Path,
    tokenizer_json_path: Path,
    dtype: np.dtype,
) -> int:
    try:
        from tokenizers import Tokenizer as FastTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Fast tokenization requires the `tokenizers` package. "
            "Install it with `pip install tokenizers` or use --backend simple."
        ) from exc

    tokenizer = FastTokenizer.from_file(str(tokenizer_json_path))
    token_count = 0
    buffer: list[int] = []
    with input_path.open("r", encoding="utf-8") as source, output_path.open("wb") as sink:
        for line in source:
            buffer.extend(tokenizer.encode(line, add_special_tokens=False).ids)
            if len(buffer) >= 1_000_000:
                token_count += flush_buffer(buffer, sink, dtype)
        token_count += flush_buffer(buffer, sink, dtype)
    return token_count


def tokenize_with_simple_tokenizer(
    input_path: Path,
    output_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    dtype: np.dtype,
) -> int:
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=special_tokens,
    )
    token_count = 0
    buffer: list[int] = []
    with input_path.open("r", encoding="utf-8") as source, output_path.open("wb") as sink:
        for token_id in tokenizer.encode_iterable(source):
            buffer.append(token_id)
            if len(buffer) >= 1_000_000:
                token_count += flush_buffer(buffer, sink, dtype)
        token_count += flush_buffer(buffer, sink, dtype)
    return token_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize a UTF-8 text file into a memmap .bin file.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=Path("artifacts/tokenizer/vocab.bin"))
    parser.add_argument("--merges-path", type=Path, default=Path("artifacts/tokenizer/merges.bin"))
    parser.add_argument("--tokenizer-json-path", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--backend", choices=["auto", "fast", "simple"], default="auto")
    args = parser.parse_args()

    with args.vocab_path.open("rb") as f:
        import pickle

        loaded_vocab = pickle.load(f)
    vocab_size = len(loaded_vocab)
    dtype = choose_dtype(vocab_size)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = args.backend
    if backend == "auto":
        backend = "fast" if args.tokenizer_json_path.exists() else "simple"

    start_time = time.time()
    if backend == "fast":
        token_count = tokenize_with_fast_tokenizer(
            args.input_path,
            args.output_path,
            args.tokenizer_json_path,
            dtype,
        )
    else:
        token_count = tokenize_with_simple_tokenizer(
            args.input_path,
            args.output_path,
            args.vocab_path,
            args.merges_path,
            args.special_tokens,
            dtype,
        )

    print(f"wrote {token_count} tokens to {args.output_path} as {dtype}")
    print(f"backend: {backend}, elapsed_seconds: {time.time() - start_time:.2f}")


if __name__ == "__main__":
    main()
