from __future__ import annotations

import argparse
import pickle
import tempfile
from pathlib import Path

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def read_text(input_path: str | Path, max_bytes: int | None = None) -> str:
    if max_bytes is None:
        return Path(input_path).read_text(encoding="utf-8")

    with Path(input_path).open("rb") as f:
        raw_bytes = f.read(max_bytes)
    return raw_bytes.decode("utf-8", errors="ignore")


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    max_bytes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    text = read_text(input_path, max_bytes=max_bytes)

    special_pattern = "|".join(re.escape(token) for token in special_tokens)
    split_pattern = f"({special_pattern})" if special_pattern else None
    chunks = re.split(split_pattern, text) if split_pattern else [text]

    word_freq: dict[tuple[bytes, ...], int] = {}
    for chunk in chunks:
        if chunk in special_tokens:
            continue
        for match in re.finditer(PAT, chunk):
            word = match.group().encode("utf-8")
            key = tuple(bytes([byte]) for byte in word)
            word_freq[key] = word_freq.get(key, 0) + 1

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []
    target_merges = vocab_size - len(vocab)
    while len(vocab) < vocab_size:
        if len(merges) % 50 == 0:
            print(f"merge progress: {len(merges)}/{target_merges}")

        pair_freq: dict[tuple[bytes, bytes], int] = {}
        for word, freq in word_freq.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + freq

        if not pair_freq:
            break

        best_pair = max(pair_freq, key=lambda pair: (pair_freq[pair], pair))
        merged_token = best_pair[0] + best_pair[1]

        new_word_freq: dict[tuple[bytes, ...], int] = {}
        for word, freq in word_freq.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tuple = tuple(new_word)
            new_word_freq[new_word_tuple] = new_word_freq.get(new_word_tuple, 0) + freq

        word_freq = new_word_freq
        vocab[len(vocab)] = merged_token
        merges.append(best_pair)

    return vocab, merges


def bytes_to_unicode() -> dict[int, str]:
    visible_bytes = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_points = visible_bytes[:]
    next_point = 0
    for byte in range(256):
        if byte not in visible_bytes:
            visible_bytes.append(byte)
            unicode_points.append(256 + next_point)
            next_point += 1
    return {byte: chr(point) for byte, point in zip(visible_bytes, unicode_points)}


BYTE_DECODER = {char: byte for byte, char in bytes_to_unicode().items()}


def byte_level_token_to_bytes(token: str, special_tokens: set[str]) -> bytes:
    if token in special_tokens:
        return token.encode("utf-8")
    try:
        return bytes(BYTE_DECODER[char] for char in token)
    except KeyError as exc:
        raise ValueError(f"Unsupported byte-level token {token!r}") from exc


def convert_hf_tokenizer(
    hf_vocab: dict[str, int],
    hf_merges: list[tuple[str, str]],
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_token_set = set(special_tokens)
    vocab: dict[int, bytes] = {}
    for token, idx in hf_vocab.items():
        vocab[idx] = byte_level_token_to_bytes(token, special_token_set)

    merges = [
        (
            byte_level_token_to_bytes(left, special_token_set),
            byte_level_token_to_bytes(right, special_token_set),
        )
        for left, right in hf_merges
    ]
    return vocab, merges


def train_bpe_fast(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    max_bytes: int | None = None,
    min_frequency: int = 2,
    tokenizer_json_path: str | Path | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Fast tokenizer training requires the `tokenizers` package. "
            "Install it with `pip install tokenizers`."
        ) from exc

    train_path = Path(input_path)
    temp_dir = None
    if max_bytes is not None:
        temp_dir = tempfile.TemporaryDirectory()
        sample_path = Path(temp_dir.name) / "tokenizer_sample.txt"
        with train_path.open("rb") as source, sample_path.open("wb") as sink:
            sink.write(source.read(max_bytes))
        train_path = sample_path

    try:
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[str(train_path)],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )
        if tokenizer_json_path is not None:
            tokenizer_json_path = Path(tokenizer_json_path)
            tokenizer_json_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(tokenizer_json_path))

        with tempfile.TemporaryDirectory() as model_dir:
            model_paths = tokenizer.save_model(model_dir)
            vocab_json = Path(model_paths[0])
            merges_txt = Path(model_paths[1])

            import json

            hf_vocab = json.loads(vocab_json.read_text(encoding="utf-8"))
            hf_merges = []
            for line in merges_txt.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith("#"):
                    continue
                left, right = line.split()
                hf_merges.append((left, right))
        return convert_hf_tokenizer(hf_vocab, hf_merges, special_tokens)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | Path,
    merges_path: str | Path,
) -> None:
    vocab_path = Path(vocab_path)
    merges_path = Path(merges_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    merges_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_list = [None] * len(vocab)
    for idx, token_bytes in vocab.items():
        vocab_list[idx] = token_bytes

    with vocab_path.open("wb") as f:
        pickle.dump(vocab_list, f)
    with merges_path.open("wb") as f:
        pickle.dump(merges, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument(
        "--backend",
        choices=["fast", "simple"],
        default="fast",
        help="Use `fast` for the tokenizers Rust backend, or `simple` for the teaching implementation.",
    )
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Only read this many bytes from input-path. Useful for quick experiments.",
    )
    parser.add_argument("--vocab-path", type=Path, default=Path("artifacts/tokenizer/vocab.bin"))
    parser.add_argument("--merges-path", type=Path, default=Path("artifacts/tokenizer/merges.bin"))
    parser.add_argument(
        "--tokenizer-json-path",
        type=Path,
        default=Path("artifacts/tokenizer/tokenizer.json"),
        help="Optional fast tokenizer JSON artifact written when --backend fast is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "fast":
        vocab, merges = train_bpe_fast(
            input_path=args.input_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            max_bytes=args.max_bytes,
            min_frequency=args.min_frequency,
            tokenizer_json_path=args.tokenizer_json_path,
        )
    else:
        vocab, merges = train_bpe(
            input_path=args.input_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            max_bytes=args.max_bytes,
        )
    save_tokenizer(vocab, merges, args.vocab_path, args.merges_path)
    print(f"vocab size: {len(vocab)}, merges count: {len(merges)}")
    print(f"saved vocab to {args.vocab_path}")
    print(f"saved merges to {args.merges_path}")


if __name__ == "__main__":
    main()
