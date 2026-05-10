from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm


EOT = "<|endoftext|>"


@dataclass
class DatasetSpec:
    name: str
    config: str | None
    split: str
    text_field: str
    label: str


def clean_text(text: str, min_chars: int, max_chars: int) -> str | None:
    text = text.replace("\x00", "").strip()
    if len(text) < min_chars:
        return None
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0].strip()
    if not text:
        return None
    return text


def stream_texts(spec: DatasetSpec, min_chars: int, max_chars: int) -> Iterable[str]:
    kwargs = {
        "path": spec.name,
        "split": spec.split,
        "streaming": True,
    }
    if spec.config:
        kwargs["name"] = spec.config

    dataset = load_dataset(**kwargs)
    for row in dataset:
        value = row.get(spec.text_field)
        if not isinstance(value, str):
            continue
        text = clean_text(value, min_chars=min_chars, max_chars=max_chars)
        if text is not None:
            yield text


def write_until_bytes(
    sink,
    texts: Iterable[str],
    target_bytes: int,
    label: str,
    rng: random.Random,
) -> int:
    written = 0
    with tqdm(total=target_bytes, unit="B", unit_scale=True, desc=label) as pbar:
        for text in texts:
            if rng.random() < 0.5:
                text = text.replace("\r\n", "\n").replace("\r", "\n")
            payload = (text + EOT + "\n").encode("utf-8")
            sink.write(payload.decode("utf-8", errors="ignore"))
            written += len(payload)
            pbar.update(len(payload))
            if written >= target_bytes:
                break
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream English and Chinese datasets into raw pretraining text files."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--valid-bytes", type=int, default=100_000_000)
    parser.add_argument("--en-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=20000)

    parser.add_argument("--en-dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--en-config", default="sample-10BT")
    parser.add_argument("--en-split", default="train")
    parser.add_argument("--en-text-field", default="text")

    parser.add_argument("--zh-dataset", default="Morton-Li/ChineseWebText2.0-HighQuality")
    parser.add_argument("--zh-config", default=None)
    parser.add_argument("--zh-split", default="train")
    parser.add_argument("--zh-text-field", default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.en_ratio <= 1.0:
        raise ValueError("--en-ratio must be between 0 and 1")

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    en_spec = DatasetSpec(
        name=args.en_dataset,
        config=args.en_config,
        split=args.en_split,
        text_field=args.en_text_field,
        label="en",
    )
    zh_spec = DatasetSpec(
        name=args.zh_dataset,
        config=args.zh_config,
        split=args.zh_split,
        text_field=args.zh_text_field,
        label="zh",
    )

    budgets = {
        "train_en": int(args.train_bytes * args.en_ratio),
        "train_zh": args.train_bytes - int(args.train_bytes * args.en_ratio),
        "valid_en": int(args.valid_bytes * args.en_ratio),
        "valid_zh": args.valid_bytes - int(args.valid_bytes * args.en_ratio),
    }

    train_path = args.output_dir / "train.txt"
    valid_path = args.output_dir / "valid.txt"
    en_texts = stream_texts(en_spec, args.min_chars, args.max_chars)
    zh_texts = stream_texts(zh_spec, args.min_chars, args.max_chars)

    with train_path.open("w", encoding="utf-8") as train_sink:
        train_en = write_until_bytes(train_sink, en_texts, budgets["train_en"], "train/en", rng)
        train_zh = write_until_bytes(train_sink, zh_texts, budgets["train_zh"], "train/zh", rng)

    with valid_path.open("w", encoding="utf-8") as valid_sink:
        valid_en = write_until_bytes(valid_sink, en_texts, budgets["valid_en"], "valid/en", rng)
        valid_zh = write_until_bytes(valid_sink, zh_texts, budgets["valid_zh"], "valid/zh", rng)

    print("done")
    print(f"train: {train_path} ({train_en + train_zh} bytes)")
    print(f"valid: {valid_path} ({valid_en + valid_zh} bytes)")
    print(f"train/en bytes: {train_en}")
    print(f"train/zh bytes: {train_zh}")
    print(f"valid/en bytes: {valid_en}")
    print(f"valid/zh bytes: {valid_zh}")


if __name__ == "__main__":
    main()
