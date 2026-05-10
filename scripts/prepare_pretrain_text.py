from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
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
    data_files: list[str] | None = None


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
    if spec.data_files:
        kwargs = {
            "path": "parquet",
            "data_files": spec.data_files,
            "split": spec.split,
            "streaming": True,
        }
    else:
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
) -> int:
    written = 0
    with tqdm(total=target_bytes, unit="B", unit_scale=True, desc=label) as pbar:
        for text in texts:
            record = text.replace("\r\n", "\n").replace("\r", "\n") + EOT + "\n"
            record_bytes = len(record.encode("utf-8"))
            sink.write(record)
            written += record_bytes
            pbar.update(record_bytes)
            if written >= target_bytes:
                break
    return written


def list_mirror_parquet_files(
    mirror_url: str,
    dataset_name: str,
    prefix: str,
    max_files: int,
) -> list[str]:
    api_url = (
        f"{mirror_url.rstrip('/')}/api/datasets/{dataset_name}/tree/main/"
        f"{prefix.strip('/')}?recursive=true&expand=false"
    )
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    rows = response.json()
    paths = [
        row["path"]
        for row in rows
        if row.get("type") == "file" and row.get("path", "").endswith(".parquet")
    ]
    paths.sort()
    if max_files > 0:
        paths = paths[:max_files]
    if not paths:
        raise RuntimeError(f"No parquet files found at {api_url}")
    return [
        f"{mirror_url.rstrip('/')}/datasets/{dataset_name}/resolve/main/{path}"
        for path in paths
    ]


def expand_local_glob(pattern: str | None) -> list[str] | None:
    if not pattern:
        return None
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise RuntimeError(f"No local parquet files matched {pattern!r}")
    return paths


def require_budget(label: str, actual_bytes: int, target_bytes: int) -> None:
    if actual_bytes < target_bytes:
        raise RuntimeError(
            f"{label} only wrote {actual_bytes} bytes, below target {target_bytes}. "
            "Add more parquet shards or lower the byte budget."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream English and Chinese datasets into raw pretraining text files."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--valid-bytes", type=int, default=100_000_000)
    parser.add_argument("--en-ratio", type=float, default=0.5)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument(
        "--mirror-url",
        default=None,
        help="Optional Hugging Face mirror URL. When set, parquet files are listed via the mirror API.",
    )

    parser.add_argument("--en-dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--en-config", default="sample-10BT")
    parser.add_argument("--en-split", default="train")
    parser.add_argument("--en-text-field", default="text")
    parser.add_argument("--en-data-prefix", default="sample/10BT")
    parser.add_argument("--en-max-files", type=int, default=0)
    parser.add_argument("--en-local-glob", default=None)

    parser.add_argument("--zh-dataset", default="Morton-Li/ChineseWebText2.0-HighQuality")
    parser.add_argument("--zh-config", default=None)
    parser.add_argument("--zh-split", default="train")
    parser.add_argument("--zh-text-field", default="text")
    parser.add_argument("--zh-data-prefix", default="data")
    parser.add_argument("--zh-max-files", type=int, default=0)
    parser.add_argument("--zh-local-glob", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.en_ratio <= 1.0:
        raise ValueError("--en-ratio must be between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    en_data_files = None
    zh_data_files = None
    en_local_files = expand_local_glob(args.en_local_glob)
    zh_local_files = expand_local_glob(args.zh_local_glob)
    if en_local_files or zh_local_files:
        if not en_local_files or not zh_local_files:
            raise ValueError("--en-local-glob and --zh-local-glob must be provided together")
        en_data_files = en_local_files
        zh_data_files = zh_local_files
        print(f"en local parquet files: {len(en_data_files)}")
        print(f"zh local parquet files: {len(zh_data_files)}")
    elif args.mirror_url:
        en_data_files = list_mirror_parquet_files(
            args.mirror_url,
            args.en_dataset,
            args.en_data_prefix,
            args.en_max_files,
        )
        zh_data_files = list_mirror_parquet_files(
            args.mirror_url,
            args.zh_dataset,
            args.zh_data_prefix,
            args.zh_max_files,
        )
        print(f"en parquet files: {len(en_data_files)}")
        print(f"zh parquet files: {len(zh_data_files)}")

    en_spec = DatasetSpec(
        name=args.en_dataset,
        config=args.en_config,
        split=args.en_split,
        text_field=args.en_text_field,
        label="en",
        data_files=en_data_files,
    )
    zh_spec = DatasetSpec(
        name=args.zh_dataset,
        config=args.zh_config,
        split=args.zh_split,
        text_field=args.zh_text_field,
        label="zh",
        data_files=zh_data_files,
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
        train_en = write_until_bytes(train_sink, en_texts, budgets["train_en"], "train/en")
        train_zh = write_until_bytes(train_sink, zh_texts, budgets["train_zh"], "train/zh")

    with valid_path.open("w", encoding="utf-8") as valid_sink:
        valid_en = write_until_bytes(valid_sink, en_texts, budgets["valid_en"], "valid/en")
        valid_zh = write_until_bytes(valid_sink, zh_texts, budgets["valid_zh"], "valid/zh")

    require_budget("train/en", train_en, budgets["train_en"])
    require_budget("train/zh", train_zh, budgets["train_zh"])
    require_budget("valid/en", valid_en, budgets["valid_en"])
    require_budget("valid/zh", valid_zh, budgets["valid_zh"])

    print("done")
    print(f"train: {train_path} ({train_en + train_zh} bytes)")
    print(f"valid: {valid_path} ({valid_en + valid_zh} bytes)")
    print(f"train/en bytes: {train_en}")
    print(f"train/zh bytes: {train_zh}")
    print(f"valid/en bytes: {valid_en}")
    print(f"valid/zh bytes: {valid_zh}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
