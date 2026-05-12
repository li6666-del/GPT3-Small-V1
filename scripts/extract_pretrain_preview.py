from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def iter_texts(pattern: str, text_field: str, limit: int):
    count = 0
    for path in sorted(Path().glob(pattern)):
        df = pd.read_parquet(path, columns=[text_field])
        for value in df[text_field]:
            if not isinstance(value, str):
                continue
            text = value.replace("\x00", "").strip()
            if not text:
                continue
            yield path, text
            count += 1
            if count >= limit:
                return


def write_section(handle, title: str, pattern: str, text_field: str, limit: int, max_chars: int) -> None:
    handle.write(f"# {title}\n\n")
    for index, (path, text) in enumerate(iter_texts(pattern, text_field, limit), start=1):
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n...[truncated]"
        handle.write(f"## {title} sample {index}\n")
        handle.write(f"source: {path}\n\n")
        handle.write(text)
        handle.write("\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract readable text samples from pretraining parquet files.")
    parser.add_argument("--en-glob", default="data/cache/fineweb_edu_10bt/**/*.parquet")
    parser.add_argument("--zh-glob", default="data/cache/chinesewebtext2_hq/**/*.parquet")
    parser.add_argument("--output-path", type=Path, default=Path("data/samples/pretrain_preview.txt"))
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--samples-per-lang", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        handle.write("Pretraining Data Preview\n")
        handle.write("========================\n\n")
        write_section(
            handle,
            "English FineWeb-Edu",
            args.en_glob,
            args.text_field,
            args.samples_per_lang,
            args.max_chars,
        )
        write_section(
            handle,
            "ChineseWebText2 HighQuality",
            args.zh_glob,
            args.text_field,
            args.samples_per_lang,
            args.max_chars,
        )
    print(args.output_path)


if __name__ == "__main__":
    main()
