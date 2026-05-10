from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import requests


def list_parquet_files(
    mirror_url: str,
    dataset_name: str,
    prefix: str,
    max_files: int,
    start_index: int,
) -> list[dict]:
    api_url = (
        f"{mirror_url.rstrip('/')}/api/datasets/{dataset_name}/tree/main/"
        f"{prefix.strip('/')}?recursive=true&expand=false"
    )
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    rows = response.json()
    files = [
        {
            "path": row["path"],
            "size": row.get("size"),
            "url": f"{mirror_url.rstrip('/')}/datasets/{dataset_name}/resolve/main/{row['path']}",
        }
        for row in rows
        if row.get("type") == "file" and row.get("path", "").endswith(".parquet")
    ]
    files.sort(key=lambda row: row["path"])
    if start_index:
        files = files[start_index:]
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No parquet files found at {api_url}")
    return files


def download_with_curl(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-L",
        "-C",
        "-",
        "--retry",
        "20",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "1024",
        "--speed-time",
        "120",
        "-o",
        str(output_path),
        url,
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Hugging Face parquet shards via a mirror.")
    parser.add_argument("--mirror-url", default="https://hf-mirror.com")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = list_parquet_files(
        mirror_url=args.mirror_url,
        dataset_name=args.dataset,
        prefix=args.prefix,
        max_files=args.max_files,
        start_index=args.start_index,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8")

    total_size = sum(row["size"] or 0 for row in files)
    print(f"dataset: {args.dataset}")
    print(f"files: {len(files)}")
    print(f"total listed bytes: {total_size}")
    print(f"manifest: {manifest_path}")

    for i, row in enumerate(files):
        output_path = args.output_dir / row["path"]
        print(f"[{i + 1}/{len(files)}] {row['path']} -> {output_path}")
        if args.dry_run:
            continue
        if output_path.exists() and row["size"] and output_path.stat().st_size == row["size"]:
            print("already complete")
            continue
        download_with_curl(row["url"], output_path)


if __name__ == "__main__":
    main()
