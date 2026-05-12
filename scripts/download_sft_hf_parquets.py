from __future__ import annotations

import argparse
from pathlib import Path

import requests


DATASETS = [
    "yahma/alpaca-cleaned",
    "BelleGroup/train_0.5M_CN",
]


def dataset_dir_name(dataset: str) -> str:
    return dataset.replace("/", "__")


def download_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"exists {path} ({path.stat().st_size} bytes)")
        return
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        total = 0
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)
                if total and total % (32 * 1024 * 1024) < 1024 * 1024:
                    print(f"downloading {path.name}: {total / (1024 ** 2):.1f} MiB")
        tmp_path.replace(path)
        print(f"downloaded {path} ({path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/sft_sources/huggingface")
    parser.add_argument("--dataset", action="append", help="Dataset repo id. Can be passed more than once.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.dataset or DATASETS
    for dataset in datasets:
        response = requests.get(
            "https://datasets-server.huggingface.co/parquet",
            params={"dataset": dataset},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        target_dir = out_dir / dataset_dir_name(dataset)
        target_dir.mkdir(parents=True, exist_ok=True)
        files = payload.get("parquet_files", [])
        if not files:
            raise RuntimeError(f"No parquet files returned for {dataset}")
        for item in files:
            filename = f"{item['split']}-{item['filename']}"
            download_file(item["url"], target_dir / filename)


if __name__ == "__main__":
    main()
