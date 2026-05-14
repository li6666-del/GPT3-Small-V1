from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MOJIBAKE_MARKERS = [
    "浣犳",
    "鎴戞",
    "鐨",
    "涓€",
    "銆",
    "锛",
    "鈥",
    "绛",
    "闂",
    "璇",
]


def iter_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(iter_text(item))
        return out
    if isinstance(value, dict):
        out = []
        for key in ("prompt", "expected", "output", "content", "response"):
            if key in value:
                out.extend(iter_text(value[key]))
        if "messages" in value:
            out.extend(iter_text(value["messages"]))
        return out
    return []


def scan_file(path: Path) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            for text in iter_text(row):
                markers = [marker for marker in MOJIBAKE_MARKERS if marker in text]
                if markers:
                    hits.append(
                        {
                            "line": line_no,
                            "markers": markers,
                            "text": text[:120],
                        }
                    )
                    break
    return {"path": str(path), "rows": total, "hits": hits, "hit_count": len(hits)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--fail-on-hit", action="store_true")
    args = parser.parse_args()

    results = [scan_file(Path(item)) for item in args.paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.fail_on_hit and any(item["hit_count"] for item in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
