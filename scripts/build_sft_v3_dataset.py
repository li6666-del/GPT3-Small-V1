from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


EN_DATASET = "yahma/alpaca-cleaned"
ZH_DATASET = "BelleGroup/train_0.5M_CN"
SOURCE = "sft_v3_real_mix"


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def too_repetitive(text: str) -> bool:
    if re.search(r"(.{1,20})\1{4,}", text):
        return True
    words = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    if len(words) >= 20:
        unique_ratio = len(set(words)) / len(words)
        return unique_ratio < 0.18
    return False


def is_good_pair(prompt: str, response: str) -> bool:
    if not (8 <= len(prompt) <= 1400 and 2 <= len(response) <= 1800):
        return False
    if too_repetitive(prompt) or too_repetitive(response):
        return False
    bad_fragments = ["http://", "https://", "www.", "<script", "</html", "### Response:"]
    lowered = response.lower()
    if any(fragment in lowered for fragment in bad_fragments):
        return False
    return True


def make_row(prompt: str, response: str, category: str, language: str, source: str = SOURCE) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "category": category,
        "language": language,
        "source": source,
    }


def alpaca_prompt(item: dict[str, Any]) -> tuple[str, str] | None:
    instruction = clean_text(item.get("instruction"))
    extra_input = clean_text(item.get("input"))
    output = clean_text(item.get("output"))
    if not instruction or not output:
        return None
    prompt = instruction if not extra_input else f"{instruction}\n{extra_input}"
    if not is_good_pair(prompt, output):
        return None
    return prompt, output


def load_hf_rows(dataset_name: str, count: int, seed: int, language: str, category: str) -> list[dict[str, object]]:
    from datasets import load_dataset

    stream = load_dataset(dataset_name, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=20_000)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in stream:
        parsed = alpaca_prompt(item)
        if parsed is None:
            continue
        prompt, output = parsed
        key = prompt[:240]
        if key in seen:
            continue
        seen.add(key)
        rows.append(make_row(prompt, output, category, language, source=dataset_name))
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise RuntimeError(f"{dataset_name} yielded only {len(rows)} good rows, wanted {count}")
    return rows


def load_parquet_rows(
    parquet_path: str | Path,
    count: int,
    seed: int,
    language: str,
    category: str,
    source: str,
) -> list[dict[str, object]]:
    import pyarrow.parquet as pq

    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(path)
    table = pq.read_table(path, columns=["instruction", "input", "output"])
    data = table.to_pylist()
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for index in indices:
        parsed = alpaca_prompt(data[index])
        if parsed is None:
            continue
        prompt, output = parsed
        key = prompt[:240]
        if key in seen:
            continue
        seen.add(key)
        rows.append(make_row(prompt, output, category, language, source=source))
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise RuntimeError(f"{path} yielded only {len(rows)} good rows, wanted {count}")
    return rows


ZH_CONCEPTS = {
    "机器学习": "机器学习是一种让计算机从数据中学习规律的方法。它通过样本训练模型，让模型学会分类、预测或生成内容。",
    "过拟合": "过拟合是指模型把训练数据记得太死，导致在新数据上表现变差。",
    "光合作用": "光合作用是植物利用阳光、水和二氧化碳制造有机物，并释放氧气的过程。",
    "验证集": "验证集是在训练过程中用来检查模型泛化效果的数据，它不参与参数更新。",
    "缓存": "缓存是把常用数据临时存到更快的位置，从而减少重复计算或网络请求。",
    "数据库索引": "数据库索引像书的目录，可以帮助数据库更快找到需要的数据。",
}

EN_CONCEPTS = {
    "machine learning": "Machine learning teaches computers patterns from data so they can make predictions or generate useful outputs.",
    "overfitting": "Overfitting happens when a model memorizes training examples too closely and performs poorly on new data.",
    "photosynthesis": "Photosynthesis is the process plants use to turn sunlight, water, and carbon dioxide into food and oxygen.",
    "validation set": "A validation set checks how well a model generalizes during training and is not used to update parameters.",
    "cache": "A cache stores frequently used data in a faster place so repeated access is quicker.",
    "database index": "A database index is like a book index. It helps find rows faster, but it can add storage and write cost.",
}

ZH_TRANSLATIONS = [
    ("训练结束后请检查验证集 loss。", "Please check the validation loss after training finishes."),
    ("不要从已经崩坏的 checkpoint 继续训练。", "Do not continue training from a collapsed checkpoint."),
    ("请保存原始生成结果。", "Please save the raw generation results."),
    ("如果输出开始重复，请降低训练强度。", "If the output starts repeating, reduce the training intensity."),
    ("这个脚本会生成第三版 SFT 数据。", "This script generates the third SFT dataset."),
]

EN_TRANSLATIONS = [
    ("The model should answer concisely and stay on topic.", "模型应该简洁回答，并且保持主题一致。"),
    ("Use the same prompts for comparison.", "使用相同的 prompt 进行对比。"),
    ("The code compiles, but generation quality still matters.", "代码可以编译，但生成质量仍然很重要。"),
    ("A good assistant admits uncertainty when needed.", "好的助手会在必要时承认不确定。"),
    ("Please send me the latest training log.", "请把最新训练日志发给我。"),
]


def synthetic_rows(count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    while len(rows) < count:
        kind = rng.choice(["zh_concept", "en_concept", "math", "translation", "refusal", "email", "summary"])
        if kind == "zh_concept":
            term, answer = rng.choice(list(ZH_CONCEPTS.items()))
            prompt = rng.choice([f"请用中文简单解释什么是{term}。", f"{term}是什么意思？请简短回答。"])
            rows.append(make_row(prompt, answer, "verified_concept", "zh", "synthetic_verified_v3"))
        elif kind == "en_concept":
            term, answer = rng.choice(list(EN_CONCEPTS.items()))
            prompt = rng.choice([f"Explain {term} in simple terms.", f"What is {term}? Answer briefly."])
            rows.append(make_row(prompt, answer, "verified_concept", "en", "synthetic_verified_v3"))
        elif kind == "math":
            if rng.random() < 0.5:
                hours = rng.randint(2, 6)
                speed = rng.choice([25, 30, 40, 50, 60, 75])
                distance = hours * speed
                rows.append(make_row(
                    f"If a train travels {distance} miles in {hours} hours, what is its average speed?",
                    f"Average speed = distance ÷ time = {distance} ÷ {hours} = {speed} miles per hour.",
                    "verified_math",
                    "en",
                    "synthetic_verified_v3",
                ))
            else:
                a, b = rng.randint(20, 500), rng.randint(20, 500)
                rows.append(make_row(f"{a} 加 {b} 等于多少？", f"{a} + {b} = {a + b}。", "verified_math", "zh", "synthetic_verified_v3"))
        elif kind == "translation":
            if rng.random() < 0.5:
                zh, en = rng.choice(ZH_TRANSLATIONS)
                rows.append(make_row(f"把下面这句话翻译成英文：{zh}", en, "verified_translation", "mixed", "synthetic_verified_v3"))
            else:
                en, zh = rng.choice(EN_TRANSLATIONS)
                rows.append(make_row(f"Translate this sentence into Chinese: {en}", zh, "verified_translation", "mixed", "synthetic_verified_v3"))
        elif kind == "refusal":
            if rng.random() < 0.5:
                name = rng.choice(["赵星河", "顾清和", "林亦安", "陈若星"])
                year = rng.randint(2070, 2099)
                rows.append(make_row(
                    f"请告诉我{name}在{year}年获得了什么国际大奖。",
                    "我无法确认这个信息。仅凭这个问题无法判断人物和奖项是否真实存在，因此不应该编造答案。",
                    "refusal",
                    "zh",
                    "synthetic_verified_v3",
                ))
            else:
                name = rng.choice(["Ariana Wells", "Jonas Venn", "Mira Stone", "Leo Hart"])
                year = rng.randint(2070, 2099)
                rows.append(make_row(
                    f"What international award did {name} win in {year}?",
                    "I cannot verify that information from the question alone, so I should not invent an award or present it as fact.",
                    "refusal",
                    "en",
                    "synthetic_verified_v3",
                ))
        elif kind == "email":
            if rng.random() < 0.5:
                rows.append(make_row(
                    "Write a short polite email asking a teammate to send the latest training log.",
                    "Hi, could you please send me the latest training log when you have a moment? I want to review the recent loss and eval results. Thanks!",
                    "email",
                    "en",
                    "synthetic_verified_v3",
                ))
            else:
                rows.append(make_row(
                    "写一封简短礼貌的中文消息，请同事发一下最新训练日志。",
                    "你好，方便的话请把最新训练日志发我一下。我想看一下最近的 loss 和 eval 结果，谢谢。",
                    "email",
                    "zh",
                    "synthetic_verified_v3",
                ))
        else:
            rows.append(make_row(
                "请用一句话概括：第一轮 SFT 的 loss 很低，但生成样本出现重复和模板串台，因此下一轮应该从 base checkpoint 重新开始。",
                "第一轮 SFT 虽然 loss 很低，但生成质量不稳，下一轮应从 base checkpoint 重新开始。",
                "summary",
                "zh",
                "synthetic_verified_v3",
            ))
    return rows


def is_mostly_chinese(text: str) -> bool:
    chinese = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    letters = sum(1 for char in text if char.isalpha())
    return chinese > max(6, letters * 0.25)


def corpus_continuation_rows(count: int, seed: int, corpus_path: str | Path = "data/clean/train.txt") -> list[dict[str, object]]:
    path = Path(corpus_path)
    if not path.exists():
        sample_path = Path("data/samples/pretrain_preview.txt")
        if sample_path.exists():
            path = sample_path
        else:
            return []

    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    size = path.stat().st_size
    if size < 10_000:
        text = path.read_text(encoding="utf-8", errors="ignore")
        candidates = [part.strip() for part in re.split(r"\n{2,}|\n", text) if len(part.strip()) > 240]
        rng.shuffle(candidates)
        for text in candidates:
            if len(rows) >= count:
                break
            rows.extend(_make_continuation_from_text(text, rng))
        return rows[:count]

    with path.open("rb") as f:
        attempts = 0
        while len(rows) < count and attempts < count * 80:
            attempts += 1
            offset = rng.randint(0, max(0, size - 4096))
            f.seek(offset)
            f.readline()
            raw = f.readline(5000)
            text = raw.decode("utf-8", errors="ignore")
            text = clean_text(text)
            if len(text) < 260 or len(text) > 2500:
                continue
            made = _make_continuation_from_text(text, rng)
            if made:
                rows.append(made[0])
    return rows


def _make_continuation_from_text(text: str, rng: random.Random) -> list[dict[str, object]]:
    if too_repetitive(text):
        return []
    if text.count("http") > 0 or text.count("|") > 8:
        return []
    text = text[:1200]
    min_prompt = 80 if is_mostly_chinese(text) else 140
    if len(text) < min_prompt + 120:
        return []
    split = rng.randint(min_prompt, min(len(text) - 80, min_prompt + 180))
    prompt_text = text[:split].strip()
    response = text[split : split + rng.randint(80, 220)].strip()
    if len(response) < 40:
        return []
    if is_mostly_chinese(text):
        prompt = rng.choice([
            f"请自然续写下面这段文字：\n{prompt_text}",
            f"继续写这段中文，不要突然换主题：\n{prompt_text}",
        ])
        return [make_row(prompt, response, "corpus_continuation", "zh", "pretrain_corpus_continuation")]
    prompt = rng.choice([
        f"Continue the following text naturally:\n{prompt_text}",
        f"Write the next part of this passage without changing the topic:\n{prompt_text}",
    ])
    return [make_row(prompt, response, "corpus_continuation", "en", "pretrain_corpus_continuation")]


def split_rows(rows: list[dict[str, object]], valid_count: int, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[valid_count:], rows[:valid_count]


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--en-examples", type=int, default=45000)
    parser.add_argument("--zh-examples", type=int, default=45000)
    parser.add_argument("--synthetic-examples", type=int, default=12000)
    parser.add_argument("--valid-examples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--offline", action="store_true", help="Skip HuggingFace datasets and use local corpus + synthetic rows.")
    parser.add_argument("--en-parquet", help="Local parquet file for yahma/alpaca-cleaned.")
    parser.add_argument("--zh-parquet", help="Local parquet file for BelleGroup/train_0.5M_CN.")
    args = parser.parse_args()

    rows = []
    target_total = args.en_examples + args.zh_examples + args.synthetic_examples
    try:
        if args.offline:
            raise RuntimeError("offline mode requested")
        if args.en_parquet and args.zh_parquet:
            rows.extend(load_parquet_rows(args.en_parquet, args.en_examples, args.seed, "en", "alpaca_en", EN_DATASET))
            rows.extend(load_parquet_rows(args.zh_parquet, args.zh_examples, args.seed + 1, "zh", "belle_zh", ZH_DATASET))
        else:
            rows.extend(load_hf_rows(EN_DATASET, args.en_examples, args.seed, "en", "alpaca_en"))
            rows.extend(load_hf_rows(ZH_DATASET, args.zh_examples, args.seed + 1, "zh", "belle_zh"))
        rows.extend(synthetic_rows(args.synthetic_examples, args.seed + 2))
    except Exception as exc:
        print(f"HF_DATASET_LOAD_FAILED: {exc!r}")
        corpus_count = min(target_total, min(35_000, max(10_000, target_total // 3)))
        corpus_rows = corpus_continuation_rows(corpus_count, args.seed + 10)
        rows.extend(corpus_rows)
        synthetic_needed = max(0, target_total - len(rows))
        rows.extend(synthetic_rows(synthetic_needed, args.seed + 20))
        print(f"offline fallback rows: corpus={len(corpus_rows)} synthetic={synthetic_needed}")
    train, valid = split_rows(rows, args.valid_examples, args.seed + 3)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    print(f"wrote {len(train)} train examples to {out_dir / 'train.jsonl'}")
    print(f"wrote {len(valid)} valid examples to {out_dir / 'valid.jsonl'}")


if __name__ == "__main__":
    main()
