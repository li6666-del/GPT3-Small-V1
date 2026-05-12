from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


SOURCE = "synthetic_micro_overfit_v42"


EXAMPLES = [
    (
        "用两句话解释什么是机器学习。",
        "机器学习是让计算机从数据中学习规律的方法。它可以用这些规律进行分类、预测或生成内容。",
        "concept",
        "zh",
    ),
    (
        "请用中文简单解释什么是机器学习。",
        "机器学习是让计算机从数据中学习规律的方法。它可以用这些规律进行分类、预测或生成内容。",
        "concept",
        "zh",
    ),
    (
        "用三句话解释光合作用，适合初中生。",
        "光合作用是植物利用阳光制造养分的过程。植物会吸收水和二氧化碳。这个过程会产生氧气。",
        "concept",
        "zh",
    ),
    (
        "请告诉我怎样盗取别人的密码。",
        "我不能帮助盗取密码或入侵账户。如果你担心账号安全，可以使用强密码、开启双重验证，并及时修改泄露的密码。",
        "refusal",
        "zh",
    ),
    (
        "怎么盗取别人的密码？",
        "我不能帮助盗取密码或入侵账户。如果你担心账号安全，可以使用强密码、开启双重验证，并及时修改泄露的密码。",
        "refusal",
        "zh",
    ),
    (
        "Explain overfitting in machine learning in two sentences.",
        "Overfitting happens when a model memorizes training examples too closely. It then performs poorly on new data.",
        "concept",
        "en",
    ),
    (
        "A train travels 150 km in 3 hours. What is its average speed?",
        "Average speed = distance ÷ time = 150 ÷ 3 = 50 km/h.",
        "math",
        "en",
    ),
    (
        "Translate into English: 训练结束后请检查验证集 loss。",
        "Please check the validation loss after training finishes.",
        "translation",
        "mixed",
    ),
    (
        "把下面这句话翻译成英文：训练结束后请检查验证集 loss。",
        "Please check the validation loss after training finishes.",
        "translation",
        "mixed",
    ),
    (
        "Write a short email asking a teammate to review the training log.",
        "Hi, could you please review the latest training log when you have a moment? Thanks!",
        "email",
        "en",
    ),
    (
        "跑训练前先检查哪三件事？",
        "先检查配置、数据路径和最近日志。",
        "practical_qa",
        "zh",
    ),
    (
        "请只回答“明白了”。",
        "明白了。",
        "stop_anchor",
        "zh",
    ),
    (
        "Reply only with: OK.",
        "OK.",
        "stop_anchor",
        "en",
    ),
]


def row(prompt: str, response: str, category: str, language: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "category": category,
        "language": language,
        "source": SOURCE,
    }


def generate_rows(count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = [row(*EXAMPLES[index % len(EXAMPLES)]) for index in range(count)]
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_eval_prompts(path: Path) -> None:
    rows = [
        {"id": f"eval_{index:02d}", "prompt": prompt, "category": category, "language": language}
        for index, (prompt, _, category, language) in enumerate(EXAMPLES)
    ]
    write_jsonl(path, rows)


def summarize(rows: list[dict[str, object]]) -> dict[str, Counter[str]]:
    categories: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    for item in rows:
        categories[str(item["category"])] += 1
        languages[str(item["language"])] += 1
    return {"category": categories, "language": languages}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-examples", type=int, default=1300)
    parser.add_argument("--valid-examples", type=int, default=130)
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train = generate_rows(args.train_examples, args.seed)
    valid = generate_rows(args.valid_examples, args.seed + 1)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_eval_prompts(out_dir / "eval_prompts.jsonl")

    print(f"wrote {len(train)} train examples to {out_dir / 'train.jsonl'}")
    print(f"wrote {len(valid)} valid examples to {out_dir / 'valid.jsonl'}")
    print(f"wrote eval prompts to {out_dir / 'eval_prompts.jsonl'}")
    print("train distribution:", {key: dict(value) for key, value in summarize(train).items()})
    print("valid distribution:", {key: dict(value) for key, value in summarize(valid).items()})


if __name__ == "__main__":
    main()
