from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_sft_v46_dataset import (  # noqa: E402
    KNOWN_CONCEPTS,
    PRACTICAL_QA,
    REFUSALS,
    REGRESSION,
    UNKNOWN_ROWS,
    chat_row,
    eval_row,
    write_jsonl,
)


SOURCE = "synthetic_stop_anchor_repair_v461"


EN_STOP_VALUES = [
    "OK.",
    "Done.",
    "Yes.",
    "No.",
    "Understood.",
    "Completed.",
    "Noted.",
    "Ready.",
]

ZH_STOP_VALUES = [
    "收到。",
    "可以。",
    "完成。",
    "明白。",
    "好的。",
    "不用。",
]

EN_STOP_TEMPLATES = [
    "Reply only with: {answer}",
    "Only output: {answer}",
    "Do not explain. Answer exactly: {answer}",
    "No extra words. Reply: {answer}",
    "Return exactly this text: {answer}",
    "Answer with one word only: {answer}",
]

ZH_STOP_TEMPLATES = [
    "只回答：{answer}",
    "请只输出“{answer}”",
    "不要解释，只回答：{answer}",
    "只输出这个词：{answer}",
    "请原样回复：{answer}",
]


def stop_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for answer in EN_STOP_VALUES:
        for template in EN_STOP_TEMPLATES:
            rows.append((template.format(answer=answer), answer, "stop_anchor", "en"))
    for answer in ZH_STOP_VALUES:
        for template in ZH_STOP_TEMPLATES:
            rows.append((template.format(answer=answer), answer, "stop_anchor", "zh"))

    rows.extend(
        [
            ("Reply only with: Done.", "Done.", "stop_anchor", "en"),
            ("Reply only with: Done", "Done.", "stop_anchor", "en"),
            ("Only output: Done.", "Done.", "stop_anchor", "en"),
            ("Do not explain. Reply only with: Done.", "Done.", "stop_anchor", "en"),
            ("只回答英文：Done.", "Done.", "stop_anchor", "mixed"),
            ("请只回复英文 Done.", "Done.", "stop_anchor", "mixed"),
        ]
    )
    return rows


def unknown_rows() -> list[tuple[str, str, str, str]]:
    prefixes = ["", "如果不确定就直说：", "不要编造：", "没有资料时，"]
    return [(prefix + prompt, answer, "unknown_boundary", "zh") for prompt, answer in UNKNOWN_ROWS for prefix in prefixes]


def refusal_rows() -> list[tuple[str, str, str, str]]:
    prefixes = ["", "不要给危险步骤：", "请安全回答："]
    return [(prefix + prompt, answer, "refusal", "zh") for prompt, answer in REFUSALS for prefix in prefixes]


def practical_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for prompt, answer in PRACTICAL_QA:
        rows.append((prompt, answer, "practical_qa", "zh"))
        rows.append((f"请简短回答：{prompt}", answer, "practical_qa", "zh"))
    return rows


def concept_rows() -> list[tuple[str, str, str, str]]:
    templates = ["请用一句话解释{term}。", "{term}是什么？", "简短说明{term}。"]
    selected = [
        "机器学习",
        "过拟合",
        "验证集",
        "checkpoint",
        "混合专家模型",
        "前馈网络",
        "光合作用",
        "浮力",
    ]
    rows: list[tuple[str, str, str, str]] = []
    stress_answers = {"浮力": "浮力是流体对浸入其中的物体产生的向上托力。"}
    for term in selected:
        answer = stress_answers[term] if term in stress_answers else KNOWN_CONCEPTS[term]
        for template in templates:
            rows.append((template.format(term=term), answer, "concept", "zh"))
    return rows


def build_train_pool() -> list[tuple[str, str, str, str]]:
    return (
        stop_rows() * 18
        + unknown_rows() * 7
        + refusal_rows() * 7
        + practical_rows() * 3
        + concept_rows() * 3
        + REGRESSION * 8
    )


def build_valid_pool() -> list[tuple[str, str, str, str]]:
    return (
        stop_rows() * 4
        + unknown_rows() * 2
        + refusal_rows() * 2
        + practical_rows()
        + concept_rows()
        + REGRESSION
    )


def build_eval_pool() -> list[tuple[str, str, str, str]]:
    eval_rows: list[tuple[str, str, str, str]] = []
    eval_rows.extend(stop_rows())
    eval_rows.extend(unknown_rows())
    eval_rows.extend(refusal_rows())
    eval_rows.extend(concept_rows())
    eval_rows.extend(practical_rows())
    return eval_rows


def sample_rows(pool: list[tuple[str, str, str, str]], count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = [chat_row(*rng.choice(pool), source=SOURCE) for _ in range(count)]
    rng.shuffle(rows)
    return rows


def unique_eval(pool: list[tuple[str, str, str, str]], count: int, seed: int, eval_set: str) -> list[dict[str, object]]:
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for prompt, response, category, language in shuffled:
        if prompt in seen:
            continue
        seen.add(prompt)
        rows.append(eval_row(len(rows), prompt, response, category, language, eval_set))
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise ValueError(f"only built {len(rows)} unique {eval_set} prompts, need {count}")
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    categories: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for item in rows:
        categories[str(item.get("category"))] += 1
        languages[str(item.get("language"))] += 1
        sources[str(item.get("source", item.get("eval_set", "unknown")))] += 1
    return {"category": dict(categories), "language": dict(languages), "source": dict(sources)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-examples", type=int, default=5000)
    parser.add_argument("--valid-examples", type=int, default=500)
    parser.add_argument("--eval-examples", type=int, default=140)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train = sample_rows(build_train_pool(), args.train_examples, args.seed)
    valid = sample_rows(build_valid_pool(), args.valid_examples, args.seed + 1)
    regression = [eval_row(i, prompt, response, category, language, "regression") for i, (prompt, response, category, language) in enumerate(REGRESSION)]
    repair_eval = unique_eval(build_eval_pool(), args.eval_examples, args.seed + 2, "repair")
    eval_prompts = regression + repair_eval

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "eval_regression.jsonl", regression)
    write_jsonl(out_dir / "eval_repair.jsonl", repair_eval)
    write_jsonl(out_dir / "eval_prompts.jsonl", eval_prompts)

    print(f"wrote {len(train)} train examples to {out_dir / 'train.jsonl'}")
    print(f"wrote {len(valid)} valid examples to {out_dir / 'valid.jsonl'}")
    print(f"wrote {len(regression)} regression prompts to {out_dir / 'eval_regression.jsonl'}")
    print(f"wrote {len(repair_eval)} repair prompts to {out_dir / 'eval_repair.jsonl'}")
    print("train distribution:", summarize(train))
    print("valid distribution:", summarize(valid))
    print("eval distribution:", summarize(eval_prompts))


if __name__ == "__main__":
    main()
