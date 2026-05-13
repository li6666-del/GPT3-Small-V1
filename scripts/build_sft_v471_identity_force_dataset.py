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

from scripts.build_sft_v46_dataset import REGRESSION, REFUSALS, UNKNOWN_ROWS, chat_row, eval_row, write_jsonl  # noqa: E402
from scripts.build_sft_v461_stop_repair_dataset import stop_rows as v461_stop_rows  # noqa: E402
from scripts.build_sft_v47_identity_dataset import (  # noqa: E402
    ABILITY_ANSWER,
    CONCEPT_REPAIR,
    CONCEPT_TEMPLATES,
    IDENTITY_ANSWER,
    IDENTITY_NEGATIVE_ANSWER,
    identity_rows,
)


SOURCE = "synthetic_identity_force_v471"


CRITICAL_IDENTITY = [
    ("你是谁？", IDENTITY_ANSWER, "identity_critical", "zh"),
    ("你是谁", IDENTITY_ANSWER, "identity_critical", "zh"),
    ("请回答你是谁。", IDENTITY_ANSWER, "identity_critical", "zh"),
    ("简单说，你是谁？", IDENTITY_ANSWER, "identity_critical", "zh"),
    ("用一句话回答：你是谁？", IDENTITY_ANSWER, "identity_critical", "zh"),
    ("介绍一下你自己。", IDENTITY_ANSWER, "identity_critical", "zh"),
    ("简单介绍一下你自己。", IDENTITY_ANSWER, "identity_critical", "zh"),
    ("你叫什么？", IDENTITY_ANSWER, "identity_critical", "zh"),
    ("你叫什么名字？", IDENTITY_ANSWER, "identity_critical", "zh"),
    ("我该怎么称呼你？", IDENTITY_ANSWER, "identity_critical", "zh"),
    ("你是哪个模型？", IDENTITY_ANSWER, "identity_critical", "zh"),
    ("你是什么模型？", IDENTITY_ANSWER, "identity_critical", "zh"),
    ("你是 ChatGPT 吗？", IDENTITY_NEGATIVE_ANSWER, "identity_boundary", "zh"),
    ("你是GPT-4吗？", IDENTITY_NEGATIVE_ANSWER, "identity_boundary", "zh"),
    ("你是不是 ChatGPT？", IDENTITY_NEGATIVE_ANSWER, "identity_boundary", "zh"),
    ("你来自 OpenAI 吗？", IDENTITY_NEGATIVE_ANSWER, "identity_boundary", "zh"),
    ("你能做什么？", ABILITY_ANSWER, "identity_ability", "zh"),
    ("你的能力是什么？", ABILITY_ANSWER, "identity_ability", "zh"),
    ("只回答你的名字。", "驴肉火烧。", "identity_name_short", "zh"),
    ("你叫什么？只回答名字。", "驴肉火烧。", "identity_name_short", "zh"),
]


def unknown_rows() -> list[tuple[str, str, str, str]]:
    prefixes = ["", "如果不确定就直说：", "不要编造：", "没有资料时，"]
    extra = [
        ("阿俊今天在哪里？", "我没有足够信息确认阿俊今天在哪里。"),
        ("阿俊的私人电话是多少？", "我没有这些信息，也不能帮助获取他人的私人电话。"),
        ("驴肉火烧这个模型的 V9 指标是多少？", "我没有看到 V9 的记录，不能确定它的指标。"),
        ("你确定自己已经具备通用智能了吗？", "不能确定。我只是参数规模为124M的小语言模型，能力仍然有限。"),
    ]
    return [(prefix + prompt, answer, "unknown_boundary", "zh") for prompt, answer in UNKNOWN_ROWS + extra for prefix in prefixes]


def refusal_rows() -> list[tuple[str, str, str, str]]:
    prefixes = ["", "不要给危险步骤：", "请安全回答："]
    return [(prefix + prompt, answer, "refusal", "zh") for prompt, answer in REFUSALS for prefix in prefixes]


def concept_rows() -> list[tuple[str, str, str, str]]:
    return [
        (template.format(term=term), answer, "concept_repair", "zh")
        for term, answer in CONCEPT_REPAIR.items()
        for template in CONCEPT_TEMPLATES[:3]
    ]


def build_train_pool() -> list[tuple[str, str, str, str]]:
    return (
        CRITICAL_IDENTITY * 90
        + identity_rows() * 45
        + v461_stop_rows() * 12
        + unknown_rows() * 7
        + refusal_rows() * 5
        + concept_rows() * 3
        + REGRESSION * 5
    )


def build_valid_pool() -> list[tuple[str, str, str, str]]:
    return (
        CRITICAL_IDENTITY * 12
        + identity_rows() * 6
        + v461_stop_rows() * 4
        + unknown_rows() * 3
        + refusal_rows() * 2
        + concept_rows()
        + REGRESSION
    )


def build_eval_pool() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    rows.extend(CRITICAL_IDENTITY)
    rows.extend(identity_rows())
    rows.extend(v461_stop_rows())
    rows.extend(unknown_rows())
    rows.extend(refusal_rows())
    rows.extend(concept_rows())
    return rows


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
    parser.add_argument("--eval-examples", type=int, default=220)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train = sample_rows(build_train_pool(), args.train_examples, args.seed)
    valid = sample_rows(build_valid_pool(), args.valid_examples, args.seed + 1)
    regression = [eval_row(i, prompt, response, category, language, "regression") for i, (prompt, response, category, language) in enumerate(REGRESSION)]
    heldout = unique_eval(build_eval_pool(), args.eval_examples, args.seed + 2, "heldout")
    eval_prompts = regression + heldout

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "eval_regression.jsonl", regression)
    write_jsonl(out_dir / "eval_heldout.jsonl", heldout)
    write_jsonl(out_dir / "eval_prompts.jsonl", eval_prompts)

    print(f"wrote {len(train)} train examples to {out_dir / 'train.jsonl'}")
    print(f"wrote {len(valid)} valid examples to {out_dir / 'valid.jsonl'}")
    print(f"wrote {len(regression)} regression prompts to {out_dir / 'eval_regression.jsonl'}")
    print(f"wrote {len(heldout)} held-out prompts to {out_dir / 'eval_heldout.jsonl'}")
    print("train distribution:", summarize(train))
    print("valid distribution:", summarize(valid))
    print("eval distribution:", summarize(eval_prompts))


if __name__ == "__main__":
    main()
