from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_sft_v46_dataset import REGRESSION, REFUSALS, UNKNOWN_ROWS, chat_row, eval_row, write_jsonl  # noqa: E402
from scripts.build_sft_v461_stop_repair_dataset import stop_rows as v461_stop_rows  # noqa: E402
from scripts.build_sft_v48_identity_format_dataset import (  # noqa: E402
    ABILITY,
    IDENTITY_FULL,
    IDENTITY_NEGATIVE,
    IDENTITY_NAME,
    IDENTITY_SIZE,
    IDENTITY_SIZE_SENTENCE,
    IDENTITY_TRAINER,
    LIMITS,
)


SOURCE = "synthetic_identity_short_force_v481"


NAME_SHORT = [
    "你叫什么？只回答名字。",
    "只回答你的名字。",
    "你的名字是什么？只回答名字。",
    "你叫什么名字？只说名字。",
    "我该怎么称呼你？只回答称呼。",
    "不要解释，只回答你的名字。",
    "只输出你的名字。",
    "只说模型名字。",
    "名字？只答名字。",
    "你的称呼是什么？只回答称呼。",
    "请只回答：你的名字。",
]

TRAINER_SHORT = [
    "你是谁训练的？只回答训练者。",
    "只回答是谁训练了你。",
    "谁训练了你？只回答训练者。",
    "你由谁训练？只回答训练者。",
    "不要解释，只说训练者。",
    "训练者是谁？只回答训练者。",
    "谁训练的你？只说名字和单位。",
]

SIZE_SHORT = [
    "你是多大参数量？只回答数字。",
    "只回答你的参数规模。",
    "你的参数量是多少？只回答参数量。",
    "参数规模？只回答。",
    "你有多少参数？只回答参数量。",
    "不要解释，只说参数规模。",
]

FULL = [
    "你是谁？",
    "介绍一下你自己。",
    "简单介绍一下你自己。",
    "你是什么模型？",
    "你是哪个模型？",
    "请简短说说你是谁。",
]

NAME_NORMAL = [
    "你叫什么？",
    "你叫什么名字？",
    "你的名字是什么？",
    "我该怎么称呼你？",
]

TRAINER_NORMAL = [
    "你是谁训练的？",
    "谁训练了你？",
    "你由谁训练？",
]

SIZE_NORMAL = [
    "你是多大参数量？",
    "你的参数量是多少？",
    "你的参数规模是多少？",
]

NEGATIVE = [
    "你是 ChatGPT 吗？",
    "你是不是 ChatGPT？",
    "你是GPT-4吗？",
    "你是不是 GPT-4？",
    "你来自 OpenAI 吗？",
]

ABILITY_LIMIT = [
    ("你能做什么？", ABILITY, "identity_ability", "zh"),
    ("你的能力是什么？", ABILITY, "identity_ability", "zh"),
    ("你有什么限制？", LIMITS, "identity_limit", "zh"),
    ("你能保证答案都正确吗？", LIMITS, "identity_limit", "zh"),
]


def critical_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    rows.extend((prompt, IDENTITY_NAME, "identity_name_short", "zh") for prompt in NAME_SHORT)
    rows.extend((prompt, IDENTITY_TRAINER, "identity_trainer_short", "zh") for prompt in TRAINER_SHORT)
    rows.extend((prompt, IDENTITY_SIZE, "identity_size_short", "zh") for prompt in SIZE_SHORT)
    rows.extend((prompt, IDENTITY_FULL, "identity_full", "zh") for prompt in FULL)
    rows.extend((prompt, IDENTITY_FULL, "identity_name_normal", "zh") for prompt in NAME_NORMAL)
    rows.extend((prompt, IDENTITY_FULL, "identity_trainer_normal", "zh") for prompt in TRAINER_NORMAL)
    rows.extend((prompt, IDENTITY_SIZE_SENTENCE, "identity_size_normal", "zh") for prompt in SIZE_NORMAL)
    rows.extend((prompt, IDENTITY_NEGATIVE, "identity_boundary", "zh") for prompt in NEGATIVE)
    rows.extend(ABILITY_LIMIT)
    return rows


def unknown_rows() -> list[tuple[str, str, str, str]]:
    prefixes = ["", "如果不确定就直说：", "不要编造："]
    extra = [
        ("阿俊今天在哪里？", "我没有足够信息确认阿俊今天在哪里。"),
        ("驴肉火烧这个模型的 V9 指标是多少？", "我没有看到 V9 的记录，不能确定它的指标。"),
        ("你能确定所有回答都正确吗？", "不能确定。复杂事实需要验证，不确定时我会说明不能确定。"),
    ]
    return [(prefix + prompt, answer, "unknown_boundary", "zh") for prompt, answer in UNKNOWN_ROWS + extra for prefix in prefixes]


def refusal_rows() -> list[tuple[str, str, str, str]]:
    return [(prompt, answer, "refusal", "zh") for prompt, answer in REFUSALS]


def build_train_pool() -> list[tuple[str, str, str, str]]:
    crit = critical_rows()
    short = [row for row in crit if row[2] in {"identity_name_short", "identity_trainer_short", "identity_size_short"}]
    return (
        short * 70
        + crit * 18
        + v461_stop_rows() * 8
        + unknown_rows() * 5
        + refusal_rows() * 4
        + REGRESSION * 3
    )


def build_valid_pool() -> list[tuple[str, str, str, str]]:
    return (
        critical_rows() * 5
        + v461_stop_rows() * 3
        + unknown_rows() * 2
        + refusal_rows() * 2
        + REGRESSION
    )


def build_eval_pool() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    rows.extend(critical_rows())
    rows.extend(v461_stop_rows())
    rows.extend(unknown_rows())
    rows.extend(refusal_rows())
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
    parser.add_argument("--train-examples", type=int, default=3500)
    parser.add_argument("--valid-examples", type=int, default=400)
    parser.add_argument("--eval-examples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train = sample_rows(build_train_pool(), args.train_examples, args.seed)
    valid = sample_rows(build_valid_pool(), args.valid_examples, args.seed + 1)
    regression = [eval_row(i, prompt, response, category, language, "regression") for i, (prompt, response, category, language) in enumerate(REGRESSION)]
    critical_eval = [eval_row(i, prompt, response, category, language, "critical") for i, (prompt, response, category, language) in enumerate(critical_rows())]
    heldout = unique_eval(build_eval_pool(), args.eval_examples, args.seed + 2, "heldout")
    eval_prompts = regression + critical_eval + heldout

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "eval_regression.jsonl", regression)
    write_jsonl(out_dir / "eval_critical.jsonl", critical_eval)
    write_jsonl(out_dir / "eval_heldout.jsonl", heldout)
    write_jsonl(out_dir / "eval_prompts.jsonl", eval_prompts)

    print(f"wrote {len(train)} train examples to {out_dir / 'train.jsonl'}")
    print(f"wrote {len(valid)} valid examples to {out_dir / 'valid.jsonl'}")
    print(f"wrote {len(regression)} regression prompts to {out_dir / 'eval_regression.jsonl'}")
    print(f"wrote {len(critical_eval)} critical prompts to {out_dir / 'eval_critical.jsonl'}")
    print(f"wrote {len(heldout)} held-out prompts to {out_dir / 'eval_heldout.jsonl'}")
    print("train distribution:", summarize(train))
    print("valid distribution:", summarize(valid))
    print("eval distribution:", summarize(eval_prompts))


if __name__ == "__main__":
    main()
