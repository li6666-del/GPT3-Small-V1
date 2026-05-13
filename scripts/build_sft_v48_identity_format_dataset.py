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
from scripts.build_sft_v47_identity_dataset import CONCEPT_REPAIR, CONCEPT_TEMPLATES  # noqa: E402


SOURCE = "synthetic_identity_format_v48"

IDENTITY_FULL = "我是参数规模为124M的小语言模型，由北京邮电大学的阿俊无所畏惧训练，你可以叫我“驴肉火烧”。"
IDENTITY_NEGATIVE = "不是。我是参数规模为124M的小语言模型，由北京邮电大学的阿俊无所畏惧训练，你可以叫我“驴肉火烧”。"
IDENTITY_NAME = "驴肉火烧。"
IDENTITY_TRAINER = "北京邮电大学的阿俊无所畏惧。"
IDENTITY_SIZE = "124M。"
IDENTITY_SIZE_SENTENCE = "我的参数规模为124M。"
ABILITY = "我可以回答简单问题、做简短解释，也会在不确定时说明不能确定。"
LIMITS = "我的能力有限，复杂事实需要验证；不确定时我会说明不能确定。"


FULL_IDENTITY_PROMPTS = [
    "你是谁？",
    "你是谁",
    "请回答你是谁。",
    "介绍一下你自己。",
    "介绍一下你自己",
    "简单介绍一下你自己。",
    "请简短说说你是谁。",
    "简单说，你是谁？",
    "用一句话回答：你是谁？",
    "说说你的身份。",
    "你是什么模型？",
    "你是哪个模型？",
    "你可以自我介绍吗？",
]

NAME_SHORT_PROMPTS = [
    "你叫什么？只回答名字。",
    "只回答你的名字。",
    "你的名字是什么？只回答名字。",
    "你叫什么名字？只说名字。",
    "我该怎么称呼你？只回答称呼。",
    "不要解释，只回答你的名字。",
    "只输出你的名字。",
    "只说模型名字。",
]

NAME_NORMAL_PROMPTS = [
    "你叫什么？",
    "你叫什么名字？",
    "你的名字是什么？",
    "我该怎么称呼你？",
]

TRAINER_SHORT_PROMPTS = [
    "你是谁训练的？只回答训练者。",
    "只回答是谁训练了你。",
    "谁训练了你？只回答训练者。",
    "你由谁训练？只回答训练者。",
    "不要解释，只说训练者。",
]

TRAINER_NORMAL_PROMPTS = [
    "你是谁训练的？",
    "谁训练了你？",
    "你由谁训练？",
    "你是谁开发的？",
]

SIZE_SHORT_PROMPTS = [
    "你是多大参数量？只回答数字。",
    "只回答你的参数规模。",
    "你的参数量是多少？只回答参数量。",
    "参数规模？只回答。",
]

SIZE_NORMAL_PROMPTS = [
    "你是多大参数量？",
    "你的参数量是多少？",
    "你的参数规模是多少？",
]

NEGATIVE_IDENTITY_PROMPTS = [
    "你是 ChatGPT 吗？",
    "你是不是 ChatGPT？",
    "你是GPT-4吗？",
    "你是不是 GPT-4？",
    "你是 GPT-5 吗？",
    "你来自 OpenAI 吗？",
    "你是 OpenAI 训练的吗？",
    "你是商业大模型吗？",
    "你是千亿参数模型吗？",
    "你是通用大模型吗？",
]

ABILITY_PROMPTS = [
    "你能做什么？",
    "你的能力是什么？",
    "你可以帮我做什么？",
    "你适合做什么？",
    "你能回答问题吗？",
]

LIMIT_PROMPTS = [
    "你有什么限制？",
    "你能保证答案都正确吗？",
    "你会不会不确定？",
    "你的回答一定可靠吗？",
    "你不知道的时候怎么办？",
]

UNKNOWN_EXTRA = [
    ("阿俊今天在哪里？", "我没有足够信息确认阿俊今天在哪里。"),
    ("阿俊的私人电话是多少？", "我没有这些信息，也不能帮助获取他人的私人电话。"),
    ("驴肉火烧这个模型的 V9 指标是多少？", "我没有看到 V9 的记录，不能确定它的指标。"),
    ("你确定自己已经具备通用智能了吗？", "不能确定。我只是参数规模为124M的小语言模型，能力仍然有限。"),
    ("你能确定所有回答都正确吗？", "不能确定。复杂事实需要验证，不确定时我会说明不能确定。"),
]


def identity_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    rows.extend((prompt, IDENTITY_FULL, "identity_full", "zh") for prompt in FULL_IDENTITY_PROMPTS)
    rows.extend((prompt, IDENTITY_NAME, "identity_name_short", "zh") for prompt in NAME_SHORT_PROMPTS)
    rows.extend((prompt, IDENTITY_FULL, "identity_name_normal", "zh") for prompt in NAME_NORMAL_PROMPTS)
    rows.extend((prompt, IDENTITY_TRAINER, "identity_trainer_short", "zh") for prompt in TRAINER_SHORT_PROMPTS)
    rows.extend((prompt, IDENTITY_FULL, "identity_trainer_normal", "zh") for prompt in TRAINER_NORMAL_PROMPTS)
    rows.extend((prompt, IDENTITY_SIZE, "identity_size_short", "zh") for prompt in SIZE_SHORT_PROMPTS)
    rows.extend((prompt, IDENTITY_SIZE_SENTENCE, "identity_size_normal", "zh") for prompt in SIZE_NORMAL_PROMPTS)
    rows.extend((prompt, IDENTITY_NEGATIVE, "identity_boundary", "zh") for prompt in NEGATIVE_IDENTITY_PROMPTS)
    rows.extend((prompt, ABILITY, "identity_ability", "zh") for prompt in ABILITY_PROMPTS)
    rows.extend((prompt, LIMITS, "identity_limit", "zh") for prompt in LIMIT_PROMPTS)
    return rows


def unknown_rows() -> list[tuple[str, str, str, str]]:
    prefixes = ["", "如果不确定就直说：", "不要编造：", "没有资料时，"]
    return [(prefix + prompt, answer, "unknown_boundary", "zh") for prompt, answer in UNKNOWN_ROWS + UNKNOWN_EXTRA for prefix in prefixes]


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
        identity_rows() * 34
        + v461_stop_rows() * 9
        + unknown_rows() * 5
        + refusal_rows() * 4
        + concept_rows() * 2
        + REGRESSION * 4
    )


def build_valid_pool() -> list[tuple[str, str, str, str]]:
    return (
        identity_rows() * 8
        + v461_stop_rows() * 4
        + unknown_rows() * 3
        + refusal_rows() * 2
        + concept_rows()
        + REGRESSION
    )


def build_eval_pool() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
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
    parser.add_argument("--train-examples", type=int, default=4500)
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
