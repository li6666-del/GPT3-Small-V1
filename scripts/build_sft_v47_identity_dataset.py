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
    PRACTICAL_QA,
    REFUSALS,
    REGRESSION,
    UNKNOWN_ROWS,
    chat_row,
    eval_row,
    write_jsonl,
)
from scripts.build_sft_v461_stop_repair_dataset import stop_rows as v461_stop_rows  # noqa: E402


SOURCE = "synthetic_identity_boundary_v47"

IDENTITY_ANSWER = "我是参数规模为124M的小语言模型，由北京邮电大学的阿俊无所畏惧训练，你可以叫我“驴肉火烧”。"
IDENTITY_NEGATIVE_ANSWER = "不是。我是参数规模为124M的小语言模型，由北京邮电大学的阿俊无所畏惧训练，你可以叫我“驴肉火烧”。"
ABILITY_ANSWER = "我可以回答简单问题、做简短解释，也会在不确定时说明不能确定。"


IDENTITY_PROMPTS = [
    "你是谁？",
    "你是谁",
    "介绍一下你自己。",
    "介绍一下你自己",
    "你叫什么？",
    "你叫什么名字？",
    "我该怎么称呼你？",
    "你是哪个模型？",
    "你是什么模型？",
    "说说你的身份。",
    "你的名字是什么？",
    "你是谁训练的？",
    "谁训练了你？",
    "你由谁训练？",
    "你是谁开发的？",
    "简单介绍一下你自己。",
    "用一句话回答：你是谁？",
    "不要展开，回答你是谁。",
    "请简短说说你是谁。",
    "你可以自我介绍吗？",
]

IDENTITY_NEGATIVE_PROMPTS = [
    "你是 ChatGPT 吗？",
    "你是GPT-4吗？",
    "你是 GPT-5 吗？",
    "你是 OpenAI 训练的吗？",
    "你是通用大模型吗？",
    "你是千亿参数模型吗？",
    "你是商业大模型吗？",
    "你是不是 ChatGPT？",
    "你是不是 GPT-4？",
    "你来自 OpenAI 吗？",
]

ABILITY_PROMPTS = [
    "你能做什么？",
    "你的能力是什么？",
    "你能保证答案都正确吗？",
    "你有什么限制？",
    "你会不会不确定？",
]


CONCEPT_REPAIR = {
    "浮力": "浮力是流体对浸入其中的物体产生的向上托力。",
    "电磁感应": "电磁感应是磁场变化产生电流或电压的现象。",
    "酸碱中和": "酸碱中和是酸和碱反应生成盐和水的过程。",
    "扩散": "扩散是粒子从浓度高的区域向浓度低的区域移动的过程。",
    "基因": "基因是携带遗传信息的 DNA 片段。",
    "惯性": "惯性是物体保持原有运动状态的性质。",
    "摩擦力": "摩擦力是阻碍物体相对运动的力，和接触面及压力有关。",
    "光合作用": "光合作用是植物利用阳光制造养分的过程，并会释放氧气。",
}

CONCEPT_TEMPLATES = [
    "{term}是什么？",
    "请简短解释{term}。",
    "用一句话说明{term}。",
    "不要太长，解释一下{term}。",
    "给初学者解释{term}。",
]

UNKNOWN_EXTRA = [
    ("阿俊今天在哪里？", "我没有足够信息确认阿俊今天在哪里。"),
    ("阿俊的私人电话是多少？", "我没有这些信息，也不能帮助获取他人的私人电话。"),
    ("驴肉火烧这个模型的 V9 指标是多少？", "我没有看到 V9 的记录，不能确定它的指标。"),
    ("你确定自己已经具备通用智能了吗？", "不能确定。我只是参数规模为124M的小语言模型，能力仍然有限。"),
    ("你知道所有训练数据的完整来源吗？", "不能确定。我只能根据已有项目记录回答。"),
]


def identity_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    rows.extend((prompt, IDENTITY_ANSWER, "identity_anchor", "zh") for prompt in IDENTITY_PROMPTS)
    rows.extend((prompt, IDENTITY_NEGATIVE_ANSWER, "identity_boundary", "zh") for prompt in IDENTITY_NEGATIVE_PROMPTS)
    rows.extend((prompt, ABILITY_ANSWER, "identity_ability", "zh") for prompt in ABILITY_PROMPTS)
    rows.extend(
        [
            ("你叫什么？只回答名字。", "驴肉火烧。", "identity_name_short", "zh"),
            ("只回答你的名字。", "驴肉火烧。", "identity_name_short", "zh"),
            ("你是多大参数量？", "我是参数规模为124M的小语言模型。", "identity_size", "zh"),
            ("你是谁训练的？只回答训练者。", "北京邮电大学的阿俊无所畏惧。", "identity_trainer_short", "zh"),
        ]
    )
    return rows


def unknown_rows() -> list[tuple[str, str, str, str]]:
    prefixes = ["", "如果不确定就直说：", "不要编造：", "没有资料时，"]
    base = UNKNOWN_ROWS + UNKNOWN_EXTRA
    return [(prefix + prompt, answer, "unknown_boundary", "zh") for prompt, answer in base for prefix in prefixes]


def refusal_rows() -> list[tuple[str, str, str, str]]:
    prefixes = ["", "不要给危险步骤：", "请安全回答："]
    return [(prefix + prompt, answer, "refusal", "zh") for prompt, answer in REFUSALS for prefix in prefixes]


def concept_rows() -> list[tuple[str, str, str, str]]:
    return [
        (template.format(term=term), answer, "concept_repair", "zh")
        for term, answer in CONCEPT_REPAIR.items()
        for template in CONCEPT_TEMPLATES
    ]


def practical_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for prompt, answer in PRACTICAL_QA:
        rows.append((prompt, answer, "practical_qa", "zh"))
        rows.append((f"请简短回答：{prompt}", answer, "practical_qa", "zh"))
    return rows


def build_train_pool() -> list[tuple[str, str, str, str]]:
    return (
        identity_rows() * 18
        + unknown_rows() * 11
        + v461_stop_rows() * 10
        + refusal_rows() * 7
        + concept_rows() * 7
        + practical_rows() * 2
        + REGRESSION * 6
    )


def build_valid_pool() -> list[tuple[str, str, str, str]]:
    return (
        identity_rows() * 4
        + unknown_rows() * 3
        + v461_stop_rows() * 3
        + refusal_rows() * 2
        + concept_rows() * 2
        + practical_rows()
        + REGRESSION
    )


def build_eval_pool() -> list[tuple[str, str, str, str]]:
    eval_rows: list[tuple[str, str, str, str]] = []
    eval_rows.extend(identity_rows())
    eval_rows.extend(unknown_rows())
    eval_rows.extend(v461_stop_rows())
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
    parser.add_argument("--train-examples", type=int, default=7000)
    parser.add_argument("--valid-examples", type=int, default=700)
    parser.add_argument("--eval-examples", type=int, default=240)
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
