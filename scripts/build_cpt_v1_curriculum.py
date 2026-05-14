from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


EOT = "<|endoftext|>"


FACTS = [
    ("法国的首都是巴黎。", "地理"),
    ("中国的首都是北京。", "地理"),
    ("日本的首都是东京。", "地理"),
    ("英国的首都是伦敦。", "地理"),
    ("德国的首都是柏林。", "地理"),
    ("一周有七天。", "时间"),
    ("一年通常有十二个月。", "时间"),
    ("一天通常有二十四小时。", "时间"),
    ("一分钟有六十秒。", "时间"),
    ("水的化学式是 H2O。", "科学"),
    ("标准大气压下，水通常在 100 摄氏度沸腾。", "科学"),
    ("水通常在 0 摄氏度结冰。", "科学"),
    ("地球绕太阳运行。", "科学"),
    ("月亮本身不会像太阳一样发光。", "科学"),
    ("人呼吸主要需要氧气。", "科学"),
    ("植物通过光合作用吸收二氧化碳并释放氧气。", "科学"),
    ("键盘通常属于输入设备。", "计算机"),
    ("显示器通常属于输出设备。", "计算机"),
    ("CPU 是计算机中负责执行指令的重要部件。", "计算机"),
    ("内存用于临时保存正在使用的数据。", "计算机"),
    ("春节是中国重要的传统节日。", "常识"),
    ("太阳通常从东方升起。", "常识"),
    ("分子可以由多个原子组成。", "科学"),
    ("过拟合通常说明模型对训练集记得太死，泛化能力不足。", "机器学习"),
    ("训练集用于训练模型，验证集用于观察训练过程中的泛化表现。", "机器学习"),
    ("测试集或 held-out 数据不能混入训练集。", "机器学习"),
    ("checkpoint 通常用于保存模型参数、优化器状态或训练进度。", "机器学习"),
    ("valid loss 不是唯一标准，还要结合生成评测和 held-out 表现。", "机器学习"),
    ("generation_eval 用于检查模型在固定提示上的生成结果。", "机器学习"),
    ("best-step selection 用于从多个 checkpoint 中选择表现更稳的 step。", "机器学习"),
]


FACT_QUESTIONS = [
    ("法国的首都叫什么？", "法国的首都是巴黎。"),
    ("中国首都是哪座城市？", "中国的首都是北京。"),
    ("一周总共有几天？", "一周总共有七天。"),
    ("一年通常有几个月？", "一年通常有十二个月。"),
    ("一天通常有多少小时？", "一天通常有二十四小时。"),
    ("一分钟有多少秒？", "一分钟有六十秒。"),
    ("水的化学式是什么？", "水的化学式是 H2O。"),
    ("标准大气压下水通常多少摄氏度沸腾？", "标准大气压下水通常约 100 摄氏度沸腾。"),
    ("checkpoint 通常用来保存什么？", "checkpoint 通常用来保存模型参数和训练状态。"),
    ("过拟合通常说明模型对训练集记得太死吗？", "是的，过拟合通常说明模型对训练集记得太死。"),
]


PROJECT_TERMS = [
    "held-out 数据用于评估泛化能力，不能进入训练集。",
    "valid.jsonl 通常用于训练过程中的验证，不等同于最终 held-out。",
    "generation_eval 记录模型对固定 prompt 的生成结果。",
    "failure memory 记录失败原因，方便下一轮避免重复错误。",
    "main gate 用于保护主线能力，stage gate 用于检查当前阶段目标。",
    "observe 指标用于观察风险，不一定否决 checkpoint。",
    "latest checkpoint 不一定最好，best-step selection 要结合评测结果。",
    "SFT 教模型按照助手格式回答，continued pretrain 补充语言分布和知识。",
]


SHORT_EXPLANATIONS = [
    "简单事实问题适合用短句回答，回答应直接、完整、不过度展开。",
    "如果缺少上下文，稳妥的做法是说明不能确定，而不是编造答案。",
    "小模型容量有限，适合先稳定短问答、拒答、未知边界和停止行为。",
    "中文短事实语料可以帮助模型更习惯完整的一句话表达。",
    "基础算术语料可以提供数字关系，但不能保证复杂推理能力。",
]


EN_SNIPPETS = [
    "The capital of France is Paris.",
    "A week has seven days.",
    "A year usually has twelve months.",
    "Water is H2O.",
    "The answer is YES.",
    "The answer is OK.",
    "Done means the task is complete.",
    "Ready means prepared to continue.",
]


def arithmetic_records(rng: random.Random) -> list[str]:
    records: list[str] = []
    op = rng.choices(["add", "sub", "mul"], weights=[0.45, 0.30, 0.25], k=1)[0]
    if op == "add":
        a = rng.randint(0, 30)
        b = rng.randint(0, 30)
        c = a + b
        records.extend(
            [
                f"{a} + {b} = {c}。",
                f"{a} 加 {b} 等于 {c}。",
                f"计算 {a} 加 {b}，结果是 {c}。",
            ]
        )
    elif op == "sub":
        a = rng.randint(0, 40)
        b = rng.randint(0, a)
        c = a - b
        records.extend(
            [
                f"{a} - {b} = {c}。",
                f"{a} 减 {b} 等于 {c}。",
                f"计算 {a} 减 {b}，结果是 {c}。",
            ]
        )
    else:
        a = rng.randint(0, 12)
        b = rng.randint(0, 12)
        c = a * b
        records.extend(
            [
                f"{a} × {b} = {c}。",
                f"{a} 乘以 {b} 等于 {c}。",
                f"计算 {a} 乘以 {b}，结果是 {c}。",
            ]
        )
    return records


def fact_record(rng: random.Random) -> str:
    sentence, label = rng.choice(FACTS)
    variants = [
        sentence,
        f"关于{label}，可以记住：{sentence}",
        f"一个常见事实是：{sentence}",
        f"{sentence}这个事实适合用简短中文回答。",
    ]
    return rng.choice(variants)


def qa_record(rng: random.Random) -> str:
    question, answer = rng.choice(FACT_QUESTIONS)
    variants = [
        f"问题：{question}\n答案：{answer}",
        f"{question}简短回答是：{answer}",
        f"对于“{question}”，稳定回答应包含：{answer}",
    ]
    return rng.choice(variants)


def project_record(rng: random.Random) -> str:
    first = rng.choice(PROJECT_TERMS)
    if rng.random() < 0.35:
        second = rng.choice(PROJECT_TERMS)
        if second != first:
            return first + "\n" + second
    return first


def explanation_record(rng: random.Random) -> str:
    first = rng.choice(SHORT_EXPLANATIONS)
    if rng.random() < 0.5:
        second = rng.choice(SHORT_EXPLANATIONS)
        if second != first:
            return first + "\n" + second
    return first


def english_record(rng: random.Random) -> str:
    return rng.choice(EN_SNIPPETS)


def make_record(rng: random.Random) -> tuple[str, str]:
    category = rng.choices(
        ["fact", "qa", "arithmetic", "project", "explain", "english"],
        weights=[0.34, 0.20, 0.26, 0.08, 0.08, 0.04],
        k=1,
    )[0]
    if category == "fact":
        text = fact_record(rng)
    elif category == "qa":
        text = qa_record(rng)
    elif category == "arithmetic":
        text = "\n".join(arithmetic_records(rng))
    elif category == "project":
        text = project_record(rng)
    elif category == "explain":
        text = explanation_record(rng)
    else:
        text = english_record(rng)
    return text, category


def write_split(path: Path, target_bytes: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    counts: Counter[str] = Counter()
    bytes_written = 0
    records = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        while bytes_written < target_bytes:
            text, category = make_record(rng)
            record = text.strip() + EOT + "\n"
            f.write(record)
            encoded_len = len(record.encode("utf-8"))
            bytes_written += encoded_len
            counts[category] += 1
            records += 1
    return {"bytes": bytes_written, "records": records, **dict(counts)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V1 continued-pretrain curriculum text.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/cpt/v1_curriculum"))
    parser.add_argument("--train-bytes", type=int, default=180_000_000)
    parser.add_argument("--valid-bytes", type=int, default=8_000_000)
    parser.add_argument("--seed", type=int, default=20260514)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_stats = write_split(args.output_dir / "train.txt", args.train_bytes, args.seed)
    valid_stats = write_split(args.output_dir / "valid.txt", args.valid_bytes, args.seed + 1)
    manifest = {
        "name": "cpt_v1_curriculum",
        "seed": args.seed,
        "train": train_stats,
        "valid": valid_stats,
        "notes": [
            "Plain-text continued pretrain curriculum, not SFT JSONL.",
            "Focused on Chinese short facts, simple QA statements, arithmetic text, and small project-term dose.",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
