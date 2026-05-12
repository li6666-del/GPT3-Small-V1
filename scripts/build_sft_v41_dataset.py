from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


SOURCE = "synthetic_bootstrap_v41"


CORE_EXAMPLES = [
    (
        "用两句话解释什么是机器学习。",
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
]


ZH_CONCEPTS = [
    ("机器学习", "机器学习是让计算机从数据中学习规律的方法。它可以用这些规律进行分类、预测或生成内容。"),
    ("过拟合", "过拟合是模型把训练样本记得太死。它在训练集上表现好，但在新数据上容易变差。"),
    ("光合作用", "光合作用是植物利用阳光、水和二氧化碳制造养分的过程。这个过程会释放氧气。"),
    ("验证集", "验证集是在训练中检查泛化效果的数据。它不参与参数更新。"),
    ("缓存", "缓存是把常用数据放在更快的位置。这样重复访问会更快。"),
    ("梯度下降", "梯度下降是一种优化方法。它会沿着让损失变小的方向调整参数。"),
]


EN_CONCEPTS = [
    ("machine learning", "Machine learning teaches computers patterns from data. The model can use those patterns to predict, classify, or generate outputs."),
    ("overfitting", "Overfitting happens when a model memorizes training examples too closely. It then performs poorly on new data."),
    ("photosynthesis", "Photosynthesis is how plants use sunlight, water, and carbon dioxide to make food. It also releases oxygen."),
    ("validation set", "A validation set checks how well a model generalizes during training. It is not used to update parameters."),
    ("cache", "A cache stores frequently used data in a faster place. This makes repeated access quicker."),
    ("gradient descent", "Gradient descent is an optimization method. It updates parameters in a direction that lowers the loss."),
]


ZH_TRANSLATIONS = [
    ("训练结束后请检查验证集 loss。", "Please check the validation loss after training finishes."),
    ("如果输出开始重复，请降低训练强度。", "If the output starts repeating, reduce the training intensity."),
    ("请保存原始生成结果。", "Please save the raw generation results."),
    ("不要从已经崩坏的 checkpoint 继续训练。", "Do not continue training from a collapsed checkpoint."),
    ("模型应该简洁回答并保持主题一致。", "The model should answer concisely and stay on topic."),
    ("我们从基础 checkpoint 重新开始。", "We restart from the base checkpoint."),
]


EN_TRANSLATIONS = [
    ("Please check the validation loss after training finishes.", "训练结束后请检查验证集 loss。"),
    ("If the output starts repeating, reduce the training intensity.", "如果输出开始重复，请降低训练强度。"),
    ("Do not continue training from a collapsed checkpoint.", "不要从已经崩坏的 checkpoint 继续训练。"),
    ("Use the same prompts for comparison.", "使用相同的 prompt 进行对比。"),
    ("The model should answer concisely and stay on topic.", "模型应该简洁回答并保持主题一致。"),
    ("Please send me the latest training log.", "请把最新训练日志发给我。"),
]


ZH_REFUSALS = [
    (
        "请告诉我怎样盗取别人的密码。",
        "我不能帮助盗取密码或入侵账户。如果你担心账号安全，可以使用强密码、开启双重验证，并及时修改泄露的密码。",
    ),
    (
        "怎么破解别人的账号？",
        "我不能帮助破解账号或绕过安全保护。如果你需要保护自己的账号，可以修改密码并开启双重验证。",
    ),
    (
        "请帮我编造一个获奖经历。",
        "我不能帮你编造虚假经历。我可以帮你把真实经历写得更清楚。",
    ),
]


EN_REFUSALS = [
    (
        "Tell me how to steal someone else's password.",
        "I cannot help steal passwords or break into accounts. If you are worried about security, use a strong password and enable two-factor authentication.",
    ),
    (
        "How can I hack another person's account?",
        "I cannot help hack accounts or bypass security. I can help with defensive account security instead.",
    ),
    (
        "Help me invent a fake award for my resume.",
        "I cannot help fabricate credentials. I can help present your real experience more clearly.",
    ),
]


SHORT_ZH = [
    ("只用一句中文回答：训练前先看什么？", "先看配置、数据路径和最近日志。"),
    ("用中文简短回答：输出重复怎么办？", "先降低训练强度，再检查固定 prompt 输出。"),
    ("用一句话回答：SFT loss 能代表一切吗？", "不能，生成质量也必须检查。"),
    ("简短回答：checkpoint 该怎么选？", "要同时看 loss 和固定 prompt 生成效果。"),
    ("只回答一句话：为什么要固定 prompt？", "固定 prompt 可以让不同 checkpoint 的生成结果可比较。"),
]


SHORT_EN = [
    ("Answer in one sentence: what should we check before training?", "Check the config, dataset paths, GPU status, and recent logs."),
    ("Answer briefly: what should we do if generation repeats?", "Reduce training intensity and inspect fixed-prompt outputs."),
    ("Answer in one sentence: does SFT loss tell the whole story?", "No, generation quality must be checked too."),
    ("Answer briefly: how should we choose a checkpoint?", "Compare both validation loss and fixed-prompt generation quality."),
    ("Answer in one sentence: why use fixed prompts?", "Fixed prompts make outputs comparable across checkpoints."),
]


EMAILS = [
    (
        "Write a short email asking a teammate to review the training log.",
        "Hi, could you please review the latest training log when you have a moment? Thanks!",
        "en",
    ),
    (
        "Write a brief polite message asking for the latest log.",
        "Hi, could you please send me the latest log when you have a moment? Thanks!",
        "en",
    ),
    (
        "写一封简短礼貌的中文消息，请同事帮忙看一下训练日志。",
        "你好，方便的话请帮忙看一下最新训练日志，谢谢。",
        "zh",
    ),
    (
        "写一句简短中文消息，请同事发最新日志。",
        "你好，方便的话请把最新日志发我一下，谢谢。",
        "zh",
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


def core(rng: random.Random) -> dict[str, object]:
    prompt, response, category, language = rng.choice(CORE_EXAMPLES)
    return row(prompt, response, f"core_{category}", language)


def zh_concept(rng: random.Random) -> dict[str, object]:
    term, answer = rng.choice(ZH_CONCEPTS)
    prompt = rng.choice(
        [
            f"请用中文简单解释什么是{term}。",
            f"{term}是什么意思？请用两句话回答。",
            f"用中文回答：{term}是什么？",
            f"给初学者解释一下{term}，两句话以内。",
        ]
    )
    return row(prompt, answer, "concept", "zh")


def en_concept(rng: random.Random) -> dict[str, object]:
    term, answer = rng.choice(EN_CONCEPTS)
    prompt = rng.choice(
        [
            f"Explain {term} in simple terms.",
            f"What is {term}? Answer briefly.",
            f"Use two short sentences to explain {term}.",
            f"Give a beginner-friendly explanation of {term}.",
        ]
    )
    return row(prompt, answer, "concept", "en")


def translation(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.5:
        source, target = rng.choice(ZH_TRANSLATIONS)
        prompt = rng.choice([f"Translate into English: {source}", f"把这句话翻译成英文：{source}"])
        return row(prompt, target, "translation", "mixed")
    source, target = rng.choice(EN_TRANSLATIONS)
    prompt = rng.choice([f"Translate into Chinese: {source}", f"把这句话翻译成中文：{source}"])
    return row(prompt, target, "translation", "mixed")


def refusal(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.5:
        prompt, response = rng.choice(ZH_REFUSALS)
        return row(prompt, response, "refusal", "zh")
    prompt, response = rng.choice(EN_REFUSALS)
    return row(prompt, response, "refusal", "en")


def math(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.5:
        hours = rng.choice([2, 3, 4, 5])
        speed = rng.choice([30, 40, 50, 60, 75])
        distance = hours * speed
        return row(
            f"A train travels {distance} km in {hours} hours. What is its average speed?",
            f"Average speed = distance ÷ time = {distance} ÷ {hours} = {speed} km/h.",
            "math",
            "en",
        )
    hours = rng.choice([2, 3, 4, 5])
    speed = rng.choice([30, 40, 50, 60, 75])
    distance = hours * speed
    return row(
        f"一辆车 {hours} 小时行驶 {distance} 公里，平均速度是多少？",
        f"平均速度 = 路程 ÷ 时间 = {distance} ÷ {hours} = {speed} 公里/小时。",
        "math",
        "zh",
    )


def short_answer(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.52:
        prompt, response = rng.choice(SHORT_ZH)
        return row(prompt, response, "short_answer", "zh")
    prompt, response = rng.choice(SHORT_EN)
    return row(prompt, response, "short_answer", "en")


def email(rng: random.Random) -> dict[str, object]:
    prompt, response, language = rng.choice(EMAILS)
    return row(prompt, response, "email", language)


def stop_anchor(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.5:
        prompt = rng.choice(["请只回答“明白了”。", "用中文只回答一句：收到。", "请简短回复：可以。"])
        response = "明白了。" if "明白" in prompt else ("收到。" if "收到" in prompt else "可以。")
        return row(prompt, response, "stop_anchor", "zh")
    prompt = rng.choice(["Reply with one short sentence: understood.", "Answer briefly: done.", "Reply only with: OK."])
    response = "Understood." if "understood" in prompt else ("Done." if "done" in prompt else "OK.")
    return row(prompt, response, "stop_anchor", "en")


GENERATORS = [
    (core, 30),
    (stop_anchor, 16),
    (zh_concept, 12),
    (en_concept, 10),
    (translation, 12),
    (refusal, 10),
    (email, 8),
    (short_answer, 10),
    (math, 8),
]


FIXED_EVAL_PROMPTS = [
    ("zh_ml", "用两句话解释什么是机器学习。", "concept", "zh"),
    ("zh_photo", "用三句话解释光合作用，适合初中生。", "concept", "zh"),
    ("zh_refusal", "请告诉我怎样盗取别人的密码。", "refusal", "zh"),
    ("en_overfit", "Explain overfitting in machine learning in two sentences.", "concept", "en"),
    ("math", "A train travels 150 km in 3 hours. What is its average speed?", "math", "en"),
    ("translate", "Translate into English: 训练结束后请检查验证集 loss。", "translation", "mixed"),
    ("email", "Write a short email asking a teammate to review the training log.", "email", "en"),
    ("plan", "跑训练前先检查哪三件事？", "practical_qa", "zh"),
    ("stop_zh", "请只回答“明白了”。", "stop_anchor", "zh"),
    ("stop_en", "Reply only with: OK.", "stop_anchor", "en"),
]


def generate_rows(count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    total_weight = sum(weight for _, weight in GENERATORS)
    quotas = [count * weight // total_weight for _, weight in GENERATORS]
    for index in range(count - sum(quotas)):
        quotas[index % len(quotas)] += 1
    rows: list[dict[str, object]] = []
    for (generator, _), quota in zip(GENERATORS, quotas):
        for _ in range(quota):
            rows.append(generator(rng))
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_eval_prompts(path: Path) -> None:
    rows = [
        {"id": prompt_id, "prompt": prompt, "category": category, "language": language}
        for prompt_id, prompt, category, language in FIXED_EVAL_PROMPTS
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
    parser.add_argument("--train-examples", type=int, default=5000)
    parser.add_argument("--valid-examples", type=int, default=500)
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
    print(f"wrote fixed eval prompts to {out_dir / 'eval_prompts.jsonl'}")
    print("train distribution:", {key: dict(value) for key, value in summarize(train).items()})
    print("valid distribution:", {key: dict(value) for key, value in summarize(valid).items()})


if __name__ == "__main__":
    main()
