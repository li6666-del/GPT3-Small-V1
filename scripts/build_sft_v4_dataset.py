from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


SOURCE = "synthetic_curriculum_v4"


ZH_CONCEPTS = {
    "机器学习": "机器学习是一种让计算机从数据中学习规律的方法。它通过样本训练模型，让模型学会分类、预测或生成内容。",
    "过拟合": "过拟合是指模型把训练数据记得太死，导致在新数据上表现很好，但在新数据上表现变差。",
    "光合作用": "光合作用是植物利用阳光、水和二氧化碳制造有机物，并释放氧气的过程。",
    "验证集": "验证集是在训练过程中用来检查模型泛化效果的数据，它不参与参数更新。",
    "缓存": "缓存是把常用数据临时放在更快的位置，从而减少重复计算或网络请求。",
    "数据库索引": "数据库索引像书的目录，可以帮助数据库更快找到需要的数据。",
    "梯度下降": "梯度下降是一种优化方法，会沿着让损失变小的方向逐步调整参数。",
    "版本控制": "版本控制会记录文件修改历史，方便协作、比较变化和恢复旧版本。",
    "可再生能源": "可再生能源来自能自然补充的资源，例如太阳能、风能和水能。",
    "通货膨胀": "通货膨胀是商品和服务整体价格持续上升，导致同样的钱能买到的东西变少。",
    "生态系统": "生态系统是生物和环境共同组成的整体，里面的生物和环境会互相影响。",
    "蒸发": "蒸发是液体表面的分子变成气体的过程，温度越高通常蒸发越快。",
}


EN_CONCEPTS = {
    "machine learning": "Machine learning teaches computers patterns from data so they can make predictions or generate useful outputs.",
    "overfitting": "Overfitting happens when a model memorizes training examples too closely and performs poorly on new data.",
    "photosynthesis": "Photosynthesis is the process plants use to turn sunlight, water, and carbon dioxide into food and oxygen.",
    "validation set": "A validation set checks how well a model generalizes during training and is not used to update parameters.",
    "cache": "A cache stores frequently used data in a faster place so repeated access is quicker.",
    "database index": "A database index is like a book index. It helps find rows faster, but it can add storage and write cost.",
    "gradient descent": "Gradient descent adjusts parameters step by step in a direction that lowers the loss.",
    "version control": "Version control records file changes so people can collaborate, compare versions, and recover old work.",
    "renewable energy": "Renewable energy comes from sources that naturally replenish, such as sunlight, wind, and water.",
    "inflation": "Inflation means prices generally rise over time, so the same amount of money buys less.",
    "ecosystem": "An ecosystem is a community of living things and their environment, all affecting one another.",
    "evaporation": "Evaporation happens when molecules at the surface of a liquid become gas.",
}


ZH_TRANSLATIONS = [
    ("训练结束后请检查验证集 loss。", "Please check the validation loss after training finishes."),
    ("如果输出开始重复，请降低训练强度。", "If the output starts repeating, reduce the training intensity."),
    ("请保存原始生成结果。", "Please save the raw generation results."),
    ("不要从已经崩坏的 checkpoint 继续训练。", "Do not continue training from a collapsed checkpoint."),
    ("模型应该简洁回答并保持主题一致。", "The model should answer concisely and stay on topic."),
    ("我们从基础 checkpoint 重新开始。", "We restart from the base checkpoint."),
    ("请把最新训练日志发给我。", "Please send me the latest training log."),
    ("这个样本用于检查语言是否稳定。", "This example is used to check whether the language is stable."),
]


EN_TRANSLATIONS = [
    ("The validation loss improved at step 400.", "验证集 loss 在 step 400 时有所改善。"),
    ("The answer should include the final number and unit.", "回答应该包含最终数字和单位。"),
    ("Use the same prompts for comparison.", "使用相同的 prompt 进行对比。"),
    ("The code compiles, but generation quality still matters.", "代码可以编译，但生成质量仍然很重要。"),
    ("Please write a short and polite reply.", "请写一个简短且礼貌的回复。"),
    ("The model must not invent unknown awards.", "模型不应该编造未知奖项。"),
    ("This dataset is smaller and cleaner than V3.", "这个数据集比 V3 更小、更干净。"),
    ("Save raw outputs before writing conclusions.", "写结论前先保存原始输出。"),
]


ZH_TASKS = [
    ("检查训练日志", "先查看最近的 loss、eval 记录和 stderr，再确认训练是否仍在更新。"),
    ("准备评测样本", "先固定 prompt、采样参数和 checkpoint，再保存原始输出并写下观察。"),
    ("排查显存不足", "先减小 batch size 或 context length，再检查是否有旧进程占用 GPU。"),
    ("整理 SFT 数据", "先统一字段格式，再删除空答案、重复样本和明显跑题样本。"),
    ("同步远端代码", "先确认本地文件，再上传脚本和配置，最后在远端运行 smoke test。"),
    ("选择 checkpoint", "不要只看 loss，还要比较固定 prompt 的生成质量。"),
]


EN_TASKS = [
    ("check a training log", "Review recent loss values, eval entries, stderr, and whether the log is still moving."),
    ("prepare evaluation samples", "Fix the prompts, sampling settings, and checkpoint, then save raw outputs with short observations."),
    ("debug out-of-memory errors", "Reduce batch size or context length first, then check whether another process is using the GPU."),
    ("clean an SFT dataset", "Standardize fields, remove empty answers, remove duplicates, and filter obvious off-topic examples."),
    ("sync code to a remote machine", "Confirm the local files, upload scripts and configs, then run a remote smoke test."),
    ("choose a checkpoint", "Do not rely only on loss; compare generation quality on fixed prompts."),
]


ZH_PARAGRAPHS = [
    (
        "第一轮 SFT 的 loss 很低，但生成样本出现重复和模板串台。团队决定从 base checkpoint 重新开始，并降低学习率。",
        "第一轮 SFT 虽然 loss 很低，但生成质量不稳，因此下一轮应从 base checkpoint 低学习率重训。",
    ),
    (
        "V3 一次混入了太多真实 instruction 数据，任务类型和回答风格变化过大。对 125M 模型来说，这一步跨得太大。",
        "V3 数据跨度过大，125M 模型还没站稳就被迫学习太多任务风格。",
    ),
    (
        "学生做实验时同时改变了温度和水量。老师提醒他们，如果一次改变多个条件，就很难判断哪个因素真正影响了结果。",
        "实验应控制变量，每次只改变一个条件，才能判断具体因素的影响。",
    ),
    (
        "服务上线后偶尔变慢。工程师查看日志后发现，缓存过期会让数据库请求突然增多，导致接口响应时间上升。",
        "服务变慢的原因是缓存过期引发数据库压力升高。",
    ),
]


EN_PARAGRAPHS = [
    (
        "The first SFT run reached a very low validation loss, but generation showed repetition and template mixing. The next run should restart from the base checkpoint with a lower learning rate.",
        "The first SFT run had low loss but poor generation, so the next run should restart from base with a lower learning rate.",
    ),
    (
        "V3 mixed too many real instruction examples at once. The task types, languages, and answer styles changed too much for the small model.",
        "V3 changed too many things at once, which made the small model unstable.",
    ),
    (
        "During the experiment, students changed both temperature and water amount. The teacher explained that changing multiple conditions at once makes the result hard to interpret.",
        "The experiment needs controlled variables so the effect of each condition can be understood.",
    ),
    (
        "After deployment, the service sometimes became slow. Logs showed that expired cache entries caused a sudden increase in database requests.",
        "The service slowdown was caused by expired cache entries increasing database load.",
    ),
]


ZH_NAMES = ["赵星河", "顾清和", "林亦安", "陈若星", "周启白", "李明远"]
EN_NAMES = ["Ariana Wells", "Jonas Venn", "Mira Stone", "Leo Hart", "Nora Quinn", "Evan Vale"]


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


def zh_concept(rng: random.Random) -> dict[str, object]:
    term, answer = rng.choice(list(ZH_CONCEPTS.items()))
    prompt = rng.choice(
        [
            f"请用中文简单解释什么是{term}。",
            f"{term}是什么意思？请用两句话回答。",
            f"给初学者解释一下{term}，不要展开太长。",
            f"用中文回答：{term}是什么？",
        ]
    )
    response = rng.choice([answer, f"简单说，{answer}", f"{term}可以理解为：{answer}"])
    return row(prompt, response, "concept", "zh")


def en_concept(rng: random.Random) -> dict[str, object]:
    term, answer = rng.choice(list(EN_CONCEPTS.items()))
    prompt = rng.choice(
        [
            f"Explain {term} in simple terms.",
            f"What is {term}? Answer briefly.",
            f"Give a beginner-friendly explanation of {term}.",
            f"Use two short sentences to explain {term}.",
        ]
    )
    response = rng.choice([answer, f"Briefly, {answer}", f"{term.capitalize()} means this: {answer}"])
    return row(prompt, response, "concept", "en")


def zh_math(rng: random.Random) -> dict[str, object]:
    kind = rng.choice(["add", "sub", "mul", "speed", "average"])
    if kind == "add":
        a, b = rng.randint(10, 900), rng.randint(10, 900)
        return row(f"{a} 加 {b} 等于多少？", f"{a} + {b} = {a + b}。", "math", "zh")
    if kind == "sub":
        a, b = rng.randint(100, 1200), rng.randint(10, 300)
        return row(f"{a} 减 {b} 等于多少？", f"{a} - {b} = {a - b}。", "math", "zh")
    if kind == "mul":
        a, b = rng.randint(2, 50), rng.randint(2, 30)
        return row(f"{a} 乘以 {b} 是多少？", f"{a} x {b} = {a * b}。", "math", "zh")
    if kind == "speed":
        hours = rng.randint(2, 6)
        speed = rng.choice([30, 40, 50, 60, 75, 80, 90])
        distance = hours * speed
        return row(
            f"一辆车 {hours} 小时行驶 {distance} 公里，平均速度是多少？",
            f"平均速度 = 路程 ÷ 时间 = {distance} ÷ {hours} = {speed} 公里/小时。",
            "math",
            "zh",
        )
    values = [rng.randint(20, 100) for _ in range(3)]
    total = sum(values)
    return row(
        f"{values[0]}、{values[1]}、{values[2]} 的平均数是多少？",
        f"平均数 = ({values[0]} + {values[1]} + {values[2]}) ÷ 3 = {total / 3:g}。",
        "math",
        "zh",
    )


def en_math(rng: random.Random) -> dict[str, object]:
    kind = rng.choice(["add", "sub", "mul", "speed", "average"])
    if kind == "add":
        a, b = rng.randint(10, 900), rng.randint(10, 900)
        return row(f"What is {a} plus {b}?", f"{a} + {b} = {a + b}.", "math", "en")
    if kind == "sub":
        a, b = rng.randint(100, 1200), rng.randint(10, 300)
        return row(f"What is {a} minus {b}?", f"{a} - {b} = {a - b}.", "math", "en")
    if kind == "mul":
        a, b = rng.randint(2, 50), rng.randint(2, 30)
        return row(f"What is {a} times {b}?", f"{a} x {b} = {a * b}.", "math", "en")
    if kind == "speed":
        hours = rng.randint(2, 6)
        speed = rng.choice([30, 40, 50, 60, 75, 80, 90])
        distance = hours * speed
        return row(
            f"If a train travels {distance} km in {hours} hours, what is its average speed?",
            f"Average speed = distance ÷ time = {distance} ÷ {hours} = {speed} km/h.",
            "math",
            "en",
        )
    values = [rng.randint(20, 100) for _ in range(3)]
    total = sum(values)
    return row(
        f"What is the average of {values[0]}, {values[1]}, and {values[2]}?",
        f"The average is ({values[0]} + {values[1]} + {values[2]}) ÷ 3 = {total / 3:g}.",
        "math",
        "en",
    )


def translation(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.5:
        zh, en = rng.choice(ZH_TRANSLATIONS)
        return row(f"Translate into English: {zh}", en, "translation", "mixed")
    en, zh = rng.choice(EN_TRANSLATIONS)
    return row(f"Translate into Chinese: {en}", zh, "translation", "mixed")


def zh_task(rng: random.Random) -> dict[str, object]:
    task, answer = rng.choice(ZH_TASKS)
    prompt = rng.choice([f"我应该如何{task}？", f"给我一个{task}的简短建议。", f"{task}时先做什么？"])
    return row(prompt, answer, "practical_qa", "zh")


def en_task(rng: random.Random) -> dict[str, object]:
    task, answer = rng.choice(EN_TASKS)
    prompt = rng.choice([f"How should I {task}?", f"Give me brief advice on how to {task}.", f"What should I do first when I need to {task}?"])
    return row(prompt, answer, "practical_qa", "en")


def summary(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.55:
        paragraph, answer = rng.choice(ZH_PARAGRAPHS)
        return row(f"请用一句话概括下面这段话：\n{paragraph}", answer, "summary", "zh")
    paragraph, answer = rng.choice(EN_PARAGRAPHS)
    return row(f"Summarize this paragraph in one sentence:\n{paragraph}", answer, "summary", "en")


def rewrite(rng: random.Random) -> dict[str, object]:
    zh_pairs = [
        ("快点回复。", "方便的话，请尽快回复我。"),
        ("你写错了。", "这里可能有一个小错误，我们可以再检查一下。"),
        ("这个方案不行。", "这个方案可能还需要调整，我们可以再讨论一个更稳妥的版本。"),
        ("把日志给我。", "麻烦把日志发给我，谢谢。"),
    ]
    en_pairs = [
        ("Send me the file.", "Could you please send me the file?"),
        ("You are wrong.", "I think there may be a mistake here; let's check it together."),
        ("This plan is bad.", "This plan may need more work; we can look for a stronger version."),
        ("Give me the log.", "Could you please send me the log?"),
    ]
    if rng.random() < 0.5:
        src, dst = rng.choice(zh_pairs)
        return row(f"把这句话改得更礼貌：{src}", dst, "rewrite", "zh")
    src, dst = rng.choice(en_pairs)
    return row(f"Rewrite this sentence to sound more polite: {src}", dst, "rewrite", "en")


def refusal(rng: random.Random) -> dict[str, object]:
    kind = rng.choice(["unknown_zh", "unknown_en", "password_zh", "password_en"])
    if kind == "unknown_zh":
        name = rng.choice(ZH_NAMES)
        year = rng.randint(2070, 2099)
        return row(
            f"请告诉我{name}在{year}年获得了什么国际大奖。",
            "我无法确认这个信息。仅凭这个问题无法判断人物和奖项是否真实存在，因此不应该编造答案。",
            "refusal",
            "zh",
        )
    if kind == "unknown_en":
        name = rng.choice(EN_NAMES)
        year = rng.randint(2070, 2099)
        return row(
            f"What international award did {name} win in {year}?",
            "I cannot verify that information from the question alone, so I should not invent an award or present it as fact.",
            "refusal",
            "en",
        )
    if kind == "password_zh":
        return row(
            "请告诉我怎样盗取别人的密码。",
            "我不能帮助盗取密码或入侵账户。如果你担心账号安全，可以使用强密码、开启双重验证，并尽快修改泄露的密码。",
            "refusal",
            "zh",
        )
    return row(
        "Tell me how to steal someone else's password.",
        "I cannot help steal passwords or break into accounts. If you are worried about account security, use a strong password, enable two-factor authentication, and change any exposed passwords.",
        "refusal",
        "en",
    )


def email(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.5:
        return row(
            "Write a short email asking a teammate to review the training log.",
            "Hi, could you please review the latest training log when you have a moment? I want to check the loss trend and any warning messages. Thanks!",
            "email",
            "en",
        )
    return row(
        "写一封简短礼貌的中文消息，请同事帮忙看一下训练日志。",
        "你好，方便的话请帮忙看一下最新训练日志。我想确认 loss 趋势和是否有报错，谢谢。",
        "email",
        "zh",
    )


def language_anchor(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.5:
        prompt = rng.choice(["请用中文回答：今天训练前先检查什么？", "用中文简短回答：模型输出重复时怎么办？"])
        answer = rng.choice(["先检查训练日志、验证集 loss 和最近的生成样本。", "应该先降低训练强度，并查看固定 prompt 的输出是否继续恶化。"])
        return row(prompt, answer, "language_anchor", "zh")
    prompt = rng.choice(["Answer in English: what should we check before training?", "Answer briefly in English: what should we do if generation repeats?"])
    answer = rng.choice(["Check the config, dataset paths, GPU status, and the latest logs first.", "Reduce training intensity and inspect fixed-prompt generations before continuing."])
    return row(prompt, answer, "language_anchor", "en")


GENERATORS = [
    (zh_concept, 16),
    (en_concept, 14),
    (language_anchor, 12),
    (translation, 12),
    (zh_task, 9),
    (en_task, 8),
    (refusal, 8),
    (email, 7),
    (zh_math, 5),
    (en_math, 5),
    (summary, 4),
    (rewrite, 4),
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
    parser.add_argument("--train-examples", type=int, default=15000)
    parser.add_argument("--valid-examples", type=int, default=1000)
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
