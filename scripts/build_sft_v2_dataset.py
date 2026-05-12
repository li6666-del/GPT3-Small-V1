from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ZH_CONCEPTS = {
    "机器学习": "机器学习是一种让计算机从数据中学习规律的方法。它通过样本训练模型，让模型学会分类、预测或生成内容。",
    "过拟合": "过拟合是指模型把训练数据记得太死，导致在新数据上表现变差。解决办法包括增加数据、简化模型和使用正则化。",
    "光合作用": "光合作用是植物利用阳光、水和二氧化碳制造有机物，并释放氧气的过程。",
    "数据库索引": "数据库索引像书的目录，可以帮助数据库更快找到需要的数据，但会增加存储和写入成本。",
    "梯度下降": "梯度下降是一种优化方法，会沿着让损失变小的方向逐步调整模型参数。",
    "缓存": "缓存是把常用数据临时存到更快的位置，从而减少重复计算或网络请求。",
    "通货膨胀": "通货膨胀是商品和服务整体价格持续上升，导致同样的钱能买到的东西变少。",
    "生态系统": "生态系统是生物和环境共同组成的整体，里面的植物、动物、微生物、水和土壤会互相影响。",
    "蒸发": "蒸发是液体表面的分子变成气体的过程，温度越高、空气越干燥，通常蒸发越快。",
    "可再生能源": "可再生能源来自能自然补充的资源，例如太阳能、风能和水能。",
    "版本控制": "版本控制用于记录文件的修改历史，方便多人协作、回退错误和比较变化。",
    "验证集": "验证集是在训练过程中用来检查模型泛化效果的数据，它不应该参与参数更新。",
}

EN_CONCEPTS = {
    "machine learning": "Machine learning teaches computers patterns from data so they can make predictions or generate useful outputs.",
    "overfitting": "Overfitting happens when a model memorizes the training data too closely and performs poorly on new examples.",
    "photosynthesis": "Photosynthesis is the process plants use to turn sunlight, water, and carbon dioxide into food and oxygen.",
    "database index": "A database index is like a book index. It helps find rows faster, though it can use more storage and slow down writes.",
    "gradient descent": "Gradient descent is an optimization method that adjusts parameters in the direction that lowers the loss.",
    "cache": "A cache stores frequently used data in a faster place so repeated access is quicker.",
    "inflation": "Inflation means the general price level rises over time, so the same amount of money buys less.",
    "ecosystem": "An ecosystem is a community of living things and their environment, all affecting one another.",
    "evaporation": "Evaporation happens when molecules at the surface of a liquid become gas.",
    "renewable energy": "Renewable energy comes from sources that naturally replenish, such as sunlight, wind, and water.",
    "version control": "Version control records changes to files so people can collaborate, compare versions, and recover old work.",
    "validation set": "A validation set checks how well a model generalizes during training and should not be used for parameter updates.",
}

ZH_TASKS = [
    ("检查训练日志", "先查看最近的 loss 和 eval 记录，再确认是否有报错或长时间不更新。"),
    ("准备评测样本", "先固定 prompt、采样参数和 checkpoint，再保存原始输出，最后写下人工观察。"),
    ("排查显存不足", "先减小 batch size 或 context length，再确认是否有旧进程占用 GPU。"),
    ("整理数据集", "先统一字段格式，再去掉空答案和明显重复样本，最后划分 train/valid。"),
    ("同步远端代码", "先备份远端代码，再上传小文件，最后运行编译或 smoke test。"),
]

EN_TASKS = [
    ("check a training log", "Look at recent loss values, eval entries, errors, and whether the log is still moving."),
    ("prepare evaluation samples", "Fix the prompts, sampling settings, and checkpoint, then save raw outputs with short observations."),
    ("debug out-of-memory errors", "Reduce batch size or context length first, then check whether another process is using the GPU."),
    ("organize a dataset", "Standardize the fields, remove empty or duplicated examples, and split the data into train and validation sets."),
    ("sync code to a remote machine", "Back up the remote files, upload the small source files, and run a quick smoke test."),
]

ZH_TRANSLATIONS = [
    ("训练结束后请检查验证集 loss。", "Please check the validation loss after training finishes."),
    ("这个脚本会生成第一批 SFT 数据。", "This script generates the first batch of SFT data."),
    ("如果输出开始重复，请降低训练强度。", "If the output starts repeating, reduce the training intensity."),
    ("请把最新日志发给我。", "Please send me the latest log."),
    ("模型应该简洁回答并保持主题一致。", "The model should answer concisely and stay on topic."),
    ("我们从基础 checkpoint 重新开始。", "We restart from the base checkpoint."),
    ("这个样本用于测试拒绝胡编。", "This example is used to test refusal to fabricate information."),
    ("请保存原始生成结果。", "Please save the raw generation results."),
]

EN_TRANSLATIONS = [
    ("The validation loss improved at step 400.", "验证集 loss 在 step 400 时有所改善。"),
    ("The answer should include the final number and unit.", "回答应该包含最终数字和单位。"),
    ("Do not continue training from a collapsed checkpoint.", "不要从已经崩坏的 checkpoint 继续训练。"),
    ("This dataset is broader than the first bootstrap set.", "这个数据集比第一版 bootstrap 数据覆盖面更广。"),
    ("Please write a short and polite reply.", "请写一个简短且礼貌的回复。"),
    ("The model must not invent unknown awards.", "模型不应该编造未知奖项。"),
    ("Use the same prompts for comparison.", "使用相同的 prompt 进行对比。"),
    ("The code compiles, but generation quality still matters.", "代码可以编译，但生成质量仍然很重要。"),
]

ZH_PARAGRAPHS = [
    (
        "团队发现第一轮 SFT 的 loss 很低，但生成样本出现了重复和模板串台。为了避免继续放大问题，他们决定从 base checkpoint 重新开始，并降低学习率。",
        "第一轮 SFT 虽然 loss 很低，但生成质量不稳，因此第二轮应从 base checkpoint 低学习率重训。",
    ),
    (
        "学生做实验时同时改变了温度和水量。老师提醒他们，如果一次改变多个条件，就很难判断哪个因素真正影响了结果。",
        "实验应控制变量，每次只改变一个条件，才能判断具体因素的影响。",
    ),
    (
        "服务上线后偶尔变慢。工程师查看日志后发现，缓存过期会让数据库请求突然增多，导致接口响应时间上升。",
        "服务变慢的原因是缓存过期引发数据库压力升高。",
    ),
    (
        "社区准备增加夜间照明。居民希望道路更安全，但也担心灯光太亮影响休息，于是物业决定先做小范围试点。",
        "社区将试点夜间照明，在安全和休息之间寻找平衡。",
    ),
]

EN_PARAGRAPHS = [
    (
        "The first SFT run reached a very low validation loss, but generation showed repetition and template mixing. The next run should restart from the base checkpoint with broader data and a lower learning rate.",
        "The first SFT run had low loss but poor generation, so the next run should restart from base with broader data and a lower learning rate.",
    ),
    (
        "During the experiment, students changed both temperature and water amount. The teacher explained that changing multiple conditions at once makes the result hard to interpret.",
        "The experiment needs controlled variables so the effect of each condition can be understood.",
    ),
    (
        "After deployment, the service sometimes became slow. Logs showed that expired cache entries caused a sudden increase in database requests.",
        "The service slowdown was caused by expired cache entries increasing database load.",
    ),
    (
        "The neighborhood wants better lighting at night. Residents want safer paths, but they also worry that bright lights may disturb sleep.",
        "The neighborhood wants safer night lighting without disturbing nearby residents.",
    ),
]

NAMES_ZH = ["李明远", "陈若星", "赵星河", "周启白", "林亦安", "顾清和"]
NAMES_EN = ["Ariana Wells", "Jonas Venn", "Mira Stone", "Leo Hart", "Nora Quinn", "Evan Vale"]


def row(prompt: str, response: str, category: str, language: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "category": category,
        "language": language,
        "source": "synthetic_sft_v2_broad",
    }


def zh_concept(rng: random.Random) -> dict[str, object]:
    term, answer = rng.choice(list(ZH_CONCEPTS.items()))
    prompt = rng.choice([
        f"请用中文简单解释什么是{term}。",
        f"{term}是什么意思？请用两句话回答。",
        f"给初学者解释一下{term}。",
        f"用通俗语言说明：{term}。",
    ])
    response = rng.choice([answer, f"{term}可以理解为：{answer}", f"简单说，{answer}"])
    return row(prompt, response, "concept", "zh")


def en_concept(rng: random.Random) -> dict[str, object]:
    term, answer = rng.choice(list(EN_CONCEPTS.items()))
    prompt = rng.choice([
        f"Explain {term} in simple terms.",
        f"What is {term}? Answer briefly.",
        f"Give a beginner-friendly explanation of {term}.",
        f"Use two sentences to explain {term}.",
    ])
    response = rng.choice([answer, f"In simple terms, {answer}", f"{term.capitalize()} means this: {answer}"])
    return row(prompt, response, "concept", "en")


def zh_math(rng: random.Random) -> dict[str, object]:
    kind = rng.choice(["add", "sub", "mul", "speed", "average", "discount"])
    if kind == "add":
        a, b = rng.randint(11, 999), rng.randint(11, 999)
        return row(f"{a} 加 {b} 等于多少？", f"{a} + {b} = {a + b}。", "math", "zh")
    if kind == "sub":
        a, b = rng.randint(100, 1200), rng.randint(10, 300)
        return row(f"{a} 减 {b} 等于多少？", f"{a} - {b} = {a - b}。", "math", "zh")
    if kind == "mul":
        a, b = rng.randint(3, 60), rng.randint(3, 40)
        return row(f"{a} 乘以 {b} 是多少？", f"{a} x {b} = {a * b}。", "math", "zh")
    if kind == "speed":
        hours = rng.randint(2, 6)
        speed = rng.randint(25, 95)
        distance = hours * speed
        return row(
            f"一辆车 {hours} 小时行驶 {distance} 公里，平均速度是多少？",
            f"平均速度 = 路程 ÷ 时间 = {distance} ÷ {hours} = {speed} 公里/小时。",
            "math",
            "zh",
        )
    if kind == "average":
        values = [rng.randint(50, 100) for _ in range(3)]
        total = sum(values)
        return row(f"{values[0]}、{values[1]}、{values[2]} 的平均数是多少？", f"平均数 = ({values[0]} + {values[1]} + {values[2]}) ÷ 3 = {total / 3:g}。", "math", "zh")
    price = rng.randint(40, 500)
    discount = rng.choice([5, 6, 7, 8, 9])
    final = price * discount / 10
    return row(f"原价 {price} 元的商品打 {discount} 折后多少钱？", f"价格 = {price} x {discount}/10 = {final:g} 元。", "math", "zh")


def en_math(rng: random.Random) -> dict[str, object]:
    kind = rng.choice(["add", "sub", "mul", "speed", "average", "discount"])
    if kind == "add":
        a, b = rng.randint(11, 999), rng.randint(11, 999)
        return row(f"What is {a} plus {b}?", f"{a} + {b} = {a + b}.", "math", "en")
    if kind == "sub":
        a, b = rng.randint(100, 1200), rng.randint(10, 300)
        return row(f"What is {a} minus {b}?", f"{a} - {b} = {a - b}.", "math", "en")
    if kind == "mul":
        a, b = rng.randint(3, 60), rng.randint(3, 40)
        return row(f"What is {a} times {b}?", f"{a} x {b} = {a * b}.", "math", "en")
    if kind == "speed":
        hours = rng.randint(2, 6)
        speed = rng.randint(25, 95)
        distance = hours * speed
        return row(
            f"If a train travels {distance} miles in {hours} hours, what is its average speed?",
            f"Average speed = distance ÷ time = {distance} ÷ {hours} = {speed} miles per hour.",
            "math",
            "en",
        )
    if kind == "average":
        values = [rng.randint(50, 100) for _ in range(3)]
        return row(f"What is the average of {values[0]}, {values[1]}, and {values[2]}?", f"The average is ({values[0]} + {values[1]} + {values[2]}) ÷ 3 = {sum(values) / 3:g}.", "math", "en")
    price = rng.randint(40, 500)
    discount = rng.choice([10, 15, 20, 25, 30])
    final = price * (100 - discount) / 100
    return row(f"A ${price} item is discounted by {discount}%. What is the final price?", f"The final price is ${final:g}.", "math", "en")


def zh_task(rng: random.Random) -> dict[str, object]:
    task, answer = rng.choice(ZH_TASKS)
    prompt = rng.choice([f"我应该如何{task}？", f"给我一个{task}的简短建议。", f"{task}时先做什么？"])
    return row(prompt, answer, "practical_qa", "zh")


def en_task(rng: random.Random) -> dict[str, object]:
    task, answer = rng.choice(EN_TASKS)
    prompt = rng.choice([f"How should I {task}?", f"Give me brief advice on how to {task}.", f"What should I do first when I need to {task}?"])
    return row(prompt, answer, "practical_qa", "en")


def translation(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.5:
        zh, en = rng.choice(ZH_TRANSLATIONS)
        return row(f"把下面这句话翻译成英文：{zh}", en, "translation", "mixed")
    en, zh = rng.choice(EN_TRANSLATIONS)
    return row(f"Translate this sentence into Chinese: {en}", zh, "translation", "mixed")


def summary(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.52:
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
        ("你必须今天完成。", "如果可以的话，请尽量在今天完成。"),
    ]
    en_pairs = [
        ("Send me the file.", "Could you please send me the file?"),
        ("You are wrong.", "I think there may be a mistake here; let's check it together."),
        ("This plan is bad.", "This plan may need more work; we can look for a stronger version."),
        ("Give me the log.", "Could you please send me the log?"),
        ("You must finish today.", "If possible, please try to finish it today."),
    ]
    if rng.random() < 0.55:
        src, dst = rng.choice(zh_pairs)
        return row(f"把这句话改得更礼貌：{src}", dst, "rewrite", "zh")
    src, dst = rng.choice(en_pairs)
    return row(f"Rewrite this sentence to sound more polite: {src}", dst, "rewrite", "en")


def refusal(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.55:
        name = rng.choice(NAMES_ZH)
        year = rng.randint(2070, 2099)
        prompt = f"请告诉我{name}在{year}年获得了什么国际大奖。"
        answer = "我无法确认这个信息。仅凭这个问题无法判断人物和奖项是否真实存在，因此不应该编造答案。"
        return row(prompt, answer, "refusal", "zh")
    name = rng.choice(NAMES_EN)
    year = rng.randint(2070, 2099)
    prompt = f"What international award did {name} win in {year}?"
    answer = "I cannot verify that information from the question alone, so I should not invent an award or present it as fact."
    return row(prompt, answer, "refusal", "en")


def story(rng: random.Random) -> dict[str, object]:
    zh_places = ["图书馆", "山村", "海边", "校园", "花园", "小镇"]
    zh_chars = ["小女孩", "小男孩", "小猫", "年轻的邮递员", "新来的学生"]
    en_places = ["library", "mountain village", "shore", "school", "garden", "small town"]
    en_chars = ["a young girl", "a young boy", "a small cat", "a careful messenger", "a new student"]
    if rng.random() < 0.55:
        char = rng.choice(zh_chars)
        place = rng.choice(zh_places)
        mood = rng.choice(["雨后", "清晨", "黄昏", "午后", "春天"])
        value = rng.choice(["耐心", "勇气", "合作", "善意", "好奇心"])
        prompt = f"写一个简短中文故事，开头是：{char}在{mood}走进{place}。"
        answer = f"{char}在{mood}走进{place}，发现一件小事和想象中不一样。他没有急着离开，而是停下来观察并尝试帮忙。故事结束时，他明白了{value}往往藏在普通的一天里。"
        return row(prompt, answer, "story", "zh")
    char = rng.choice(en_chars)
    place = rng.choice(en_places)
    time = rng.choice(["after the rain", "in the morning", "at sunset", "in the afternoon", "in spring"])
    value = rng.choice(["patience", "courage", "teamwork", "kindness", "curiosity"])
    prompt = f"Write a short story that begins with: {char} entered the {place} {time}."
    answer = f"{char.capitalize()} entered the {place} {time} and noticed a small problem others had missed. Instead of walking away, they paid attention and helped. By the end, everyone saw how {value} can grow from a simple choice."
    return row(prompt, answer, "story", "en")


def email(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.5:
        prompt = "Write a short polite email asking a teammate to send the latest training log."
        answer = "Hi, could you please send me the latest training log when you have a moment? I want to review the recent loss and eval results. Thanks!"
        return row(prompt, answer, "email", "en")
    prompt = "写一封简短礼貌的中文消息，请同事发一下最新训练日志。"
    answer = "你好，方便的话请把最新训练日志发我一下。我想看一下最近的 loss 和 eval 结果，谢谢。"
    return row(prompt, answer, "email", "zh")


GENERATORS = [
    (zh_concept, 12),
    (en_concept, 10),
    (zh_task, 8),
    (en_task, 7),
    (zh_math, 5),
    (en_math, 5),
    (translation, 12),
    (summary, 10),
    (rewrite, 8),
    (refusal, 6),
    (story, 8),
    (email, 5),
]


def generate_rows(count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    total = sum(weight for _, weight in GENERATORS)
    quotas = [count * weight // total for _, weight in GENERATORS]
    for i in range(count - sum(quotas)):
        quotas[i % len(quotas)] += 1
    rows: list[dict[str, object]] = []
    for (fn, _), quota in zip(GENERATORS, quotas):
        for _ in range(quota):
            rows.append(fn(rng))
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-examples", type=int, default=80000)
    parser.add_argument("--valid-examples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = generate_rows(args.train_examples, args.seed)
    valid = generate_rows(args.valid_examples, args.seed + 1)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    print(f"wrote {len(train)} train examples to {out_dir / 'train.jsonl'}")
    print(f"wrote {len(valid)} valid examples to {out_dir / 'valid.jsonl'}")


if __name__ == "__main__":
    main()
