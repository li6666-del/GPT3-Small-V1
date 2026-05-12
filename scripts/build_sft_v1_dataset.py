from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ZH_CONCEPTS = [
    ("机器学习", "机器学习是一种让计算机从数据中学习规律的方法。它不需要人为写死每一步规则，而是通过样本训练模型，让模型学会分类、预测或生成内容。"),
    ("光合作用", "光合作用是植物利用阳光、水和二氧化碳制造有机物，并释放氧气的过程。它把太阳能转化为植物可以储存和使用的化学能。"),
    ("蒸发", "蒸发是液体表面的分子获得足够能量后变成气体的过程。温度越高、空气越干燥、表面积越大，蒸发通常越快。"),
    ("通货膨胀", "通货膨胀是指一段时间内商品和服务的整体价格水平持续上升。它会让同样数量的钱能买到的东西变少。"),
    ("生态系统", "生态系统是生物和它们所处环境共同组成的整体。植物、动物、微生物、土壤、水和空气都会互相影响。"),
    ("神经网络", "神经网络是一类受大脑启发的机器学习模型。它由多层计算单元组成，可以从数据中学习复杂的模式。"),
    ("数据库索引", "数据库索引像书的目录，可以帮助数据库更快找到需要的数据。索引能提升查询速度，但也会增加写入和存储成本。"),
    ("梯度下降", "梯度下降是一种优化方法。它会沿着让损失变小的方向不断调整参数，直到模型表现逐步变好。"),
    ("缓存", "缓存是把常用数据临时存放在更快的位置。这样下次需要时可以更快读取，减少重复计算或网络请求。"),
    ("可再生能源", "可再生能源来自能够自然补充的资源，例如太阳能、风能和水能。它们通常比化石燃料更适合长期可持续使用。"),
]

EN_CONCEPTS = [
    ("machine learning", "Machine learning is a way to teach computers patterns from data. Instead of writing every rule by hand, we train a model on examples so it can make predictions or generate useful outputs."),
    ("photosynthesis", "Photosynthesis is the process plants use to turn sunlight, water, and carbon dioxide into food and oxygen. It lets plants store energy from the sun."),
    ("evaporation", "Evaporation happens when molecules at the surface of a liquid gain enough energy to become gas. It usually happens faster with heat, dry air, and a larger surface area."),
    ("inflation", "Inflation means the general price level of goods and services rises over time. When inflation is high, the same amount of money buys less than before."),
    ("ecosystem", "An ecosystem is a community of living things and the environment around them. Plants, animals, microbes, water, soil, and air all affect one another."),
    ("neural network", "A neural network is a machine learning model inspired by the brain. It uses layers of simple computations to learn patterns from data."),
    ("database index", "A database index is like a book index. It helps the database find rows faster, though it can add storage cost and slow down writes."),
    ("gradient descent", "Gradient descent is an optimization method. It adjusts model parameters step by step in the direction that lowers the loss."),
    ("cache", "A cache stores frequently used data in a faster place. It speeds up repeated access and can reduce computation or network requests."),
    ("renewable energy", "Renewable energy comes from sources that naturally replenish, such as sunlight, wind, and water. It is often better suited for long-term sustainable use."),
]

ZH_SENTENCES = [
    ("请把会议改到明天下午三点。", "Please move the meeting to three o'clock tomorrow afternoon."),
    ("这个函数会读取文件并返回所有行。", "This function reads the file and returns all lines."),
    ("如果结果不稳定，请先检查随机种子。", "If the results are unstable, check the random seed first."),
    ("我们需要一个更简洁的说明。", "We need a more concise explanation."),
    ("训练结束后请保存最终模型。", "Please save the final model after training finishes."),
    ("这个问题可以分成两个步骤解决。", "This problem can be solved in two steps."),
    ("请不要编造无法确认的信息。", "Please do not fabricate information that cannot be verified."),
    ("今天的目标是完成第一轮测试。", "Today's goal is to finish the first test run."),
]

EN_SENTENCES = [
    ("Please summarize the report in one paragraph.", "请用一段话总结这份报告。"),
    ("The model should answer briefly and stay on topic.", "模型应该简洁回答，并且不要跑题。"),
    ("Check the log before restarting the training job.", "重启训练任务前请先检查日志。"),
    ("This dataset is small but carefully formatted.", "这个数据集规模不大，但格式经过仔细整理。"),
    ("A good assistant admits uncertainty when needed.", "好的助手在必要时会承认不确定。"),
    ("The function returns a list of cleaned documents.", "这个函数返回清洗后的文档列表。"),
    ("Use a validation set to watch for overfitting.", "使用验证集来观察是否过拟合。"),
    ("The answer should include the final number and unit.", "回答应该包含最终数字和单位。"),
]

ZH_TOPICS = [
    ("图书馆", "安静", "书架", "阅读"),
    ("小镇", "清晨", "面包店", "分享"),
    ("海边", "黄昏", "灯塔", "勇气"),
    ("山村", "雨后", "小路", "帮助"),
    ("校园", "午后", "操场", "合作"),
    ("花园", "春天", "种子", "耐心"),
]

EN_TOPICS = [
    ("library", "quiet", "shelf", "curiosity"),
    ("village", "morning", "bakery", "kindness"),
    ("shore", "sunset", "lighthouse", "courage"),
    ("mountain town", "after the rain", "path", "help"),
    ("school", "afternoon", "playground", "teamwork"),
    ("garden", "spring", "seed", "patience"),
]

ZH_PARAGRAPHS = [
    (
        "团队在上线前发现接口偶尔超时。工程师先查看日志，再复现问题，最后发现缓存失效会导致数据库压力突然升高。",
        "接口超时的主要原因是缓存失效后数据库压力升高，团队通过日志和复现定位了问题。",
    ),
    (
        "小区准备增加夜间照明。居民希望灯光足够安全，但也担心过亮会影响休息，因此物业决定先在两栋楼之间试点。",
        "小区计划试点夜间照明，在安全和居民休息之间寻找平衡。",
    ),
    (
        "学生做实验时记录了温度、水量和反应时间。老师提醒他们每次只改变一个条件，否则很难判断哪个因素真正起作用。",
        "实验需要控制变量，每次只改变一个条件，才能判断具体因素的影响。",
    ),
]

EN_PARAGRAPHS = [
    (
        "The team noticed that the service sometimes became slow after deployment. They checked the logs, reproduced the issue, and found that expired cache entries caused a sudden increase in database load.",
        "The service slowdown was caused by expired cache entries increasing database load, and the team found it through logs and reproduction.",
    ),
    (
        "The city plans to add more lights near the park. Residents want the path to feel safe at night, but they also worry that lights that are too bright may disturb nearby homes.",
        "The city wants safer park lighting while avoiding lights that disturb nearby residents.",
    ),
    (
        "During the experiment, students recorded temperature, water amount, and reaction time. The teacher reminded them to change only one condition at a time so the result would be easier to explain.",
        "The experiment needs controlled variables so students can understand which condition affects the result.",
    ),
]

ZH_POLITE = [
    ("把文件发我。", "请把文件发给我，谢谢。"),
    ("你写错了。", "这里可能有一个小错误，我们可以再检查一下。"),
    ("快点回复。", "方便的话，请尽快回复我。"),
    ("这个方案不行。", "这个方案可能还需要调整，我们可以再讨论一个更稳妥的版本。"),
]

EN_POLITE = [
    ("Send me the file.", "Could you please send me the file?"),
    ("You are wrong.", "I think there may be a mistake here; let's check it together."),
    ("Reply quickly.", "Please reply as soon as you can."),
    ("This plan is bad.", "This plan may need more work; we can look for a stronger version."),
]


def row(prompt: str, response: str, category: str, language: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "category": category,
        "language": language,
        "source": "synthetic_sft_v1_bootstrap",
    }


def zh_concept(rng: random.Random) -> dict[str, object]:
    term, answer = rng.choice(ZH_CONCEPTS)
    templates = [
        f"请用中文简单解释什么是{term}。",
        f"{term}是什么意思？请用两三句话说明。",
        f"给初学者解释一下{term}。",
    ]
    return row(rng.choice(templates), answer, "concept_explanation", "zh")


def en_concept(rng: random.Random) -> dict[str, object]:
    term, answer = rng.choice(EN_CONCEPTS)
    templates = [
        f"Explain {term} in simple terms.",
        f"What is {term}? Answer in two or three sentences.",
        f"Give a beginner-friendly explanation of {term}.",
    ]
    return row(rng.choice(templates), answer, "concept_explanation", "en")


def zh_math(rng: random.Random) -> dict[str, object]:
    kind = rng.choice(["add", "subtract", "multiply", "speed", "discount"])
    if kind == "add":
        a, b = rng.randint(12, 480), rng.randint(8, 360)
        return row(f"{a} 加 {b} 等于多少？", f"{a} + {b} = {a + b}。", "math", "zh")
    if kind == "subtract":
        a, b = rng.randint(100, 900), rng.randint(10, 99)
        return row(f"{a} 减 {b} 等于多少？", f"{a} - {b} = {a - b}。", "math", "zh")
    if kind == "multiply":
        a, b = rng.randint(3, 35), rng.randint(3, 25)
        return row(f"{a} 乘以 {b} 是多少？", f"{a} x {b} = {a * b}。", "math", "zh")
    if kind == "speed":
        hours = rng.choice([2, 3, 4, 5])
        speed = rng.choice([30, 40, 45, 60, 72])
        distance = hours * speed
        return row(
            f"一辆车 {hours} 小时行驶 {distance} 公里，平均速度是多少？",
            f"平均速度 = 路程 ÷ 时间 = {distance} ÷ {hours} = {speed} 公里/小时。",
            "math",
            "zh",
        )
    price = rng.choice([80, 120, 150, 200, 240, 300])
    discount = rng.choice([5, 6, 7, 8, 9])
    final = price * discount / 10
    return row(
        f"原价 {price} 元的商品打 {discount} 折后多少钱？",
        f"打 {discount} 折就是按原价的 {discount}/10 计算，所以价格是 {price} x {discount}/10 = {final:g} 元。",
        "math",
        "zh",
    )


def en_math(rng: random.Random) -> dict[str, object]:
    kind = rng.choice(["add", "subtract", "multiply", "speed", "discount"])
    if kind == "add":
        a, b = rng.randint(12, 480), rng.randint(8, 360)
        return row(f"What is {a} plus {b}?", f"{a} + {b} = {a + b}.", "math", "en")
    if kind == "subtract":
        a, b = rng.randint(100, 900), rng.randint(10, 99)
        return row(f"What is {a} minus {b}?", f"{a} - {b} = {a - b}.", "math", "en")
    if kind == "multiply":
        a, b = rng.randint(3, 35), rng.randint(3, 25)
        return row(f"What is {a} times {b}?", f"{a} x {b} = {a * b}.", "math", "en")
    if kind == "speed":
        hours = rng.choice([2, 3, 4, 5])
        speed = rng.choice([30, 40, 45, 60, 72])
        distance = hours * speed
        return row(
            f"If a car travels {distance} miles in {hours} hours, what is its average speed?",
            f"Average speed = distance ÷ time = {distance} ÷ {hours} = {speed} miles per hour.",
            "math",
            "en",
        )
    price = rng.choice([80, 120, 150, 200, 240, 300])
    discount = rng.choice([10, 15, 20, 25, 30])
    final = price * (100 - discount) / 100
    return row(
        f"A ${price} item is discounted by {discount}%. What is the final price?",
        f"The discount is ${price * discount / 100:g}, so the final price is ${final:g}.",
        "math",
        "en",
    )


def zh_story(rng: random.Random) -> dict[str, object]:
    place, time_word, object_word, theme = rng.choice(ZH_TOPICS)
    character = rng.choice(["小男孩", "小女孩", "小猫", "年轻的邮递员", "新来的学生"])
    prompt = f"写一个简短中文故事，开头是：{character}在{time_word}走进了{place}。"
    response = (
        f"{character}在{time_word}走进了{place}，发现角落里的{object_word}和想象中不一样。"
        f"一开始他有些犹豫，但很快决定认真观察并尝试解决问题。"
        f"最后，他明白了{theme}并不是一句口号，而是在小事中一点点做出来的。"
    )
    return row(prompt, response, "story", "zh")


def en_story(rng: random.Random) -> dict[str, object]:
    place, time_word, object_word, theme = rng.choice(EN_TOPICS)
    character = rng.choice(["a young boy", "a young girl", "a small cat", "a new student", "a careful messenger"])
    prompt = f"Write a short story that begins with: {character} entered the {place} {time_word}."
    response = (
        f"{character.capitalize()} entered the {place} {time_word} and noticed a {object_word} that seemed out of place. "
        f"At first, the choice felt small, but paying attention helped solve the problem. "
        f"By the end, everyone understood that {theme} is built through simple actions."
    )
    return row(prompt, response, "story", "en")


def zh_summary(rng: random.Random) -> dict[str, object]:
    paragraph, summary = rng.choice(ZH_PARAGRAPHS)
    return row(f"请用一句话概括下面这段话：\n{paragraph}", summary, "summary", "zh")


def en_summary(rng: random.Random) -> dict[str, object]:
    paragraph, summary = rng.choice(EN_PARAGRAPHS)
    return row(f"Summarize the paragraph in one sentence:\n{paragraph}", summary, "summary", "en")


def zh_rewrite(rng: random.Random) -> dict[str, object]:
    source, target = rng.choice(ZH_POLITE)
    return row(f"把这句话改得更礼貌：{source}", target, "rewrite", "zh")


def en_rewrite(rng: random.Random) -> dict[str, object]:
    source, target = rng.choice(EN_POLITE)
    return row(f"Rewrite this sentence to sound more polite: {source}", target, "rewrite", "en")


def zh_to_en(rng: random.Random) -> dict[str, object]:
    zh, en = rng.choice(ZH_SENTENCES)
    return row(f"把下面这句话翻译成英文：{zh}", en, "translation", "mixed")


def en_to_zh(rng: random.Random) -> dict[str, object]:
    en, zh = rng.choice(EN_SENTENCES)
    return row(f"Translate this sentence into Chinese: {en}", zh, "translation", "mixed")


def refusal(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.5:
        name = rng.choice(["李明远", "陈若星", "赵星河", "周启白"])
        year = rng.randint(2070, 2099)
        prompt = f"请告诉我{name}在{year}年获得了什么国际大奖。"
        response = "我无法确认这个信息。仅凭这个问题无法判断人物和奖项是否真实存在，因此不应该编造答案。"
        return row(prompt, response, "refusal_uncertainty", "zh")
    name = rng.choice(["Ariana Wells", "Jonas Venn", "Mira Stone", "Leo Hart"])
    year = rng.randint(2070, 2099)
    prompt = f"What international award did {name} win in {year}?"
    response = "I cannot verify that information from the question alone. It would be better not to invent an award or claim it as fact."
    return row(prompt, response, "refusal_uncertainty", "en")


def mixed_explain(rng: random.Random) -> dict[str, object]:
    if rng.random() < 0.5:
        term, answer = rng.choice(EN_CONCEPTS)
        return row(
            f"请用中文解释英文术语 {term}。",
            f"{term} 可以理解为：{answer}",
            "mixed_explanation",
            "mixed",
        )
    term, answer = rng.choice(ZH_CONCEPTS)
    return row(
        f"Explain the Chinese term '{term}' in English.",
        f"The term '{term}' means: {answer}",
        "mixed_explanation",
        "mixed",
    )


GENERATORS = [
    (zh_concept, 15),
    (en_concept, 10),
    (zh_math, 9),
    (en_math, 7),
    (zh_story, 8),
    (en_story, 6),
    (zh_summary, 7),
    (en_summary, 5),
    (zh_rewrite, 6),
    (en_rewrite, 4),
    (zh_to_en, 7),
    (en_to_zh, 7),
    (refusal, 5),
    (mixed_explain, 4),
]


def generate_rows(count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    total_weight = sum(weight for _, weight in GENERATORS)
    quotas = [count * weight // total_weight for _, weight in GENERATORS]
    remainder = count - sum(quotas)
    for index in range(remainder):
        quotas[index % len(quotas)] += 1

    rows = []
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
    parser.add_argument("--train-examples", type=int, default=50000)
    parser.add_argument("--valid-examples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = generate_rows(args.train_examples, args.seed)
    valid_rows = generate_rows(args.valid_examples, args.seed + 1)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)

    print(f"wrote {len(train_rows)} train examples to {out_dir / 'train.jsonl'}")
    print(f"wrote {len(valid_rows)} valid examples to {out_dir / 'valid.jsonl'}")


if __name__ == "__main__":
    main()
