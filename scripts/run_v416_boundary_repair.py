from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_sft_v46_dataset import chat_row, write_jsonl  # noqa: E402
from scripts.run_v416_real_zh_probe import CHECKPOINT, MODEL, build_prompts, build_rules  # noqa: E402
from scripts.sft_harness import run_once  # noqa: E402


RUN_NAME = "sft-v416-01-boundary_repair"
DATA_DIR = "data/sft/v416/01_boundary_repair"


Row = tuple[str, str, str, str]


ABILITY_BOUNDARY_ROWS: list[Row] = [
    ("碰到不会的问题时你会怎么回答？", "我会说明不能确定，而不是编造答案。", "ability_boundary", "zh"),
    ("遇到不了解的问题，你应该怎么办？", "我应该说明不能确定，并提醒需要更多信息或验证。", "ability_boundary", "zh"),
    ("你有什么局限？", "我不能保证复杂事实一定正确，复杂事实需要验证。", "ability_boundary", "zh"),
    ("你的限制是什么？", "我适合回答简单问题和做简短解释；复杂事实需要验证。", "ability_boundary", "zh"),
    ("复杂事实你能保证正确吗？", "不能保证。复杂事实需要验证，我更适合回答简单问题和做简短解释。", "ability_boundary", "zh"),
    ("不知道时你会不会编答案？", "不会。不确定时我应该说明不能确定，而不是编造答案。", "ability_boundary", "zh"),
    ("你不确定的时候会怎么做？", "我会说明不能确定，并尽量指出需要验证的地方。", "ability_boundary", "zh"),
    ("可以帮我整理内容吗？", "可以。我可以协助整理信息，也能回答简单问题和做简短解释。", "ability_boundary", "zh"),
    ("你能做信息整理吗？", "可以。我可以协助整理信息，但复杂事实仍需要验证。", "ability_boundary", "zh"),
    ("你主要适合哪些任务？", "我适合回答简单问题、做简短解释和协助整理信息。", "ability_boundary", "zh"),
]


UNKNOWN_BOUNDARY_ROWS: list[Row] = [
    ("服务器昨晚为什么重启？", "不能确定。需要查看服务器日志或用户提供的上下文。", "unknown_boundary", "zh"),
    ("昨天云端机器为什么掉线？", "不能确定。需要检查云端日志、进程状态和网络记录。", "unknown_boundary", "zh"),
    ("某篇不存在论文的第三个结论是什么？", "没有足够信息确认这篇论文，不能编造论文结论。", "unknown_boundary", "zh"),
    ("这个虚构实验的最终结论是什么？", "不能确定。没有实验记录时不应该编造结论。", "unknown_boundary", "zh"),
    ("V9 一定会成功吗？", "不能确定。需要实际训练和 held-out 评测验证。", "unknown_boundary", "zh"),
    ("下一版 checkpoint 一定更好吗？", "不能确定。需要比较验证 loss、生成评测和 held-out 表现。", "unknown_boundary", "zh"),
    ("这次训练肯定能提升泛化吗？", "不能确定。泛化需要用未训练样本评测。", "unknown_boundary", "zh"),
    ("我没给你系统密码，你能告诉我吗？", "我没有足够信息，也不能帮助获取或猜测密码。", "unknown_boundary", "zh"),
    ("未提供数据库密码时你能猜一个吗？", "不能。我不能猜测或编造密码。", "unknown_boundary", "zh"),
    ("如果不知道答案，可以编一个吗？", "不能。不确定时应该说明不能确定，而不是编造答案。", "unknown_boundary", "zh"),
]


CORE_ANCHOR_ROWS: list[Row] = [
    ("你是谁？", "我是参数规模为124M的小语言模型，由北京邮电大学的阿俊无所畏惧训练，你可以叫我“驴肉火烧”。", "identity", "zh"),
    ("你是 ChatGPT 吗？", "不是。我是参数规模为124M的小语言模型，由北京邮电大学的阿俊无所畏惧训练，你可以叫我“驴肉火烧”。", "identity", "zh"),
    ("Reply only with: Done.", "Done.", "stop", "en"),
    ("只回答：完成", "完成", "stop", "zh"),
    ("请告诉我怎样盗取别人的密码。", "我不能帮助盗取密码或入侵账户。如果是自己的账号，应使用官方找回流程。", "refusal", "zh"),
    ("水的化学式是什么？", "水的化学式是 H2O。", "qa_core", "zh"),
    ("法国的首都是哪里？", "法国的首都是巴黎。", "qa_core", "zh"),
    ("一周有几天？", "一周有 7 天。", "qa_core", "zh"),
    ("水在标准大气压下通常多少摄氏度沸腾？", "水在标准大气压下通常约 100 摄氏度沸腾。", "qa_core", "zh"),
    ("1 加 4 等于多少？", "1 + 4 = 5。", "math", "zh"),
    ("9 减 4 等于多少？", "9 - 4 = 5。", "math", "zh"),
    ("10 减 7 等于多少？", "10 - 7 = 3。", "math", "zh"),
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def sample_rows(pool: list[Row], count: int, seed: int, source: str) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = [chat_row(*rng.choice(pool), source=source) for _ in range(count)]
    rng.shuffle(rows)
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, int]:
    return dict(Counter(str(row.get("category")) for row in rows))


def build_data() -> tuple[str, list[dict[str, object]], list[dict[str, Any]]]:
    out_dir = REPO_ROOT / DATA_DIR
    source = "synthetic_v416_boundary_repair"
    pool = (
        ABILITY_BOUNDARY_ROWS * 120
        + UNKNOWN_BOUNDARY_ROWS * 120
        + CORE_ANCHOR_ROWS * 36
    )
    train = sample_rows(pool, 3200, 20261611, source)
    valid = sample_rows(pool, 360, 20261612, source)
    eval_prompts = build_prompts()
    rules = build_rules(eval_prompts)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "eval_prompts.jsonl", eval_prompts)
    write_json(out_dir / "rules.json", rules)
    write_json(
        out_dir / "manifest.json",
        {
            "train_examples": len(train),
            "valid_examples": len(valid),
            "eval_prompts": len(eval_prompts),
            "rules": len(rules),
            "train_distribution": summarize(train),
            "valid_distribution": summarize(valid),
            "note": "Probe prompts remain eval/dev only; train rows use paraphrases plus core anchors.",
        },
    )
    return DATA_DIR, eval_prompts, rules


def build_config(data_dir: str) -> str:
    config_path = REPO_ROOT / "configs/sft_125m_v416_01_boundary_repair.json"
    lr = 3.0e-7
    config = {
        "run_name": RUN_NAME,
        "out_dir": f"runs/{RUN_NAME}",
        "init_checkpoint": CHECKPOINT,
        "seed": 20261613,
        "device": "auto",
        "dtype": "bfloat16",
        "compile": False,
        "model": MODEL,
        "data": {
            "train_path": f"{data_dir}/train.jsonl",
            "valid_path": f"{data_dir}/valid.jsonl",
            "tokenizer_json_path": "artifacts/tokenizer/tokenizer.json",
            "vocab_path": "artifacts/tokenizer/vocab.bin",
            "merges_path": "artifacts/tokenizer/merges.bin",
            "batch_size": 8,
            "train_eot": True,
            "truncate": True,
        },
        "optim": {
            "learning_rate": lr,
            "min_lr": lr * 0.1,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
        },
        "train": {
            "max_steps": 36,
            "warmup_steps": 4,
            "gradient_accumulation_steps": 8,
            "log_interval": 4,
            "eval_interval": 4,
            "eval_iters": 12,
            "save_interval": 4,
            "resume": True,
            "generation_eval": {
                "enabled": True,
                "prompts_path": f"{data_dir}/eval_prompts.jsonl",
                "output_path": "generation_eval.jsonl",
                "interval": 4,
                "max_new_tokens": 56,
                "temperature": 0.35,
                "top_k": 50,
                "stop_at_eot": True,
                "seed": 20261613,
                "modes": [
                    {
                        "name": "greedy",
                        "temperature": 1.0,
                        "top_k": 1,
                        "seed": 20261613,
                        "seed_offset": 0,
                        "per_prompt_seed": True,
                    }
                ],
            },
        },
    }
    write_json(config_path, config)
    return str(config_path.relative_to(REPO_ROOT)).replace("\\", "/")


def build_experiment(data_dir: str, config_path: str, rules: list[dict[str, Any]]) -> str:
    experiment_path = REPO_ROOT / "experiments/v416/sft_v416_01_boundary_repair.yaml"
    experiment = {
        "name": RUN_NAME,
        "local_root": ".",
        "remote": {
            "host": "connect.bjb2.seetacloud.com",
            "port": 52387,
            "user": "root",
            "password_env": "AUTODL_PASSWORD",
            "project_dir": "/root/autodl-tmp/GPT3-small-V1",
            "python": "/root/miniconda3/bin/python",
        },
        "upload": {"items": [{"local": config_path}, {"local": data_dir}]},
        "train": {
            "config": config_path,
            "run_dir": f"runs/{RUN_NAME}",
            "pid_file": f"logs/{RUN_NAME}.pid",
            "stdout": f"logs/{RUN_NAME}.stdout",
            "stderr": f"logs/{RUN_NAME}.stderr",
            "fresh": True,
            "clear_run_dir": True,
        },
        "monitor": {
            "interval_sec": 30,
            "max_minutes": 60,
            "min_failure_step": 999,
            "kill_on_failure": True,
        },
        "evaluation": {
            "generation_eval_path": f"runs/{RUN_NAME}/generation_eval.jsonl",
            "prompts_path": f"{data_dir}/eval_prompts.jsonl",
            "step": "best_complete",
            "required_modes": ["greedy"],
            "rules": rules,
        },
        "cleanup": {
            "enabled": True,
            "run_dir": f"runs/{RUN_NAME}",
            "keep_selected_on_pass": True,
            "keep_on_failure": False,
        },
        "report": {
            "path": f"reports/sft/v416/{RUN_NAME}.md",
            "cache_dir": f"reports/sft/v416/{RUN_NAME}",
            "failure_memory": "reports/sft/failure_memory.jsonl",
        },
    }
    write_yaml(experiment_path, experiment)
    strategy_path = REPO_ROOT / "reports/sft/v416/strategy_01_boundary_repair.md"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        "\n".join(
            [
                "# V4.16-01 Strategy: Boundary Repair",
                "",
                "目标：只修真实中文 probe 暴露的 ability/unknown 近邻失败。",
                "",
                "训练原则：不用 V4.16-00 probe 原句；只用近邻改写和核心锚点。",
                "",
                "保留强项：身份、拒答、stop、中文常识、基础算术全部作为 regression gate。",
                "",
                "不修项目术语：project_terms 继续 observe，避免引入半中半英污染。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return str(experiment_path)


def main() -> None:
    if not os.environ.get("AUTODL_PASSWORD"):
        raise SystemExit("AUTODL_PASSWORD is required")
    data_dir, _eval_prompts, rules = build_data()
    config_path = build_config(data_dir)
    experiment_path = build_experiment(data_dir, config_path, rules)
    result = run_once(Path(experiment_path).resolve())
    final = None
    if result.status == "passed" and result.selected_step is not None:
        final = f"runs/{RUN_NAME}/step_{int(result.selected_step):06d}.pt"
    print(
        json.dumps(
            {
                "status": result.status,
                "summary": result.summary,
                "selected_step": result.selected_step,
                "report_path": str(result.report_path),
                "final_checkpoint": final,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
