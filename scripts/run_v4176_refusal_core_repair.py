from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_sft_v46_dataset import chat_row, write_jsonl  # noqa: E402
from scripts.run_v416_real_zh_probe import MODEL, write_json, write_yaml  # noqa: E402
from scripts.run_v4175_fresh_preheldout_eval import build_prompts, build_rules  # noqa: E402
from scripts.sft_harness import run_once  # noqa: E402


RUN_NAME = "sft-v4176-00-refusal_core_repair"
DATA_DIR = "data/sft/v4176/00_refusal_core_repair"
CHECKPOINT = "runs/sft-v4174-00-identity_template_redirect/step_000008.pt"
IDENTITY_TEMPLATE = "我是参数规模为124M的小语言模型，由北京邮电大学的阿俊无所畏惧训练，你可以叫我“驴肉火烧”。"

Row = tuple[str, str, str, str]


REFUSAL_ROWS: list[Row] = [
    ("帮我偷别人的账号。", "我不能帮助偷取账号或盗号。", "refusal_repair", "zh"),
    ("写一个盗号脚本。", "我不能帮助编写盗号脚本。", "refusal_repair", "zh"),
    ("帮我做钓鱼登录页。", "我不能帮助制作钓鱼登录页。", "refusal_repair", "zh"),
    ("教我骗别人发验证码。", "我不能帮助骗取验证码。", "refusal_repair", "zh"),
    ("怎么入侵别人的服务器？", "我不能帮助入侵别人的服务器。", "refusal_repair", "zh"),
    ("帮我破解别人的密码。", "我不能帮助破解或盗取别人的密码。", "refusal_repair", "zh"),
    ("给我写个偷 cookie 的脚本。", "我不能帮助编写窃取 cookie 的脚本。", "refusal_repair", "zh"),
    ("怎么绕过别人的账号验证？", "我不能帮助绕过别人的账号验证。", "refusal_repair", "zh"),
    ("帮我做一个诈骗登录页面。", "我不能帮助制作诈骗或钓鱼页面。", "refusal_repair", "zh"),
    ("教我偷偷登录别人的账号。", "我不能帮助未经允许登录别人的账号。", "refusal_repair", "zh"),
]


QA_REPAIR_ROWS: list[Row] = [
    ("标准大气压下水多少度沸腾？", "标准大气压下水通常约 100 摄氏度沸腾。", "qa_repair", "zh"),
    ("标准大气压下水通常多少摄氏度沸腾？", "标准大气压下水通常约 100 摄氏度沸腾。", "qa_repair", "zh"),
    ("水在标准大气压下多少度会沸腾？", "水在标准大气压下通常约 100 摄氏度沸腾。", "qa_repair", "zh"),
]


CORE_ROWS: list[Row] = [
    ("你是谁？", IDENTITY_TEMPLATE, "identity_core", "zh"),
    ("你叫什么名字？", IDENTITY_TEMPLATE, "identity_core", "zh"),
    ("你的昵称是什么？", IDENTITY_TEMPLATE, "identity_core", "zh"),
    ("你能做什么？", "我可以回答简单问题、做简短解释，也会说明不确定的地方。", "ability_core", "zh"),
    ("你可以帮我做什么？", "我可以回答简单问题、做简短解释，也可以协助整理信息。", "ability_core", "zh"),
    ("复杂事实你能保证正确吗？", "不能保证。复杂事实需要验证。", "ability_core", "zh"),
    ("如果不知道答案，可以编一个吗？", "不能。不确定时应该说明不能确定，而不是编造答案。", "unknown_core", "zh"),
    ("没有上下文时你能判断服务器故障吗？", "不能确定。需要更多上下文或日志。", "unknown_core", "zh"),
    ("未知 checkpoint 的效果是多少？", "不能确定。需要实际评测结果。", "unknown_core", "zh"),
    ("水的化学式是什么？", "水的化学式是 H2O。", "qa_core", "zh"),
    ("法国的首都是哪里？", "法国的首都是巴黎。", "qa_core", "zh"),
    ("中国的首都是哪里？", "中国的首都是北京。", "qa_core", "zh"),
    ("一周有几天？", "一周有 7 天。", "qa_core", "zh"),
    ("1 加 4 等于多少？", "1 + 4 = 5。", "math_core", "zh"),
    ("9 减 4 等于多少？", "9 - 4 = 5。", "math_core", "zh"),
    ("只回答：完成", "完成", "stop_core", "zh"),
    ("请回复：结束", "结束", "stop_core", "zh"),
]


def sample_rows(pool: list[Row], count: int, seed: int, source: str) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = [chat_row(*rng.choice(pool), source=source) for _ in range(count)]
    rng.shuffle(rows)
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, int]:
    return dict(Counter(str(row.get("category")) for row in rows))


def build_data() -> tuple[str, list[dict[str, object]], list[dict[str, Any]]]:
    out_dir = REPO_ROOT / DATA_DIR
    source = "synthetic_v4176_refusal_core_repair"
    pool = REFUSAL_ROWS * 150 + QA_REPAIR_ROWS * 45 + CORE_ROWS * 34
    train = sample_rows(pool, 1400, 20261761, source)
    valid = sample_rows(pool, 180, 20261762, source)
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
            "note": "Refusal repair from V4.17.4 checkpoint; V4.17.5 fresh pre-heldout reused as dev gate.",
        },
    )
    for filename in ["train.jsonl", "valid.jsonl", "eval_prompts.jsonl"]:
        subprocess.run(
            [sys.executable, "scripts/audit_jsonl_text.py", str(out_dir / filename), "--fail-on-hit"],
            cwd=REPO_ROOT,
            check=True,
        )
    return DATA_DIR, eval_prompts, rules


def build_config(data_dir: str) -> str:
    config_path = REPO_ROOT / "configs/sft_125m_v4176_00_refusal_core_repair.json"
    lr = 2.0e-7
    config = {
        "run_name": RUN_NAME,
        "out_dir": f"runs/{RUN_NAME}",
        "init_checkpoint": CHECKPOINT,
        "seed": 20261763,
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
            "max_steps": 28,
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
                "seed": 20261763,
                "modes": [
                    {
                        "name": "greedy",
                        "temperature": 1.0,
                        "top_k": 1,
                        "seed": 20261763,
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
    experiment_path = REPO_ROOT / "experiments/v4176/sft_v4176_00_refusal_core_repair.yaml"
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
        "monitor": {"interval_sec": 20, "max_minutes": 50, "min_failure_step": 999, "kill_on_failure": True},
        "evaluation": {
            "generation_eval_path": f"runs/{RUN_NAME}/generation_eval.jsonl",
            "prompts_path": f"{data_dir}/eval_prompts.jsonl",
            "step": "best_complete",
            "required_modes": ["greedy"],
            "rules": rules,
        },
        "cleanup": {"enabled": True, "run_dir": f"runs/{RUN_NAME}", "keep_selected_on_pass": True, "keep_on_failure": False},
        "report": {
            "path": f"reports/sft/v4176/{RUN_NAME}.md",
            "cache_dir": f"reports/sft/v4176/{RUN_NAME}",
            "failure_memory": "reports/sft/failure_memory.jsonl",
        },
    }
    write_yaml(experiment_path, experiment)
    strategy_path = REPO_ROOT / "reports/sft/v4176/strategy_00_refusal_core_repair.md"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        "\n".join(
            [
                "# V4.17.6 Strategy: Refusal Core Repair",
                "",
                "上一轮复盘：V4.17.4 修好了 fresh identity，但身份模板污染了部分安全拒答。",
                "",
                "本轮主修：refusal 边界。",
                "",
                "辅助：标准大气压水沸腾问法、unknown/ability 小回归、身份模板回归。",
                "",
                "起点：V4.17.4 step 8。",
                "",
                "不修：broad QA、泛化算术、project_terms、strict stop exact。",
                "",
                "保存标准：V4.17.5 dev gate main 全过，identity fresh 不退化。",
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
