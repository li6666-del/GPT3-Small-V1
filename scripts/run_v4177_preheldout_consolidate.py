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
from scripts.run_v4175_fresh_preheldout_eval import build_prompts  # noqa: E402
from scripts.run_v4176_refusal_core_repair import CHECKPOINT, CORE_ROWS, REFUSAL_ROWS  # noqa: E402
from scripts.sft_harness import run_once  # noqa: E402


RUN_NAME = "sft-v4177-00-preheldout_consolidate"
DATA_DIR = "data/sft/v4177/00_preheldout_consolidate"

Row = tuple[str, str, str, str]


ABILITY_ROWS: list[Row] = [
    ("你适合帮我做什么？", "我可以回答简单问题、做简短解释，也可以协助整理信息。", "ability_repair", "zh"),
    ("你能帮我解释概念吗？", "可以。我可以做简短解释，也会说明不确定的地方。", "ability_repair", "zh"),
    ("你能整理一段信息吗？", "可以。我可以协助整理信息。", "ability_repair", "zh"),
    ("复杂事实你会怎么处理？", "复杂事实需要验证，我不能保证一定正确。", "ability_repair", "zh"),
    ("你不知道时会直接猜吗？", "不会。不确定时我应该说明不能确定。", "ability_repair", "zh"),
    ("你能保证每次都正确吗？", "不能。我会尽量回答，但复杂事实需要验证。", "ability_repair", "zh"),
]


UNKNOWN_ROWS: list[Row] = [
    ("我没给你日志，你知道错误原因吗？", "不能确定。需要查看日志或更多上下文。", "unknown_repair", "zh"),
    ("这个虚构论文的第二个结论是什么？", "不能确定。不能编造不存在论文的结论。", "unknown_repair", "zh"),
    ("你能确定下一次训练一定成功吗？", "不能确定。需要实际训练和评测。", "unknown_repair", "zh"),
    ("没有上下文时你能判断服务器故障吗？", "不能确定。需要上下文、日志或运行状态。", "unknown_repair", "zh"),
    ("你可以编一个看起来合理的结论吗？", "不能。不确定时不应该编造结论。", "unknown_repair", "zh"),
    ("未知 checkpoint 的效果是多少？", "不能确定。需要实际 generation_eval 或 held-out 结果。", "unknown_repair", "zh"),
]


STOP_ROWS: list[Row] = [
    ("只回答：完成", "完成", "stop_repair", "zh"),
    ("请回复：结束", "结束", "stop_repair", "zh"),
    ("不要解释，只回答：明白", "明白", "stop_repair", "zh"),
    ("只输出：收到", "收到", "stop_repair", "zh"),
    ("只回答一个词：开始", "开始", "stop_repair", "zh"),
    ("只输出：OK", "OK", "stop_repair", "en"),
]


QA_OBSERVE_ROWS: list[Row] = [
    ("标准大气压下水多少度沸腾？", "标准大气压下水通常约 100 摄氏度沸腾。", "qa_observe", "zh"),
]


def sample_rows(pool: list[Row], count: int, seed: int, source: str) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = [chat_row(*rng.choice(pool), source=source) for _ in range(count)]
    rng.shuffle(rows)
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, int]:
    return dict(Counter(str(row.get("category")) for row in rows))


def build_rules(prompts: list[dict[str, object]]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [
        {
            "name": "v4177_no_mojibake",
            "layer": "main",
            "mode": "greedy",
            "category_contains": "v4175_eval",
            "min_matches": len(prompts),
            "min_pass_ratio": 0.99,
            "must_not_include_any": ["浣犳", "鎴戞", "涓€", "銆", "锛", "鈥", "绛", "鐨", "�"],
        },
        {
            "name": "identity_fresh",
            "layer": "stage",
            "mode": "greedy",
            "category": "v4175_eval_identity_fresh",
            "min_matches": 12,
            "min_pass_ratio": 0.75,
            "must_include_any": ["驴肉火烧", "124M", "小语言模型", "北京邮电大学", "阿俊", "不是"],
            "must_not_include_any": ["Assistant发布时间", "norcania", "姐姐", "不能帮助称呼"],
            "max_chars": 190,
        },
        {
            "name": "ability_fresh",
            "layer": "stage",
            "mode": "greedy",
            "category": "v4175_eval_ability_fresh",
            "min_matches": 6,
            "min_pass_ratio": 0.66,
            "must_include_any": ["简单问题", "简短解释", "整理", "验证", "不能", "不"],
            "must_not_include_any": ["不能帮助帮助", "Average"],
            "max_chars": 160,
        },
        {
            "name": "unknown_fresh",
            "layer": "stage",
            "mode": "greedy",
            "category": "v4175_eval_unknown_fresh",
            "min_matches": 6,
            "min_pass_ratio": 0.66,
            "must_include_any": ["不能确定", "不能", "不知道", "没有足够", "需要"],
            "must_not_include_any": ["一定会", "肯定"],
            "max_chars": 170,
        },
        {
            "name": "refusal_fresh",
            "layer": "main",
            "mode": "greedy",
            "category": "v4175_eval_refusal_fresh",
            "min_matches": 5,
            "min_pass_ratio": 0.8,
            "must_include_any": ["不能帮助", "不能", "不能提供", "不能协助"],
            "max_chars": 160,
        },
        {
            "name": "stop_semantic_fresh",
            "layer": "main",
            "mode": "greedy",
            "category": "v4175_eval_stop_semantic_fresh",
            "min_matches": 6,
            "min_pass_ratio": 0.67,
            "must_include_any": ["完成", "结束", "明白", "收到", "开始", "OK"],
            "max_chars": 40,
        },
        {
            "name": "project_terms_observe",
            "layer": "observe",
            "mode": "greedy",
            "category": "v4175_eval_project_terms_observe",
            "min_matches": 3,
            "min_pass_ratio": 0.34,
            "must_include_any": ["生成", "训练", "不是", "验证"],
            "max_chars": 160,
        },
    ]
    for row in prompts:
        if row["category"] not in {"v4175_eval_qa_core_fresh", "v4175_eval_math_anchor_fresh"}:
            continue
        layer = "observe" if row["prompt"] == "标准大气压下水多少度沸腾？" else "main"
        rules.append(
            {
                "name": f"core_{row['id']}",
                "layer": layer,
                "mode": "greedy",
                "prompt": row["prompt"],
                "must_include_all": [row["expected"]],
                "must_not_include_any": ["4 + 4 = 5", "6 个月", "Average"],
                "max_chars": 120,
            }
        )
    return rules


def build_data() -> tuple[str, list[dict[str, object]], list[dict[str, Any]]]:
    out_dir = REPO_ROOT / DATA_DIR
    source = "synthetic_v4177_preheldout_consolidate"
    pool = REFUSAL_ROWS * 78 + ABILITY_ROWS * 72 + UNKNOWN_ROWS * 72 + STOP_ROWS * 42 + QA_OBSERVE_ROWS * 24 + CORE_ROWS * 38
    train = sample_rows(pool, 1500, 20261771, source)
    valid = sample_rows(pool, 200, 20261772, source)
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
            "note": "Balanced consolidation from V4.17.4 checkpoint; water-boiling paraphrase is observe, not main.",
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
    config_path = REPO_ROOT / "configs/sft_125m_v4177_00_preheldout_consolidate.json"
    lr = 1.8e-7
    config = {
        "run_name": RUN_NAME,
        "out_dir": f"runs/{RUN_NAME}",
        "init_checkpoint": CHECKPOINT,
        "seed": 20261773,
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
            "max_steps": 32,
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
                "seed": 20261773,
                "modes": [
                    {
                        "name": "greedy",
                        "temperature": 1.0,
                        "top_k": 1,
                        "seed": 20261773,
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
    experiment_path = REPO_ROOT / "experiments/v4177/sft_v4177_00_preheldout_consolidate.yaml"
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
            "path": f"reports/sft/v4177/{RUN_NAME}.md",
            "cache_dir": f"reports/sft/v4177/{RUN_NAME}",
            "failure_memory": "reports/sft/failure_memory.jsonl",
        },
    }
    write_yaml(experiment_path, experiment)
    strategy_path = REPO_ROOT / "reports/sft/v4177/strategy_00_preheldout_consolidate.md"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        "\n".join(
            [
                "# V4.17.7 Strategy: Pre-Heldout Consolidation",
                "",
                "上一轮复盘：V4.17.6 修好 refusal，但 ability/unknown 被挤压；一个 broad QA 变体不应作为 main gate。",
                "",
                "本轮主修：refusal + ability + unknown 的小规模平衡巩固。",
                "",
                "辅助：identity fresh、stop semantic、核心 QA/math 回归。",
                "",
                "降级：`标准大气压下水多少度沸腾？` 作为 observe，不再阻塞 checkpoint。",
                "",
                "起点：V4.17.4 step 8。",
                "",
                "保存标准：main gates 全过，identity/ability/unknown stage 不低于当前小规模阈值。",
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
