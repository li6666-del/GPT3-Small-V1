from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v416_real_zh_probe import MODEL, write_json, write_yaml  # noqa: E402
from scripts.run_v4176_refusal_core_repair import CHECKPOINT  # noqa: E402
from scripts.run_v4177_preheldout_consolidate import build_data  # noqa: E402
from scripts.sft_harness import run_once  # noqa: E402


RUN_NAME = "sft-v4178-00-preheldout_mainline_gate"


def downgrade_ability_to_observe(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for rule in rules:
        item = dict(rule)
        if item.get("name") == "ability_fresh":
            item["layer"] = "observe"
        adjusted.append(item)
    return adjusted


def build_config(data_dir: str) -> str:
    config_path = REPO_ROOT / "configs/sft_125m_v4178_00_preheldout_mainline_gate.json"
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
    experiment_path = REPO_ROOT / "experiments/v4178/sft_v4178_00_preheldout_mainline_gate.yaml"
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
            "path": f"reports/sft/v4178/{RUN_NAME}.md",
            "cache_dir": f"reports/sft/v4178/{RUN_NAME}",
            "failure_memory": "reports/sft/failure_memory.jsonl",
        },
    }
    write_yaml(experiment_path, experiment)
    strategy_path = REPO_ROOT / "reports/sft/v4178/strategy_00_preheldout_mainline_gate.md"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        "\n".join(
            [
                "# V4.17.8 Strategy: Pre-Heldout Mainline Gate",
                "",
                "上一轮复盘：V4.17.7 main gates 已过，identity/unknown/refusal/stop 基本盘稳定，失败只剩 ability fresh 近邻。",
                "",
                "决策：ability fresh 降为 observe，不再阻塞 checkpoint。",
                "",
                "理由：ability 近邻连续多轮不稳，继续强修会污染主线；保留核心能力说明，不把泛化能力说明当作当前硬门槛。",
                "",
                "起点：V4.17.4 step 8。",
                "",
                "保存标准：main gates 全过，identity/unknown stage 过线，ability 仅记录。",
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
    rules = downgrade_ability_to_observe(rules)
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
