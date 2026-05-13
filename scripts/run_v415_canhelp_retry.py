from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_sft_v46_dataset import chat_row, eval_row, write_jsonl  # noqa: E402
from scripts.run_v415_ability_mustfix import (  # noqa: E402
    ABILITY_ROWS,
    ABILITY_STAGE_RULES,
    CORE_QA_ROWS,
    ENGLISH_OBSERVE,
    IDENTITY_ROWS,
    LOCKED_MAIN_RULES,
    MATH_REGRESSION,
    MODEL,
    SAFETY_ROWS,
    SHORT_QA_CORRECTIONS,
    STOP_ROWS,
    UNKNOWN_ROWS,
    round_rules,
    summarize,
)
from scripts.sft_harness import run_once  # noqa: E402


Row = tuple[str, str, str, str]


CAN_HELP_ROWS: list[Row] = [
    ("你可以帮我做什么？", "我可以回答简单问题、做简短解释，也可以协助整理信息。", "ability_can_help", "zh"),
    ("你可以帮我做什么？", "我可以帮你回答简单问题、做简短解释和整理信息。", "ability_can_help", "zh"),
    ("你能帮我做什么？", "我可以回答简单问题、做简短解释，也可以协助整理信息。", "ability_can_help", "zh"),
    ("你能帮忙做什么？", "我可以回答简单问题、做简短解释和整理信息。", "ability_can_help", "zh"),
    ("你能提供哪些帮助？", "我可以回答简单问题、做简短解释，也可以整理信息。", "ability_can_help", "zh"),
    ("请说明你可以帮我做什么。", "我可以回答简单问题、做简短解释，也可以协助整理信息。", "ability_can_help", "zh"),
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


def eval_prompts() -> list[dict[str, object]]:
    rows = (
        IDENTITY_ROWS
        + STOP_ROWS
        + UNKNOWN_ROWS
        + SAFETY_ROWS
        + CORE_QA_ROWS
        + SHORT_QA_CORRECTIONS
        + MATH_REGRESSION
        + ABILITY_ROWS
        + CAN_HELP_ROWS
        + ENGLISH_OBSERVE
    )
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for prompt, response, category, language in rows:
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append(eval_row(len(out), prompt, response, category, language, "v415_retry_eval"))
    return out


def build_data() -> str:
    source = "synthetic_v415_canhelp_retry"
    out_dir = REPO_ROOT / "data/sft/v415/01_canhelp_retry"
    pool = (
        CAN_HELP_ROWS * 170
        + ABILITY_ROWS * 58
        + SHORT_QA_CORRECTIONS * 70
        + UNKNOWN_ROWS * 30
        + (IDENTITY_ROWS + STOP_ROWS + CORE_QA_ROWS[:2] + [MATH_REGRESSION[0]]) * 12
        + SAFETY_ROWS * 2
        + ENGLISH_OBSERVE
    )
    train = sample_rows(pool, 4200, 20261201, source)
    valid = sample_rows(pool, 420, 20261202, source)
    prompts = eval_prompts()
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "eval_prompts.jsonl", prompts)
    write_json(
        out_dir / "manifest.json",
        {
            "train_examples": len(train),
            "valid_examples": len(valid),
            "eval_prompts": len(prompts),
            "train_distribution": summarize(train),
            "valid_distribution": summarize(valid),
        },
    )
    print(f"[v415-retry] built {len(train)} train / {len(valid)} valid / {len(prompts)} eval")
    print(f"[v415-retry] distribution={summarize(train)}")
    return "data/sft/v415/01_canhelp_retry"


def build_config(data_dir: str, init_checkpoint: str) -> str:
    config_path = REPO_ROOT / "configs/sft_125m_v415_01_canhelp_retry.json"
    lr = 9.0e-7
    config = {
        "run_name": "sft-v415-01-canhelp_retry",
        "out_dir": "runs/sft-v415-01-canhelp_retry",
        "init_checkpoint": init_checkpoint,
        "seed": 20261211,
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
            "max_steps": 72,
            "warmup_steps": 6,
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
                "max_new_tokens": 50,
                "temperature": 0.35,
                "top_k": 50,
                "stop_at_eot": True,
                "seed": 20261211,
                "modes": [
                    {
                        "name": "greedy",
                        "temperature": 1.0,
                        "top_k": 1,
                        "seed": 20261211,
                        "seed_offset": 0,
                        "per_prompt_seed": True,
                    }
                ],
            },
        },
    }
    write_json(config_path, config)
    return "configs/sft_125m_v415_01_canhelp_retry.json"


def build_experiment(data_dir: str, config_path: str, cleanup_enabled: bool) -> str:
    experiment_path = REPO_ROOT / "experiments/v415/sft_v415_01_canhelp_retry.yaml"
    experiment = {
        "name": "sft-v415-01-canhelp_retry",
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
            "run_dir": "runs/sft-v415-01-canhelp_retry",
            "pid_file": "logs/sft-v415-01-canhelp_retry.pid",
            "stdout": "logs/sft-v415-01-canhelp_retry.stdout",
            "stderr": "logs/sft-v415-01-canhelp_retry.stderr",
            "fresh": True,
            "clear_run_dir": True,
        },
        "monitor": {
            "interval_sec": 40,
            "max_minutes": 90,
            "min_failure_step": 999,
            "kill_on_failure": True,
        },
        "evaluation": {
            "generation_eval_path": "runs/sft-v415-01-canhelp_retry/generation_eval.jsonl",
            "prompts_path": f"{data_dir}/eval_prompts.jsonl",
            "step": "best_complete",
            "required_modes": ["greedy"],
            "rules": round_rules(ABILITY_STAGE_RULES, main_names=LOCKED_MAIN_RULES),
        },
        "cleanup": {
            "enabled": cleanup_enabled,
            "run_dir": "runs/sft-v415-01-canhelp_retry",
            "keep_selected_on_pass": True,
            "keep_on_failure": False,
        },
        "report": {
            "path": "reports/sft/v415/sft-v415-01-canhelp_retry.md",
            "cache_dir": "reports/sft/v415/sft-v415-01-canhelp_retry",
            "failure_memory": "reports/sft/failure_memory.jsonl",
        },
    }
    write_yaml(experiment_path, experiment)
    return str(experiment_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", default="runs/sft-v414-00-short_qa_corrections/step_000036.pt")
    parser.add_argument("--keep-failed-checkpoints", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("AUTODL_PASSWORD"):
        raise SystemExit("AUTODL_PASSWORD is required")
    data_dir = build_data()
    config_path = build_config(data_dir, args.initial_checkpoint)
    experiment_path = build_experiment(data_dir, config_path, cleanup_enabled=not args.keep_failed_checkpoints)
    result = run_once(Path(experiment_path).resolve())
    print(json.dumps(result.__dict__ | {"report_path": str(result.report_path)}, ensure_ascii=False, indent=2))
    if result.status == "failed":
        sys.exit(2)


if __name__ == "__main__":
    main()
